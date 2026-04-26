import os
import logging
import time
import threading
import secrets
import sys
import subprocess
from typing import List, Tuple, Callable, Optional, Dict, Set
from datetime import datetime, date, timedelta
from statistics import mean
from types import SimpleNamespace
from flask import Flask, render_template, redirect, url_for, request, session, flash, abort, jsonify
from sqlalchemy import func, desc, select
from sqlalchemy.orm import sessionmaker, load_only
import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load
from models import (
    Base,
    Match,
    PhysicalAssessment,
    Player,
    Coach,
    Director,
    User,
    PlayerAvailability,
    PlayerStat,
    PlayerAttributeHistory,
    PlayerMatchParticipation,
    ScoutReport,
)
from train_model import PlayerNet
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from db_utils import normalize_db_url, create_app_engine, ensure_player_columns
from preprocessing import (
    aggregate_attribute_history_dataframe,
    aggregate_availability_dataframe,
    aggregate_match_participation_dataframe,
    aggregate_physical_assessment_dataframe,
    aggregate_scout_report_dataframe,
    aggregate_stats_dataframe,
    availability_feature_defaults,
    attribute_history_feature_defaults,
    dataframe_from_players,
    load_preprocessor as load_saved_preprocessor,
    match_feature_defaults,
    physical_feature_defaults,
    player_base_dataframe_from_players,
    preprocessor_input_dim,
    scout_report_feature_defaults,
    stats_feature_defaults,
    transform_features,
)
from player_logic import (
    ATTRIBUTE_FIELDS,
    ATTRIBUTE_LABELS,
    POSITION_CHOICES,
    normalized_position,
    position_weights,
    recommend_position_from_attrs,
    weighted_score_from_attrs,
    is_valid_attribute,
    is_valid_eval_age,
    default_player_photo_url,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Límite de payload para requests (mitiga abusos y errores por uploads grandes)
# Default: 2MB. Ajustable por env var `MAX_CONTENT_LENGTH`.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", str(2 * 1024 * 1024)))


# --- Observabilidad mínima ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
try:
    app.logger.setLevel(LOG_LEVEL)
except Exception:
    app.logger.setLevel(logging.INFO)

# Logger root para librerías (opcional)
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))


# --- Cache in-memory con TTL (MVP) ---
_CACHE: dict = {}
_CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "60"))
_LOGIN_ATTEMPTS: Dict[str, List[float]] = {}
_LOGIN_RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("LOGIN_RATE_LIMIT_WINDOW_SECONDS", "900"))
_LOGIN_RATE_LIMIT_MAX_ATTEMPTS = int(os.environ.get("LOGIN_RATE_LIMIT_MAX_ATTEMPTS", "5"))


# --- Guardrails pipeline (evita doble ejecución concurrente) ---
_PIPELINE_LOCK = threading.Lock()
_LOGIN_ATTEMPT_LOCK = threading.Lock()


def _cache_get(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    expires_at, value = item
    if time.time() >= expires_at:
        _CACHE.pop(key, None)
        return None
    return value

def _cache_set(key: str, value):
    _CACHE[key] = (time.time() + _CACHE_TTL_SECONDS, value)


def _cache_invalidate_prefix(prefix: str) -> None:
    keys = [key for key in _CACHE.keys() if key.startswith(prefix)]
    for key in keys:
        _CACHE.pop(key, None)


def invalidate_dashboard_cache() -> None:
    _cache_invalidate_prefix("dashboard:")


def client_ip() -> str:
    forwarded_for = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    return forwarded_for or request.remote_addr or "unknown"


def login_attempt_key(username: Optional[str]) -> str:
    normalized_username = (username or "").strip().lower() or "-"
    return f"{client_ip()}:{normalized_username}"


def _prune_login_attempts(now_ts: Optional[float] = None) -> None:
    now_ts = now_ts if now_ts is not None else time.time()
    cutoff = now_ts - _LOGIN_RATE_LIMIT_WINDOW_SECONDS
    expired_keys = []
    for key, attempts in _LOGIN_ATTEMPTS.items():
        fresh_attempts = [attempt for attempt in attempts if attempt >= cutoff]
        if fresh_attempts:
            _LOGIN_ATTEMPTS[key] = fresh_attempts
        else:
            expired_keys.append(key)
    for key in expired_keys:
        _LOGIN_ATTEMPTS.pop(key, None)


def is_login_rate_limited(username: Optional[str]) -> bool:
    with _LOGIN_ATTEMPT_LOCK:
        _prune_login_attempts()
        return len(_LOGIN_ATTEMPTS.get(login_attempt_key(username), [])) >= _LOGIN_RATE_LIMIT_MAX_ATTEMPTS


def register_failed_login(username: Optional[str]) -> None:
    with _LOGIN_ATTEMPT_LOCK:
        _prune_login_attempts()
        key = login_attempt_key(username)
        attempts = _LOGIN_ATTEMPTS.setdefault(key, [])
        attempts.append(time.time())


def clear_failed_logins(username: Optional[str]) -> None:
    with _LOGIN_ATTEMPT_LOCK:
        _LOGIN_ATTEMPTS.pop(login_attempt_key(username), None)


# --- Secret key obligatoria en producción ---
_env2 = (os.environ.get("FLASK_ENV") or os.environ.get("ENV") or "").lower()
_secret = os.environ.get("APP_SECRET_KEY", "")
if _env2 in ("production", "prod") and (not _secret or _secret == "reemplazar-esta-clave"):
    raise RuntimeError("APP_SECRET_KEY must be set in production")
if _secret and _secret != "reemplazar-esta-clave":
    app.secret_key = _secret
else:
    app.secret_key = secrets.token_urlsafe(32)
    if _env2 not in ("production", "prod"):
        app.logger.warning("APP_SECRET_KEY no configurada. Se genero una clave efimera para esta ejecucion.")


# --- Cookies de sesión (hardening mínimo) ---
_env = (os.environ.get("FLASK_ENV") or os.environ.get("ENV") or "").lower()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
if _env in ("production", "prod"):
    app.config["SESSION_COOKIE_SECURE"] = True


# --- CSRF mínimo (session token) ---
def _csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token

@app.context_processor
def inject_csrf_token():
    return {"csrf_token": _csrf_token()}

def _require_csrf():
    # Solo para endpoints críticos (llamar manualmente en POST)
    token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if not token or token != session.get("csrf_token"):
        abort(400)

APP_DB_URL = normalize_db_url(
    os.environ.get("APP_DB_URL", "sqlite:///players_updated_v2.db"),
    base_dir=BASE_DIR,
)
if APP_DB_URL.rsplit("/", 1)[-1] == "players.db":
    app.logger.warning("APP_DB_URL apunta a players.db (legacy). Se recomienda players_updated_v2.db. Evidencia: scouting_app/players.db")

TRAINING_DB_URL = normalize_db_url(
    os.environ.get("TRAINING_DB_URL", "sqlite:///players_training.db"),
    base_dir=BASE_DIR,
)
try:
    EVAL_POOL_MAX = max(1, int(os.environ.get("EVAL_POOL_MAX", "100")))
except ValueError:
    EVAL_POOL_MAX = 100

SYNC_SHORTLIST_ENABLED = (os.environ.get("SYNC_SHORTLIST_ENABLED", "0").strip().lower() in {
    "1", "true", "yes", "y", "si", "s", "on"
})

ENFORCE_EVAL_POOL_LIMIT = (os.environ.get("ENFORCE_EVAL_POOL_LIMIT", "1").strip().lower() in {
    "1", "true", "yes", "y", "si", "s", "on"
})

engine = create_app_engine(APP_DB_URL)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(engine)
ensure_player_columns(engine)


def trim_operational_player_pool(db_session, max_players: int = EVAL_POOL_MAX) -> int:
    """Mantiene la base operativa en un maximo de jugadores evaluables."""
    total = db_session.query(func.count(Player.id)).scalar() or 0
    if total <= max_players:
        return 0

    stat_subq = (
        db_session.query(
            PlayerStat.player_id.label("player_id"),
            func.max(PlayerStat.record_date).label("last_stat_date"),
        )
        .group_by(PlayerStat.player_id)
        .subquery()
    )
    attr_subq = (
        db_session.query(
            PlayerAttributeHistory.player_id.label("player_id"),
            func.max(PlayerAttributeHistory.record_date).label("last_attr_date"),
        )
        .group_by(PlayerAttributeHistory.player_id)
        .subquery()
    )

    rows = (
        db_session.query(
            Player.id,
            Player.age,
            stat_subq.c.last_stat_date,
            attr_subq.c.last_attr_date,
        )
        .outerjoin(stat_subq, stat_subq.c.player_id == Player.id)
        .outerjoin(attr_subq, attr_subq.c.player_id == Player.id)
        .all()
    )

    def sort_key(row):
        dates = [d for d in (row.last_stat_date, row.last_attr_date) if d is not None]
        last_activity = max(dates) if dates else date.min
        has_history = 1 if dates else 0
        in_eval_range = 1 if is_valid_eval_age(int(row.age or 0)) else 0
        return (has_history, in_eval_range, last_activity, row.id)

    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    keep_ids: Set[int] = {row.id for row in rows_sorted[:max_players]}
    drop_ids = [row.id for row in rows_sorted if row.id not in keep_ids]

    if not drop_ids:
        return 0

    db_session.query(PlayerStat).filter(PlayerStat.player_id.in_(drop_ids)).delete(synchronize_session=False)
    db_session.query(PlayerAttributeHistory).filter(PlayerAttributeHistory.player_id.in_(drop_ids)).delete(synchronize_session=False)
    db_session.query(Player).filter(Player.id.in_(drop_ids)).delete(synchronize_session=False)
    return len(drop_ids)


def backfill_player_photo_urls(db_session) -> int:
    players = (
        db_session.query(Player)
        .filter((Player.photo_url == None) | (Player.photo_url == ""))  # noqa: E711
        .all()
    )
    updated = 0
    for player in players:
        player.photo_url = default_player_photo_url(
            name=player.name,
            national_id=player.national_id,
            fallback=str(player.id),
        )
        updated += 1
    return updated


def compute_operational_data_quality(db_session) -> Dict[str, int]:
    """Resume consistencia basica de la base operativa."""
    players_total = db_session.query(func.count(Player.id)).scalar() or 0
    missing_national_id = (
        db_session.query(func.count(Player.id))
        .filter((Player.national_id == None) | (func.trim(Player.national_id) == ""))  # noqa: E711
        .scalar()
        or 0
    )
    invalid_age = (
        db_session.query(func.count(Player.id))
        .filter((Player.age < 12) | (Player.age > 18))
        .scalar()
        or 0
    )
    missing_name = (
        db_session.query(func.count(Player.id))
        .filter((Player.name == None) | (func.trim(Player.name) == ""))  # noqa: E711
        .scalar()
        or 0
    )
    missing_position = (
        db_session.query(func.count(Player.id))
        .filter((Player.position == None) | (func.trim(Player.position) == ""))  # noqa: E711
        .scalar()
        or 0
    )
    missing_photo_url = (
        db_session.query(func.count(Player.id))
        .filter((Player.photo_url == None) | (func.trim(Player.photo_url) == ""))  # noqa: E711
        .scalar()
        or 0
    )
    orphan_stats = (
        db_session.query(func.count(PlayerStat.id))
        .outerjoin(Player, Player.id == PlayerStat.player_id)
        .filter(Player.id == None)  # noqa: E711
        .scalar()
        or 0
    )
    orphan_attribute_history = (
        db_session.query(func.count(PlayerAttributeHistory.id))
        .outerjoin(Player, Player.id == PlayerAttributeHistory.player_id)
        .filter(Player.id == None)  # noqa: E711
        .scalar()
        or 0
    )
    over_limit_players = max(players_total - EVAL_POOL_MAX, 0)

    return {
        "players_total": int(players_total),
        "missing_national_id": int(missing_national_id),
        "invalid_age": int(invalid_age),
        "missing_name": int(missing_name),
        "missing_position": int(missing_position),
        "missing_photo_url": int(missing_photo_url),
        "orphan_stats": int(orphan_stats),
        "orphan_attribute_history": int(orphan_attribute_history),
        "over_limit_players": int(over_limit_players),
    }


def cleanup_operational_data(db_session) -> Dict[str, int]:
    """Limpia inconsistencias legacy sin inventar datos faltantes."""
    invalid_player_ids = [
        player_id
        for (player_id,) in db_session.query(Player.id)
        .filter(
            (Player.name == None)  # noqa: E711
            | (func.trim(Player.name) == "")
            | (Player.position == None)  # noqa: E711
            | (func.trim(Player.position) == "")
            | (Player.national_id == None)  # noqa: E711
            | (func.trim(Player.national_id) == "")
            | (Player.age < 12)
            | (Player.age > 18)
        )
        .all()
    ]

    removed_invalid_players = len(invalid_player_ids)
    if invalid_player_ids:
        db_session.query(PlayerStat).filter(PlayerStat.player_id.in_(invalid_player_ids)).delete(synchronize_session=False)
        db_session.query(PlayerAttributeHistory).filter(
            PlayerAttributeHistory.player_id.in_(invalid_player_ids)
        ).delete(synchronize_session=False)
        db_session.query(Player).filter(Player.id.in_(invalid_player_ids)).delete(synchronize_session=False)

    photo_updates = backfill_player_photo_urls(db_session)
    history_updates = 0
    history_sync_fn = globals().get("sync_attribute_history_baseline")
    if callable(history_sync_fn):
        history_updates = history_sync_fn(db_session)

    trimmed = 0
    if ENFORCE_EVAL_POOL_LIMIT:
        trimmed = trim_operational_player_pool(db_session, EVAL_POOL_MAX)

    quality_after = compute_operational_data_quality(db_session)
    return {
        "removed_invalid_players": int(removed_invalid_players),
        "photo_updates": int(photo_updates),
        "history_updates": int(history_updates),
        "trimmed_players": int(trimmed),
        **quality_after,
    }


def enforce_operational_pool_limit_on_startup():
    if not ENFORCE_EVAL_POOL_LIMIT:
        return
    db = Session()
    try:
        removed = trim_operational_player_pool(db, EVAL_POOL_MAX)
        if removed:
            db.commit()
            app.logger.warning(
                "Base operativa recortada a %s jugadores (eliminados: %s).",
                EVAL_POOL_MAX,
                removed,
            )
        photo_updates = backfill_player_photo_urls(db)
        if photo_updates:
            db.commit()
            app.logger.info("Fotos de jugadores completadas: %s", photo_updates)
        history_sync_fn = globals().get("sync_attribute_history_baseline")
        if callable(history_sync_fn):
            history_updates = history_sync_fn(db)
            if history_updates:
                db.commit()
                app.logger.info("Historial tecnico sincronizado desde ficha actual: %s jugadores", history_updates)
    except Exception:
        db.rollback()
        app.logger.exception("No se pudo aplicar el limite de jugadores operativos al iniciar.")
    finally:
        db.close()


def run_subprocess(command: List[str], description: str) -> Tuple[bool, str]:
    """Ejecuta un comando externo y devuelve (exito, mensaje)."""
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        log = f"[{description}] {'OK' if result.returncode == 0 else 'ERROR'}"
        if output:
            log += f"\n{output}"
        if error:
            log += f"\n{error}"
        return result.returncode == 0, log
    except Exception as exc:
        return False, f"[{description}] ERROR: {exc}"


def ensure_training_dataset(min_players: int = 1) -> Tuple[bool, List[str]]:
    """Verifica que la BD de entrenamiento tenga datos; genera si esta vacia."""
    logs: List[str] = []
    training_engine = create_app_engine(TRAINING_DB_URL)
    TrainingSession = sessionmaker(bind=training_engine)
    Base.metadata.create_all(training_engine)
    ensure_player_columns(training_engine)
    session = TrainingSession()
    try:
        count = session.query(func.count(Player.id)).scalar() or 0
    finally:
        session.close()
    logs.append(f"Jugadores disponibles en base de referencia: {count}")
    if count >= min_players:
        return True, logs
    cmd = [
        sys.executable,
        "generate_data.py",
        "--num-players",
        "20000",
        "--db-url",
        TRAINING_DB_URL,
    ]
    success, log = run_subprocess(cmd, "Preparar base de referencia")
    logs.append(log)
    return success, logs


def update_database_pipeline(limit: int = EVAL_POOL_MAX, sync_shortlist: bool = SYNC_SHORTLIST_ENABLED) -> Tuple[bool, List[str]]:
    """Ejecuta entrenamiento del modelo y sincronizacion opcional para la base operativa."""
    overall_logs: List[str] = []
    ok, logs = ensure_training_dataset()
    overall_logs.extend(logs)
    if not ok:
        return False, overall_logs
    train_cmd = [
        sys.executable,
        "train_model.py",
        "--db-url",
        TRAINING_DB_URL,
        "--model-out",
        MODEL_PATH,
        "--preprocessor-out",
        PREPROCESSOR_PATH,
        "--calibrator-out",
        CALIBRATOR_PATH,
        "--metadata-out",
        TRAINING_METADATA_PATH,
        "--epochs",
        "30",
    ]
    ok, train_log = run_subprocess(train_cmd, "Actualizacion de puntajes")
    overall_logs.append(train_log)
    if not ok:
        return False, overall_logs

    if sync_shortlist:
        ensure_player_columns(engine)
        sync_cmd = [
            sys.executable,
            "sync_shortlist.py",
            "--src-db",
            TRAINING_DB_URL,
            "--dst-db",
            APP_DB_URL,
            "--limit",
            str(limit),
        ]
        ok, sync_log = run_subprocess(sync_cmd, "Sincronizacion de base operativa")
        overall_logs.append(sync_log)
        if not ok:
            return False, overall_logs
    else:
        overall_logs.append("Sincronizacion de base operativa omitida.")

    # Guardrail final: la base operativa no debe superar EVAL_POOL_MAX.
    db = Session()
    try:
        removed = trim_operational_player_pool(db, EVAL_POOL_MAX)
        if removed:
            db.commit()
            overall_logs.append(f"Base operativa recortada a {EVAL_POOL_MAX} jugadores (eliminados {removed}).")
        else:
            db.rollback()
            overall_logs.append(f"Base operativa dentro del limite ({EVAL_POOL_MAX} jugadores).")
    except Exception as exc:
        db.rollback()
        overall_logs.append(f"No se pudo aplicar el recorte de base operativa: {exc}")
        return False, overall_logs
    finally:
        db.close()

    try:
        global model, preprocessor, probability_calibrator
        model, preprocessor, probability_calibrator = load_runtime_artifacts(
            MODEL_PATH,
            PREPROCESSOR_PATH,
            CALIBRATOR_PATH,
            allow_retrain=False,
        )
        invalidate_dashboard_cache()
        overall_logs.append("Modelo, preprocesador y calibrador recargados en memoria; cache del dashboard invalidado.")
    except Exception as exc:
        overall_logs.append(f"No se pudieron recargar los artefactos del modelo despues del entrenamiento: {exc}")
        return False, overall_logs

    return True, overall_logs

# ----------------------------------------------------
# Usuario administrador inicial (opcional en desarrollo)
def init_admin_user():
    username = (os.environ.get("ADMIN_USERNAME") or "admin").strip()
    password = (os.environ.get("ADMIN_PASSWORD") or "").strip()
    allow_default = (os.environ.get("ALLOW_DEFAULT_ADMIN") or "").strip().lower() in {
        "1", "true", "yes", "y", "si", "s", "on"
    }
    is_prod = (os.environ.get("FLASK_ENV") or os.environ.get("ENV") or "").lower() in {"production", "prod"}

    if not password:
        # En produccion no crear admin por defecto.
        if is_prod:
            app.logger.warning("ADMIN_PASSWORD no configurado. No se crea usuario admin inicial en produccion.")
            return
        # En desarrollo, solo permitir admin por defecto si se habilita explicito.
        if not allow_default:
            app.logger.info(
                "ADMIN_PASSWORD no configurado. Se omite bootstrap de admin. "
                "Setear ADMIN_PASSWORD o ALLOW_DEFAULT_ADMIN=true para desarrollo local."
            )
            return
        password = "admin"

    if len(username) < 3 or len(password) < 6:
        app.logger.warning("Credenciales de bootstrap invalidas. No se crea usuario admin inicial.")
        return
    if not is_strong_password(password):
        app.logger.warning(
            "La contraseña de bootstrap no cumple el minimo de seguridad (8+ caracteres, letra y numero)."
        )
        return

    db = Session()
    existing = db.query(User).filter(User.username == username).first()
    if not existing:
        user = User(username=username,
                    password_hash=generate_password_hash(password),
                    role="administrador")
        db.add(user)
        db.commit()
    db.close()

init_admin_user()
enforce_operational_pool_limit_on_startup()

# ----------------------------------------------------
# Landing
@app.route("/")
def landing():
    db = Session()
    total_players = db.query(func.count(Player.id)).scalar() or 0
    avg_age = db.query(func.avg(Player.age)).scalar()
    countries = db.query(Player.country).distinct().count()
    positions = db.query(Player.position).distinct().count()
    db.close()

    metrics = {
        "total_players": int(total_players),
        "avg_age": float(avg_age) if avg_age else None,
        "countries": countries,
        "positions": positions,
    }

    if session.get("user_id"):
        call_to_action_url = url_for("index")
        call_to_action_label = "Ir al panel"
    else:
        call_to_action_url = url_for("login")
        call_to_action_label = "Iniciar sesión"

    return render_template(
        "landing.html",
        metrics=metrics,
        call_to_action_url=call_to_action_url,
        call_to_action_label=call_to_action_label,
    )

# ----------------------------------------------------

@app.context_processor
def navbar_url_helpers():
    def first_url(*endpoints, **values):
        """Devuelve la primera URL válida entre una lista de endpoints.
        Si ninguno existe, devuelve '#'.
        """
        for ep in endpoints:
            try:
                return url_for(ep, **values)
            except Exception:
                continue
        return "#"
    return dict(first_url=first_url)

@app.context_processor
def auth_flags():
    role = current_role()
    return {
        "is_authenticated": bool(session.get("user_id")),
        "current_username": session.get("username", "admin"),
        "current_role": role,
        "current_role_label": role_display_label(role),
        "is_admin": role == "administrador",
        "can_edit_player_data": can_edit_player_data(),
        "can_manage_players": can_manage_players(),
        "can_manage_staff": can_manage_staff(),
        "can_manage_users": can_manage_users(),
    }


def display_position_label(value: Optional[str]) -> str:
    normalized = normalized_position(value)
    return "Arquero" if normalized == "Portero" else normalized


def is_strong_password(password: str) -> bool:
    if len(password or "") < 8:
        return False
    has_letter = any(char.isalpha() for char in password)
    has_digit = any(char.isdigit() for char in password)
    return has_letter and has_digit


ROLE_ADMIN = "administrador"
ROLE_SCOUT = "scout"
ROLE_DIRECTOR = "director"
ROLE_ALIASES = {
    "admin": ROLE_ADMIN,
    "administrador": ROLE_ADMIN,
    "scout": ROLE_SCOUT,
    "ojeador": ROLE_SCOUT,
    "director": ROLE_DIRECTOR,
}


def normalize_role(role: Optional[str]) -> str:
    raw = (role or "").strip().lower()
    return ROLE_ALIASES.get(raw, ROLE_SCOUT)


def current_role() -> str:
    return normalize_role(session.get("role"))


def role_display_label(role: Optional[str]) -> str:
    normalized = normalize_role(role)
    return {
        ROLE_ADMIN: "Administrador",
        ROLE_SCOUT: "Scout",
        ROLE_DIRECTOR: "Director",
    }.get(normalized, "Scout")


def has_any_role(*roles: str) -> bool:
    expected = {normalize_role(role) for role in roles}
    return current_role() in expected


def can_edit_player_data() -> bool:
    return bool(session.get("user_id")) and has_any_role(ROLE_ADMIN, ROLE_SCOUT)


def can_manage_players() -> bool:
    return can_edit_player_data()


def can_manage_staff() -> bool:
    return bool(session.get("user_id")) and has_any_role(ROLE_ADMIN)


def can_manage_users() -> bool:
    return bool(session.get("user_id")) and has_any_role(ROLE_ADMIN)


@app.context_processor
def position_labels():
    return {"display_position": display_position_label}

# Decorador de login
def login_required(view_func: Callable) -> Callable:
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login', next=request.url))
        return view_func(*args, **kwargs)
    return wrapper


def roles_required(*roles: str) -> Callable:
    normalized_roles = {normalize_role(role) for role in roles}

    def decorator(view_func: Callable) -> Callable:
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if not session.get('user_id'):
                return redirect(url_for('login', next=request.url))
            if current_role() not in normalized_roles:
                abort(403)
            return view_func(*args, **kwargs)

        return wrapper

    return decorator

# ----------------------------------------------------
# Modelo
def load_model_state(model_path: str, input_dim: int) -> PlayerNet:
    model = PlayerNet(input_dim=input_dim)
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_probability_calibrator(calibrator_path: str):
    if not os.path.exists(calibrator_path):
        return None
    try:
        return joblib_load(calibrator_path)
    except Exception:
        return None


def apply_probability_calibrator(calibrator, probabilities: List[float]) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if calibrator is None:
        return raw
    try:
        if hasattr(calibrator, "predict_proba"):
            return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1].astype(np.float32)
        return np.clip(np.asarray(calibrator.predict(raw), dtype=np.float32), 0.0, 1.0)
    except Exception:
        return raw


def load_runtime_artifacts(
    model_path: str,
    preprocessor_path: str,
    calibrator_path: Optional[str] = None,
    allow_retrain: bool = True,
) -> Tuple[Optional[PlayerNet], Optional[object], Optional[object]]:
    def retrain_or_raise(message: str, original_exc: Optional[Exception] = None):
        if not allow_retrain:
            if original_exc is not None:
                raise original_exc
            raise FileNotFoundError(message)
        print(message)
        success, logs = update_database_pipeline()
        for log in logs:
            print(log)
        if not success:
            raise RuntimeError(
                "No se pudo reentrenar el modelo automaticamente. Ejecute train_model.py manualmente."
            ) from original_exc

    if not os.path.exists(model_path):
        retrain_or_raise("Advertencia: modelo no encontrado. Intentando reentrenar automaticamente.")
    if not os.path.exists(preprocessor_path):
        retrain_or_raise("Advertencia: preprocesador no encontrado. Intentando reentrenar automaticamente.")

    preprocessor = load_saved_preprocessor(preprocessor_path)
    input_dim = preprocessor_input_dim(preprocessor)
    try:
        model = load_model_state(model_path, input_dim)
    except RuntimeError as exc:
        retrain_or_raise(
            "Advertencia: modelo incompatible con el preprocesador actual. Intentando reentrenar automaticamente.",
            original_exc=exc,
        )
        preprocessor = load_saved_preprocessor(preprocessor_path)
        input_dim = preprocessor_input_dim(preprocessor)
        model = load_model_state(model_path, input_dim)
    calibrator = load_probability_calibrator(calibrator_path) if calibrator_path else None
    return model, preprocessor, calibrator


MODEL_PATH = os.path.join(BASE_DIR, "model.pt")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "probability_calibrator.joblib")
TRAINING_METADATA_PATH = os.path.join(BASE_DIR, "training_metadata.json")
try:
    model, preprocessor, probability_calibrator = load_runtime_artifacts(MODEL_PATH, PREPROCESSOR_PATH, CALIBRATOR_PATH)
except FileNotFoundError:
    model = None
    preprocessor = None
    probability_calibrator = None
    print("Advertencia: modelo o preprocesador no encontrados.")

def legacy_health_endpoint():
    """Healthcheck básico: app viva + conectividad DB."""
    try:
        # Validación mínima de conectividad
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        app.logger.exception("Healthcheck failed")
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/health")
def health():
    """Healthcheck basico: app viva + conectividad DB + calidad operativa."""
    db = Session()
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        data_quality = compute_operational_data_quality(db)
        return jsonify(
            {
                "status": "ok",
                "database": "ok",
                "data_quality": data_quality,
                "limits": {
                    "eval_pool_max": EVAL_POOL_MAX,
                },
            }
        ), 200
    except Exception as e:
        app.logger.exception("Healthcheck failed")
        return jsonify({"status": "error", "detail": str(e)}), 500
    finally:
        db.close()


def prepare_input(player: Player) -> torch.Tensor:
    return players_to_model_tensor([player])

def compute_suggestions(player: Player, threshold=14, top_n=3) -> List[Tuple[str, int]]:
    attrs = {
        "Ritmo (pace)": player.pace,
        "Disparo (shooting)": player.shooting,
        "Pase (passing)": player.passing,
        "Regate (dribbling)": player.dribbling,
        "Defensa (defending)": player.defending,
        "Físico (physical)": player.physical,
        "Visión (vision)": player.vision,
        "Marcaje (tackling)": player.tackling,
        "Determinación (determination)": player.determination,
        "Técnica (technique)": player.technique,
    }
    gaps = {name: threshold - value for name, value in attrs.items() if value < threshold}
    return sorted(gaps.items(), key=lambda item: item[1], reverse=True)[:top_n]


def score_band(score: float) -> str:
    if score >= 15:
        return "Alto"
    if score >= 10:
        return "Medio"
    return "Bajo"

POSITION_OPTIONS = POSITION_CHOICES


def player_attribute_map(player: Player) -> Dict[str, int]:
    return {field: getattr(player, field) for field in ATTRIBUTE_FIELDS}


def fetch_player_stat_feature_map(player_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    if not player_ids:
        return {}
    query = (
        select(
            PlayerStat.player_id,
            PlayerStat.record_date,
            PlayerStat.pass_accuracy,
            PlayerStat.final_score,
        )
        .where(PlayerStat.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        stats_df = pd.read_sql(query, connection)
    aggregated_df = aggregate_stats_dataframe(stats_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = stats_feature_defaults(row)
    return feature_map


def fetch_player_attribute_feature_map(players: List[Player]) -> Dict[int, Dict[str, Optional[float]]]:
    player_ids = [player.id for player in players if player.id]
    if not player_ids:
        return {}
    query = (
        select(
            PlayerAttributeHistory.player_id,
            PlayerAttributeHistory.record_date,
            PlayerAttributeHistory.pace,
            PlayerAttributeHistory.shooting,
            PlayerAttributeHistory.passing,
            PlayerAttributeHistory.dribbling,
            PlayerAttributeHistory.defending,
            PlayerAttributeHistory.physical,
            PlayerAttributeHistory.vision,
            PlayerAttributeHistory.tackling,
            PlayerAttributeHistory.determination,
            PlayerAttributeHistory.technique,
        )
        .where(PlayerAttributeHistory.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        history_df = pd.read_sql(query, connection)
    players_df = player_base_dataframe_from_players(players)
    aggregated_df = aggregate_attribute_history_dataframe(players_df, history_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = attribute_history_feature_defaults(row)
    return feature_map


def fetch_player_match_feature_map(players: List[Player]) -> Dict[int, Dict[str, Optional[float]]]:
    player_ids = [player.id for player in players if player.id]
    if not player_ids:
        return {}
    query = (
        select(
            PlayerMatchParticipation.player_id,
            Match.match_date,
            Match.opponent_level,
            PlayerMatchParticipation.started,
            PlayerMatchParticipation.position_played,
            PlayerMatchParticipation.minutes_played,
            PlayerMatchParticipation.final_score,
        )
        .join(Match, PlayerMatchParticipation.match_id == Match.id)
        .where(PlayerMatchParticipation.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        participation_df = pd.read_sql(query, connection)
    players_df = player_base_dataframe_from_players(players)
    aggregated_df = aggregate_match_participation_dataframe(players_df, participation_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = match_feature_defaults(row)
    return feature_map


def fetch_player_scout_report_feature_map(player_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    if not player_ids:
        return {}
    query = (
        select(
            ScoutReport.player_id,
            ScoutReport.report_date,
            ScoutReport.decision_making,
            ScoutReport.tactical_reading,
            ScoutReport.mental_profile,
            ScoutReport.adaptability,
            ScoutReport.observed_projection_score,
        )
        .where(ScoutReport.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        scout_report_df = pd.read_sql(query, connection)
    aggregated_df = aggregate_scout_report_dataframe(scout_report_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = scout_report_feature_defaults(row)
    return feature_map


def fetch_player_physical_feature_map(player_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    if not player_ids:
        return {}
    query = (
        select(
            PhysicalAssessment.player_id,
            PhysicalAssessment.assessment_date,
            PhysicalAssessment.height_cm,
            PhysicalAssessment.weight_kg,
            PhysicalAssessment.dominant_foot,
            PhysicalAssessment.estimated_speed,
            PhysicalAssessment.endurance,
            PhysicalAssessment.in_growth_spurt,
        )
        .where(PhysicalAssessment.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        physical_df = pd.read_sql(query, connection)
    aggregated_df = aggregate_physical_assessment_dataframe(physical_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = physical_feature_defaults(row)
    return feature_map


def fetch_player_availability_feature_map(player_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    if not player_ids:
        return {}
    query = (
        select(
            PlayerAvailability.player_id,
            PlayerAvailability.record_date,
            PlayerAvailability.availability_pct,
            PlayerAvailability.fatigue_pct,
            PlayerAvailability.training_load_pct,
            PlayerAvailability.missed_days,
            PlayerAvailability.injury_flag,
        )
        .where(PlayerAvailability.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        availability_df = pd.read_sql(query, connection)
    aggregated_df = aggregate_availability_dataframe(availability_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = availability_feature_defaults(row)
    return feature_map


def players_to_model_tensor(
    players: List[Player],
    stats_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    attribute_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    match_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    scout_report_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    physical_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    availability_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
) -> torch.Tensor:
    if preprocessor is None:
        raise RuntimeError("No hay preprocesador cargado para inferencia.")
    player_ids = [player.id for player in players if player.id]
    stats_feature_map = stats_feature_map or fetch_player_stat_feature_map(player_ids)
    attribute_feature_map = attribute_feature_map or fetch_player_attribute_feature_map(players)
    match_feature_map = match_feature_map or fetch_player_match_feature_map(players)
    scout_report_feature_map = scout_report_feature_map or fetch_player_scout_report_feature_map(player_ids)
    physical_feature_map = physical_feature_map or fetch_player_physical_feature_map(player_ids)
    availability_feature_map = availability_feature_map or fetch_player_availability_feature_map(player_ids)
    features_df = dataframe_from_players(
        players,
        stats_feature_map=stats_feature_map,
        attribute_feature_map=attribute_feature_map,
        match_feature_map=match_feature_map,
        scout_report_feature_map=scout_report_feature_map,
        physical_feature_map=physical_feature_map,
        availability_feature_map=availability_feature_map,
    )
    features_array = transform_features(features_df, preprocessor)
    return torch.tensor(features_array, dtype=torch.float32)


def batch_project_players(
    players: List[Player],
    avg_score_map: Optional[Dict[int, Optional[float]]] = None,
    stats_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    attribute_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    match_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    scout_report_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    physical_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    availability_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
) -> Dict[int, Dict[str, object]]:
    if not players or model is None or preprocessor is None:
        return {}

    avg_score_map = avg_score_map or {}
    player_ids = [player.id for player in players if player.id]
    stats_feature_map = stats_feature_map or fetch_player_stat_feature_map(player_ids)
    attribute_feature_map = attribute_feature_map or fetch_player_attribute_feature_map(players)
    match_feature_map = match_feature_map or fetch_player_match_feature_map(players)
    scout_report_feature_map = scout_report_feature_map or fetch_player_scout_report_feature_map(player_ids)
    physical_feature_map = physical_feature_map or fetch_player_physical_feature_map(player_ids)
    availability_feature_map = availability_feature_map or fetch_player_availability_feature_map(player_ids)
    with torch.no_grad():
        probs_tensor = torch.sigmoid(
            model(
                players_to_model_tensor(
                    players,
                    stats_feature_map=stats_feature_map,
                    attribute_feature_map=attribute_feature_map,
                    match_feature_map=match_feature_map,
                    scout_report_feature_map=scout_report_feature_map,
                    physical_feature_map=physical_feature_map,
                    availability_feature_map=availability_feature_map,
                )
            )
        )
    raw_base_probs = np.asarray(probs_tensor.detach().cpu().numpy()).reshape(-1)
    calibrated_base_probs = (
        apply_probability_calibrator(probability_calibrator, raw_base_probs.tolist())
        if probability_calibrator is not None
        else None
    )

    projections: Dict[int, Dict[str, object]] = {}
    for idx, player in enumerate(players):
        base_prob = float(raw_base_probs[idx])
        attr_map = player_attribute_map(player)
        best_position, best_score = recommend_position_from_attrs(attr_map)
        fit_score = weighted_score_from_attrs(attr_map, player.position)
        player_stats_features = stats_feature_map.get(player.id, {})
        avg_final_score = avg_score_map.get(player.id)
        if avg_final_score is None:
            avg_final_score = player_stats_features.get("avg_final_score_hist")
        stats_summary = {"avg_final_score": avg_final_score}
        combined = combine_probability(base_prob, stats_summary, fit_score=fit_score)
        calibrated_base_prob = (
            float(calibrated_base_probs[idx])
            if calibrated_base_probs is not None
            else None
        )
        calibrated_combined = (
            combine_probability(calibrated_base_prob, stats_summary, fit_score=fit_score)
            if calibrated_base_prob is not None
            else None
        )
        projections[player.id] = {
            "base_prob": base_prob,
            "base_prob_source": "raw_pytorch_sigmoid",
            "calibrated_base_prob": calibrated_base_prob,
            "calibrated_combined_prob": calibrated_combined,
            "combined_prob": combined,
            "category": categorize_probability(combined),
            "stats_summary": stats_summary,
            "fit_score": fit_score,
            "recommended_position": best_position,
            "recommended_score": best_score,
        }
    return projections


def player_fit_score(player: Player, target_position: Optional[str] = None) -> float:
    return weighted_score_from_attrs(player_attribute_map(player), target_position)


def normalize_identifier(raw_value: Optional[str]) -> Optional[str]:
    if not raw_value:
        return None
    digits = "".join(ch for ch in raw_value if ch.isdigit())
    if len(digits) < 6:
        return None
    return digits



def fetch_player_stats(player_id: int, db_session = None) -> List[PlayerStat]:
    close_session = False
    if db_session is None:
        db_session = Session()
        close_session = True
    stats = (db_session.query(PlayerStat)
             .filter(PlayerStat.player_id == player_id)
             .order_by(PlayerStat.record_date.asc(), PlayerStat.id.asc())
             .all())
    if close_session:
        db_session.close()
    return stats


def summarize_stats(stats: List[PlayerStat]) -> Dict[str, Optional[float]]:
    if not stats:
        return {
            "entries": 0,
            "total_matches": 0,
            "total_goals": 0,
            "total_assists": 0,
            "total_minutes": 0,
            "avg_pass_accuracy": None,
            "avg_shot_accuracy": None,
            "avg_duels": None,
            "avg_final_score": None,
            "latest_final_score": None,
            "latest_date": None,
        }

    def avg(values: List[Optional[float]]) -> Optional[float]:
        filtered = [v for v in values if v is not None]
        return round(mean(filtered), 2) if filtered else None

    return {
        "entries": len(stats),
        "total_matches": sum(s.matches_played for s in stats),
        "total_goals": sum(s.goals for s in stats),
        "total_assists": sum(s.assists for s in stats),
        "total_minutes": sum(s.minutes_played for s in stats),
        "avg_pass_accuracy": avg([s.pass_accuracy for s in stats]),
        "avg_shot_accuracy": avg([s.shot_accuracy for s in stats]),
        "avg_duels": avg([s.duels_won_pct for s in stats]),
        "avg_final_score": avg([s.final_score for s in stats if s.final_score is not None]),
        "latest_final_score": stats[-1].final_score,
        "latest_date": stats[-1].record_date.isoformat(),
    }


def fetch_attribute_history(player_id: int, db_session = None) -> List[PlayerAttributeHistory]:
    close_session = False
    if db_session is None:
        db_session = Session()
        close_session = True
    history = (db_session.query(PlayerAttributeHistory)
               .filter(PlayerAttributeHistory.player_id == player_id)
               .order_by(PlayerAttributeHistory.record_date.asc(), PlayerAttributeHistory.id.asc())
               .all())
    if close_session:
        db_session.close()
    return history


def _attribute_row_from_player(player: Player) -> Dict[str, int]:
    return {field: int(getattr(player, field) or 0) for field in ATTRIBUTE_FIELDS}


def _attribute_row_from_entry(entry: PlayerAttributeHistory) -> Dict[str, int]:
    return {field: int(getattr(entry, field) or 0) for field in ATTRIBUTE_FIELDS}


def sync_player_attribute_history(player: Player, db_session, note: str = "Sincronizacion automatica de ficha") -> bool:
    """Asegura que el ultimo registro del historial tecnico refleje la ficha actual.

    Devuelve True si crea un registro nuevo.
    """
    latest = (
        db_session.query(PlayerAttributeHistory)
        .filter(PlayerAttributeHistory.player_id == player.id)
        .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
        .first()
    )
    current_values = _attribute_row_from_player(player)
    if latest and _attribute_row_from_entry(latest) == current_values:
        return False

    entry = PlayerAttributeHistory(
        player_id=player.id,
        record_date=date.today(),
        notes=note,
        **current_values,
    )
    db_session.add(entry)
    return True


def sync_attribute_history_baseline(db_session) -> int:
    """Sincroniza historial tecnico para jugadores con ficha desfasada o sin historial."""
    players = db_session.query(Player).all()
    created = 0
    for player in players:
        if sync_player_attribute_history(player, db_session):
            created += 1
    return created


def summarize_attribute_history(history: List[PlayerAttributeHistory]) -> Dict[str, Optional[int]]:
    if not history:
        summary: Dict[str, Optional[int]] = {field: None for field in ATTRIBUTE_FIELDS}
        summary["entries"] = 0
        summary["latest_date"] = None
        return summary
    latest = history[-1]
    summary = {field: getattr(latest, field) for field in ATTRIBUTE_FIELDS}
    summary["entries"] = len(history)
    summary["latest_date"] = latest.record_date.isoformat()
    return summary


def combine_probability(base_prob: float, stats_summary: Dict[str, Optional[float]], fit_score: Optional[float] = None) -> float:
    """Combina la probabilidad del modelo con señales simples del historial y del fit del jugador.

    - base_prob: salida del modelo (0..1)
    - avg_final_score: promedio histórico (1..10) si existe
    - fit_score: puntaje ponderado por posición (0..20) si existe
    """
    avg_score = stats_summary.get("avg_final_score")
    rating_weight = None if avg_score is None else min(max(float(avg_score) / 10.0, 0.0), 1.0)

    fit_weight = None
    if fit_score is not None:
        try:
            fit_weight = min(max(float(fit_score) / 20.0, 0.0), 1.0)
        except Exception:
            fit_weight = None

    # Pesos (tuneables vía env vars)
    w_model = float(os.environ.get("POT_W_MODEL", "0.35"))
    w_rating = float(os.environ.get("POT_W_RATING", "0.35"))
    w_fit = float(os.environ.get("POT_W_FIT", "0.30"))

    # Si no hay rating o fit, re-normalizamos para no castigar por falta de datos
    components = [(w_model, base_prob)]
    if rating_weight is not None:
        components.append((w_rating, rating_weight))
    if fit_weight is not None:
        components.append((w_fit, fit_weight))

    weight_sum = sum(w for w, _ in components) or 1.0
    combined = sum(w * v for w, v in components) / weight_sum

    return max(0.0, min(combined, 0.99))
def stats_chart_payload(stats: List[PlayerStat]) -> Dict[str, List]:
    labels = []
    final_scores = []
    pass_pct = []
    shot_pct = []
    duel_pct = []
    for entry in stats:
        labels.append(entry.record_date.strftime("%Y-%m-%d"))
        final_scores.append(entry.final_score if entry.final_score is not None else None)
        pass_pct.append(entry.pass_accuracy if entry.pass_accuracy is not None else None)
        shot_pct.append(entry.shot_accuracy if entry.shot_accuracy is not None else None)
        duel_pct.append(entry.duels_won_pct if entry.duels_won_pct is not None else None)
    return {
        "labels": labels,
        "final_scores": final_scores,
        "pass_pct": pass_pct,
        "shot_pct": shot_pct,
        "duel_pct": duel_pct,
    }


def calculate_stats_rating(metrics: Dict[str, Optional[float]]) -> float:
    matches = metrics.get("matches", 0) or 0
    goals = metrics.get("goals", 0) or 0
    assists = metrics.get("assists", 0) or 0
    minutes = metrics.get("minutes", 0) or 0
    pass_pct = metrics.get("pass_pct") or 0.0
    shot_pct = metrics.get("shot_pct") or 0.0
    duels_pct = metrics.get("duels_pct") or 0.0

    minutes_factor = min(minutes / 90.0, 1.5)
    scoring_factor = min(goals, 3) * 1.5 + min(assists, 3) * 1.2
    accuracy_factor = (pass_pct / 100.0) * 2.0 + (shot_pct / 100.0) * 1.5 + (duels_pct / 100.0) * 1.3
    consistency_factor = min(matches, 3) * 0.5

    raw_score = 1.5 + minutes_factor + scoring_factor + accuracy_factor + consistency_factor
    return round(max(1.0, min(10.0, raw_score)), 2)


def attribute_chart_payload(history: List[PlayerAttributeHistory]) -> Dict[str, List[Optional[int]]]:
    labels: List[str] = []
    series: Dict[str, List[Optional[int]]] = {field: [] for field in ATTRIBUTE_FIELDS}
    for entry in history:
        labels.append(entry.record_date.strftime("%Y-%m-%d"))
        for field in ATTRIBUTE_FIELDS:
            series[field].append(getattr(entry, field))
    return {"labels": labels, "series": series}


def potential_thresholds() -> Tuple[float, float]:
    high = float(os.environ.get("POTENTIAL_HIGH_THRESHOLD", "0.60"))
    medium = float(os.environ.get("POTENTIAL_MEDIUM_THRESHOLD", "0.35"))
    if medium >= high:
        medium = max(0.0, high - 0.05)
    return medium, high


def is_high_potential_probability(probability: float) -> bool:
    _, high = potential_thresholds()
    return probability >= high


def categorize_probability(probability: float) -> str:
    medium, high = potential_thresholds()

    if probability >= high:
        return "Alto potencial"
    if probability >= medium:
        return "Potencial medio"
    return "Bajo potencial"
def compute_projection(player: Player, stats: Optional[List[PlayerStat]] = None, db_session = None) -> Optional[Dict[str, object]]:
    if model is None:
        return None
    if stats is None:
        stats_list = fetch_player_stats(player.id, db_session=db_session)
    else:
        stats_list = stats
    summary = summarize_stats(stats_list)
    avg_score = summary.get("avg_final_score")
    stats_feature_map = {
        player.id: stats_feature_defaults(
            {
                "stats_entry_count": summary.get("entries"),
                "avg_final_score_hist": avg_score,
                "avg_pass_accuracy_hist": summary.get("avg_pass_accuracy"),
                "latest_final_score_hist": summary.get("latest_final_score"),
            }
        )
    }
    attribute_feature_map = fetch_player_attribute_feature_map([player])
    match_feature_map = fetch_player_match_feature_map([player])
    scout_report_feature_map = fetch_player_scout_report_feature_map([player.id])
    projection = batch_project_players(
        [player],
        {player.id: avg_score},
        stats_feature_map=stats_feature_map,
        attribute_feature_map=attribute_feature_map,
        match_feature_map=match_feature_map,
        scout_report_feature_map=scout_report_feature_map,
    ).get(player.id)
    if not projection:
        return None
    projection["history"] = stats_list
    return projection


def refresh_player_potential(player: Player, db_session = None) -> Optional[Dict[str, object]]:
    projection = compute_projection(player, db_session=db_session)
    if projection:
        player.potential_label = is_high_potential_probability(projection["combined_prob"])
    return projection

# ----------------------------------------------------
# LOGIN
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        _require_csrf()

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if is_login_rate_limited(username):
            return (
                render_template(
                    'login.html',
                    error='Se bloquearon temporalmente los intentos de acceso. Espera unos minutos antes de reintentar.',
                ),
                429,
            )
        db = Session()
        user = db.query(User).filter(User.username == username).first()
        db.close()
        if user and check_password_hash(user.password_hash, password):
            clear_failed_logins(username)
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = normalize_role(user.role)
            return redirect(request.args.get('next') or url_for('index'))
        register_failed_login(username)
        return render_template('login.html', error='Usuario o contraseña inválidos')
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))  # <- antes apuntaba al login; ahora a la web pública


# ----------------------------------------------------
# REGISTRO
@app.route('/register', methods=['GET', 'POST'])
@roles_required(ROLE_ADMIN)
def register():
    if request.method == "POST":
        _require_csrf()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = normalize_role(request.form.get('role'))
        if not username or not password or not role:
            return render_template('register.html', error='Todos los campos son obligatorios')
        if not is_strong_password(password):
            return render_template(
                'register.html',
                error='La contraseña debe tener al menos 8 caracteres e incluir letras y numeros',
            )
        if role not in {ROLE_ADMIN, ROLE_SCOUT, ROLE_DIRECTOR}:
            return render_template('register.html', error='Rol inválido')
        db = Session()
        if db.query(User).filter(User.username == username).first():
            db.close()
            return render_template('register.html', error='El usuario ya existe')
        user = User(username=username,
                    password_hash=generate_password_hash(password),
                    role=role)
        db.add(user)
        db.commit()
        db.close()
        return redirect(url_for('index'))
    return render_template('register.html')

# ----------------------------------------------------
# DASHBOARD
@app.route('/dashboard')
@login_required
def dashboard():
    period = request.args.get('period', 'month')
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    dashboard_cache_key = f"dashboard:{period}:{start_str}:{end_str}"
    cached_html = _cache_get(dashboard_cache_key)
    if cached_html is not None:
        return cached_html

    today = date.today()
    if period == 'custom':
        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else today - timedelta(days=30)
        except ValueError:
            start_date = today - timedelta(days=30)
        try:
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else today
        except ValueError:
            end_date = today
    else:
        end_date = today
        delta = {
            'week': 7,
            'month': 30,
            'year': 365
        }.get(period, 30)
        start_date = end_date - timedelta(days=delta)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    db = Session()
    pos_rows = db.query(Player.position, func.count(Player.id)).group_by(Player.position).all()
    positions = [display_position_label(row[0]) if row[0] else "Sin posicion" for row in pos_rows]
    pos_values = [row[1] for row in pos_rows]
    total_players = db.query(func.count(Player.id)).scalar() or 0
    avg_attrs = db.query(
        func.avg(Player.pace), func.avg(Player.shooting), func.avg(Player.passing),
        func.avg(Player.dribbling), func.avg(Player.defending), func.avg(Player.physical),
        func.avg(Player.vision), func.avg(Player.tackling), func.avg(Player.determination), func.avg(Player.technique)
    ).one()
    avg_labels = [ATTRIBUTE_LABELS[field] for field in ATTRIBUTE_FIELDS]
    avg_values = [round(float(value), 2) if value is not None else 0 for value in avg_attrs]

    players = (
        db.query(Player)
        .options(
            load_only(
                Player.id,
                Player.name,
                Player.age,
                Player.position,
                Player.pace,
                Player.shooting,
                Player.passing,
                Player.dribbling,
                Player.defending,
                Player.physical,
                Player.vision,
                Player.tackling,
                Player.determination,
                Player.technique,
            )
        )
        .all()
    )
    player_avg_rows = db.query(
        PlayerStat.player_id, func.avg(PlayerStat.final_score)
    ).group_by(PlayerStat.player_id).all()
    avg_score_map = {
        player_id: (float(avg_score) if avg_score is not None else None)
        for player_id, avg_score in player_avg_rows
    }
    top_potential = []
    category_counts = {'Alto potencial': 0, 'Potencial medio': 0, 'Bajo potencial': 0, 'Sin datos': 0}
    if players and model is not None:
        projections = batch_project_players(players, avg_score_map)
        for player in players:
            projection = projections.get(player.id)
            if not projection:
                continue
            top_potential.append({
                "id": player.id,
                "name": player.name,
                "probability": projection["combined_prob"],
                "category": projection["category"],
            })
            category_counts[projection["category"]] += 1
        top_potential.sort(key=lambda item: item["probability"], reverse=True)
        top_potential = top_potential[:10]
    else:
        category_counts['Sin datos'] = total_players

    stats_in_range = (db.query(PlayerStat)
                      .filter(PlayerStat.record_date >= start_date,
                              PlayerStat.record_date <= end_date,
                              PlayerStat.final_score != None)
                      .order_by(PlayerStat.player_id.asc(),
                                PlayerStat.record_date.asc(),
                                PlayerStat.id.asc())
                      .all())
    player_map = {player.id: player for player in players}
    evolution_map: Dict[int, Dict[str, object]] = {}
    for stat in stats_in_range:
        entry = evolution_map.setdefault(
            stat.player_id,
            {
                "first_score": stat.final_score,
                "first_date": stat.record_date,
                "last_score": stat.final_score,
                "last_date": stat.record_date,
            }
        )
        if stat.record_date < entry["first_date"]:
            entry["first_date"] = stat.record_date
            entry["first_score"] = stat.final_score
        if stat.record_date >= entry["last_date"]:
            entry["last_date"] = stat.record_date
            entry["last_score"] = stat.final_score

    top_evolution = []
    for player_id, values in evolution_map.items():
        first = values["first_score"]
        last = values["last_score"]
        if first is None or last is None:
            continue
        delta = round(float(last - first), 2)
        top_evolution.append({
            "id": player_id,
            "name": player_map[player_id].name if player_id in player_map else f"Jugador {player_id}",
            "delta": delta,
            "start": values["first_date"],
            "end": values["last_date"],
        })
    top_evolution.sort(key=lambda item: item["delta"], reverse=True)
    top_evolution = top_evolution[:10]
    final_score_avg = db.query(func.avg(PlayerStat.final_score)).filter(PlayerStat.final_score != None).scalar()
    final_score_avg = round(float(final_score_avg), 2) if final_score_avg is not None else None
    db.close()

    pot_labels = list(category_counts.keys())
    pot_values = [category_counts[label] for label in pot_labels]
    html = render_template(
        'dashboard.html',
        positions=positions,
        pos_values=pos_values,
        pot_labels=pot_labels,
        pot_values=pot_values,
        avg_labels=avg_labels,
        avg_values=avg_values,
        total_players=total_players,
        final_score_avg=final_score_avg,
        top_potential=top_potential,
        top_evolution=top_evolution,
        selected_period=period,
        start_date_str=start_str,
        end_date_str=end_str,
    )
    _cache_set(dashboard_cache_key, html)
    return html
# ----------------------------------------------------
# LISTA JUGADORES
@app.route("/players")
@login_required
def index():
    search_term = request.args.get('q')
    pos_filter = request.args.get('position')
    club_filter = request.args.get('club')
    country_filter = request.args.get('country')
    top_potential = request.args.get('top_potential')
    order_attr = request.args.get('order_attr')
    page = request.args.get('page', 1, type=int)
    per_page = 50
    db = Session()
    player_list_columns = [
        Player.id,
        Player.name,
        Player.national_id,
        Player.age,
        Player.position,
        Player.club,
        Player.country,
        Player.photo_url,
        Player.pace,
        Player.shooting,
        Player.passing,
        Player.dribbling,
        Player.defending,
        Player.physical,
        Player.vision,
        Player.tackling,
        Player.determination,
        Player.technique,
        Player.potential_label,
    ]
    query = db.query(Player).options(load_only(*player_list_columns))
    pos_list = [r[0] for r in db.query(Player.position).distinct().all()]
    club_list = [r[0] for r in db.query(Player.club).distinct().all() if r[0]]
    country_list = [r[0] for r in db.query(Player.country).distinct().all() if r[0]]
    if search_term:
        query = query.filter(Player.name.ilike(f"%{search_term}%"))
    if pos_filter:
        query = query.filter(Player.position == pos_filter)
    if club_filter:
        query = query.filter(Player.club == club_filter)
    if country_filter:
        query = query.filter(Player.country == country_filter)
    if order_attr and hasattr(Player, order_attr):
        query = query.order_by(desc(getattr(Player, order_attr)))
    total = query.count()
    total_pages = (total + per_page - 1) // per_page
    players = query.offset((page - 1) * per_page).limit(per_page).all()
    if top_potential:
        # Para el filtro de alto potencial, ordenar por potencial real (mayor->menor)
        # y paginar luego del ordenamiento para que sea consistente entre páginas.
        players = query.all()
    player_ids = [player.id for player in players]
    avg_score_map: Dict[int, Optional[float]] = {}
    if player_ids:
        player_avg_rows = (
            db.query(PlayerStat.player_id, func.avg(PlayerStat.final_score))
            .filter(PlayerStat.player_id.in_(player_ids))
            .group_by(PlayerStat.player_id)
            .all()
        )
        avg_score_map = {
            player_id: (float(avg_score) if avg_score is not None else None)
            for player_id, avg_score in player_avg_rows
        }
    projections = batch_project_players(players, avg_score_map)
    player_rows = []
    for player in players:
        projection = projections.get(player.id)
        if projection:
            combined_pct = projection["combined_prob"] * 100
            row = {
                "player": player,
                "photo_url": player.photo_url or default_player_photo_url(
                    name=player.name,
                    national_id=player.national_id,
                    fallback=str(player.id),
                ),
                "category": projection["category"],
                "probability": f"{combined_pct:.1f}%",
                "prob_value": combined_pct,
                "fit_score": projection["fit_score"],
                "best_position": projection["recommended_position"],
                "best_score": projection["recommended_score"],
            }
        else:
            attr_map = player_attribute_map(player)
            best_position, best_score = recommend_position_from_attrs(attr_map)
            fit_score = weighted_score_from_attrs(attr_map, player.position)
            row = {
                "player": player,
                "photo_url": player.photo_url or default_player_photo_url(
                    name=player.name,
                    national_id=player.national_id,
                    fallback=str(player.id),
                ),
                "category": "Sin datos suficientes",
                "probability": "--",
                "prob_value": None,
                "fit_score": fit_score,
                "best_position": best_position,
                "best_score": best_score,
            }
        player_rows.append(row)

    if top_potential:
        player_rows = [
            item for item in player_rows
            if item["prob_value"] is not None and is_high_potential_probability(item["prob_value"] / 100.0)
        ]
        player_rows.sort(
            key=lambda item: (item["prob_value"] is not None, item["prob_value"] or -1.0),
            reverse=True,
        )
        total = len(player_rows)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(max(page, 1), total_pages)
        start = (page - 1) * per_page
        end = start + per_page
        player_rows = player_rows[start:end]
    db.close()
    return render_template("players.html",
                           players=player_rows,
                           search_term=search_term,
                           pos_list=pos_list, club_list=club_list, country_list=country_list,
                           pos_filter=pos_filter, club_filter=club_filter, country_filter=country_filter,
                           top_potential=top_potential, order_attr=order_attr,
                           page=page, total_pages=total_pages, total_results=total)

# ----------------------------------------------------
# DETALLE
@app.route("/player/<int:player_id>")
@login_required
def player_detail(player_id: int):
    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)
    history_synced = False
    if can_edit_player_data():
        history_synced = sync_player_attribute_history(player, db, note="Sincronizacion automatica al abrir ficha")
    if history_synced:
        db.commit()
    stats = fetch_player_stats(player_id, db_session=db)
    recent_stats = list(reversed(stats[-3:])) if stats else []
    attr_history = fetch_attribute_history(player_id, db_session=db)
    recent_attributes = list(reversed(attr_history[-3:])) if attr_history else []
    attribute_summary = summarize_attribute_history(attr_history)
    stats_summary = summarize_stats(stats)
    db.close()  # player_detail_db_closed
    player_photo_url = player.photo_url or default_player_photo_url(
        name=player.name,
        national_id=player.national_id,
        fallback=str(player.id),
    )
    projection = compute_projection(player, stats)
    attr_map = player_attribute_map(player)
    best_position, best_position_score = recommend_position_from_attrs(attr_map)
    current_fit = weighted_score_from_attrs(attr_map, player.position)
    position_ranking = [
        {
            "position": pos,
            "score": weighted_score_from_attrs(attr_map, pos),
            "is_current": normalized_position(player.position) == pos,
        }
        for pos in POSITION_OPTIONS
    ]
    position_ranking.sort(key=lambda item: item["score"], reverse=True)
    technical_attributes = [
        {"label": "Ritmo", "value": player.pace},
        {"label": "Disparo", "value": player.shooting},
        {"label": "Pase", "value": player.passing},
        {"label": "Regate", "value": player.dribbling},
        {"label": "Defensa", "value": player.defending},
        {"label": "Físico", "value": player.physical},
        {"label": "Visión", "value": player.vision},
        {"label": "Marcaje", "value": player.tackling},
        {"label": "Determinación", "value": player.determination},
        {"label": "Técnica", "value": player.technique},
    ]

    def build_trait(name: str, score: float, strengths: str, follow_up: str, improvement: str) -> dict:
        band = score_band(score)
        messaging = {"Alto": strengths, "Medio": follow_up, "Bajo": improvement}
        return {
            "name": name,
            "score": round(score, 1),
            "band": band,
            "message": messaging[band],
        }

    psychological_profile = [
        build_trait(
            "Resiliencia competitiva",
            player.determination,
            "Sostiene el esfuerzo bajo presión; indicado para partidos decisivos.",
            "Trabajar rutinas de respiración y feedback constante para fortalecer su respuesta en escenarios adversos.",
            "Recomendar intervención del área psicológica y refuerzo en hábitos de disciplina diaria.",
        ),
        build_trait(
            "Visión táctica",
            (player.vision + player.passing) / 2,
            "Lee espacios y acelera cambios de juego, facilita la progresión del equipo.",
            "Incrementar análisis de vídeo para mejorar la toma de decisiones en el último tercio.",
            "Diseñar ejercicios de toma de decisiones en superioridad/inferioridad numérica.",
        ),
        build_trait(
            "Creatividad ofensiva",
            (player.technique + player.dribbling) / 2,
            "Desborde y control diferenciales; puede romper líneas defensivas.",
            "Trabajar gestos técnicos específicos a alta velocidad para trasladar virtudes al contexto profesional.",
            "Enfocar sesiones en conducción orientada y confianza en el uno contra uno.",
        ),
        build_trait(
            "Liderazgo comunicacional",
            (player.vision + player.determination + player.passing) / 3,
            "Influye en sus compañeros y ordena fases ofensivas.",
            "Definir responsabilidades puntuales dentro del equipo para ganar protagonismo progresivo.",
            "Establecer mentoría con referentes del plantel y dinámicas de comunicación en cancha.",
        ),
    ]

    development_focus = compute_suggestions(player, threshold=15, top_n=3)

    return render_template(
        "player_detail.html",
        player=player,
        player_photo_url=player_photo_url,
        technical_attributes=technical_attributes,
        radar_labels=[attr["label"] for attr in technical_attributes],
        radar_values=[attr["value"] for attr in technical_attributes],
        psychological_profile=psychological_profile,
        development_focus=development_focus,
        recent_stats=recent_stats,
        stats_summary=stats_summary,
        history_payload=stats_chart_payload(stats),
        attribute_history=recent_attributes,
        attribute_summary=attribute_summary,
        attribute_payload=attribute_chart_payload(attr_history),
        attribute_labels=ATTRIBUTE_LABELS,
        projection=projection,
        best_position=best_position,
        best_position_score=best_position_score,
        current_fit=current_fit,
        position_ranking=position_ranking[:3],
    )


@app.route("/player/<int:player_id>/stats", methods=["GET", "POST"])
@login_required
def player_stats(player_id: int):
    if request.method == "POST":
        _require_csrf()
        if not can_edit_player_data():
            abort(403)

    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)

    if request.method == "POST":
        action = request.form.get("action", "add")
        if action == "recalculate":
            refresh_player_potential(player, db)
            db.commit()
            invalidate_dashboard_cache()
            db.close()
            flash("Listo: se actualizo la proyeccion con los ultimos datos.", "success")
            return redirect(url_for("predict_player", player_id=player_id))
        errors: List[str] = []
        record_date = parse_date_field(request.form.get("record_date"), errors, "La fecha del registro")
        matches_played = validate_non_negative_int_field(request.form.get("matches_played"), "Partidos jugados", errors)
        goals = validate_non_negative_int_field(request.form.get("goals"), "Goles", errors)
        assists = validate_non_negative_int_field(request.form.get("assists"), "Asistencias", errors)
        minutes_played = validate_non_negative_int_field(request.form.get("minutes_played"), "Minutos jugados", errors)
        yellow_cards = validate_non_negative_int_field(request.form.get("yellow_cards"), "Tarjetas amarillas", errors)
        red_cards = validate_non_negative_int_field(request.form.get("red_cards"), "Tarjetas rojas", errors)
        pass_accuracy = validate_optional_float_range(
            request.form.get("pass_accuracy"),
            "Precision de pase",
            errors,
            0,
            100,
        )
        shot_accuracy = validate_optional_float_range(
            request.form.get("shot_accuracy"),
            "Precision de remate",
            errors,
            0,
            100,
        )
        duels_won_pct = validate_optional_float_range(
            request.form.get("duels_won_pct"),
            "Duelos ganados",
            errors,
            0,
            100,
        )
        final_score = validate_optional_float_range(
            request.form.get("final_score"),
            "Valoracion final",
            errors,
            1,
            10,
        )
        if errors:
            for message in errors:
                flash(message, "danger")
            stats = (
                db.query(PlayerStat)
                .filter(PlayerStat.player_id == player_id)
                .order_by(PlayerStat.record_date.desc(), PlayerStat.id.desc())
                .all()
            )
            summary = summarize_stats(list(reversed(stats)))
            db.close()
            return render_template(
                "player_stats.html",
                player=player,
                stats=stats,
                summary=summary,
            )

        stat = PlayerStat(
            player_id=player_id,
            record_date=record_date,
            matches_played=matches_played,
            goals=goals,
            assists=assists,
            minutes_played=minutes_played,
            yellow_cards=yellow_cards,
            red_cards=red_cards,
            pass_accuracy=pass_accuracy,
            shot_accuracy=shot_accuracy,
            duels_won_pct=duels_won_pct,
            final_score=final_score,
            notes=request.form.get("notes") or None,
        )
        if stat.final_score is None:
            stat.final_score = calculate_stats_rating(
                {
                    "matches": stat.matches_played,
                    "goals": stat.goals,
                    "assists": stat.assists,
                    "minutes": stat.minutes_played,
                    "pass_pct": stat.pass_accuracy,
                    "shot_pct": stat.shot_accuracy,
                    "duels_pct": stat.duels_won_pct,
                }
            )
        db.add(stat)
        db.commit()
        refresh_player_potential(player, db)
        db.commit()
        invalidate_dashboard_cache()
        db.close()
        flash("Listo: se agrego el registro al historial del jugador.", "success")
        return redirect(url_for("player_stats", player_id=player_id))

    stats = (db.query(PlayerStat)
             .filter(PlayerStat.player_id == player_id)
             .order_by(PlayerStat.record_date.desc(), PlayerStat.id.desc())
             .all())
    db.close()
    summary = summarize_stats(list(reversed(stats)))
    return render_template(
        "player_stats.html",
        player=player,
        stats=stats,
        summary=summary,
    )

# ----------------------------------------------------
# HISTORIAL DE ATRIBUTOS
@app.route("/player/<int:player_id>/attributes", methods=["GET", "POST"])
@login_required
def player_attributes(player_id: int):
    if request.method == "POST":
        _require_csrf()
        if not can_edit_player_data():
            abort(403)

    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)

    if request.method == "POST":
        action = request.form.get("action", "add")
        if action == "recalculate":
            refresh_player_potential(player, db)
            db.commit()
            invalidate_dashboard_cache()
            db.close()
            flash("Listo: se actualizo la proyeccion con los nuevos atributos.", "success")
            return redirect(url_for('predict_player', player_id=player_id))
        errors: List[str] = []
        record_date = parse_date_field(request.form.get("record_date"), errors, "La fecha del historial")

        entry = PlayerAttributeHistory(player_id=player_id, record_date=record_date, notes=request.form.get("notes") or None)
        has_any_value = False
        for field in ATTRIBUTE_FIELDS:
            raw = request.form.get(field)
            value = parse_int_field(raw, default=-1) if raw not in (None, "") else None
            if value is not None and not is_valid_attribute(value):
                errors.append(f"{ATTRIBUTE_LABELS[field]} debe estar entre 0 y 20.")
                value = None
            setattr(entry, field, value)
            if value is not None:
                has_any_value = True
                setattr(player, field, value)
        if not has_any_value:
            errors.append("Debes cargar al menos un atributo para guardar el historial.")
        if errors:
            for message in errors:
                flash(message, "danger")
            history = (
                db.query(PlayerAttributeHistory)
                .filter(PlayerAttributeHistory.player_id == player_id)
                .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
                .all()
            )
            ascending_history = list(reversed(history))
            summary = summarize_attribute_history(ascending_history)
            payload = attribute_chart_payload(ascending_history)
            db.close()
            return render_template(
                "player_attributes.html",
                player=player,
                history=history,
                summary=summary,
                attribute_labels=ATTRIBUTE_LABELS,
                payload=payload,
            )
        db.add(entry)
        refresh_player_potential(player, db)
        db.commit()
        invalidate_dashboard_cache()
        db.close()
        flash("Listo: se guardo el historial de atributos.", "success")
        return redirect(url_for("player_attributes", player_id=player_id))

    history = (db.query(PlayerAttributeHistory)
               .filter(PlayerAttributeHistory.player_id == player_id)
               .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
               .all())
    ascending_history = list(reversed(history))
    summary = summarize_attribute_history(ascending_history)
    payload = attribute_chart_payload(ascending_history)
    db.close()
    return render_template(
        "player_attributes.html",
        player=player,
        history=history,
        summary=summary,
        attribute_labels=ATTRIBUTE_LABELS,
        payload=payload,
    )

# ----------------------------------------------------
# EDITAR JUGADOR
@app.route('/edit_player/<int:player_id>', methods=['GET', 'POST'])
@roles_required(ROLE_ADMIN, ROLE_SCOUT)
def edit_player(player_id):
    if request.method == "POST":
        _require_csrf()

    db = Session()
    player = db.get(Player, player_id)
    if not player:
        db.close()
        abort(404)
    if request.method == 'POST':
        errors: List[str] = []
        name = (request.form.get('name') or '').strip()
        national_id = normalize_identifier(request.form.get("national_id"))
        age = parse_int_field(request.form.get('age'))
        position = normalize_position_choice(request.form.get('position'))
        club = (request.form.get('club') or '').strip() or None
        country = (request.form.get('country') or '').strip() or None
        photo_url = (request.form.get('photo_url') or '').strip() or None
        if not name:
            errors.append("El nombre es obligatorio.")
        if not national_id:
            errors.append("El DNI debe contener solo números.")
        else:
            repeated = (
                db.query(Player.id)
                .filter(Player.national_id == national_id, Player.id != player.id)
                .first()
            )
            if repeated:
                errors.append("El DNI ingresado pertenece a otro jugador.")
        if not is_valid_eval_age(age):
            errors.append("La edad debe estar entre 12 y 18 años.")
        attr_values: Dict[str, int] = {}
        for field in ATTRIBUTE_FIELDS:
            value = parse_int_field(request.form.get(field), getattr(player, field))
            if not is_valid_attribute(value):
                errors.append(f"{ATTRIBUTE_LABELS[field]} debe estar entre 0 y 20.")
            attr_values[field] = value
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return render_template('edit_player.html', player=player, form_data=request.form, position_options=POSITION_OPTIONS)
        player.name = name
        player.national_id = national_id
        player.age = age
        player.position = position
        player.club = club
        player.country = country
        player.photo_url = photo_url or default_player_photo_url(
            name=name,
            national_id=national_id,
            fallback=str(player.id),
        )
        for field, value in attr_values.items():
            setattr(player, field, value)
        player.potential_label = True if request.form.get('potential_label') == '1' else False
        sync_player_attribute_history(player, db, note="Actualizacion de ficha")
        refresh_player_potential(player, db)
        db.commit()
        invalidate_dashboard_cache()
        db.close()
        flash("Listo: se actualizo la ficha del jugador.", "success")
        return redirect(url_for('player_detail', player_id=player_id))
    db.close()
    return render_template('edit_player.html', player=player, position_options=POSITION_OPTIONS)

# ----------------------------------------------------
# ELIMINAR JUGADOR
@app.route('/delete_player/<int:player_id>', methods=['POST'])
@roles_required(ROLE_ADMIN, ROLE_SCOUT)
def delete_player(player_id):
    _require_csrf()

    db = Session()
    player = db.get(Player, player_id)
    if not player:
        db.close()
        abort(404)
    db.delete(player)
    db.commit()
    invalidate_dashboard_cache()
    db.close()
    flash("Listo: se elimino el jugador del seguimiento.", "success")
    return redirect(url_for('index'))

@app.route("/player/<int:player_id>/predict")
@login_required
def predict_player(player_id: int):
    if model is None:
        return "No hay calculo disponible todavia. Actualiza los datos desde Configuracion.", 500

    db = Session()
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        db.close()
        abort(404)
    if can_edit_player_data():
        projection = refresh_player_potential(player, db)
    else:
        projection = compute_projection(player, db_session=db)
    player_view = SimpleNamespace(**player.to_dict())
    player_view.photo_url = player.photo_url or default_player_photo_url(
        name=player.name,
        national_id=player.national_id,
        fallback=str(player.id),
    )
    player_view.potential_label = (
        is_high_potential_probability(projection["combined_prob"])
        if projection else player.potential_label
    )
    suggestions = compute_suggestions(player_view)
    attr_map = player_attribute_map(player)
    best_position, best_position_score = recommend_position_from_attrs(attr_map)
    current_fit = weighted_score_from_attrs(attr_map, player.position)

    if not projection:
        stats_summary = summarize_stats([])
        prob_base = prob_combined = 0.0
        prob_calibrated_combined = None
        category = "Sin datos suficientes"
        stats_history = []
    else:
        prob_base = projection["base_prob"]
        prob_combined = projection["combined_prob"]
        prob_calibrated_combined = projection.get("calibrated_combined_prob")
        category = projection["category"]
        stats_summary = projection["stats_summary"]
        stats_history = projection["history"]
    history_payload = stats_chart_payload(stats_history)
    attribute_payload = attribute_chart_payload(fetch_attribute_history(player_id))
    if can_edit_player_data():
        db.commit()
        invalidate_dashboard_cache()
    else:
        db.rollback()
    db.close()

    return render_template(
        "prediction.html",
        player=player_view,
        probability=f"{prob_combined*100:.1f}%",
        probability_base=f"{prob_base*100:.1f}%",
        probability_calibrated=(
            f"{float(prob_calibrated_combined)*100:.1f}%"
            if prob_calibrated_combined is not None
            else None
        ),
        probability_delta=(prob_combined - prob_base) * 100,
        category=category,
        suggestions=suggestions,
        stats_summary=stats_summary,
        history_payload=history_payload,
        attribute_payload=attribute_payload,
        attribute_labels=ATTRIBUTE_LABELS,
        best_position=best_position,
        best_position_score=best_position_score,
        current_fit=current_fit,
    )
# ----------------------------------------------------
# COMPARADORES
@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare_players():
    db = Session()

    # Limitar la cantidad de jugadores que se cargan en el combo
    MAX_COMPARE_PLAYERS = 2000
    rows = (
        db.query(Player.id, Player.name, Player.position)
        .order_by(Player.name.asc())
        .limit(MAX_COMPARE_PLAYERS + 1)
        .all()
    )
    truncated = len(rows) > MAX_COMPARE_PLAYERS
    if truncated:
        rows = rows[:MAX_COMPARE_PLAYERS]

    # Usamos SimpleNamespace para tener player.id / player.name / player.position
    players = [
        SimpleNamespace(id=row[0], name=row[1], position=row[2])
        for row in rows
    ]

    selected_one = None
    selected_two = None
    comparison = None
    target_position = None

    if request.method == "POST":
        selected_one = request.form.get("player_one", type=int)
        selected_two = request.form.get("player_two", type=int)
        target_position_raw = request.form.get("target_position")
        target_position = normalized_position(target_position_raw) if target_position_raw else None

        if selected_one and selected_two and selected_one != selected_two:
            # Traemos SOLO los dos jugadores seleccionados
            selected = (
                db.query(Player)
                .filter(Player.id.in_([selected_one, selected_two]))
                .all()
            )
            players_map = {p.id: p for p in selected}
            player_one = players_map.get(selected_one)
            player_two = players_map.get(selected_two)

            if player_one and player_two:
                # Stats solo de estos dos jugadores
                stats_one = fetch_player_stats(player_one.id, db_session=db)
                stats_two = fetch_player_stats(player_two.id, db_session=db)

                summary_one = summarize_stats(stats_one)
                summary_two = summarize_stats(stats_two)
                avg_score_map = {
                    player_one.id: summary_one.get("avg_final_score"),
                    player_two.id: summary_two.get("avg_final_score"),
                }

                projections = batch_project_players([player_one, player_two], avg_score_map)
                projection_one = projections.get(player_one.id)
                projection_two = projections.get(player_two.id)

                prob_one = projection_one["combined_prob"] if projection_one else 0.0
                prob_two = projection_two["combined_prob"] if projection_two else 0.0

                # Comparación atributo por atributo
                attr_rows = []
                score_one = 0
                score_two = 0

                base_position = target_position or normalized_position(player_one.position or player_two.position)
                weights_map = position_weights(base_position)

                for field in ATTRIBUTE_FIELDS:
                    label = ATTRIBUTE_LABELS[field]
                    value_one = getattr(player_one, field)
                    value_two = getattr(player_two, field)
                    weight = weights_map.get(field, 0.0)

                    if value_one > value_two:
                        winner = 1
                        score_one += weight
                    elif value_two > value_one:
                        winner = 2
                        score_two += weight
                    else:
                        winner = 0

                    attr_rows.append(
                        {
                            "label": label,
                            "value_one": value_one,
                            "value_two": value_two,
                            "weight": weight,
                            "winner": winner,
                        }
                    )

                avg_one = summary_one.get("avg_final_score")
                avg_two = summary_two.get("avg_final_score")

                attr_map_one = player_attribute_map(player_one)
                attr_map_two = player_attribute_map(player_two)
                total_one = weighted_score_from_attrs(attr_map_one, base_position) + (avg_one or 0)
                total_two = weighted_score_from_attrs(attr_map_two, base_position) + (avg_two or 0)
                best_pos_one, best_pos_one_score = recommend_position_from_attrs(attr_map_one)
                best_pos_two, best_pos_two_score = recommend_position_from_attrs(attr_map_two)
                fit_one = weighted_score_from_attrs(attr_map_one, base_position)
                fit_two = weighted_score_from_attrs(attr_map_two, base_position)

                if total_one > total_two:
                    conclusion = (
                        f"{player_one.name} presenta mejores indicadores generales "
                        f"respecto a {player_two.name}."
                    )
                elif total_two > total_one:
                    conclusion = (
                        f"{player_two.name} presenta mejores indicadores generales "
                        f"respecto a {player_one.name}."
                    )
                else:
                    conclusion = (
                        "Ambos jugadores presentan indicadores equivalentes "
                        "en la comparación."
                    )

                comparison = {
                    "player_one": {
                        "name": player_one.name,
                        "position": player_one.position,
                        "probability": prob_one,
                        "avg_score": avg_one,
                        "fit_score": fit_one,
                        "best_position": best_pos_one,
                        "best_position_score": best_pos_one_score,
                    },
                    "player_two": {
                        "name": player_two.name,
                        "position": player_two.position,
                        "probability": prob_two,
                        "avg_score": avg_two,
                        "fit_score": fit_two,
                        "best_position": best_pos_two,
                        "best_position_score": best_pos_two_score,
                    },
                    "attributes": attr_rows,
                    "score_one": score_one,
                    "score_two": score_two,
                    "conclusion": conclusion,
                    "target_position": base_position,
                }

    db.close()
    return render_template(
        "compare.html",
        players=players,
        selected_one=selected_one,
        selected_two=selected_two,
        comparison=comparison,
        truncated=truncated,
        max_players=MAX_COMPARE_PLAYERS,
        target_position=target_position,
    )


@app.route("/compare/multi", methods=["GET", "POST"])
@login_required
def compare_multi():
    db = Session()

    # Igual que en el comparador 1vs1: limitamos la lista de jugadores
    MAX_COMPARE_PLAYERS = 2000
    rows = (
        db.query(Player.id, Player.name, Player.position)
        .order_by(Player.name.asc())
        .limit(MAX_COMPARE_PLAYERS + 1)
        .all()
    )
    truncated = len(rows) > MAX_COMPARE_PLAYERS
    if truncated:
        rows = rows[:MAX_COMPARE_PLAYERS]

    players = [
        SimpleNamespace(id=row[0], name=row[1], position=row[2])
        for row in rows
    ]

    selected_ids: List[int] = []
    comparison = None
    target_position = None

    # Ranking global por puesto (4 pestañas): Arquero, Defensa, Mediocampista, Delantero.
    # Nota: Lateral se agrupa en Defensa para simplificar lectura táctica.
    position_tabs = [
        {"key": "arquero", "label": "Arqueros", "bucket": "Portero"},
        {"key": "defensa", "label": "Defensas", "bucket": "Defensa"},
        {"key": "mediocampo", "label": "Mediocampistas", "bucket": "Mediocampista"},
        {"key": "delantera", "label": "Delanteros", "bucket": "Delantero"},
    ]
    ranking_by_position: Dict[str, List[Dict[str, object]]] = {tab["key"]: [] for tab in position_tabs}

    all_players_for_ranking = (
        db.query(Player)
        .options(
            load_only(
                Player.id,
                Player.name,
                Player.age,
                Player.position,
                Player.club,
                Player.pace,
                Player.shooting,
                Player.passing,
                Player.dribbling,
                Player.defending,
                Player.physical,
                Player.vision,
                Player.tackling,
                Player.determination,
                Player.technique,
            )
        )
        .all()
    )
    ranking_avg_rows = (
        db.query(PlayerStat.player_id, func.avg(PlayerStat.final_score))
        .group_by(PlayerStat.player_id)
        .all()
    )
    ranking_avg_score_map = {
        player_id: (float(avg_score) if avg_score is not None else None)
        for player_id, avg_score in ranking_avg_rows
    }
    ranking_projections = batch_project_players(all_players_for_ranking, ranking_avg_score_map)
    for player in all_players_for_ranking:
        projection = ranking_projections.get(player.id)
        if not projection:
            continue
        normalized_pos = normalized_position(player.position)
        if normalized_pos == "Lateral":
            normalized_pos = "Defensa"

        tab = next((tab for tab in position_tabs if tab["bucket"] == normalized_pos), None)
        if not tab:
            continue

        ranking_by_position[tab["key"]].append(
            {
                "id": player.id,
                "name": player.name,
                "position": player.position,
                "club": player.club,
                "age": player.age,
                "probability": round(float(projection["combined_prob"]) * 100, 1),
                "category": projection["category"],
            }
        )

    for tab in position_tabs:
        rows = ranking_by_position[tab["key"]]
        rows.sort(key=lambda item: item["probability"], reverse=True)
        ranking_by_position[tab["key"]] = rows[:10]

    if request.method == "POST":
        raw_ids = request.form.getlist("players")
        try:
            selected_ids = [int(pid) for pid in raw_ids][:10]
        except ValueError:
            selected_ids = []
        target_position_raw = request.form.get("target_position")
        target_position = normalized_position(target_position_raw) if target_position_raw else None

        if selected_ids:
            selected_players = (
                db.query(Player)
                .filter(Player.id.in_(selected_ids))
                .order_by(Player.name.asc())
                .all()
            )

            if selected_players:
                # Promedios de final_score solo para estos jugadores (se usa para el total)
                score_rows = (
                    db.query(PlayerStat.player_id, func.avg(PlayerStat.final_score))
                    .filter(
                        PlayerStat.player_id.in_([p.id for p in selected_players])
                    )
                    .group_by(PlayerStat.player_id)
                    .all()
                )
                avg_score_map = {
                    player_id: (float(avg) if avg is not None else None)
                    for player_id, avg in score_rows
                }
                selected_projections = batch_project_players(selected_players, avg_score_map)

                # Atributos a graficar (eran los del radar)
                labels = [ATTRIBUTE_LABELS[field] for field in ATTRIBUTE_FIELDS]

                # Datasets para gráfico de barras (uno por jugador)
                datasets = []
                colors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                ]

                ranking: List[dict] = []

                base_position = target_position
                for idx, player in enumerate(selected_players):
                    values = [getattr(player, field) for field in ATTRIBUTE_FIELDS]
                    color = colors[idx % len(colors)]

                    datasets.append(
                        {
                            "label": player.name,
                            "data": values,
                            "backgroundColor": color,
                            "borderColor": color,
                            "borderWidth": 1,
                        }
                    )

                    # Puntaje total = suma atributos + promedio histórico (si existe)
                    attr_map = player_attribute_map(player)
                    attribute_sum = weighted_score_from_attrs(attr_map, base_position or player.position)
                    avg_score = avg_score_map.get(player.id)
                    projection = selected_projections.get(player.id)
                    total = attribute_sum + (avg_score or 0)

                    # Mapa atributo -> valor (con los labels “bonitos”)
                    attributes_map = {
                        ATTRIBUTE_LABELS[f]: getattr(player, f) for f in ATTRIBUTE_FIELDS
                    }

                    ranking.append(
                        {
                            "name": player.name,
                            "position": player.position,
                            "attributes_map": attributes_map,
                            "total": round(total, 2),
                            "probability": round(float(projection["combined_prob"]) * 100, 1) if projection else None,
                        }
                    )

                # Ordenamos ranking por total (desc)
                ranking.sort(key=lambda item: item["total"], reverse=True)

                # Gráfico de barras horizontal con promedios históricos (igual que antes)
                score_labels = [r["name"] for r in ranking]
                # Para no mostrar la columna "promedio" en la tabla, lo usamos solo en el gráfico
                score_values = []
                for r in ranking:
                    # reconstruimos avg desde attributes_map+suma si quisiéramos, pero ya lo tenemos en avg_score_map
                    # buscamos por nombre
                    p = next((p for p in selected_players if p.name == r["name"]), None)
                    avg_val = avg_score_map.get(p.id) if p else None
                    score_values.append(avg_val or 0)

                comparison = {
                    "chart_payload": {
                        "labels": labels,
                        "datasets": datasets,
                    },
                    "score_payload": {
                        "labels": score_labels,
                        "datasets": [
                            {
                                "label": "Promedio historial",
                                "data": score_values,
                                "backgroundColor": "rgba(13, 110, 253, 0.6)",
                                "borderColor": "#0d6efd",
                            }
                        ],
                    },
                    "ranking": ranking,   # ahora trae attributes_map y total
                    "target_position": base_position,
                }

    db.close()
    return render_template(
        "compare_multi.html",
        players=players,
        selected_ids=selected_ids,
        comparison=comparison,
        truncated=truncated,
        max_players=MAX_COMPARE_PLAYERS,
        target_position=target_position,
        position_tabs=position_tabs,
        ranking_by_position=ranking_by_position,
    )




# ----------------------------------------------------
# CRUD COACHES (igual estrategia)
@app.route("/coaches")
@login_required
def list_coaches():
    db = Session()
    coaches = db.query(Coach).all()
    db.close()
    return render_template("coaches.html", coaches=coaches)

@app.route("/coaches/new", methods=["GET", "POST"])
@roles_required(ROLE_ADMIN)
def new_coach():
    if request.method == "POST":
        _require_csrf()

    if request.method == "POST":
        db = Session()
        coach = Coach(
            name=request.form["name"],
            role=request.form["role"],
            age=request.form.get("age"),
            club=request.form.get("club"),
            country=request.form.get("country")
        )
        db.add(coach)
        db.commit()
        db.close()
        return redirect(url_for("list_coaches"))
    return render_template("coach_form.html", coach=None)

@app.route("/coaches/edit/<int:coach_id>", methods=["GET", "POST"])
@roles_required(ROLE_ADMIN)
def edit_coach(coach_id):
    if request.method == "POST":
        _require_csrf()

    db = Session()
    coach = db.get(Coach, coach_id)
    if not coach:
        db.close()
        abort(404)
    if request.method == "POST":
        coach.name = request.form["name"]
        coach.role = request.form["role"]
        coach.age = request.form.get("age")
        coach.club = request.form.get("club")
        coach.country = request.form.get("country")
        db.commit()
        db.close()
        return redirect(url_for("list_coaches"))
    db.close()
    return render_template("coach_form.html", coach=coach)

@app.route("/coaches/delete/<int:coach_id>", methods=["POST"])
@roles_required(ROLE_ADMIN)
def delete_coach(coach_id):
    _require_csrf()

    db = Session()
    coach = db.get(Coach, coach_id)
    if not coach:
        db.close()
        abort(404)
    db.delete(coach)
    db.commit()
    db.close()
    return redirect(url_for("list_coaches"))


# ----------------------------------------------------
# CRUD DIRECTORES
@app.route("/directors")
@login_required
def list_directors():
    db = Session()
    directors = db.query(Director).all()
    db.close()
    return render_template("directors.html", directors=directors)


@app.route("/directors/new", methods=["GET", "POST"])
@roles_required(ROLE_ADMIN)
def new_director():
    if request.method == "POST":
        _require_csrf()

    if request.method == "POST":
        db = Session()
        director = Director(
            name=request.form["name"],
            position=request.form["position"],
            age=request.form.get("age"),
            club=request.form.get("club"),
            country=request.form.get("country"),
        )
        db.add(director)
        db.commit()
        db.close()
        return redirect(url_for("list_directors"))
    return render_template("director_form.html", director=None)


@app.route("/directors/edit/<int:director_id>", methods=["GET", "POST"])
@roles_required(ROLE_ADMIN)
def edit_director(director_id):
    if request.method == "POST":
        _require_csrf()

    db = Session()
    director = db.get(Director, director_id)
    if not director:
        db.close()
        abort(404)
    if request.method == "POST":
        director.name = request.form["name"]
        director.position = request.form["position"]
        director.age = request.form.get("age")
        director.club = request.form.get("club")
        director.country = request.form.get("country")
        db.commit()
        db.close()
        return redirect(url_for("list_directors"))
    db.close()
    return render_template("director_form.html", director=director)


@app.route("/directors/delete/<int:director_id>", methods=["POST"])
@roles_required(ROLE_ADMIN)
def delete_director(director_id):
    _require_csrf()

    db = Session()
    director = db.get(Director, director_id)
    if not director:
        db.close()
        abort(404)
    db.delete(director)
    db.commit()
    db.close()
    return redirect(url_for("list_directors"))


@app.route("/settings", methods=["GET", "POST"])
@roles_required(ROLE_ADMIN)
def settings():
    # CSRF mínimo para POST
    if request.method == "POST":
        _require_csrf()

    status_messages: List[str] = []
    modal_message: Optional[str] = None

    if request.method == "POST":
        action = request.form.get("action")
        if action == "update_database":
            start = time.time()
            if not _PIPELINE_LOCK.acquire(blocking=False):
                status_messages = ["Ya hay una actualizacion en curso. Reintenta en unos minutos."]
                flash("Ya hay una actualizacion en curso. Reintenta en unos minutos.", "warning")
            else:
                try:
                    success, logs = update_database_pipeline(
                        limit=EVAL_POOL_MAX,
                        sync_shortlist=SYNC_SHORTLIST_ENABLED,
                    )
                    status_messages = logs
                finally:
                    duration = round(time.time() - start, 2)
                    status_messages.append(f"Duracion total de la actualizacion: {duration}s")
                    try:
                        app.logger.info("Pipeline update_database finished in %ss", duration)
                    except Exception:
                        pass
                    _PIPELINE_LOCK.release()

                if success:
                    if SYNC_SHORTLIST_ENABLED:
                        modal_message = "Se actualizaron puntajes y se sincronizo la base operativa."
                    else:
                        modal_message = "Se actualizaron puntajes (sin sincronizar jugadores operativos)."
                    flash("Listo: la actualizacion general finalizo correctamente.", "success")
                else:
                    flash("No se pudo completar la actualizacion. Revisa el detalle.", "danger")
        elif action == "cleanup_demo_data":
            db = Session()
            try:
                summary = cleanup_operational_data(db)
                db.commit()
                invalidate_dashboard_cache()
                status_messages = [
                    "Auditoria y limpieza de base operativa completadas.",
                    f"Jugadores invalidos removidos: {summary['removed_invalid_players']}",
                    f"Fotos completadas: {summary['photo_updates']}",
                    f"Historial tecnico sincronizado: {summary['history_updates']}",
                    f"Jugadores recortados por limite operativo: {summary['trimmed_players']}",
                    (
                        "Calidad actual -> "
                        f"total={summary['players_total']}, "
                        f"sin_dni={summary['missing_national_id']}, "
                        f"edad_invalida={summary['invalid_age']}, "
                        f"sin_nombre={summary['missing_name']}, "
                        f"sin_posicion={summary['missing_position']}, "
                        f"sin_foto={summary['missing_photo_url']}, "
                        f"stats_huerfanos={summary['orphan_stats']}, "
                        f"historial_huerfano={summary['orphan_attribute_history']}, "
                        f"exceso_limite={summary['over_limit_players']}"
                    ),
                ]
                if summary["removed_invalid_players"] or summary["trimmed_players"] or summary["photo_updates"]:
                    flash("Listo: la base operativa quedo auditada y consistente para la demo.", "success")
                else:
                    flash("No se detectaron inconsistencias nuevas en la base operativa.", "info")
            except Exception as exc:
                db.rollback()
                status_messages = [f"No se pudo completar la limpieza operativa: {exc}"]
                flash("No se pudo completar la limpieza de la base operativa.", "danger")
            finally:
                db.close()

    db = Session()
    try:
        data_quality = compute_operational_data_quality(db)
    finally:
        db.close()
    return render_template(
        "settings.html",
        status_messages=status_messages,
        modal_message=modal_message,
        data_quality=data_quality,
        eval_pool_max=EVAL_POOL_MAX,
    )



def parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    value = value.strip().lower()
    return value in {"1", "true", "t", "yes", "y", "si", "sí", "alto"}


def parse_int_field(value: Optional[str], default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_date_field(value: Optional[str], errors: List[str], label: str = "La fecha") -> date:
    if not value:
        return date.today()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        errors.append(f"{label} debe tener formato YYYY-MM-DD.")
        return date.today()


def validate_non_negative_int_field(
    value: Optional[str],
    label: str,
    errors: List[str],
    allow_blank: bool = True,
) -> int:
    if value in (None, ""):
        if allow_blank:
            return 0
        errors.append(f"{label} es obligatorio.")
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} debe ser un numero entero.")
        return 0
    if parsed < 0:
        errors.append(f"{label} no puede ser negativo.")
    return parsed


def validate_optional_float_range(
    value: Optional[str],
    label: str,
    errors: List[str],
    min_value: float,
    max_value: float,
) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} debe ser un numero valido.")
        return None
    if parsed < min_value or parsed > max_value:
        errors.append(f"{label} debe estar entre {min_value:g} y {max_value:g}.")
        return None
    return parsed


def normalize_position_choice(value: Optional[str]) -> str:
    return normalized_position(value)


@app.route("/players/manage", methods=["GET", "POST"])
@roles_required(ROLE_ADMIN, ROLE_SCOUT)
def manage_players():
    if request.method == "POST":
        _require_csrf()

    if request.method == "POST":
        mode = request.form.get("mode", "single")
        db = Session()
        created: List[str] = []
        errors: List[str] = []
        try:
            if mode == "single":
                current_total = db.query(func.count(Player.id)).scalar() or 0
                if current_total >= EVAL_POOL_MAX:
                    errors.append(
                        f"Se alcanzo el maximo de {EVAL_POOL_MAX} jugadores evaluables. "
                        "Elimina o actualiza jugadores existentes."
                    )
                name = (request.form.get("name") or "").strip()
                national_id = normalize_identifier(request.form.get("national_id"))
                age = parse_int_field(request.form.get("age"))
                position = normalize_position_choice(request.form.get("position"))
                club = (request.form.get("club") or "").strip() or None
                country = (request.form.get("country") or "").strip() or None
                photo_url = (request.form.get("photo_url") or "").strip() or None
                if not name:
                    errors.append("El nombre es obligatorio.")
                if not national_id:
                    errors.append("Ingresa un DNI/ID valido (solo numeros).")
                else:
                    repeated = (
                        db.query(Player.id)
                        .filter(Player.national_id == national_id)
                        .first()
                    )
                    if repeated:
                        errors.append("Ese DNI ya esta registrado en la base.")
                if not is_valid_eval_age(age):
                    errors.append("La edad debe estar entre 12 y 18 anos.")
                attr_values: Dict[str, int] = {}
                for field in ATTRIBUTE_FIELDS:
                    value = parse_int_field(request.form.get(field), 10)
                    if not is_valid_attribute(value):
                        errors.append(f"{ATTRIBUTE_LABELS[field]} debe ubicarse entre 0 y 20.")
                    attr_values[field] = value
                if not errors:
                    player = Player(
                        name=name,
                        national_id=national_id,
                        age=age,
                        position=position,
                        club=club,
                        country=country,
                        photo_url=photo_url or default_player_photo_url(name=name, national_id=national_id),
                        potential_label=parse_bool(request.form.get("potential_label")),
                        **attr_values,
                    )
                    db.add(player)
                    db.flush()
                    sync_player_attribute_history(player, db, note="Alta de jugador")
                    refresh_player_potential(player, db)
                    db.commit()
                    invalidate_dashboard_cache()
                    created.append(player.name)
                    flash(f"Listo: se agrego a {player.name} al seguimiento.", "success")
                    return redirect(url_for("manage_players"))
            elif mode == "bulk":
                bulk_input = request.form.get("bulk_input") or ""
                lines = [line.strip() for line in bulk_input.splitlines() if line.strip()]
                current_total = db.query(func.count(Player.id)).scalar() or 0
                available_slots = max(EVAL_POOL_MAX - current_total, 0)
                if available_slots <= 0:
                    errors.append(
                        f"No hay cupo disponible. Limite actual: {EVAL_POOL_MAX} jugadores evaluables."
                    )
                if not lines:
                    errors.append("Ingresa al menos una fila en la carga masiva.")
                else:
                    seen_ids = set()
                    for idx, line in enumerate(lines, start=1):
                        if len(created) >= available_slots:
                            errors.append(
                                f"Se alcanzo el limite de {EVAL_POOL_MAX} jugadores. "
                                "El resto de filas fue omitido."
                            )
                            break
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) < 17:
                            errors.append(f"Linea {idx}: se esperaban 17 o 18 columnas (recibido {len(parts)}).")
                            continue
                        try:
                            name = parts[0]
                            national_id = normalize_identifier(parts[1])
                            age = int(parts[2])
                            position = normalize_position_choice(parts[3])
                            club = parts[4] or None
                            country = parts[5] or None
                            raw_photo_url = parts[17] if len(parts) > 17 else ""
                            if not national_id:
                                raise ValueError("DNI no valido.")
                            if national_id in seen_ids:
                                raise ValueError("DNI repetido en la carga.")
                            exists = (
                                db.query(Player.id)
                                .filter(Player.national_id == national_id)
                                .first()
                            )
                            if exists:
                                raise ValueError("DNI ya registrado en la base.")
                            if not is_valid_eval_age(age):
                                raise ValueError("Edad fuera del rango permitido (12-18).")
                            attr_values = {}
                            for offset, field in enumerate(ATTRIBUTE_FIELDS, start=6):
                                value = int(parts[offset])
                                if not is_valid_attribute(value):
                                    raise ValueError(f"{ATTRIBUTE_LABELS[field]} fuera de rango.")
                                attr_values[field] = value
                            potential_flag = parse_bool(parts[16]) if len(parts) > 16 else False
                            player = Player(
                                name=name,
                                national_id=national_id,
                                age=age,
                                position=position,
                                club=club,
                                country=country,
                                photo_url=(raw_photo_url.strip() or default_player_photo_url(name=name, national_id=national_id)),
                                potential_label=potential_flag,
                                **attr_values,
                            )
                            seen_ids.add(national_id)
                        except (ValueError, IndexError) as exc:
                            errors.append(f"Linea {idx}: {exc}")
                            continue
                        db.add(player)
                        db.flush()
                        sync_player_attribute_history(player, db, note="Alta masiva de jugador")
                        refresh_player_potential(player, db)
                        created.append(player.name)
                    if created and not errors:
                        db.commit()
                        invalidate_dashboard_cache()
                        flash(f"Listo: se agregaron {len(created)} jugadores.", "success")
                    elif created and errors:
                        db.commit()
                        invalidate_dashboard_cache()
                        flash(f"Se agregaron {len(created)} jugadores, pero quedaron advertencias para revisar.", "warning")
                    else:
                        db.rollback()
            else:
                errors.append("Modo de carga desconocido.")
        finally:
            db.close()
        if errors:
            for message in errors:
                flash(message, "danger")
        if created and mode == "bulk":
            return redirect(url_for("manage_players"))
    attribute_sequence = [(field, ATTRIBUTE_LABELS[field]) for field in ATTRIBUTE_FIELDS]
    db_metrics = Session()
    current_total = db_metrics.query(func.count(Player.id)).scalar() or 0
    db_metrics.close()
    return render_template("manage_players.html",
                           attribute_labels=ATTRIBUTE_LABELS,
                           attribute_sequence=attribute_sequence,
                           position_options=POSITION_OPTIONS,
                           current_total=current_total,
                           max_players=EVAL_POOL_MAX)



# --- Error handlers (observabilidad mínima) ---

@app.errorhandler(400)
def handle_400(e):
    app.logger.warning("400 Bad Request - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=400, message="Solicitud inválida"), 400

@app.errorhandler(413)
def handle_413(e):
    app.logger.warning("413 Payload Too Large - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=413, message="Request demasiado grande"), 413



@app.errorhandler(403)
def handle_403(e):
    app.logger.warning("403 Forbidden - path=%s user_id=%s role=%s", request.path, session.get("user_id"), session.get("role"))
    return render_template("error.html", code=403, message="Acceso denegado"), 403

@app.errorhandler(404)
def handle_404(e):
    app.logger.info("404 Not Found - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=404, message="Recurso no encontrado"), 404

@app.errorhandler(500)
def handle_500(e):
    app.logger.exception("500 Internal Server Error - path=%s user_id=%s role=%s", request.path, session.get("user_id"), session.get("role"))
    return render_template("error.html", code=500, message="Error interno del servidor"), 500
if __name__ == "__main__":
    app.run(debug=True)
