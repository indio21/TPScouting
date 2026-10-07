"""Run the Block 8 demo smoke against a disposable SQLite database."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "scouting_app"
EVIDENCE_JSON = ROOT / "docs/revision_octubre_2026/evidencia_demo_bloque8.json"
EVIDENCE_MD = ROOT / "docs/revision_octubre_2026/evidencia_demo_bloque8.md"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from iniciar_demo import prepare_demo


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "no instalado"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tpscouting_block8_") as directory:
        db_path = Path(directory) / "demo_vacia.db"
        summary = prepare_demo(db_path, 60, 42, "profesor_demo", "DemoProfesor123")

        import app as app_module
        from models import (
            PhysicalAssessment,
            Player,
            PlayerAttributeHistory,
            PlayerAvailability,
            PlayerMatchParticipation,
            PlayerStat,
            ScoutReport,
        )

        app_module.app.config.update(TESTING=True)
        session = app_module.Session()
        try:
            history_models = [
                PlayerStat,
                PlayerAttributeHistory,
                PlayerMatchParticipation,
                ScoutReport,
                PhysicalAssessment,
                PlayerAvailability,
            ]
            id_sets = [
                {row[0] for row in session.query(model.player_id).distinct().all()}
                for model in history_models
            ]
            complete_ids = set.intersection(*id_sets)
            if not complete_ids:
                raise RuntimeError("La base demo no contiene un jugador con todos los historiales.")
            player = session.get(Player, min(complete_ids))
            if player is None:
                raise RuntimeError("No se pudo recuperar el jugador de prueba.")
            player.photo_url = None
            session.commit()
            player_id = int(player.id)
            player_name = str(player.name)
            current_age = int(player.current_age)
            stored_age = int(player.age)
            category_year = int(player.category_year)
            history_counts = {
                model.__tablename__: int(session.query(model).filter(model.player_id == player_id).count())
                for model in history_models
            }
        finally:
            session.close()

        if current_age != stored_age or not 12 <= current_age <= 18:
            raise RuntimeError("La edad del jugador seleccionado no es coherente.")
        if any(value < 1 for value in history_counts.values()):
            raise RuntimeError(f"Faltan historiales para el jugador seleccionado: {history_counts}")

        client = app_module.app.test_client()
        login_page = client.get("/login")
        csrf_match = re.search(rb'name="csrf_token"[^>]*value="([^"]+)"', login_page.data)
        if login_page.status_code != 200 or not csrf_match:
            raise RuntimeError("No se pudo abrir el login u obtener el token CSRF.")

        login = client.post(
            "/login",
            data={
                "username": "profesor_demo",
                "password": "DemoProfesor123",
                "csrf_token": csrf_match.group(1).decode("utf-8"),
            },
            follow_redirects=False,
        )
        if login.status_code != 302 or "/players" not in login.headers.get("Location", ""):
            raise RuntimeError(f"El login no redirigió al listado: {login.status_code}")

        responses = {
            "players": client.get("/players"),
            "detail": client.get(f"/player/{player_id}"),
            "stats": client.get(f"/player/{player_id}/stats"),
            "attributes": client.get(f"/player/{player_id}/attributes"),
            "prediction": client.get(f"/player/{player_id}/predict"),
            "silhouette": client.get("/static/img/player-silhouette.svg"),
        }
        statuses = {name: response.status_code for name, response in responses.items()}
        if any(status != 200 for status in statuses.values()):
            raise RuntimeError(f"Una ruta de la demo falló: {statuses}")
        if player_name.encode("utf-8") not in responses["detail"].data:
            raise RuntimeError("La ficha no muestra el jugador seleccionado.")
        if b"/static/img/player-silhouette.svg" not in responses["detail"].data:
            raise RuntimeError("La ficha no utiliza la silueta local cuando falta la foto.")
        if player_name.encode("utf-8") not in responses["prediction"].data:
            raise RuntimeError("La vista de predicción no corresponde al jugador seleccionado.")
        if b"<svg" not in responses["silhouette"].data.lower():
            raise RuntimeError("La silueta local no es un SVG válido.")

        artifacts = {}
        for name in ("model.pt", "preprocessor.joblib", "probability_calibrator.joblib"):
            path = APP_DIR / name
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

        evidence = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "database": "SQLite temporal eliminada al finalizar",
            "seed": 42,
            "summary": summary,
            "selected_player": {
                "id": player_id,
                "age": current_age,
                "stored_age": stored_age,
                "category_year": category_year,
                "history_counts": history_counts,
            },
            "route_statuses": statuses,
            "login_redirect": login.headers.get("Location"),
            "local_silhouette_verified": True,
            "prediction_page_verified": True,
            "versions": {
                "python": sys.version.split()[0],
                "flask": package_version("Flask"),
                "sqlalchemy": package_version("SQLAlchemy"),
                "torch": package_version("torch"),
                "scikit-learn": package_version("scikit-learn"),
            },
            "artifacts": artifacts,
        }
        EVIDENCE_JSON.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        app_module.engine.dispose()

    lines = [
        "# Evidencia de demo desde base vacía — Bloque 8",
        "",
        f"Fecha UTC: {evidence['timestamp_utc']}.",
        "",
        "- Base SQLite temporal creada desde cero y eliminada al finalizar.",
        f"- Jugadores sintéticos: {summary['players']}; semilla: 42.",
        f"- Jugador controlado: ID {player_id}, edad {current_age}, categoría {category_year}.",
        f"- Historiales del jugador: `{history_counts}`.",
        f"- Rutas verificadas: `{statuses}`.",
        "- Login con CSRF y redirección interna: correcto.",
        "- Silueta SVG local ante foto ausente: correcta.",
        "- Vista de predicción con artefactos existentes: correcta.",
        "",
        "La prueba confirma el flujo técnico con datos sintéticos. No constituye validación con jugadores reales.",
    ]
    EVIDENCE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
