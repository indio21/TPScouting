import importlib
import json
from datetime import date
from types import SimpleNamespace

from werkzeug.security import generate_password_hash
from sqlalchemy.orm import sessionmaker


def _create_user(db, User, username, password, role="scout"):
    user = User(username=username, password_hash=generate_password_hash(password), role=role)
    db.add(user)
    db.commit()
    return user


def _get_csrf_token(client, path="/login"):
    client.get(path)
    with client.session_transaction() as sess:
        return sess.get("csrf_token")


def _login(client, username, password):
    csrf_token = _get_csrf_token(client, "/login")
    return client.post("/login", data={"username": username, "password": password, "csrf_token": csrf_token})


def _create_player(app_module, db, **overrides):
    payload = {
        "name": "Jugador Base",
        "national_id": "30123456",
        "age": 16,
        "position": "Defensa",
        "club": "Club Base",
        "country": "Argentina",
        "photo_url": "",
        "pace": 10,
        "shooting": 9,
        "passing": 11,
        "dribbling": 10,
        "defending": 12,
        "physical": 11,
        "vision": 10,
        "tackling": 12,
        "determination": 13,
        "technique": 9,
        "potential_label": False,
    }
    payload.update(overrides)
    player = app_module.Player(**payload)
    db.add(player)
    db.commit()
    return player


def test_refresh_player_potential_uses_same_high_threshold(app_module):
    dummy_player = SimpleNamespace(potential_label=False)
    original = app_module.compute_projection
    app_module.compute_projection = lambda player, db_session=None: {"combined_prob": 0.60}
    try:
        projection = app_module.refresh_player_potential(dummy_player)
    finally:
        app_module.compute_projection = original

    assert projection["combined_prob"] == 0.60
    assert dummy_player.potential_label is True


def test_register_rejects_weak_password(client, app_module, db):
    _create_user(db, app_module.User, "adminuser", "admin1234", role="administrador")
    _login(client, "adminuser", "admin1234")
    csrf_token = _get_csrf_token(client, "/register")

    response = client.post(
        "/register",
        data={
            "username": "nuevo",
            "password": "abc",
            "role": "scout",
            "csrf_token": csrf_token,
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "al menos 8 caracteres" in response.get_data(as_text=True)


def test_director_cannot_access_manage_players(client, app_module, db):
    _create_user(db, app_module.User, "director1", "director123", role="director")
    _login(client, "director1", "director123")

    response = client.get("/players/manage")

    assert response.status_code == 403


def test_scout_can_access_manage_players(client, app_module, db):
    _create_user(db, app_module.User, "scout1", "scout1234", role="scout")
    _login(client, "scout1", "scout1234")

    response = client.get("/players/manage")

    assert response.status_code == 200


def test_director_cannot_post_player_stats(client, app_module, db):
    director = _create_user(db, app_module.User, "director2", "director123", role="director")
    player = app_module.Player(
        name="Jugador Test",
        national_id="30111222",
        age=16,
        position="Defensa",
        club="Club Test",
        country="Argentina",
        photo_url="",
        pace=10,
        shooting=10,
        passing=10,
        dribbling=10,
        defending=10,
        physical=10,
        vision=10,
        tackling=10,
        determination=10,
        technique=10,
        potential_label=False,
    )
    db.add(player)
    db.commit()

    _login(client, director.username, "director123")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")
    response = client.post(
        f"/player/{player.id}/stats",
        data={
            "csrf_token": csrf_token,
            "action": "add",
            "record_date": "2026-04-15",
            "matches_played": 1,
            "minutes_played": 90,
        },
    )

    assert response.status_code == 403


def test_login_rate_limit_blocks_after_repeated_failures(client, app_module, db):
    _create_user(db, app_module.User, "blocked_user", "blocked123", role="scout")

    for _ in range(5):
        csrf_token = _get_csrf_token(client, "/login")
        response = client.post(
            "/login",
            data={"username": "blocked_user", "password": "mal", "csrf_token": csrf_token},
        )
        assert response.status_code == 200

    csrf_token = _get_csrf_token(client, "/login")
    blocked = client.post(
        "/login",
        data={"username": "blocked_user", "password": "mal", "csrf_token": csrf_token},
    )

    assert blocked.status_code == 429
    assert "bloquearon temporalmente" in blocked.get_data(as_text=True)


def test_timestamps_are_populated_on_new_records(app_module, db):
    player = app_module.Player(
        name="Jugador Audit",
        national_id="44555666",
        age=17,
        position="Delantero",
        club="Club Audit",
        country="Argentina",
        photo_url="",
        pace=12,
        shooting=13,
        passing=11,
        dribbling=14,
        defending=8,
        physical=12,
        vision=11,
        tackling=7,
        determination=15,
        technique=13,
        potential_label=True,
    )
    db.add(player)
    db.commit()
    db.refresh(player)

    assert player.created_at is not None
    assert player.updated_at is not None


def test_cleanup_operational_data_removes_legacy_players_without_national_id(app_module, db):
    invalid_player = app_module.Player(
        name="Legacy Sin DNI",
        national_id=None,
        age=16,
        position="Defensa",
        club="Club Demo",
        country="Argentina",
        photo_url="",
        pace=10,
        shooting=9,
        passing=11,
        dribbling=10,
        defending=12,
        physical=11,
        vision=10,
        tackling=12,
        determination=13,
        technique=9,
        potential_label=False,
    )
    valid_player = app_module.Player(
        name="Jugador Valido",
        national_id="55444333",
        age=15,
        position="Mediocampista",
        club="Club Demo",
        country="Argentina",
        photo_url="",
        pace=12,
        shooting=10,
        passing=13,
        dribbling=12,
        defending=9,
        physical=11,
        vision=13,
        tackling=8,
        determination=14,
        technique=12,
        potential_label=True,
    )
    db.add_all([invalid_player, valid_player])
    db.commit()

    summary = app_module.cleanup_operational_data(db)
    db.commit()

    assert summary["removed_invalid_players"] == 1
    assert summary["missing_national_id"] == 0
    assert db.query(app_module.Player).filter_by(name="Legacy Sin DNI").count() == 0
    assert db.query(app_module.Player).filter_by(name="Jugador Valido").count() == 1


def test_health_reports_operational_data_quality(client, app_module, db):
    player = app_module.Player(
        name="Legacy Health",
        national_id=None,
        age=16,
        position="Delantero",
        club="Club Test",
        country="Argentina",
        photo_url="",
        pace=11,
        shooting=12,
        passing=9,
        dribbling=13,
        defending=7,
        physical=11,
        vision=10,
        tackling=6,
        determination=14,
        technique=12,
        potential_label=False,
    )
    db.add(player)
    db.commit()

    response = client.get("/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["database"] == "ok"
    assert payload["data_quality"]["missing_national_id"] == 1


def test_register_creates_user_with_valid_role(client, app_module, db):
    _create_user(db, app_module.User, "adminok", "admin1234", role="administrador")
    _login(client, "adminok", "admin1234")
    csrf_token = _get_csrf_token(client, "/register")

    response = client.post(
        "/register",
        data={
            "username": "director_ok",
            "password": "Director123",
            "role": "director",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    created = db.query(app_module.User).filter_by(username="director_ok").first()
    assert created is not None
    assert created.role == "director"


def test_register_requires_csrf_token(client, app_module, db):
    _create_user(db, app_module.User, "admincsrf", "admin1234", role="administrador")
    _login(client, "admincsrf", "admin1234")

    response = client.post(
        "/register",
        data={
            "username": "sin_csrf",
            "password": "Password123",
            "role": "scout",
        },
    )

    assert response.status_code == 400


def test_manage_players_creates_valid_player(client, app_module, db):
    _create_user(db, app_module.User, "scout_create", "scout1234", role="scout")
    _login(client, "scout_create", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data={
            "mode": "single",
            "name": "Nuevo Talento",
            "national_id": "40111222",
            "age": "15",
            "position": "Delantero",
            "club": "Club Nuevo",
            "country": "Argentina",
            "photo_url": "",
            "pace": "14",
            "shooting": "15",
            "passing": "10",
            "dribbling": "14",
            "defending": "6",
            "physical": "12",
            "vision": "11",
            "tackling": "5",
            "determination": "16",
            "technique": "13",
            "potential_label": "1",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    created = db.query(app_module.Player).filter_by(national_id="40111222").first()
    assert created is not None
    assert created.name == "Nuevo Talento"


def test_manage_players_rejects_duplicate_national_id(client, app_module, db):
    _create_user(db, app_module.User, "scout_dup", "scout1234", role="scout")
    _create_player(app_module, db, name="Existente", national_id="41122334")
    _login(client, "scout_dup", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data={
            "mode": "single",
            "name": "Duplicado",
            "national_id": "41122334",
            "age": "16",
            "position": "Defensa",
            "club": "Club Dup",
            "country": "Argentina",
            "pace": "10",
            "shooting": "9",
            "passing": "10",
            "dribbling": "9",
            "defending": "12",
            "physical": "11",
            "vision": "10",
            "tackling": "12",
            "determination": "13",
            "technique": "9",
            "csrf_token": csrf_token,
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "ya esta registrado" in response.get_data(as_text=True)
    assert db.query(app_module.Player).filter_by(name="Duplicado").count() == 0


def test_edit_player_updates_core_fields(client, app_module, db):
    _create_user(db, app_module.User, "scout_edit", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Editar Base", national_id="42233445")
    _login(client, "scout_edit", "scout1234")
    csrf_token = _get_csrf_token(client, f"/edit_player/{player.id}")

    response = client.post(
        f"/edit_player/{player.id}",
        data={
            "name": "Editar Final",
            "national_id": "42233446",
            "age": "17",
            "position": "Mediocampista",
            "club": "Club Editado",
            "country": "Uruguay",
            "photo_url": "",
            "pace": "13",
            "shooting": "10",
            "passing": "15",
            "dribbling": "14",
            "defending": "11",
            "physical": "12",
            "vision": "15",
            "tackling": "9",
            "determination": "16",
            "technique": "14",
            "potential_label": "1",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.refresh(player)
    assert player.name == "Editar Final"
    assert player.national_id == "42233446"
    assert player.position == "Mediocampista"


def test_delete_player_removes_record(client, app_module, db):
    _create_user(db, app_module.User, "scout_delete", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Eliminar Base", national_id="43344556")
    _login(client, "scout_delete", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/delete_player/{player.id}",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    assert db.query(app_module.Player).filter_by(id=player.id).count() == 0


def test_player_stats_reject_invalid_percentages(client, app_module, db):
    _create_user(db, app_module.User, "scout_stats", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Stats Base", national_id="44455667")
    _login(client, "scout_stats", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats",
        data={
            "csrf_token": csrf_token,
            "action": "add",
            "record_date": "2026-04-15",
            "matches_played": "1",
            "minutes_played": "90",
            "pass_accuracy": "150",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Precision de pase debe estar entre 0 y 100" in response.get_data(as_text=True)
    assert db.query(app_module.PlayerStat).filter_by(player_id=player.id).count() == 0


def test_player_attributes_reject_values_out_of_range(client, app_module, db):
    _create_user(db, app_module.User, "scout_attrs", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Attrs Base", national_id="45566778")
    _login(client, "scout_attrs", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/attributes")

    response = client.post(
        f"/player/{player.id}/attributes",
        data={
            "csrf_token": csrf_token,
            "action": "add",
            "record_date": "2026-04-15",
            "pace": "25",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Ritmo debe estar entre 0 y 20" in response.get_data(as_text=True)
    assert db.query(app_module.PlayerAttributeHistory).filter_by(player_id=player.id).count() == 0


def test_admin_can_create_and_delete_coach(client, app_module, db):
    _create_user(db, app_module.User, "admin_staff", "admin1234", role="administrador")
    _login(client, "admin_staff", "admin1234")

    csrf_token = _get_csrf_token(client, "/coaches/new")
    create_response = client.post(
        "/coaches/new",
        data={
            "csrf_token": csrf_token,
            "name": "Profe Uno",
            "role": "Entrenador",
            "age": "40",
            "club": "Club Staff",
            "country": "Argentina",
        },
        follow_redirects=False,
    )

    assert create_response.status_code in (301, 302)
    coach = db.query(app_module.Coach).filter_by(name="Profe Uno").first()
    assert coach is not None

    csrf_token = _get_csrf_token(client, "/coaches")
    delete_response = client.post(
        f"/coaches/delete/{coach.id}",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert delete_response.status_code in (301, 302)
    assert db.query(app_module.Coach).filter_by(id=coach.id).count() == 0


def test_prepare_input_matches_batch_transformation(app_module, db):
    player = _create_player(app_module, db, name="Tensor Uno", national_id="46677889")
    second_player = _create_player(app_module, db, name="Tensor Dos", national_id="47788990", position="Delantero")

    single_tensor = app_module.prepare_input(player).detach().cpu().numpy()
    batch_tensor = app_module.players_to_model_tensor([player, second_player]).detach().cpu().numpy()

    assert single_tensor.shape[1] == batch_tensor.shape[1]
    assert single_tensor[0].tolist() == batch_tensor[0].tolist()


def test_prepare_input_includes_historical_features(app_module, db):
    player = _create_player(app_module, db, name="Hist Uno", national_id="49900112")
    db.add_all(
        [
            app_module.PlayerStat(
                player_id=player.id,
                record_date=date(2026, 4, 1),
                matches_played=1,
                minutes_played=90,
                pass_accuracy=72.0,
                final_score=6.5,
            ),
            app_module.PlayerStat(
                player_id=player.id,
                record_date=date(2026, 4, 8),
                matches_played=1,
                minutes_played=90,
                pass_accuracy=78.0,
                final_score=7.0,
            ),
        ]
    )
    db.commit()

    feature_map = app_module.fetch_player_stat_feature_map([player.id])
    assert feature_map[player.id]["stats_entry_count"] == 2
    assert round(float(feature_map[player.id]["avg_final_score_hist"]), 2) == 6.75
    assert round(float(feature_map[player.id]["avg_pass_accuracy_hist"]), 2) == 75.0
    assert round(float(feature_map[player.id]["latest_final_score_hist"]), 2) == 7.0

    single_tensor = app_module.prepare_input(player).detach().cpu().numpy()
    batch_tensor = app_module.players_to_model_tensor([player], stats_feature_map=feature_map).detach().cpu().numpy()
    assert single_tensor[0].tolist() == batch_tensor[0].tolist()


def test_training_main_persists_preprocessor_artifact(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    train_module = importlib.import_module("train_model")
    models_module = importlib.import_module("models")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "train_pipeline.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    models_module.Base.metadata.create_all(engine)
    db_utils_module.ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        players = [
            models_module.Player(
                name=f"Train {idx}",
                national_id=f"488990{idx:02d}",
                age=age,
                position=position,
                club="Club Train",
                country="Argentina",
                photo_url="",
                pace=pace,
                shooting=shooting,
                passing=passing,
                dribbling=dribbling,
                defending=defending,
                physical=physical,
                vision=vision,
                tackling=tackling,
                determination=determination,
                technique=technique,
                potential_label=potential,
            )
            for idx, (
                age,
                position,
                pace,
                shooting,
                passing,
                dribbling,
                defending,
                physical,
                vision,
                tackling,
                determination,
                technique,
                potential,
            ) in enumerate(
                [
                    (16, "Defensa", 10, 8, 11, 9, 14, 12, 10, 13, 12, 9, False),
                    (17, "Mediocampista", 12, 10, 14, 13, 10, 11, 14, 9, 15, 13, True),
                    (15, "Delantero", 15, 16, 10, 15, 6, 12, 10, 5, 16, 14, True),
                    (16, "Lateral", 11, 8, 10, 11, 12, 11, 9, 12, 11, 9, False),
                    (14, "Portero", 6, 3, 8, 4, 14, 15, 8, 15, 12, 9, False),
                    (18, "Defensa", 11, 7, 10, 8, 15, 13, 9, 14, 13, 10, False),
                    (13, "Delantero", 16, 17, 11, 16, 4, 11, 10, 3, 17, 15, True),
                    (17, "Mediocampista", 13, 9, 15, 14, 9, 11, 15, 8, 16, 14, True),
                    (15, "Lateral", 14, 8, 12, 13, 13, 12, 11, 13, 14, 11, False),
                    (18, "Delantero", 17, 18, 12, 17, 4, 13, 11, 3, 17, 16, True),
                ],
                start=1,
            )
        ]
        session.add_all(players)
        session.flush()
        session.add_all(
            [
                models_module.PlayerStat(
                    player_id=players[0].id,
                    record_date=date(2026, 4, 1),
                    matches_played=1,
                    minutes_played=90,
                    pass_accuracy=68.0,
                    final_score=6.2,
                ),
                models_module.PlayerStat(
                    player_id=players[1].id,
                    record_date=date(2026, 4, 3),
                    matches_played=1,
                    minutes_played=90,
                    pass_accuracy=82.0,
                    final_score=7.8,
                ),
                models_module.PlayerStat(
                    player_id=players[2].id,
                    record_date=date(2026, 4, 5),
                    matches_played=1,
                    minutes_played=90,
                    pass_accuracy=74.0,
                    final_score=8.1,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    model_path = tmp_path / "model.pt"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    metadata_path = tmp_path / "training_metadata.json"
    train_module.main(
        db_url,
        str(model_path),
        str(preprocessor_path),
        str(metadata_path),
        epochs=4,
        lr=1e-3,
        patience=2,
    )

    assert model_path.exists()
    assert preprocessor_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_summary"]["age_range_filtered"] == {"min": 13, "max": 18}
    assert metadata["dataset"]["validation_size"] > 0
    assert 0.0 <= metadata["pytorch"]["selected_threshold"] <= 1.0


def test_training_dataframe_merges_historical_features(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    preprocessing_module = importlib.import_module("preprocessing")
    models_module = importlib.import_module("models")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "train_stats_features.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    models_module.Base.metadata.create_all(engine)
    db_utils_module.ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        player = models_module.Player(
            name="Train Stats",
            national_id="59999100",
            age=16,
            position="Mediocampista",
            club="Club Train",
            country="Argentina",
            photo_url="",
            pace=12,
            shooting=10,
            passing=14,
            dribbling=13,
            defending=10,
            physical=11,
            vision=14,
            tackling=9,
            determination=15,
            technique=13,
            potential_label=True,
        )
        session.add(player)
        session.flush()
        session.add_all(
            [
                models_module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2026, 3, 1),
                    matches_played=1,
                    minutes_played=90,
                    pass_accuracy=70.0,
                    final_score=6.0,
                ),
                models_module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2026, 4, 1),
                    matches_played=1,
                    minutes_played=90,
                    pass_accuracy=80.0,
                    final_score=8.0,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    df = preprocessing_module.training_dataframe_from_engine(engine)
    row = df.iloc[0]

    assert int(row["stats_entry_count"]) == 2
    assert round(float(row["avg_final_score_hist"]), 2) == 7.0
    assert round(float(row["avg_pass_accuracy_hist"]), 2) == 75.0
    assert round(float(row["latest_final_score_hist"]), 2) == 8.0


def test_load_data_filters_training_range(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    train_module = importlib.import_module("train_model")
    models_module = importlib.import_module("models")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "train_filter.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    models_module.Base.metadata.create_all(engine)
    db_utils_module.ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.add_all(
            [
                models_module.Player(
                    name="Juvenil",
                    national_id="59999001",
                    age=17,
                    position="Defensa",
                    club="Club Train",
                    country="Argentina",
                    photo_url="",
                    pace=10,
                    shooting=8,
                    passing=11,
                    dribbling=9,
                    defending=14,
                    physical=12,
                    vision=10,
                    tackling=13,
                    determination=12,
                    technique=9,
                    potential_label=False,
                ),
                models_module.Player(
                    name="Mayor",
                    national_id="59999002",
                    age=21,
                    position="Delantero",
                    club="Club Train",
                    country="Argentina",
                    photo_url="",
                    pace=16,
                    shooting=17,
                    passing=10,
                    dribbling=15,
                    defending=6,
                    physical=12,
                    vision=10,
                    tackling=5,
                    determination=16,
                    technique=14,
                    potential_label=True,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    df, y, summary = train_module.load_data(db_url)

    assert len(df) == 1
    assert int(df.iloc[0]["age"]) == 17
    assert len(y) == 1
    assert summary["raw_rows"] == 2
    assert summary["filtered_rows"] == 1


def test_label_probability_favors_younger_and_position_fit(scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))
    generate_module = importlib.import_module("generate_data")

    attrs = {
        "pace": 14,
        "shooting": 8,
        "passing": 12,
        "dribbling": 13,
        "defending": 15,
        "physical": 12,
        "vision": 11,
        "tackling": 14,
        "determination": 16,
        "technique": 12,
    }

    younger_prob, _ = generate_module.label_probability("Defensa", 13, attrs)
    older_prob, _ = generate_module.label_probability("Defensa", 18, attrs)
    aligned_prob, _ = generate_module.label_probability("Defensa", 15, attrs)
    misaligned_prob, _ = generate_module.label_probability("Delantero", 15, attrs)

    assert younger_prob > older_prob
    assert aligned_prob > misaligned_prob
