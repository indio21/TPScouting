from types import SimpleNamespace

from werkzeug.security import generate_password_hash


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
