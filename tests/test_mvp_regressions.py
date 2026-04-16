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
