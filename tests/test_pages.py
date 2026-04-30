from werkzeug.security import generate_password_hash
from flask import url_for

def _create_user(db, User, username, password, role="scout"):
    u = User(username=username, password_hash=generate_password_hash(password), role=role)
    db.add(u)
    db.commit()
    return u

def _get_csrf_token(client, path="/login"):
    client.get(path)
    with client.session_transaction() as sess:
        return sess.get("csrf_token")

def _login(client, username, password):
    csrf_token = _get_csrf_token(client, "/login")
    return client.post("/login", data={"username": username, "password": password, "csrf_token": csrf_token})

def test_players_list_ok_after_login(client, app_module, db):
    _create_user(db, app_module.User, "u", "p", role="scout")
    _login(client, "u", "p")
    resp = client.get("/players")
    assert resp.status_code == 200

def test_dashboard_ok_after_login(client, app_module, db):
    _create_user(db, app_module.User, "u2", "p2", role="scout")
    _login(client, "u2", "p2")
    resp = client.get("/dashboard")
    assert resp.status_code == 200

def test_settings_requires_admin_after_hotfix(client, app_module, db):
    _create_user(db, app_module.User, "u3", "p3", role="scout")
    _login(client, "u3", "p3")
    resp = client.get("/settings")
    assert resp.status_code == 403


def test_staff_blueprint_keeps_legacy_endpoint_names(app_module):
    with app_module.app.test_request_context():
        assert url_for("list_coaches") == "/coaches"
        assert url_for("new_coach") == "/coaches/new"
        assert url_for("edit_coach", coach_id=7) == "/coaches/edit/7"
        assert url_for("delete_coach", coach_id=7) == "/coaches/delete/7"
        assert url_for("list_directors") == "/directors"
        assert url_for("new_director") == "/directors/new"
        assert url_for("edit_director", director_id=8) == "/directors/edit/8"
        assert url_for("delete_director", director_id=8) == "/directors/delete/8"


def test_players_blueprint_keeps_legacy_endpoint_names(app_module):
    with app_module.app.test_request_context():
        assert url_for("index") == "/players"
        assert url_for("manage_players") == "/players/manage"
        assert url_for("player_detail", player_id=9) == "/player/9"
        assert url_for("player_stats", player_id=9) == "/player/9/stats"
        assert url_for("player_attributes", player_id=9) == "/player/9/attributes"
        assert url_for("predict_player", player_id=9) == "/player/9/predict"
        assert url_for("edit_player", player_id=9) == "/edit_player/9"
        assert url_for("delete_player", player_id=9) == "/delete_player/9"


def test_compare_pages_ok_after_login(client, app_module, db):
    _create_user(db, app_module.User, "u4", "p4", role="scout")
    _login(client, "u4", "p4")
    resp_compare = client.get("/compare")
    resp_compare_multi = client.get("/compare/multi")
    assert resp_compare.status_code == 200
    assert resp_compare_multi.status_code == 200
