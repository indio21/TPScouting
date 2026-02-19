from werkzeug.security import generate_password_hash

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
