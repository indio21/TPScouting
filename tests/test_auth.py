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

def test_landing_public_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200

def test_protected_requires_login_redirects_to_login(client):
    resp = client.get("/players")
    assert resp.status_code in (301, 302)
    assert "/login" in resp.headers.get("Location", "")

def test_login_success_sets_session_and_redirects(client, app_module, db):
    _create_user(db, app_module.User, "user1", "pass1", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    resp = client.post(
        "/login",
        data={"username": "user1", "password": "pass1", "csrf_token": csrf_token},
        follow_redirects=False,
    )
    assert resp.status_code in (301, 302)
    assert "/players" in resp.headers.get("Location", "")

def test_logout_clears_session_and_redirects_to_landing(client, app_module, db):
    _create_user(db, app_module.User, "user2", "pass2", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    client.post("/login", data={"username": "user2", "password": "pass2", "csrf_token": csrf_token})
    resp = client.get("/logout", follow_redirects=False)
    assert resp.status_code in (301, 302)
    assert resp.headers.get("Location", "").endswith("/")

def test_register_requires_admin_role(client, app_module, db):
    _create_user(db, app_module.User, "user3", "pass3", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    client.post("/login", data={"username": "user3", "password": "pass3", "csrf_token": csrf_token})
    resp = client.get("/register")
    assert resp.status_code == 403
