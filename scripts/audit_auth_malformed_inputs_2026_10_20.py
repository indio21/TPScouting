"""Reproduce malformed login inputs against a disposable database.

This is an audit companion, not part of the application's regression suite.
It records responses without modifying application code or user databases.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy.orm import close_all_sessions
from werkzeug.security import generate_password_hash

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/revision_octubre_2026/final/reproduccion_entradas_auth_2026-10-20.json"


def main() -> None:
    original = Path.cwd()
    spec = importlib.util.spec_from_file_location("audit_fixtures", ROOT / "tests/conftest.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    results = []
    with tempfile.TemporaryDirectory(prefix="tpscouting_audit_inputs_") as directory:
        monkeypatch = pytest.MonkeyPatch()
        module = None
        try:
            module = fixtures.app_module.__wrapped__(
                Path(directory), ROOT / "scouting_app", monkeypatch
            )
            module.app.config["TESTING"] = False
            module.app.logger.disabled = True
            database = module.Session()
            try:
                database.add(
                    module.User(
                        username="auditoria_usuario",
                        password_hash=generate_password_hash("ClavePrueba123"),
                        role="scout",
                    )
                )
                database.commit()
            finally:
                database.close()
            cases = (
                ("password_ausente", {"username": "auditoria_usuario"}),
                (
                    "csrf_no_ascii",
                    {
                        "username": "auditoria_usuario",
                        "password": "ClavePrueba123",
                        "csrf_token": "é",
                    },
                ),
            )
            for name, values in cases:
                client = module.app.test_client()
                client.get("/login")
                with client.session_transaction() as session:
                    token = session["csrf_token"]
                response = client.post("/login", data={"csrf_token": token, **values})
                with client.session_transaction() as session:
                    authenticated = bool(session.get("user_id"))
                results.append(
                    {
                        "case": name,
                        "http_status": response.status_code,
                        "authenticated": authenticated,
                    }
                )
        finally:
            close_all_sessions()
            if module is not None:
                module.engine.dispose()
            monkeypatch.undo()
            os.chdir(original)
    report = {"temporary_database_removed": True, "results": results}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
