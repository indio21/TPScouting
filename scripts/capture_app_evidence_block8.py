"""Capture one current, reproducible screenshot set for the final document."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "scouting_app"
OUTPUT_DIR = ROOT / "docs/evidencia_word_render/bloque8_2026-10-06"
MANIFEST = OUTPUT_DIR / "manifest.json"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from iniciar_demo import prepare_demo


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    original_cwd = Path.cwd()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = {}
    try:
        with tempfile.TemporaryDirectory(prefix="tpscouting_capture_block8_") as directory:
            db_path = Path(directory) / "captura.db"
            prepare_demo(db_path, 60, 42, "profesor_demo", "DemoProfesor123")

            os.chdir(APP_DIR)
            import app as app_module

            session = app_module.Session()
            try:
                players = session.query(app_module.Player).order_by(app_module.Player.id).limit(3).all()
                if len(players) < 3:
                    raise RuntimeError("La demo no contiene tres jugadores para las capturas.")
                player_id = int(players[0].id)
                players[0].photo_url = None
                session.commit()
            finally:
                session.close()

            server = make_server("127.0.0.1", 0, app_module.app)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                with sync_playwright() as playwright:
                    launch = {"headless": True}
                    if CHROME.exists():
                        launch["executable_path"] = str(CHROME)
                    browser = playwright.chromium.launch(**launch)
                    try:
                        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                        page = context.new_page()

                        def capture(name: str, route: str) -> None:
                            page.goto(f"{base_url}{route}", wait_until="networkidle", timeout=60_000)
                            path = OUTPUT_DIR / name
                            page.screenshot(path=str(path), full_page=False)
                            if path.stat().st_size < 15_000:
                                raise RuntimeError(f"La captura {name} es demasiado pequeña.")
                            evidence[name] = {
                                "route": route,
                                "bytes": path.stat().st_size,
                                "sha256": sha256(path),
                            }

                        capture("01_login.png", "/login")
                        page.fill("#username", "profesor_demo")
                        page.fill("#password", "DemoProfesor123")
                        page.get_by_role("button", name="Entrar").click()
                        page.wait_for_url("**/players**")
                        capture("02_dashboard.png", "/dashboard")
                        capture("03_players.png", "/players")
                        capture("04_player_detail.png", f"/player/{player_id}")
                        page.goto(f"{base_url}/player/{player_id}", wait_until="networkidle", timeout=60_000)
                        detail_height = page.evaluate("document.documentElement.scrollHeight")
                        for name, offset in (
                            ("04a_player_detail_perfil.png", 0),
                            ("04b_player_detail_historiales.png", min(900, max(0, detail_height - 1000))),
                            ("04c_player_detail_reportes.png", max(0, detail_height - 1000)),
                        ):
                            page.evaluate("position => window.scrollTo(0, position)", offset)
                            page.wait_for_timeout(250)
                            path = OUTPUT_DIR / name
                            page.screenshot(path=str(path), full_page=False)
                            if path.stat().st_size < 15_000:
                                raise RuntimeError(f"La captura {name} es demasiado pequeña.")
                            evidence[name] = {
                                "route": f"/player/{player_id}",
                                "scroll_y": offset,
                                "bytes": path.stat().st_size,
                                "sha256": sha256(path),
                            }
                        capture("05_prediction.png", f"/player/{player_id}/predict")
                        capture("06_compare_multi.png", "/compare/multi")
                        context.close()
                    finally:
                        browser.close()
            finally:
                server.shutdown()
                thread.join(timeout=5)
                app_module.engine.dispose()
    finally:
        os.chdir(original_cwd)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Aplicación local contra SQLite temporal, datos sintéticos, seed 42",
        "database_removed": True,
        "screenshots": evidence,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
