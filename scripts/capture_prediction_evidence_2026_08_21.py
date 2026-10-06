from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
from datetime import date
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "scouting_app"
OUTPUT = ROOT / "docs" / "evidencia_word_render" / "05_prediction_render_corregida_2026-08-21.png"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def load_app(temp_dir: Path):
    os.environ["APP_SECRET_KEY"] = "captura-evidencia-local-2026-08-21"
    os.environ["APP_DB_URL"] = f"sqlite:///{(temp_dir / 'captura_app.db').as_posix()}"
    os.environ["TRAINING_DB_URL"] = f"sqlite:///{(temp_dir / 'captura_training.db').as_posix()}"

    sys.path.insert(0, str(APP_DIR))
    os.chdir(APP_DIR)
    spec = importlib.util.spec_from_file_location("scouting_app_capture_20260821", APP_DIR / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar scouting_app/app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.app.config.update(TESTING=True)
    return module


def seed_evidence_case(module) -> int:
    session = module.Session()
    try:
        user = module.User(
            username="auditoria_visual",
            password_hash=module.generate_password_hash("visual2026"),
            role=module.ROLE_ADMIN,
        )
        player = module.Player(
            name="Pablo Martinez",
            national_id="88990011",
            age=16,
            birth_date=date(2010, 3, 21),
            position="Mediocampista",
            club="Club Juvenil",
            country="Argentina",
            photo_url="",
            pace=13,
            shooting=11,
            passing=15,
            dribbling=14,
            defending=10,
            physical=12,
            vision=16,
            tackling=9,
            determination=16,
            technique=15,
            potential_label=True,
        )
        session.add_all([user, player])
        session.flush()
        session.add_all(
            [
                module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2026, 5, 15),
                    matches_played=8,
                    goals=2,
                    assists=4,
                    minutes_played=650,
                    yellow_cards=1,
                    red_cards=0,
                    pass_accuracy=84.0,
                    shot_accuracy=58.0,
                    duels_won_pct=61.0,
                    final_score=7.6,
                    notes="Registro de prueba visual local.",
                ),
                module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2026, 7, 20),
                    matches_played=7,
                    goals=3,
                    assists=3,
                    minutes_played=590,
                    yellow_cards=0,
                    red_cards=0,
                    pass_accuracy=87.0,
                    shot_accuracy=62.0,
                    duels_won_pct=64.0,
                    final_score=8.1,
                    notes="Registro de prueba visual local.",
                ),
            ]
        )
        session.commit()
        return int(player.id)
    finally:
        session.close()


def capture(module, player_id: int) -> None:
    server = make_server("127.0.0.1", 0, module.app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as playwright:
            launch_args = {"headless": True}
            if CHROME.exists():
                launch_args["executable_path"] = str(CHROME)
            browser = playwright.chromium.launch(**launch_args)
            try:
                context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                page = context.new_page()
                page.goto(f"{base_url}/login", wait_until="networkidle")
                page.fill("#username", "auditoria_visual")
                page.fill("#password", "visual2026")
                page.get_by_role("button", name="Entrar").click()
                page.wait_for_url("**/players**")
                page.goto(f"{base_url}/player/{player_id}/predict", wait_until="networkidle")

                expect(page.get_by_text("16 anos", exact=True)).to_be_visible()
                expect(page.get_by_text("Ajuste combinado", exact=True)).to_be_visible()
                expect(page.get_by_text("Impacto de historial y ajuste posicional.", exact=True)).to_be_visible()
                expect(page.get_by_text("Referencia calibrada", exact=True)).to_be_visible()

                OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(OUTPUT), full_page=False)
                if OUTPUT.stat().st_size < 20_000:
                    raise RuntimeError("La captura generada es demasiado pequena")
                context.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        module.engine.dispose()


def main() -> int:
    original_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="tpscouting-captura-", ignore_cleanup_errors=True) as temp_name:
            module = load_app(Path(temp_name))
            player_id = seed_evidence_case(module)
            capture(module, player_id)
    finally:
        os.chdir(original_cwd)
    print(f"Captura real generada: {OUTPUT}")
    print("Base utilizada: SQLite temporal eliminada al finalizar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
