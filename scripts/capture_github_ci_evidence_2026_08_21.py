"""Capture public GitHub Actions evidence for the final audited Word file."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


RUN_URL = "https://github.com/indio21/TPScouting/actions/runs/32543415220"
EXPECTED_SHA = "bc5ddd35d0fa3bf6d85faab637772d4e9025fc98"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "evidencia_word_render"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=RUN_URL)
    parser.add_argument("--expected-sha", default=EXPECTED_SHA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--summary-name", default="08_ci_run_73_resumen.png")
    parser.add_argument("--detail-name", default="09_ci_run_73_ampliada.png")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_args = {"headless": True}
    if CHROME.exists():
        launch_args["executable_path"] = str(CHROME)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**launch_args)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(8_000)

        body = page.locator("body").inner_text()
        if "CI" not in body:
            raise RuntimeError("La pagina publica no muestra la ejecucion CI esperada.")
        if args.expected_sha[:7] not in body:
            raise RuntimeError("La pagina no corresponde al commit final esperado.")

        page.screenshot(
            path=str(output_dir / args.summary_name),
            full_page=False,
        )
        page.screenshot(
            path=str(output_dir / args.detail_name),
            full_page=True,
        )
        browser.close()


if __name__ == "__main__":
    main()
