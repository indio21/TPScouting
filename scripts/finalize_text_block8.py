"""Close the academic text review on the image-free Block 8 copy."""

from __future__ import annotations

import hashlib
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_REVISION_TEXTO_SIN_IMAGENES_2026-10-06.docx"
OUTPUT = ROOT / "docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_TEXTO_REVISADO_SIN_IMAGENES_2026-10-06.docx"
EXPECTED_SHA256 = "ae37ca668cf4a4a0861c16b8c35317c10a0e477736c10acca66201c018022b71"

OLD_SCORE = (
    "El score previo a los controles pondera crecimiento (0,15), nivel futuro (0,12), rendimiento (0,13), "
    "presión (0,13), consistencia (0,10), rol (0,09), disponibilidad (0,09), recuperación (0,09), "
    "evaluación scout (0,10) y breakout (0,14), con una penalización por inestabilidad de 0,08."
)
NEW_SCORE = (
    "El score previo a los controles pondera crecimiento (0,15), nivel futuro (0,12), rendimiento (0,13), "
    "presión (0,13), consistencia (0,10), rol (0,09), disponibilidad (0,09), recuperación (0,09), "
    "evaluación scout (0,10) y breakout (0,14). Además, aplica una penalización por inestabilidad de 0,08."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SHA256:
        raise RuntimeError("La copia textual no coincide con el hash verificado.")
    document = Document(SOURCE)

    score_matches = [p for p in document.paragraphs if OLD_SCORE in p.text]
    if len(score_matches) != 1:
        raise RuntimeError(f"Se esperaba una oración de score; se hallaron {len(score_matches)}.")
    score_matches[0].text = score_matches[0].text.replace(OLD_SCORE, NEW_SCORE)

    bibliography = next(i for i, p in enumerate(document.paragraphs) if p.text.strip() == "8. BIBLIOGRAFÍA")
    following = document.paragraphs[bibliography + 1]
    section_properties = following._p.xpath("./w:pPr/w:sectPr")
    if len(section_properties) != 1 or following.text.strip():
        raise RuntimeError("No se encontró el salto de sección aislado después de Bibliografía.")
    section_properties[0].getparent().remove(section_properties[0])

    document.save(OUTPUT)
    print(f"OUTPUT={OUTPUT}")
    print(f"SHA256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
