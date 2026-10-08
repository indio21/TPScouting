"""Apply the user-requested final font and image layout corrections."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image

EXPECTED_SOURCE_SHA256 = "aa197775fba8441260f9068b9350f1f533249b03c7e73ad5d4f194e336af66e8"

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = ROOT / "docs/evidencia_word_render/bloque8_2026-10-06"
DIAGRAM_DIR = ROOT / "docs/diagramas/export"
ROTATED_DIR = ROOT / "docs/revision_octubre_2026/recursos_finales_bloque8/diagramas_rotados"

SUBFIGURES = {
    "a) Perfil, atributos y resumen de rendimiento.": CAPTURE_DIR / "04a_player_detail_perfil.png",
    "b) Historiales, gráficos y evaluaciones periódicas.": CAPTURE_DIR / "04b_player_detail_historiales.png",
    "c) Reportes, disponibilidad y acciones de mantenimiento.": CAPTURE_DIR / "04c_player_detail_reportes.png",
}

ANNEX_DIAGRAMS = {
    "Figura 10-2. Diagrama de componentes de TPScouting ampliado.": "03_componentes.png",
    "Figura 10-3. Diagrama de clases del modelo de dominio ampliado.": "05_clases.png",
    "Figura 10-4. Diagrama de secuencia de inferencia ampliado.": "01_secuencia_prediccion.png",
    "Figura 10-5. Diagrama de secuencia del panel general ampliado.": "02_secuencia_dashboard.png",
    "Figura 10-6. Diagrama de despliegue en Render ampliado.": "04_despliegue.png",
    "Figura 10-7. Diagrama de casos de uso general ampliado.": "06_casos_uso_general.png",
    "Figura 10-8. Diagrama de casos de uso de gestión de jugadores ampliado.": "07_casos_uso_gestion_jugadores.png",
    "Figura 10-9. Diagrama de casos de uso de análisis y decisión scout ampliado.": "08_casos_uso_analisis_decision.png",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(document: Document, text: str):
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text == text]
    if len(matches) != 1:
        raise RuntimeError(f"Se esperaba una coincidencia para {text!r}; se hallaron {len(matches)}.")
    return matches[0]


def clear_content(paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def image_paragraph_before(caption):
    previous = caption._p.getprevious()
    if previous is None or not previous.tag.endswith("}p"):
        raise RuntimeError(f"No se halló el párrafo gráfico antes de {caption.text!r}.")
    from docx.text.paragraph import Paragraph

    return Paragraph(previous, caption._parent)


def rotate_diagrams() -> dict[str, Path]:
    ROTATED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for caption, filename in ANNEX_DIAGRAMS.items():
        source = DIAGRAM_DIR / filename
        target = ROTATED_DIR / f"rotado_{filename}"
        with Image.open(source) as image:
            oriented = image.rotate(90, expand=True) if image.width > image.height else image.copy()
            oriented.save(target)
        outputs[caption] = target
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sha256(args.source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("El DOCX fuente no coincide con la revisión gramatical verificada.")

    document = Document(args.source)

    changed_styles = []
    for style in document.styles:
        if hasattr(style, "font") and style.font.size and abs(style.font.size.pt - 11.0) < 0.01:
            style.font.size = Pt(14)
            if style.font.name == "Arial":
                style.font.name = "Arial"
            changed_styles.append(style.name)

    # Con Arial 14, los párrafos vacíos de la portada empujan ubicación y fecha
    # a la hoja del índice. Se conserva el texto a 14 puntos y se compactan solo
    # los separadores vacíos de la portada.
    for paragraph in document.paragraphs[:15]:
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        if not paragraph.text.strip() and not paragraph._p.xpath(".//w:drawing | .//w:pict"):
            paragraph.paragraph_format.line_spacing = Pt(4)
    exact(document, "ÍNDICE").paragraph_format.page_break_before = True
    exact(document, "LISTA DE FIGURAS").paragraph_format.page_break_before = False
    cover_student = exact(document, "Alumno: Solari, Pablo\nLegajo: [PENDIENTE DE INFORMAR]")
    cover_student.runs[1].text = ""

    # Arial 14 desplaza la Tabla 4-8 lo suficiente para dejar su leyenda sola.
    # Compactar únicamente esa tabla conserva la tipografía tabular preexistente
    # y mantiene la leyenda en la misma hoja.
    cases_table = document.tables[7]
    for row in cases_table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 0.85

    for caption_text, image_path in SUBFIGURES.items():
        caption = exact(document, caption_text)
        paragraph = image_paragraph_before(caption)
        clear_content(paragraph)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        paragraph.add_run().add_picture(str(image_path), width=Inches(6.0))

    overall_caption = exact(
        document,
        "Figura 6-5. Ficha de jugador con historial, atributos y acciones CRUD en modales.",
    )
    old_single_image = image_paragraph_before(overall_caption)
    if old_single_image._p.xpath(".//w:drawing"):
        old_single_image._p.getparent().remove(old_single_image._p)

    rotated = rotate_diagrams()
    for caption_text, image_path in rotated.items():
        caption = exact(document, caption_text)
        paragraph = image_paragraph_before(caption)
        clear_content(paragraph)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.page_break_before = True
        paragraph.paragraph_format.keep_with_next = True
        with Image.open(image_path) as image:
            ratio = image.width / image.height
        if ratio <= 6.0 / 8.0:
            paragraph.add_run().add_picture(str(image_path), height=Inches(8.0))
        else:
            paragraph.add_run().add_picture(str(image_path), width=Inches(6.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output)
    checked = Document(args.output)
    if len(checked.inline_shapes) != 28:
        raise RuntimeError(f"Se esperaban 28 imágenes y se hallaron {len(checked.inline_shapes)}.")
    if len(checked.tables) != 19:
        raise RuntimeError("La corrección alteró las tablas.")
    print(f"OUTPUT={args.output}")
    print(f"STYLES_11_TO_14={changed_styles}")
    print("SUBFIGURES_RESTORED=3")
    print("ANNEX_DIAGRAMS_ROTATED=8")
    print(f"SHA256={sha256(args.output).upper()}")


if __name__ == "__main__":
    main()
