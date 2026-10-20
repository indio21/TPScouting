"""Regenerate thesis images, update Word fields and export the final PDF.

The approved clean document is preserved. The script creates a pre-fields copy,
then uses Microsoft Word to update the index and figure/table lists, repaginate,
save the final DOCX and export its PDF.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import win32com.client
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph
from PIL import Image

from fill_word_lists_block8 import pdf_page_map, replace_list


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "docs/revision_octubre_2026/word/"
    "TRABAJO_FINAL_TPScouting_ACTUALIZADO_SEIS_PUNTOS_2026-10-19.docx"
)
EXPECTED_SOURCE_SHA256 = (
    "6b001a1f83e9bb1f1cdc899701171dfc648ab1b138b461e087165697575f2833"
)

FINAL_DIR = ROOT / "docs/revision_octubre_2026/final"
PREFIELDS = FINAL_DIR / "TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20_PREFIELDS.docx"
FINAL_DOCX = FINAL_DIR / "TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20.docx"
FINAL_PDF = FINAL_DIR / "TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20.pdf"

DIAGRAM_DIR = ROOT / "docs/diagramas/export"
EVIDENCE_DIR = ROOT / "docs/evidencia_word_render/final_2026-10-19"
ROTATED_DIR = FINAL_DIR / "diagramas_rotados"


TEXT_REPLACEMENTS = {
    "• docs/model_training_evidence.md: evidencia del entrenamiento y comparación de modelos.":
        "• docs/modelo_ml.md: descripción del modelo, preprocesamiento y artefactos incluidos en la entrega.",
    "• docs/auditoria_pendientes_2026-05-17.md: riesgos restantes y decisiones MVP.":
        "• GUIA_DEMO_PROFESOR.md: instrucciones de instalación, ejecución y evaluación de la demo local.",
    "• docs/cierre_pre_entrega_word_render_2026-05-18.md: cierre pre-entrega, deploy y smoke.":
        "• docs/REAL_DATA_READINESS.md: controles pendientes antes de cualquier uso con datos reales de menores.",
    "Las Figuras 6-2 a 6-7 fueron reproducidas el 6 de octubre de 2026 en un "
    "navegador real contra una base SQLite temporal con datos sintéticos y semilla "
    "42. La sesión autenticada verificó el acceso, el panel general, el listado, la "
    "ficha, la predicción y el comparador múltiple. La base temporal se eliminó al "
    "finalizar; estas capturas no demuestran disponibilidad continua del despliegue "
    "público.":
        "Las Figuras 6-2 a 6-7 fueron reproducidas el 19 de octubre de 2026 en un "
        "navegador real contra una base SQLite temporal con 60 jugadores sintéticos y "
        "semilla 42. La sesión autenticada verificó el acceso, el panel general, el "
        "listado, la ficha, la predicción y el comparador múltiple. La base temporal se "
        "eliminó al finalizar; estas capturas no demuestran disponibilidad continua del "
        "despliegue público.",
    "La Figura 6-6 presenta la vista de predicción reproducida en la verificación local "
    "del 6 de octubre de 2026, con la edad derivada de birth_date y el ajuste combinado "
    "respecto de PlayerNet crudo. Ese ajuste incorpora el historial disponible y la "
    "adecuación posicional.":
        "La Figura 6-6 presenta la vista de predicción reproducida en la verificación "
        "local del 19 de octubre de 2026, con la edad derivada de birth_date y el ajuste "
        "combinado respecto de PlayerNet crudo. Ese ajuste incorpora el historial "
        "disponible y la adecuación posicional.",
    "La CI #73 se conserva como evidencia histórica del commit "
    "bc5ddd35d0fa3bf6d85faab637772d4e9025fc98 en el repositorio de desarrollo. El "
    "código funcional de la entrega corresponde al commit "
    "8093fe616c9c50b72078ff4f241d9ab3427e763d, certificado por la ejecución CI "
    "38656130065 y el smoke de contenedor 38656129892. Ese código se integró en el "
    "paquete de entrega publicado en "
    "4c5ef2c88494e257a3bd8232bccba7f33f79b66a, verificado por la ejecución CI "
    "38815440730 y el smoke de contenedor 38815440745. La figura siguiente documenta "
    "la ejecución histórica y no sustituye esas evidencias posteriores.":
        "La CI #73 se conserva como evidencia histórica del repositorio de desarrollo. "
        "El código funcional de la entrega corresponde al commit "
        "8093fe616c9c50b72078ff4f241d9ab3427e763d, certificado por la ejecución CI "
        "38656130065 y el smoke de contenedor 38656129892. Ese código se integró en el "
        "paquete de entrega publicado en "
        "4c5ef2c88494e257a3bd8232bccba7f33f79b66a, verificado por la ejecución CI "
        "38815440730 y el smoke de contenedor 38815440745. La Figura 6-8 documenta la "
        "ejecución CI del paquete publicado; el identificador del commit y el alcance de "
        "la evidencia se mantienen explícitos.",
    "La Figura 10-1 conserva, con fines de trazabilidad histórica, una vista ampliada de "
    "la CI #73 del repositorio de desarrollo. Esa ejecución corresponde al commit "
    "bc5ddd35d0fa3bf6d85faab637772d4e9025fc98 y no certifica el repositorio de entrega "
    "ni las correcciones locales.":
        "La Figura 10-1 presenta una vista ampliada de la ejecución CI 38815440730 del "
        "repositorio de entrega. La ejecución certifica el paquete publicado en el "
        "commit 4c5ef2c88494e257a3bd8232bccba7f33f79b66a. La incorporación posterior de este "
        "Word y su PDF requiere una nueva ejecución CI del commit final de entrega.",
}


CAPTION_REPLACEMENTS = {
    "Figura 5-1. Secuencia real de inferencia: score combinado principal y calibración secundaria.":
        "Figura 5-1. Secuencia de inferencia: score operativo de interfaz y referencia calibrada.",
    "Figura 6-8. Resumen público de la ejecución CI #73 en GitHub Actions.":
        "Figura 6-8. Ejecución CI 38815440730 del paquete publicado en el repositorio de entrega.",
    "Figura 10-1. Evidencia histórica de la CI #73 del repositorio de desarrollo.":
        "Figura 10-1. Vista ampliada de la ejecución CI 38815440730 del repositorio de entrega.",
}


MAIN_IMAGES = {
    "Figura 4-1. Diagrama de componentes de TPScouting.": (DIAGRAM_DIR / "03_componentes.png", 6.0),
    "Figura 4-2. Diagrama de clases del modelo de dominio.": (DIAGRAM_DIR / "05_clases.png", 5.9),
    "Figura 4-3. Diagrama de casos de uso general": (DIAGRAM_DIR / "06_casos_uso_general.png", 5.9),
    "Figura 4-4. Diagrama de casos de uso de gestión de jugadores.": (DIAGRAM_DIR / "07_casos_uso_gestion_jugadores.png", 4.25),
    "Figura 4-5. Diagrama de casos de uso de análisis y decisión scout.": (DIAGRAM_DIR / "08_casos_uso_analisis_decision.png", 6.0),
    "Figura 5-1. Secuencia real de inferencia: score combinado principal y calibración secundaria.": (DIAGRAM_DIR / "01_secuencia_prediccion.png", 6.0),
    "Figura 5-2. Secuencia de carga del panel general.": (DIAGRAM_DIR / "02_secuencia_dashboard.png", 6.0),
    "Figura 5-3. Arquitectura de despliegue en Render y mecanismos operativos del MVP.": (DIAGRAM_DIR / "04_despliegue.png", 6.0),
    "Figura 6-1. Curva real de entrenamiento generada desde training_metadata.json.": (EVIDENCE_DIR / "07_training_curve_real.png", 5.75),
    "Figura 6-2. Login del MVP en la verificación local.": (EVIDENCE_DIR / "01_login.png", 6.0),
    "Figura 6-3. Panel general con métricas por rol.": (EVIDENCE_DIR / "02_dashboard.png", 6.0),
    "Figura 6-4. Listado paginado de jugadores con edad, categoría y potencial.": (EVIDENCE_DIR / "03_players.png", 6.0),
    "a) Perfil, atributos y resumen de rendimiento.": (EVIDENCE_DIR / "04a_player_detail_perfil.png", 6.0),
    "b) Historiales, gráficos y evaluaciones periódicas.": (EVIDENCE_DIR / "04b_player_detail_historiales.png", 6.0),
    "c) Reportes, disponibilidad y acciones de mantenimiento.": (EVIDENCE_DIR / "04c_player_detail_reportes.png", 6.0),
    "Figura 6-6. Vista de predicción con edad y ajuste combinado de historial y posición.": (EVIDENCE_DIR / "05_prediction.png", 6.0),
    "Figura 6-7. Comparador múltiple de jugadores.": (EVIDENCE_DIR / "06_compare_multi.png", 6.0),
    "Figura 6-8. Resumen público de la ejecución CI #73 en GitHub Actions.": (EVIDENCE_DIR / "08_ci_delivery_38815440730_resumen.png", 5.75),
    "Figura 10-1. Evidencia histórica de la CI #73 del repositorio de desarrollo.": (EVIDENCE_DIR / "09_ci_delivery_38815440730_ampliada.png", 5.75),
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


def exact(document: Document, text: str) -> Paragraph:
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text == text]
    if len(matches) != 1:
        raise RuntimeError(
            f"Se esperaba una coincidencia para {text!r}; se hallaron {len(matches)}."
        )
    return matches[0]


def clear_content(paragraph: Paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def image_paragraph_before(caption: Paragraph) -> Paragraph:
    previous = caption._p.getprevious()
    if previous is None or not previous.tag.endswith("}p"):
        raise RuntimeError(f"No se halló el párrafo gráfico antes de {caption.text!r}.")
    return Paragraph(previous, caption._parent)


def replace_image(document: Document, caption_text: str, image_path: Path, width: float) -> None:
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    caption = exact(document, caption_text)
    paragraph = image_paragraph_before(caption)
    if not paragraph._p.xpath(".//w:drawing | .//w:pict"):
        raise RuntimeError(f"No se halló una imagen antes de {caption_text!r}.")
    clear_content(paragraph)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    if not caption_text.startswith("Figura 10-"):
        paragraph.paragraph_format.page_break_before = False
        preceding = paragraph._p.getprevious()
        if preceding is not None and preceding.tag == qn("w:p"):
            introduction = Paragraph(preceding, caption._parent)
            if introduction.text.startswith(("La Figura ", "Las Figuras ")):
                introduction.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(image_path), width=Inches(width))


def rotated_diagrams() -> dict[str, Path]:
    ROTATED_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for caption, filename in ANNEX_DIAGRAMS.items():
        source = DIAGRAM_DIR / filename
        target = ROTATED_DIR / f"rotado_{filename}"
        with Image.open(source) as image:
            rotated = image.rotate(90, expand=True) if image.width > image.height else image.copy()
            rotated.save(target)
        results[caption] = target
    return results


def replace_annex_image(document: Document, caption_text: str, image_path: Path) -> None:
    caption = exact(document, caption_text)
    paragraph = image_paragraph_before(caption)
    if not paragraph._p.xpath(".//w:drawing | .//w:pict"):
        raise RuntimeError(f"No se halló una imagen antes de {caption_text!r}.")
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


def build_prefields() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("La fuente no coincide con el Word limpio aprobado.")
    document = Document(SOURCE)

    for old, new in TEXT_REPLACEMENTS.items():
        paragraph = exact(document, old)
        paragraph.text = new

    # The approved file stores figure/table lists as verified static entries.
    # Keep their captions aligned before the first PDF page mapping pass.
    for old, new in CAPTION_REPLACEMENTS.items():
        list_matches = [
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.startswith(f"{old}\t")
        ]
        if len(list_matches) != 1:
            raise RuntimeError(
                f"Se esperaba una entrada de lista para {old!r}; se hallaron {len(list_matches)}."
            )
        _, page = list_matches[0].text.rsplit("\t", 1)
        list_matches[0].text = f"{new}\t{page}"

    for caption, (image_path, width) in MAIN_IMAGES.items():
        replace_image(document, caption, image_path, width)

    for old, new in CAPTION_REPLACEMENTS.items():
        exact(document, old).text = new

    for caption, image_path in rotated_diagrams().items():
        replace_annex_image(document, caption, image_path)

    # Preserve the approved 14 pt bibliography while avoiding a short orphaned
    # continuation page. Only spacing between complete references is changed.
    bibliography = False
    for paragraph in document.paragraphs:
        if paragraph.style.name == "Heading 1":
            bibliography = paragraph.text == "8. BIBLIOGRAFÍA"
        elif bibliography and paragraph.text.strip():
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.keep_together = True
        if paragraph.style.name == "Código técnico":
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    exact(document, "Rafaela, Santa Fe, Argentina\nOctubre de 2026").alignment = WD_ALIGN_PARAGRAPH.LEFT

    # With the new 10 pt minimum, the endpoints table leaves its caption alone
    # on the following page. Compact only paragraph spacing in that table; the
    # font size remains 10 pt or larger.
    endpoints_tables = [
        table
        for table in document.tables
        if table.cell(0, 0).text.strip() == "Método"
        and table.cell(0, 1).text.strip() == "Ruta"
    ]
    if len(endpoints_tables) != 1:
        raise RuntimeError(
            f"Se esperaba una tabla de endpoints; se hallaron {len(endpoints_tables)}."
        )
    for row in endpoints_tables[0].rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = 0
                paragraph.paragraph_format.space_after = 0
                paragraph.paragraph_format.line_spacing = 0.85

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    document.save(PREFIELDS)
    checked = Document(PREFIELDS)
    if (len(checked.inline_shapes), len(checked.tables), len(checked.sections)) != (
        28,
        19,
        9,
    ):
        raise RuntimeError("La sustitución alteró imágenes, tablas o secciones.")


def word_update_and_export() -> tuple[int, int]:
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(
            str(FINAL_DOCX),
            ConfirmConversions=False,
            ReadOnly=False,
            AddToRecentFiles=False,
        )
        toc_count = document.TablesOfContents.Count
        tof_count = document.TablesOfFigures.Count
        if toc_count != 1 or tof_count != 0:
            raise RuntimeError(
                f"Estructura de campos inesperada: TOC={toc_count}, TOF={tof_count}."
            )
        for index in range(1, toc_count + 1):
            document.TablesOfContents(index).Update()
        document.Repaginate()
        pages = document.ComputeStatistics(2)
        document.Save()
        document.ExportAsFixedFormat(str(FINAL_PDF), 17)
        document.Close(False)
        document = None
        return pages, toc_count
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()


def rebuild_static_lists() -> tuple[int, int, int]:
    document = Document(FINAL_DOCX)
    figures = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.style.name == "Figure Caption Generated" and paragraph.text.strip()
    ]
    tables = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.style.name == "Table Caption Generated" and paragraph.text.strip()
    ]
    pages = pdf_page_map(FINAL_PDF, figures + tables)
    replace_list(document, "LISTA DE TABLAS", "RESUMEN", tables, pages)
    replace_list(document, "LISTA DE FIGURAS", "LISTA DE TABLAS", figures, pages)
    document.save(FINAL_DOCX)
    return len(figures), len(tables), len(pages)


def list_mismatches() -> list[tuple[str, int, int]]:
    document = Document(FINAL_DOCX)
    captions = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.style.name in {"Figure Caption Generated", "Table Caption Generated"}
        and paragraph.text.strip()
    ]
    actual = pdf_page_map(FINAL_PDF, captions)
    listed = {}
    for paragraph in document.paragraphs:
        if paragraph.style.name.lower().startswith("toc") and "\t" in paragraph.text:
            caption, page = paragraph.text.rsplit("\t", 1)
            if caption in actual and page.isdigit():
                listed[caption] = int(page)
    return [
        (caption, listed.get(caption, -1), page)
        for caption, page in actual.items()
        if listed.get(caption) != page
    ]


def main() -> None:
    build_prefields()
    shutil.copy2(PREFIELDS, FINAL_DOCX)
    pages, toc_count = word_update_and_export()
    figures = tables = mapped = 0
    mismatches = [("initial", -1, -1)]
    for _ in range(3):
        figures, tables, mapped = rebuild_static_lists()
        pages, toc_count = word_update_and_export()
        mismatches = list_mismatches()
        if not mismatches:
            break
    if mismatches:
        raise RuntimeError(f"Las listas no convergieron con las páginas reales: {mismatches[:3]}")
    checked = Document(FINAL_DOCX)
    if (len(checked.inline_shapes), len(checked.tables), len(checked.sections)) != (
        28,
        19,
        9,
    ):
        raise RuntimeError("Word alteró la estructura esperada durante la actualización.")
    print(f"PREFIELDS={PREFIELDS}")
    print(f"FINAL_DOCX={FINAL_DOCX}")
    print(f"FINAL_PDF={FINAL_PDF}")
    print(f"PAGES={pages} TOC={toc_count} STATIC_FIGURES={figures} STATIC_TABLES={tables} MAPPED={mapped}")
    print(f"DOCX_SHA256={sha256(FINAL_DOCX).upper()}")
    print(f"PDF_SHA256={sha256(FINAL_PDF).upper()}")


if __name__ == "__main__":
    main()
