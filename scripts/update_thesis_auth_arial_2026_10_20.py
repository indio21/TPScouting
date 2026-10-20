"""Update the verified thesis copy with auth evidence and Arial typography.

The functional commit is evaluated before this script runs. Its CI/container
records must be completed successfully. The document's eventual publication
commit is kept in external evidence to avoid self-referential commit hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from lxml import etree

import finalize_thesis_delivery_2026_10_19 as layout

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/revision_octubre_2026/cierre_correcciones_2026-10-20"
SOURCE = ROOT / "docs/revision_octubre_2026/final/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20.docx"
EXPECTED_SOURCE = "2ea4a6ea4c99617cb3eda36f8c9218e28d4d9b479d812f1240f81347569aad16"
STEM = "TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20"
OUTPUT = EVIDENCE / f"{STEM}.docx"
PDF = OUTPUT.with_suffix(".pdf")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = "{" + NS["w"] + "}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def body_paragraph(document, prefix):
    matches = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith(prefix)
        and not paragraph.style.name.lower().startswith("toc")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Se esperaba un párrafo para {prefix!r}: {len(matches)}")
    return matches[0]


def set_text(paragraph, text):
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(text)


def arial_package(path):
    """Normalize all editable Word text/defaults/theme fonts, including fields."""
    fonts_changed = 0
    bullet_map = {"\uf0b7": "•", "\uf0a7": "▪", "\uf0e8": "→"}
    with tempfile.TemporaryDirectory(prefix="tpscouting_arial_") as directory:
        target = Path(directory) / path.name
        with ZipFile(path) as source_zip, ZipFile(target, "w", ZIP_DEFLATED) as output_zip:
            for item in source_zip.infolist():
                data = source_zip.read(item.filename)
                if item.filename.startswith("word/") and item.filename.endswith(".xml"):
                    root = etree.fromstring(data)
                    for run in root.xpath(".//w:r[w:t or w:instrText]", namespaces=NS):
                        properties = run.find(W + "rPr")
                        if properties is None:
                            properties = etree.Element(W + "rPr")
                            run.insert(0, properties)
                        if properties.find(W + "rFonts") is None:
                            properties.insert(0, etree.Element(W + "rFonts"))
                    for node in root.xpath(".//w:rFonts", namespaces=NS):
                        node.attrib.clear()
                        for key in ("ascii", "hAnsi", "eastAsia", "cs"):
                            node.set(W + key, "Arial")
                        fonts_changed += 1
                    for node in root.xpath(".//w:lvlText", namespaces=NS):
                        value = node.get(W + "val")
                        if value in bullet_map:
                            node.set(W + "val", bullet_map[value])
                    for node in root.xpath(".//w:sz | .//w:szCs", namespaces=NS):
                        if float(node.get(W + "val")) < 20:
                            node.set(W + "val", "20")
                    drawing_ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                    for node in root.xpath(".//a:latin | .//a:ea | .//a:cs | .//a:font", namespaces=drawing_ns):
                        node.set("typeface", "Arial")
                    data = etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)
                output_zip.writestr(item, data)
        shutil.copy2(target, path)
    return fonts_changed


def check_fonts(path):
    others = []
    with ZipFile(path) as archive:
        theme_ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        theme = etree.fromstring(archive.read("word/theme/theme1.xml"))
        theme_fonts = {}
        for group in ("major", "minor"):
            for suffix, element in (("HAnsi", "latin"), ("Ascii", "latin"), ("EastAsia", "ea"), ("Bidi", "cs")):
                found = theme.xpath(f".//a:fontScheme/a:{group}Font/a:{element}/@typeface", namespaces=theme_ns)
                if found:
                    theme_fonts[group + suffix] = found[0]
        for name in archive.namelist():
            if name.startswith("word/") and name.endswith(".xml"):
                root = etree.fromstring(archive.read(name))
                for node in root.xpath(".//w:rFonts", namespaces=NS):
                    for key, value in node.attrib.items():
                        attribute = key.rsplit("}", 1)[-1]
                        resolved = theme_fonts.get(value) if "theme" in attribute.lower() else value
                        if attribute != "hint" and resolved != "Arial":
                            others.append({"part": name, "attribute": key, "value": value})
    return others


def main():
    if digest(SOURCE).lower() != EXPECTED_SOURCE:
        raise RuntimeError("El Word fuente no coincide con la entrega revisada.")
    functional = json.loads((EVIDENCE / "functional_evidence.json").read_text(encoding="utf-8"))
    for kind in ("ci", "container"):
        if functional[kind]["conclusion"] != "success":
            raise RuntimeError(f"Falta validación aprobada: {kind}")
        if functional[kind]["head_sha"] != functional["delivery_commit"]:
            raise RuntimeError("La evidencia CI no pertenece al commit declarado.")
    coverage = json.loads((EVIDENCE / "coverage.json").read_text(encoding="utf-8"))
    log = (EVIDENCE / "pytest.txt").read_text(encoding="utf-16")
    match = re.search(r"(\d+) passed, (\d+) skipped, (\d+) warnings", log)
    if not match:
        raise RuntimeError("No se encontró el resultado de la suite local.")
    passed, skipped, warnings = map(int, match.groups())
    percentage = f"{coverage['totals']['percent_covered']:.2f}".replace(".", ",")
    commit = functional["delivery_commit"]
    main_commit = functional["main_commit"]
    ci = functional["ci"]["id"]
    container = functional["container"]["id"]
    document = Document(SOURCE)
    set_text(
        body_paragraph(document, "La verificación local del"),
        f"La verificación local del 20/10/2026 obtuvo {passed} pruebas aprobadas, "
        f"{skipped} omitida y {warnings} advertencias conocidas, con una cobertura de "
        f"{percentage} %. La suite incluye regresiones para credenciales incompletas, "
        "tokens CSRF no ASCII y cierre de la sesión de base ante errores de autenticación. "
        "La prueba omitida corresponde al smoke visual Playwright opt-in; la "
        "verificación funcional adicional de la demo se realizó con SQLite temporal. "
        "Las advertencias corresponden a RuntimeWarning de scikit-learn por columnas "
        "completamente NaN en dos pruebas de preprocesamiento. La cobertura global "
        "supera el umbral configurado, aunque no es uniforme y no demuestra ausencia "
        "de defectos en rutas no cubiertas.",
    )
    set_text(
        body_paragraph(document, "• /health público sin excepciones internas"),
        "• /health público sin excepciones internas, CSP compatible con los recursos "
        "inventariados y validación central de photo_url. La cabecera HSTS se habilita "
        "en modo de producción; su efectividad depende del entorno HTTPS y de la "
        "configuración de terminación TLS y del proxy.",
    )
    set_text(
        body_paragraph(document, "• Protección CSRF con comparación"),
        "• Protección CSRF con comparación de tiempo constante y formularios POST "
        "para operaciones mutantes. Las credenciales incompletas y los tokens CSRF "
        "no válidos se rechazan con una respuesta controlada, sin iniciar sesión.",
    )
    set_text(
        body_paragraph(document, "La CI #73 se conserva"),
        "La CI #73 se conserva como antecedente histórico del repositorio de "
        "desarrollo. La versión funcional corregida del repositorio de entrega "
        f"corresponde al commit {commit}, verificado por la ejecución CI {ci} "
        f"y el smoke de contenedor {container}, ambos aprobados. La CI incluye "
        "Python 3.11 y 3.12, Ruff y auditoría de dependencias. La Figura 6-8 muestra "
        "la ejecución anterior 38815440730 del paquete 4c5ef2c y se conserva como "
        "evidencia fechada; no sustituye la validación de la versión corregida.",
    )
    set_text(
        body_paragraph(document, "Repositorio de entrega:"),
        "Repositorio de entrega: https://github.com/indio21/TPScouting-entrega. "
        f"Versión funcional evaluada: commit {commit}, CI {ci} y smoke de contenedor "
        f"{container}, aprobados. La identificación del paquete documental y los "
        "hashes del Word/PDF se registran en VERIFICACION_ENTREGA.md. El repositorio "
        "principal conserva el historial de trabajo y las evidencias internas que no "
        "forman parte de la entrega académica.",
    )
    set_text(
        body_paragraph(document, "La Figura 10-1 presenta"),
        "La Figura 10-1 presenta una vista ampliada de la ejecución anterior "
        "38815440730 del paquete 4c5ef2c. Las correcciones de autenticación "
        f"se verificaron posteriormente en el commit funcional {commit}, mediante "
        f"la CI {ci} y el smoke de contenedor {container}, aprobados. La publicación "
        "del documento actualizado se identifica por sus hashes y el historial de Git; "
        "no se atribuye a una captura histórica la validación de cambios posteriores.",
    )
    evidence = next(table for table in document.tables if table.cell(0, 0).text == "Evidencia")
    rows = {
        1: ("pytest local (20/10/2026)", f"{passed} passed, {skipped} skipped, {warnings} warnings"),
        2: ("pytest-cov local", f"{percentage} %"),
        3: ("Prueba omitida y demo adicional", "Playwright opt-in omitido en la suite; demo funcional adicional verificada con SQLite temporal y 60 jugadores sintéticos"),
        5: ("Commit funcional del repositorio principal de respaldo", main_commit),
        6: ("Commit funcional verificado del repositorio de entrega", commit),
        7: ("CI de la versión funcional corregida", f"{ci}: aprobada; Python 3.11/3.12, Ruff y auditoría de dependencias"),
        8: ("Contenedor de la versión funcional corregida", f"{container}: construcción y smoke aprobados"),
    }
    for index, values in rows.items():
        for column, value in enumerate(values):
            set_text(evidence.cell(index, column).paragraphs[0], value)
    objectives = next(table for table in document.tables if table.cell(0, 0).text == "Objetivo declarado en 1.2")
    set_text(
        objectives.cell(1, 2).paragraphs[0],
        f"Registro, historiales, paneles, comparadores y PlayerNet; {passed} pruebas "
        f"aprobadas, {skipped} omitida y {percentage} % de cobertura. Versión funcional "
        f"de entrega validada por CI {ci} y smoke de contenedor {container}.",
    )

    for paragraph in document.paragraphs:
        style = paragraph.style.name
        if style.startswith("Heading") or style in {"Figure Caption Generated", "Table Caption Generated", "Subfigura"}:
            for run in paragraph.runs:
                run.font.size = Pt(14)
        if style == "Normal" and paragraph.text.strip():
            for run in paragraph.runs:
                if run.text.strip():
                    run.font.size = Pt(14)
        if style == "Código técnico":
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in paragraph.runs:
                run.font.size = Pt(10)
    for style in document.styles:
        if style.name.startswith("Heading"):
            style.font.size = Pt(14)
    document.save(OUTPUT)
    arial_package(OUTPUT)

    layout.FINAL_DOCX = OUTPUT
    layout.FINAL_PDF = PDF
    pages, _ = layout.word_update_and_export()
    for _ in range(3):
        layout.rebuild_static_lists()
        arial_package(OUTPUT)
        pages, _ = layout.word_update_and_export()
        if not layout.list_mismatches():
            break
    if layout.list_mismatches():
        raise RuntimeError("Las listas no coinciden con las páginas reales.")
    # Word may recreate unused font definitions. Normalize those definitions
    # without changing metrics; then export once more to verify the same layout.
    arial_package(OUTPUT)
    pages, _ = layout.word_update_and_export()
    others = check_fonts(OUTPUT)
    if others:
        raise RuntimeError(f"Persisten fuentes no Arial: {others[:4]}")
    checked = Document(OUTPUT)
    if (len(checked.inline_shapes), len(checked.tables), len(checked.sections)) != (28, 19, 9):
        raise RuntimeError("Se alteró la estructura del documento.")
    print(json.dumps({"docx": str(OUTPUT), "pdf": str(PDF), "pages": pages,
                      "docx_sha256": digest(OUTPUT), "pdf_sha256": digest(PDF),
                      "passed": passed, "coverage": percentage,
                      "non_arial_word_fonts": others}, indent=2))


if __name__ == "__main__":
    main()
