"""Apply only the four authorized November corrections to a new thesis copy."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import fitz
import win32com.client
from docx import Document
from lxml import etree

import audit_final_delivery_layout_2026_10_20 as layout_audit
from update_thesis_auth_arial_2026_10_20 import check_fonts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/revision_octubre_2026/entrega_word_usuario_21-10-2026/Trabajo Final_Solari Pablo_21-10-2026.docx"
PUBLIC_SOURCE = ROOT.parent / "TPScouting-entrega/documento" / SOURCE.name
FOLDER = ROOT / "docs/revision_noviembre_2026/correcciones_menores_2026-11-12"
OUTPUT = FOLDER / "Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_12-11-2026.docx"
PDF = OUTPUT.with_suffix(".pdf")
BACKUP = FOLDER / ("BACKUP_" + SOURCE.name)
EXPECTED_SOURCE = "81A743DF38FB32820EA7DC3BEC247F57CCE6F3F826510DFEA2ED9CF3D63A131C"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = "{" + NS["w"] + "}"
ORPHAN = "Tabla 4-2. Comparación entre motores relacionales."
OLD_HEALTH = "Estado de servicio, base y calidad de datos."
NEW_HEALTH = "Comprobación mínima de disponibilidad del servicio y conectividad con la base de datos."
URL = "https://tpscouting-mvp.onrender.com"
AUTH_ANCHOR = " Esta mitigación fue verificada"
AUTH_ADDITION = (
    " Con la configuración predeterminada, cinco intentos fallidos en una ventana "
    "móvil de 15 minutos provocan el rechazo de los siguientes intentos con HTTP 429, "
    "incluso si las credenciales son correctas, hasta que venzan suficientes fallos "
    "de esa ventana. Un tercero que conozca el nombre de usuario puede agotar esa "
    "cuota e impedir temporalmente nuevos inicios de sesión en esa cuenta. "
    "El registro de intentos reside en memoria por proceso."
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def plain(element):
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


def splice(element, start, end, replacement):
    """Edit only text nodes, retaining runs, styles, links and paragraph properties."""
    spans = []
    position = 0
    for node in element.xpath(".//w:t", namespaces=NS):
        value = node.text or ""
        spans.append((node, position, position + len(value), value))
        position += len(value)
    if start == end:
        for node, left, right, value in spans:
            if left <= start <= right and right > left:
                at = start - left
                node.text = value[:at] + replacement + value[at:]
                return
        raise RuntimeError("Insertion point not found")
    inserted = False
    for node, left, right, value in spans:
        if right <= start or left >= end:
            continue
        lo, hi = max(0, start - left), min(len(value), end - left)
        node.text = value[:lo] + (replacement if not inserted else "") + value[hi:]
        inserted = True
    if not inserted:
        raise RuntimeError("Replacement range not found")


def replace(element, old, new):
    value = plain(element)
    if value.count(old) != 1:
        raise RuntimeError(f"Expected exactly one occurrence: {old!r}")
    start = value.index(old)
    splice(element, start, start + len(old), new)


def update_document_xml(editor):
    temporary = OUTPUT.with_suffix(".working.docx")
    with ZipFile(OUTPUT) as source, ZipFile(temporary, "w", ZIP_DEFLATED) as target:
        for item in source.infolist():
            content = source.read(item.filename)
            if item.filename == "word/document.xml":
                xml = etree.fromstring(content)
                editor(xml)
                content = etree.tostring(xml, encoding="UTF-8", xml_declaration=True, standalone=True)
            target.writestr(item, content)
    temporary.replace(OUTPUT)


def corrections(xml):
    paragraphs = xml.xpath(".//w:p", namespaces=NS)
    orphan = [p for p in paragraphs if plain(p).startswith(ORPHAN)]
    if len(orphan) != 2:
        raise RuntimeError("Expected body caption and list entry for orphan table")
    for paragraph in orphan:
        paragraph.getparent().remove(paragraph)
    health = [cell for cell in xml.xpath(".//w:tc", namespaces=NS) if plain(cell) == OLD_HEALTH]
    if len(health) != 1:
        raise RuntimeError("Health cell not uniquely identified")
    replace(health[0], OLD_HEALTH, NEW_HEALTH)
    url_paragraphs = [p for p in paragraphs if URL * 2 in plain(p)]
    if len(url_paragraphs) != 1:
        raise RuntimeError("Duplicated URL not uniquely identified")
    duplicate_links = [h for h in url_paragraphs[0].xpath(".//w:hyperlink", namespaces=NS) if plain(h) == URL]
    if len(duplicate_links) != 1:
        raise RuntimeError("Expected duplicate hyperlink after plain-text URL")
    # Keep the existing black command text and remove only its appended duplicate.
    duplicate_links[0].getparent().remove(duplicate_links[0])
    auth = [p for p in paragraphs if plain(p).startswith("El login aplica rate limiting local")]
    if len(auth) != 1:
        raise RuntimeError("Authentication paragraph not uniquely identified")
    start = plain(auth[0]).index(AUTH_ANCHOR)
    splice(auth[0], start, start, AUTH_ADDITION)

    # All four content corrections precede numbering changes.
    count = 0
    for paragraph in xml.xpath(".//w:p", namespaces=NS):
        for match in reversed(list(re.finditer(r"\bTabla 4-([3-8])\b", plain(paragraph)))):
            splice(paragraph, match.start(1), match.end(1), str(int(match[1]) - 1))
            count += 1
    if count != 13:
        raise RuntimeError(f"Unexpected renumbering count: {count}")


def export_with_updated_toc_pages():
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(OUTPUT), ConfirmConversions=False, ReadOnly=False, AddToRecentFiles=False)
        if document.TablesOfContents.Count != 1 or document.TablesOfFigures.Count != 0:
            raise RuntimeError("Unexpected field structure")
        document.Repaginate()
        # Headings are unchanged: update only page numbers, preserving TOC formatting.
        document.TablesOfContents(1).UpdatePageNumbers()
        document.Repaginate()
        document.Save()
        document.ExportAsFixedFormat(str(PDF), 17)
        document.Close(False)
        document = None
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()


def normalized(value):
    return re.sub(r"\s+", " ", value).strip().casefold()


def caption_pages():
    document = Document(OUTPUT)
    captions = [p.text.strip() for p in document.paragraphs
                if p.style.name in {"Figure Caption Generated", "Table Caption Generated"} and p.text.strip()]
    if len(captions) != 44:
        raise RuntimeError("Expected 25 figures and 19 table captions")
    with fitz.open(PDF) as pdf:
        texts = [normalized(page.get_text()) for page in pdf]
        mapping = {}
        for caption in captions:
            matches = [i for i, text in enumerate(texts) if normalized(caption) in text]
            if len(matches) < 2:
                raise RuntimeError(f"Caption absent from list or body: {caption}")
            page = pdf[matches[-1]]
            footer = layout_audit.footer_numbers(page)
            if len(footer) != 1:
                raise RuntimeError("Cannot identify actual displayed page")
            mapping[caption] = int(footer[0])
        return mapping


def update_list_pages(mapping):
    changes = []

    def edit(xml):
        for paragraph in xml.xpath(".//w:p", namespaces=NS):
            value = plain(paragraph)
            for caption, number in mapping.items():
                if value.startswith(caption) and value[len(caption):].isdigit():
                    old = value[len(caption):]
                    if old != str(number):
                        splice(paragraph, len(caption), len(value), str(number))
                        changes.append(caption)
                    break
    update_document_xml(edit)
    return changes


def media(path):
    with ZipFile(path) as archive:
        return Counter(hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith("word/media/"))


def body(document):
    start = next(i for i, p in enumerate(document.paragraphs) if p.text.strip() == "RESUMEN")
    return document.paragraphs[start:]


def expected_text(text):
    if text.startswith("El login aplica rate limiting local"):
        text = text.replace(AUTH_ANCHOR, AUTH_ADDITION + AUTH_ANCHOR)
    text = text.replace(URL * 2, URL)
    return re.sub(r"\bTabla 4-([3-8])\b", lambda m: "Tabla 4-" + str(int(m[1])-1), text)


def verify():
    before, after = Document(SOURCE), Document(OUTPUT)
    old_body = [p for p in body(before) if p.text != ORPHAN]
    new_body = body(after)
    expected = [expected_text(p.text) for p in old_body]
    actual = [p.text for p in new_body]
    old_tables = [[c.text for r in t.rows for c in r.cells] for t in before.tables]
    new_tables = [[c.text for r in t.rows for c in r.cells] for t in after.tables]
    for cells in old_tables:
        for i, text in enumerate(cells):
            cells[i] = text.replace(OLD_HEALTH, NEW_HEALTH)
    headings_before = [(p.text, p.style.name) for p in before.paragraphs if p.style.name.startswith("Heading")]
    headings_after = [(p.text, p.style.name) for p in after.paragraphs if p.style.name.startswith("Heading")]
    section_attrs = ("page_width", "page_height", "top_margin", "bottom_margin", "left_margin", "right_margin", "start_type")
    sections_same = [[getattr(s, k) for k in section_attrs] for s in before.sections] == [[getattr(s, k) for k in section_attrs] for s in after.sections]
    list_entries = [p for p in after.paragraphs if re.match(r"^(Figura|Tabla) \d+-\d+\..+\t\d+$", p.text)]
    mapping = caption_pages()
    mismatches = [p.text for p in list_entries if int(p.text.rsplit("\t", 1)[1]) != mapping[p.text.rsplit("\t", 1)[0]]]
    counts = {"tables": len(after.tables), "images": len(after.inline_shapes), "sections": len(after.sections),
              "table_captions": sum(p.style.name == "Table Caption Generated" and bool(p.text.strip()) for p in after.paragraphs),
              "figure_and_table_list_entries": len(list_entries)}
    checks = {
        "source_backup_and_public_word_unchanged": all(digest(p) == EXPECTED_SOURCE for p in (SOURCE, BACKUP, PUBLIC_SOURCE)),
        "only_authorized_body_text_changes": actual == expected,
        "body_paragraph_styles_unchanged": [p.style.name for p in old_body] == [p.style.name for p in new_body],
        "only_authorized_table_cell_change": old_tables == new_tables,
        "chapter_headings_unchanged": headings_before == headings_after,
        "media_unchanged": media(SOURCE) == media(OUTPUT),
        "section_geometry_unchanged": sections_same,
        "all_word_fonts_arial": not check_fonts(OUTPUT),
        "orphan_caption_removed": not any(ORPHAN in p.text for p in after.paragraphs),
        "chapter4_table_numbers_contiguous": [int(re.match(r"Tabla 4-(\d+)", p.text)[1]) for p in after.paragraphs
                                             if p.style.name == "Table Caption Generated" and p.text.startswith("Tabla 4-")] == list(range(1, 8)),
        "all_list_pages_current": len(list_entries) == 44 and not mismatches,
        "structure_preserved": (counts["tables"], counts["images"], counts["sections"], counts["table_captions"]) == (19, 28, 9, 19),
    }
    with fitz.open(PDF) as pdf:
        fonts = sorted({span["font"] for page in pdf for block in page.get_text("dict")["blocks"] if "lines" in block
                        for line in block["lines"] for span in line["spans"]})
        checks["pdf_fonts_unchanged"] = fonts == ["Arial-BoldMT", "Arial-ItalicMT", "ArialMT"]
        checks["no_broken_fields"] = not any(any(marker in page.get_text() for marker in ("Error!", "¡Error!", "Reference source not found")) for page in pdf)
        pages = len(pdf)
    report = {"source": str(SOURCE), "backup": str(BACKUP), "docx": str(OUTPUT), "pdf": str(PDF),
              "docx_sha256": digest(OUTPUT), "pdf_sha256": digest(PDF), "pages": pages,
              "counts": counts, "checks": checks, "list_mismatches": mismatches, "passed": all(checks.values())}
    if actual != expected:
        report["unexpected_body_changes"] = [{"index": i, "expected": a, "actual": b} for i, (a, b) in enumerate(zip(expected, actual)) if a != b][:5]
    (FOLDER / "verificacion_integridad.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Document integrity verification failed")
    layout_audit.DOCX, layout_audit.PDF, layout_audit.REPORT = OUTPUT, PDF, FOLDER / "control_formato.json"
    layout_audit.main()


def main():
    if digest(SOURCE) != EXPECTED_SOURCE or digest(PUBLIC_SOURCE) != EXPECTED_SOURCE:
        raise RuntimeError("The approved source differs from the verified version")
    FOLDER.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() or BACKUP.exists():
        raise RuntimeError("Output already exists; preserve previous versions")
    shutil.copy2(SOURCE, BACKUP)
    shutil.copy2(SOURCE, OUTPUT)
    update_document_xml(corrections)
    with ZipFile(SOURCE) as original, ZipFile(OUTPUT) as corrected:
        changed_parts = [n for n in original.namelist() if original.read(n) != corrected.read(n)]
        if changed_parts != ["word/document.xml"]:
            raise RuntimeError(f"Unexpected package changes: {changed_parts}")
    print("Content corrected and then renumbered; original format definitions preserved.", flush=True)
    export_with_updated_toc_pages()
    for _ in range(4):
        changes = update_list_pages(caption_pages())
        if not changes:
            break
        export_with_updated_toc_pages()
    verify()


if __name__ == "__main__":
    main()
