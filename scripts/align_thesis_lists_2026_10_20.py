"""Align final figure/table page numbers without changing approved content."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import fitz
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.shared import Pt

import finalize_thesis_delivery_2026_10_19 as layout
from update_thesis_auth_arial_2026_10_20 import arial_package, check_fonts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/revision_octubre_2026/cierre_correcciones_2026-10-20/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20.docx"
FOLDER = ROOT / "docs/revision_octubre_2026/listas_alineadas_2026-10-20"
OUTPUT = FOLDER / "TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_LISTAS_ALINEADAS_2026-10-20.docx"
PDF = OUTPUT.with_suffix(".pdf")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def align_entries():
    document = Document(OUTPUT)
    section = document.sections[0]
    width = section.page_width - section.left_margin - section.right_margin
    count = 0
    for paragraph in document.paragraphs:
        if not re.match(r"^(Figura|Tabla) \d+-\d+\..+\t\d+$", paragraph.text):
            continue
        fmt = paragraph.paragraph_format
        fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt.left_indent = Pt(0)
        fmt.first_line_indent = Pt(0)
        # Reserve space for the number on wrapped lines. The right tab itself
        # sits at the common text margin, beyond this wrapping inset.
        fmt.right_indent = Pt(24)
        fmt.tab_stops.clear_all()
        fmt.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        for run in paragraph.runs:
            run.font.name = "Arial"
        count += 1
    if count != 45:
        raise RuntimeError(f"Unexpected list entry count: {count}")
    document.save(OUTPUT)


def body(document):
    paragraphs = document.paragraphs
    start = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "RESUMEN")
    return [p.text for p in paragraphs[start:]]


def media(path):
    with ZipFile(path) as archive:
        return Counter(hashlib.sha256(archive.read(n)).hexdigest()
                       for n in archive.namelist() if n.startswith("word/media/"))


def number_geometry():
    document = Document(OUTPUT)
    entries = [p.text for p in document.paragraphs
               if re.match(r"^(Figura|Tabla) \d+-\d+\..+\t\d+$", p.text)]
    results = []
    with fitz.open(PDF) as pdf:
        for entry in entries:
            caption, number = entry.rsplit("\t", 1)
            key = re.match(r"^(Figura|Tabla) \d+-\d+\.", caption).group()
            matches = []
            for index in range(min(10, len(pdf))):
                page = pdf[index]
                words = page.get_text("words")
                words.sort(key=lambda w: (w[5], w[6], w[7]))
                # Captions start with an exact label and numbered identifier.
                label, identifier = key.split()
                for pos in range(len(words) - 1):
                    if words[pos][4] == label and words[pos+1][4] == identifier:
                        # Last number at the fixed right margin before next entry.
                        candidates = [w for w in words[pos+2:]
                                      if w[4] == number and abs(w[2] - 526.95) < 1
                                      and w[1] >= words[pos][1] - 1]
                        if candidates:
                            chosen = min(candidates, key=lambda w: w[1])
                            matches.append({"physical_page": index+1, "right_x": round(chosen[2], 3)})
            if len(matches) != 1:
                raise RuntimeError(f"Cannot locate aligned page number: {key}: {matches}")
            results.append({"entry": key, "number": int(number), **matches[0]})
        fonts = sorted({span["font"] for page in pdf for block in page.get_text("dict")["blocks"]
                        if "lines" in block for line in block["lines"] for span in line["spans"]})
        for index in sorted({item["physical_page"] - 1 for item in results}):
            pdf[index].get_pixmap(matrix=fitz.Matrix(1, 1)).save(str(FOLDER / f"lista_pagina_{index+1:03}.png"))
        return results, fonts, len(pdf)


def main():
    if digest(SOURCE) != "29ACEE4C6117C143FA7086E7F15A2FDB27F6716ADE3261BD23DC32F960BCB03E":
        raise RuntimeError("Source differs from approved document")
    FOLDER.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise RuntimeError("Output exists; preserve reviewed versions")
    shutil.copy2(SOURCE, OUTPUT)
    layout.FINAL_DOCX, layout.FINAL_PDF = OUTPUT, PDF
    align_entries()
    arial_package(OUTPUT)
    layout.word_update_and_export()
    for _ in range(4):
        layout.rebuild_static_lists()
        align_entries()
        arial_package(OUTPUT)
        layout.word_update_and_export()
        if not layout.list_mismatches():
            break
    if layout.list_mismatches():
        raise RuntimeError("Page references did not converge")
    before, after = Document(SOURCE), Document(OUTPUT)
    checks = {
        "body_text_unchanged": body(before) == body(after),
        "table_text_unchanged": [[c.text for r in t.rows for c in r.cells] for t in before.tables]
                              == [[c.text for r in t.rows for c in r.cells] for t in after.tables],
        "media_unchanged": media(SOURCE) == media(OUTPUT),
        "structure_unchanged": (len(after.inline_shapes), len(after.tables), len(after.sections)) == (28, 19, 9),
        "all_word_fonts_arial": not check_fonts(OUTPUT),
    }
    numbers, fonts, pages = number_geometry()
    checks["all_pdf_fonts_arial"] = all(f.startswith("Arial") for f in fonts)
    checks["page_numbers_aligned"] = max(n["right_x"] for n in numbers) - min(n["right_x"] for n in numbers) < 0.2
    report = {"source": str(SOURCE), "docx": str(OUTPUT), "pdf": str(PDF),
              "docx_sha256": digest(OUTPUT), "pdf_sha256": digest(PDF), "pages": pages,
              "checks": checks, "fonts": fonts, "numbers": numbers, "passed": all(checks.values())}
    (FOLDER / "alineacion_e_integridad.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "numbers"}, indent=2, ensure_ascii=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
