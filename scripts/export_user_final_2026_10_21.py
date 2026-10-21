"""Preserve the user's final Word byte for byte and export it with Word."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import fitz
import win32com.client
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\Usuario\Desktop\Trabajo Final_Solari Pablo_21-10-2026.docx")
FOLDER = ROOT / "docs/revision_octubre_2026/entrega_word_usuario_21-10-2026"
OUTPUT = FOLDER / SOURCE.name
PDF = OUTPUT.with_suffix(".pdf")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    original = digest(SOURCE)
    if original != "81A743DF38FB32820EA7DC3BEC247F57CCE6F3F826510DFEA2ED9CF3D63A131C":
        raise RuntimeError("The attached document changed since inspection")
    FOLDER.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() or PDF.exists():
        raise RuntimeError("Preserve existing exports")
    shutil.copy2(SOURCE, OUTPUT)
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(OUTPUT), ConfirmConversions=False,
                                       ReadOnly=True, AddToRecentFiles=False)
        document.Repaginate()
        pages = document.ComputeStatistics(2)
        # Do not update fields or save: the final Word is exactly the user's file.
        document.ExportAsFixedFormat(str(PDF), 17)
        document.Close(False)
        document = None
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()
    if digest(SOURCE) != original or digest(OUTPUT) != original:
        raise RuntimeError("The approved Word was altered")
    with ZipFile(OUTPUT) as archive:
        if archive.testzip():
            raise RuntimeError("Invalid DOCX package")
    docx = Document(OUTPUT)
    with fitz.open(PDF) as pdf:
        fonts = sorted({span["font"] for page in pdf for block in page.get_text("dict")["blocks"]
                        if "lines" in block for line in block["lines"] for span in line["spans"]})
        pdf[0].get_pixmap(matrix=fitz.Matrix(1, 1)).save(str(FOLDER / "portada.png"))
        report = {"source": str(SOURCE), "docx": str(OUTPUT), "pdf": str(PDF),
                  "word_preserved_byte_for_byte": True, "docx_sha256": original,
                  "pdf_sha256": digest(PDF), "word_pages": pages, "pdf_pages": len(pdf),
                  "images": len(docx.inline_shapes), "tables": len(docx.tables),
                  "sections": len(docx.sections), "pdf_fonts": fonts}
    (FOLDER / "exportacion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
