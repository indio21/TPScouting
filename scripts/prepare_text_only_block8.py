"""Create a text-only thesis copy and an inventory for restoring images later."""

from __future__ import annotations

import hashlib
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_CANDIDATO_FINAL_BLOQUE8_v4_2026-10-06.docx"
OUTPUT = ROOT / "docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_REVISION_TEXTO_SIN_IMAGENES_2026-10-06.docx"
INVENTORY = ROOT / "docs/revision_octubre_2026/inventario_imagenes_bloque8.md"
EXPECTED_SHA256 = "f66ee0c475c33f7d0a4880254f6510e590ca7531ec8f972941468e7ff78a21ee"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def nearby_caption(paragraphs, index: int) -> str:
    for distance in range(5):
        for candidate_index in (index + distance, index - distance):
            if 0 <= candidate_index < len(paragraphs):
                paragraph = paragraphs[candidate_index]
                if "Caption" in paragraph.style.name and paragraph.text.strip():
                    return paragraph.text.strip()
    return "Sin leyenda próxima identificada"


def main() -> None:
    if digest(SOURCE.read_bytes()) != EXPECTED_SHA256:
        raise RuntimeError("El candidato v4 no coincide con el hash del checkpoint.")

    document = Document(SOURCE)
    paragraphs = list(document.paragraphs)
    rows = []
    removed = 0

    for index, paragraph in enumerate(paragraphs):
        drawings = list(paragraph._p.xpath(".//w:drawing"))
        pictures = list(paragraph._p.xpath(".//w:pict"))
        for node in drawings + pictures:
            blips = node.xpath(".//a:blip")
            relationship_id = blips[0].get(qn("r:embed")) if blips else None
            part = document.part.related_parts.get(relationship_id) if relationship_id else None
            blob = part.blob if part is not None and hasattr(part, "blob") else b""
            filename = Path(str(part.partname)).name if part is not None else "sin-relacion"
            rows.append(
                {
                    "occurrence": removed + 1,
                    "paragraph": index + 1,
                    "caption": nearby_caption(paragraphs, index),
                    "file": filename,
                    "sha256": digest(blob) if blob else "no-disponible",
                    "bytes": len(blob),
                }
            )
            node.getparent().remove(node)
            removed += 1

    media = {
        row["file"]: (row["bytes"], row["sha256"])
        for row in rows
        if row["file"] != "sin-relacion"
    }
    image_relationships = [
        relationship.rId
        for relationship in document.part.rels.values()
        if not relationship.is_external and relationship.reltype.endswith("/image")
    ]
    for relationship_id in image_relationships:
        document.part.drop_rel(relationship_id)
    document.save(OUTPUT)

    lines = [
        "# Inventario de imágenes retiradas — Bloque 8",
        "",
        f"Fuente inalterada: `{SOURCE.relative_to(ROOT)}`.",
        f"Copia para revisión textual: `{OUTPUT.relative_to(ROOT)}`.",
        f"Ocurrencias retiradas del cuerpo: {removed}.",
        f"Archivos multimedia relacionados en el paquete: {len(media)}.",
        "",
        "Las imágenes se retiraron únicamente de la copia de revisión. La fuente v4 conserva todos los recursos.",
        "",
        "| # | Párrafo | Leyenda próxima | Recurso | Bytes | SHA-256 |",
        "|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        caption = row["caption"].replace("|", "\\|")
        lines.append(
            f"| {row['occurrence']} | {row['paragraph']} | {caption} | {row['file']} | "
            f"{row['bytes']} | `{row['sha256']}` |"
        )
    lines.extend(["", "## Recursos únicos", "", "| Recurso | Bytes | SHA-256 |", "|---|---:|---|"])
    for filename, (size, sha256) in sorted(media.items()):
        lines.append(f"| {filename} | {size} | `{sha256}` |")
    INVENTORY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"REMOVED={removed}")
    print(f"OUTPUT={OUTPUT}")
    print(f"OUTPUT_SHA256={digest(OUTPUT.read_bytes())}")
    print(f"INVENTORY={INVENTORY}")


if __name__ == "__main__":
    main()
