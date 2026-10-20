"""Copy only approved final Word/PDF delivery documents.

Old public Word/PDF files are backed up in the private working repository before
replacement. This script performs no Git commit, push, or application change.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT.parent / "TPScouting-entrega"
FINAL = ROOT / "docs/revision_octubre_2026/final"
BACKUP = FINAL / "backup_entrega_anterior"
STEM = "TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20"
PREVIOUS = "TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-15"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if DELIVERY.resolve() != Path(r"C:\Tesis\TPScouting-entrega").resolve():
        raise RuntimeError("El destino no coincide con el repositorio de entrega.")
    BACKUP.mkdir(parents=True, exist_ok=True)
    for extension in ("docx", "pdf"):
        old = DELIVERY / "documento" / f"{PREVIOUS}.{extension}"
        backup = BACKUP / old.name
        new = FINAL / f"{STEM}.{extension}"
        target = DELIVERY / "documento" / new.name
        if old.exists():
            if backup.exists() and digest(backup) != digest(old):
                raise RuntimeError("Ya existe un backup distinto; no se sobrescribe.")
            shutil.copy2(old, backup)
            if digest(backup) != digest(old):
                raise RuntimeError("El backup no coincide con su fuente.")
        shutil.copy2(new, target)
        if digest(new) != digest(target):
            raise RuntimeError("El archivo copiado difiere del aprobado.")
        if old.exists():
            old.unlink()
        print(f"DELIVERY={target.name} SHA256={digest(target)}")

if __name__ == "__main__":
    main()
