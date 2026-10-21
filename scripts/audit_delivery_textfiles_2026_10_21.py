"""Check every public Markdown/text manifest against the delivery files."""

from __future__ import annotations

import importlib.metadata
import json
import re
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT.parent / "TPScouting-entrega"
FOLDER = ROOT / "docs/revision_octubre_2026/entrega_word_usuario_21-10-2026"


def main():
    paths = subprocess.check_output(["git", "ls-files", "*.md", "*.txt"], cwd=DELIVERY, text=True).splitlines()
    issues, checked_links, checked_refs, requirements = [], [], [], {}
    for name in paths:
        path = DELIVERY / name
        text = path.read_text(encoding="utf-8")
        if name.endswith(".md"):
            for raw in re.findall(r"\[[^\]]*\]\((<[^>]+>|[^)]+)\)", text):
                value = raw.strip().strip("<>")
                if urlsplit(value).scheme or value.startswith("#"):
                    continue
                relative = unquote(value.split("#", 1)[0])
                target = path.parent / relative
                checked_links.append({"source": name, "target": relative, "exists": target.exists()})
                if not target.exists():
                    issues.append(f"{name}: broken link {value}")
            for value in re.findall(r"`([^`\n]+)`", text):
                if re.match(r"^(docs/|scripts/|scouting_app/|requirements[^ /]*\.txt$)", value) and value.endswith((".py", ".md", ".txt", ".svg", ".pt", ".joblib")):
                    checked_refs.append({"source": name, "target": value})
                    if not (DELIVERY / value).is_file():
                        issues.append(f"{name}: absent referenced file {value}")
        else:
            for line in text.splitlines():
                if not line.strip() or line.lstrip().startswith(("#", "--")):
                    continue
                req = Requirement(line)
                key = canonicalize_name(req.name)
                requirements.setdefault(name, {})[key] = str(req.specifier)
                if req.marker and not req.marker.evaluate():
                    continue
                installed = importlib.metadata.version(req.name)
                if not req.specifier.contains(installed):
                    issues.append(f"{name}: installed {key}=={installed} does not match {req}")
    direct = requirements["requirements.txt"]
    lock = requirements["requirements-lock.txt"]
    for key, version in direct.items():
        source = requirements["requirements-torch-cpu.txt"] if key == "torch" else lock
        if source.get(key) != version:
            issues.append(f"Direct/lock mismatch: {key}")

    # Exercise the documented backup recipe on a live, disposable WAL database.
    runbook = (DELIVERY / "RUNBOOK.md").read_text(encoding="utf-8")
    recipe = re.search(r"@'\n(.*?)\n'@", runbook, re.DOTALL).group(1)
    with tempfile.TemporaryDirectory(prefix="tpscouting_backup_docs_") as temporary:
        source = Path(temporary) / "source.db"
        target = Path(temporary) / "backup.db"
        with closing(sqlite3.connect(source)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE evidence (value TEXT)")
            connection.execute("INSERT INTO evidence VALUES ('verified')")
            connection.commit()
            code = recipe.replace('"scouting_app/players_updated_v2.db"', repr(str(source)))
            code = code.replace('"backups/operativa_YYYYMMDD.db"', repr(str(target)))
            exec(compile(code, "RUNBOOK backup recipe", "exec"), {})
            with closing(sqlite3.connect(target)) as backup:
                if backup.execute("SELECT value FROM evidence").fetchall() != [("verified",)]:
                    issues.append("Backup recipe did not preserve committed records")
                if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    issues.append("Backup integrity_check failed")
    report = {"files": paths, "markdown_files": sum(p.endswith(".md") for p in paths),
              "text_manifests": sum(p.endswith(".txt") for p in paths), "relative_links": checked_links,
              "inline_file_references": checked_refs, "manifest_versions": requirements,
              "backup_recipe_verified_on_temporary_wal_database": True,
              "issues": issues, "passed": not issues}
    (FOLDER / "auditoria_md_txt.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"relative_links", "manifest_versions", "inline_file_references"}}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
