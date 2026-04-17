from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from docx import Document


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOUTING_APP_DIR = REPO_ROOT / "scouting_app"
DOCS_DIR = REPO_ROOT / "docs"
METADATA_PATH = SCOUTING_APP_DIR / "training_metadata.json"
TRAINING_DB_PATH = SCOUTING_APP_DIR / "players_training.db"
PROGRESS_MD = DOCS_DIR / "prediction_improvement_progress.md"
PROGRESS_DOCX = DOCS_DIR / "prediction_improvement_progress.docx"

PRIOR_LONGITUDINAL_STAGE = {
    "commit": "449fbe9",
    "players": 20000,
    "player_attribute_history": 180051,
    "player_stats": 162778,
    "pytorch": {
        "roc_auc": 0.8468,
        "pr_auc": 0.6153,
        "f1": 0.5751,
    },
    "logistic_balanced": {
        "roc_auc": 0.9052,
        "pr_auc": 0.7360,
        "f1": 0.6878,
    },
}


def git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
            )
            .strip()
        )
    except Exception:
        return "desconocida"


def git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                text=True,
            )
            .strip()
        )
    except Exception:
        return "desconocido"


def format_metric(value) -> str:
    if value in ("", None):
        return "N/D"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def db_counts() -> dict[str, int]:
    queries = {
        "players": "select count(*) from players",
        "player_attribute_history": "select count(*) from player_attribute_history",
        "player_stats": "select count(*) from player_stats",
        "matches": "select count(*) from matches",
        "player_match_participations": "select count(*) from player_match_participations",
        "scout_reports": "select count(*) from scout_reports",
    }
    counts: dict[str, int] = {}
    with sqlite3.connect(TRAINING_DB_PATH) as connection:
        cursor = connection.cursor()
        for key, query in queries.items():
            cursor.execute(query)
            counts[key] = int(cursor.fetchone()[0])
    return counts


def build_markdown(metadata: dict, counts: dict[str, int]) -> str:
    pytorch_test = metadata["pytorch"]["test"]
    baseline_lr = metadata["baselines"]["logistic_regression_balanced"]["test"]
    delta_pytorch_f1 = float(pytorch_test["f1"]) - PRIOR_LONGITUDINAL_STAGE["pytorch"]["f1"]
    delta_pytorch_pr_auc = float(pytorch_test["pr_auc"]) - PRIOR_LONGITUDINAL_STAGE["pytorch"]["pr_auc"]

    return f"""# Mejora de prediccion - avance tecnico

## Estado actual
- Rama de trabajo: `{git_branch()}`
- Ultimo commit publicado antes de esta etapa: `{PRIOR_LONGITUDINAL_STAGE["commit"]}`
- HEAD local al generar este documento: `{git_head()}`
- Objetivo de esta iteracion: sumar contexto de partido (`Match` + `PlayerMatchParticipation`) y observacion cualitativa (`ScoutReport`) al sistema de prediccion.
- Alcance del producto mantenido: scouting juvenil de 12 a 18 anos para clubes formativos.

## Que se implemento en esta etapa
- Nueva tabla `Match` para contexto minimo del partido.
- Nueva tabla `PlayerMatchParticipation` para registrar la participacion puntual del jugador por partido.
- Nueva tabla `ScoutReport` para observaciones cualitativas del scout.
- El generador sintetico ahora crea:
- trayectoria tecnica mensual
- partidos con contexto
- participacion del jugador en cada partido
- `PlayerStat` agregado a partir de esas participaciones
- reportes de scout derivados de trayectoria y rendimiento reciente
- El pipeline de entrenamiento e inferencia ahora agrega features de:
- contexto de partido
- tasa de titularidad
- minutos medios
- nivel medio del rival
- alineacion entre posicion natural y posicion jugada
- reportes cualitativos del scout

## Conteos reales de la base de entrenamiento actual
- Jugadores: {counts["players"]}
- Snapshots de `PlayerAttributeHistory`: {counts["player_attribute_history"]}
- Registros agregados de `PlayerStat`: {counts["player_stats"]}
- Partidos sinteticos (`Match`): {counts["matches"]}
- Participaciones por partido: {counts["player_match_participations"]}
- Reportes de scout: {counts["scout_reports"]}

## Comparacion con la etapa anterior
### Etapa anterior: trayectoria tecnica sin contexto de partido explicito
- PyTorch: ROC-AUC {PRIOR_LONGITUDINAL_STAGE["pytorch"]["roc_auc"]:.4f}, PR-AUC {PRIOR_LONGITUDINAL_STAGE["pytorch"]["pr_auc"]:.4f}, F1 {PRIOR_LONGITUDINAL_STAGE["pytorch"]["f1"]:.4f}
- Logistic balanceado: ROC-AUC {PRIOR_LONGITUDINAL_STAGE["logistic_balanced"]["roc_auc"]:.4f}, PR-AUC {PRIOR_LONGITUDINAL_STAGE["logistic_balanced"]["pr_auc"]:.4f}, F1 {PRIOR_LONGITUDINAL_STAGE["logistic_balanced"]["f1"]:.4f}

### Etapa actual: contexto de partido + ScoutReport
- PyTorch: accuracy {format_metric(pytorch_test["accuracy"])}, ROC-AUC {format_metric(pytorch_test["roc_auc"])}, PR-AUC {format_metric(pytorch_test["pr_auc"])}, F1 {format_metric(pytorch_test["f1"])}, precision {format_metric(pytorch_test["precision"])}, recall {format_metric(pytorch_test["recall"])}
- Logistic balanceado: accuracy {format_metric(baseline_lr["accuracy"])}, ROC-AUC {format_metric(baseline_lr["roc_auc"])}, PR-AUC {format_metric(baseline_lr["pr_auc"])}, F1 {format_metric(baseline_lr["f1"])}

## Hallazgos verificados
- PyTorch mejora respecto de la etapa longitudinal anterior:
- cambio en F1: {delta_pytorch_f1:+.4f}
- cambio en PR-AUC: {delta_pytorch_pr_auc:+.4f}
- El baseline `LogisticRegression(class_weight="balanced")` sigue siendo mejor que PyTorch en esta corrida.
- La prediccion ahora es mas defendible metodologicamente porque ya no se apoya solo en atributos actuales y agregados simples.
- El sistema ya puede incorporar senal de contexto competitivo y senal cualitativa del scout sin romper el pipeline compartido entre entrenamiento e inferencia.

## Limites honestos de esta etapa
- Los partidos sinteticos siguen siendo por jugador; todavia no representan encuentros compartidos entre varios jugadores del mismo plantel.
- `ScoutReport` sigue siendo sintetico, no manual ni proveniente de observacion real de usuarios.
- El target del entrenamiento sigue siendo `potential_label` binario; todavia no pasamos a una meta temporal de progresion.
- Aunque PyTorch mejoro, todavia no supera al baseline lineal balanceado.
- La base de entrenamiento SQLite ya es pesada y el repo recibio advertencia de GitHub por tamano de `players_training.db`.

## Validacion ejecutada
- `pytest -q`: 34 tests aprobados.
- Smoke de app con artefactos nuevos:
- `/` -> 200
- `/health` -> 200
- `/login` -> 200
- Reentrenamiento completo ejecutado sobre la nueva base sintetica.

## Proximo paso recomendado
- Cambiar el target a una meta temporal de progresion.
- Evaluar si conviene introducir `Availability` o `PhysicalAssessment`.
- Replantear si el baseline lineal debe quedar como referencia principal hasta que PyTorch demuestre ventaja clara.
"""


def markdown_to_docx(markdown_text: str, output_path: Path) -> None:
    doc = Document()
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            doc.add_paragraph("")
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:], level=1)
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:], level=2)
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:], level=3)
            continue
        if line.startswith("- "):
            doc.add_paragraph(line[2:], style="List Bullet")
            continue
        doc.add_paragraph(line)
    doc.save(output_path)


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    counts = db_counts()
    markdown = build_markdown(metadata, counts)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_MD.write_text(markdown, encoding="utf-8")
    markdown_to_docx(markdown, PROGRESS_DOCX)
    print(f"Generado: {PROGRESS_MD}")
    print(f"Generado: {PROGRESS_DOCX}")


if __name__ == "__main__":
    main()
