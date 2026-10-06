# Indice de evidencias para incorporar al Word

Fecha de preparacion: 2026-07-10

Commit local auditado: `7d7680e04dd15b4c4f01d9e0aee8711aeea06f0d`

## Evidencia principal

| Archivo | Tipo | Contenido | Ubicacion sugerida en el Word |
|---|---|---|---|
| `evidencia_bloque2.md` | Informe | Auditoria integral con archivos, funciones y lineas | Anexo tecnico |
| `tablas_para_tesis.md` | Tablas | Target, 68 features, entrenamiento, metricas, pruebas y problemas | Metodologia, Desarrollo, Resultados y Anexos |
| `vacios_no_verificables.md` | Control academico | Afirmaciones que no pueden sostenerse | Limitaciones |
| `evidencia_bloque2.json` | JSON estructurado | Hallazgos procesables y verificables | Anexo digital |
| `comandos_ejecutados.txt` | Transcript resumido | Comandos seguros y resultados | Anexo de reproducibilidad |
| `manifest_evidencias.json` | Manifiesto SHA-256 | Tamano y hash de cada evidencia separada | Control de integridad del anexo digital |

## Entrenamiento y modelo

| Archivo | Naturaleza | Pie o descripcion sugerida |
|---|---|---|
| `training_metadata_snapshot_2026-07-10.json` | Copia exacta | Metadata de la corrida oficial registrada el 19/05/2026 |
| `training_history.csv` | Derivado del metadata | Evolucion por epoca de training loss, PR-AUC, F1, threshold y learning rate |
| `training_curve_snapshot_2026-07-10.png` | Copia de evidencia versionada | Curva de entrenamiento reconstruida desde la metadata persistida |
| `metricas_test.csv` | Derivado del metadata | Metricas separadas de PlayerNet calibrado, PlayerNet crudo y baselines |
| `target_distribution.csv` | Derivado verificado | Distribucion del target temporal por particion |
| `artefactos_sha256.csv` | Calculo directo | Identificacion SHA-256 del modelo, preprocesador, calibrador, metadata y splits |
| `trazabilidad_entrenamiento.md` | Documento | Alcance y limites de la evidencia de entrenamiento |

## Features

| Archivo | Naturaleza | Uso sugerido |
|---|---|---|
| `features_68.csv` | Exportacion del preprocesador | Tabla editable/importable en Word o Excel |
| `features_68.json` | Exportacion estructurada | Anexo digital con nombre, grupo, origen y transformacion de cada feature |

Control: ambos archivos contienen exactamente 68 filas/features y coinciden con `input_dim=68`.

## Pruebas y cobertura

| Archivo | Resultado conservado | Pie o descripcion sugerida |
|---|---|---|
| `pytest_2026-07-10.txt` | 83 passed, 1 skipped, 4 warnings, 38.88 s | Salida local completa de pytest sobre el codigo sincronizado de entrega |
| `coverage_2026-07-10.txt` | 83 passed, 1 skipped, 4 warnings; cobertura total 79% | Reporte completo de pytest-cov por archivo |

Comandos reales:

```powershell
python -m pytest -q -rs -p no:cacheprovider
python -m pytest -q -rs -p no:cacheprovider --cov=scouting_app --cov-report=term-missing
```

La prueba omitida es el smoke visual Playwright opt-in. Los cuatro warnings provienen de columnas all-NaN en dos pruebas de preprocesamiento.

## GitHub Actions

| Archivo | Naturaleza | Pie o descripcion sugerida |
|---|---|---|
| `github_actions_2026-07-10.png` | Captura publica real | Historial de 71 ejecuciones del workflow CI |
| `github_actions_run71_2026-07-10.png` | Captura publica real | CI #71 exitosa en Python 3.11 y 3.12, con dos artefactos de cobertura |
| `github_actions_runs_2026-07-10.json` | API publica sin transformar | Diez runs mas recientes y total de runs |
| `github_actions_run71_jobs_2026-07-10.json` | API publica sin transformar | Jobs y pasos de CI #71 |
| `ci_workflow_snapshot_2026-07-10.yml` | Copia exacta | Workflow que instala dependencias, ejecuta pytest-cov y sube coverage.xml |
| `github_actions_evidencia.md` | Documento | Lectura verificable, URLs y limitaciones de la evidencia remota |

Advertencia para la redaccion: CI #71 prueba el commit remoto `fa8a50d...`. No prueba el commit local `7d7680e...`, que al momento de la auditoria estaba un commit por delante de `origin/main`.

## Evidencia que no debe afirmarse

- No existe log crudo del entrenamiento original.
- No existe validation loss ni duracion registrada.
- El metadata no guarda el commit exacto de entrenamiento.
- Las metricas de test no evaluan el score combinado que muestra la aplicacion.
- La corrida oficial utiliza datos sinteticos; no hay evidencia de datos reales o semisinteticos.

## Orden recomendado de incorporacion

1. En Metodologia: tabla del target y composicion de 68 features.
2. En Desarrollo: configuracion de PlayerNet, flujo de inferencia y artefactos.
3. En Pruebas: resumen pytest, cobertura y captura de CI #71.
4. En Resultados: metricas separadas para salida cruda y calibrada.
5. En Limitaciones: target construido antes del split, datos sinteticos y evidencia de entrenamiento faltante.
6. En Anexos: metadata, history CSV, features CSV/JSON, hashes, transcripciones y capturas.
