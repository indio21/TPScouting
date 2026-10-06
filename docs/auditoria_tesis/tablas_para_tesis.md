# Tablas verificadas para el Trabajo Final

Todas las cifras corresponden a la inspeccion del 2026-07-10. Las tablas distinguen el modelo crudo, su calibracion y el score combinado de la aplicacion.

## 1. Definicion de la variable objetivo

| Elemento | Definicion verificable | Fuente |
|---|---|---|
| Target usado para entrenar | `temporal_target_label` | `scouting_app/preprocessing.py:106,1264-1799`; `scouting_app/train_model.py:194-211` |
| Clase positiva | Jugador seleccionado por `selected_mask` luego del score temporal, quality gates, reglas de consolidacion/breakout y cuota por cohorte posicion:edad | `scouting_app/preprocessing.py:1577-1799` |
| Clase negativa | Jugador no seleccionado por esas reglas | `scouting_app/preprocessing.py:1798-1799` |
| Senales del desenlace | crecimiento y nivel futuro de atributos, rendimiento futuro, presion/dificultad, consistencia, rol/minutos/titularidad/posicion natural, disponibilidad, fatiga/lesion/recuperacion, evaluacion scout y breakout | `scouting_app/preprocessing.py:1264-1757` |
| Tasa objetivo | 8% deseado; limites globales 5%-12% | `scouting_app/preprocessing.py:107-109,1762-1788` |
| Distribucion total | 1,597 positivas; 18,403 negativas; 20,000 total (7.985%) | `scouting_app/training_metadata.json:380-386`; cache temporal inspeccionada |
| Distribucion train | 1,118 positivas; 12,882 negativas; 14,000 total | `training_splits.json` y target temporal inspeccionados |
| Distribucion validation | 239 positivas; 2,761 negativas; 3,000 total | idem |
| Distribucion test | 240 positivas; 2,760 negativas; 3,000 total | idem |
| `potential_label` | Etiqueta sintetica almacenada, generada desde edad, posicion y diez atributos; no es el `y` de esta corrida | `scouting_app/models.py:31-63`; `scouting_app/generate_data.py:122-193`; `scouting_app/train_model.py:194-211` |
| Leakage directo | No se encontro inclusion de futuro ni labels dentro de `MODEL_FEATURE_COLUMNS`; las features historicas se cortan antes del futuro | `scouting_app/preprocessing.py:30-106,951-1012,1802-1915` |
| Riesgo metodologico | El target se construye y normaliza con cuantiles/cuotas globales antes del split; validation/test intervienen en la definicion de sus etiquetas | `scouting_app/preprocessing.py:1700-1799`; `scouting_app/train_model.py:194-211,525-540` |

Formula documentable del score previo a los gates:

`0.15*growth + 0.12*future_level + 0.13*performance + 0.13*pressure + 0.10*consistency + 0.09*role + 0.09*availability + 0.09*recovery + 0.10*scout + 0.14*breakout - 0.08*stability_penalty`

Fuente: `scouting_app/preprocessing.py`, `_temporal_target_dataframe`, aprox. lineas 1577-1641.

## 2. Composicion de las 68 features

| Grupo | Variables de origen | Transformacion | Features resultantes | Acumulado |
|---|---|---|---:|---:|
| Base numericas | `age`; `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physical`, `vision`, `tackling`, `determination`, `technique` | Imputacion mediana + MinMaxScaler | 11 | 11 |
| Estadisticas historicas | conteo, promedio de score final, promedio de precision de pase, ultimo score final | Imputacion 0 + MinMaxScaler | 4 | 15 |
| Evolucion de atributos | conteo; mejoras 90/180/365d; tendencia; mejoras ponderadas 90/180/365d; tendencia ponderada; volatilidad; gap actual-reciente | Imputacion 0 + MinMaxScaler | 11 | 26 |
| Partidos | conteo; score promedio/reciente; minutos promedio/volatilidad; tasa titular; dificultad promedio; tasa y score en alta dificultad; tasa posicion natural | Imputacion 0 + MinMaxScaler | 10 | 36 |
| Reportes scout | conteo; decision; lectura tactica; perfil mental; adaptabilidad; proyeccion reciente; tendencia de proyeccion | Imputacion 0 + MinMaxScaler | 7 | 43 |
| Evaluacion fisica | conteo; altura/peso/velocidad/resistencia recientes; BMI; crecimiento de altura; cambio de peso; estiron; uso de pie izquierdo; uso de ambos pies | Imputacion 0 + MinMaxScaler | 11 | 54 |
| Disponibilidad | conteo; disponibilidad promedio/reciente; fatiga promedio/reciente; carga media; tasa lesion; dias perdidos; tendencia | Imputacion 0 + MinMaxScaler | 9 | 63 |
| Posicion | Portero, Defensa, Lateral, Mediocampista, Delantero | Imputacion moda + one-hot fijo | 5 | 68 |

Fuentes: `scouting_app/player_logic.py:13-24,42-48`; `scouting_app/preprocessing.py:30-148`; salida real de `scouting_app/preprocessor.joblib`; `scouting_app/training_metadata.json:25-29`.

Nombres exactos de salida del preprocesador:

| Rango | Features |
|---|---|
| 1-11 | `base_numeric__age`, `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physical`, `vision`, `tackling`, `determination`, `technique` (todos con prefijo `base_numeric__`) |
| 12-15 | `historical_numeric__stats_entry_count`, `avg_final_score_hist`, `avg_pass_accuracy_hist`, `latest_final_score_hist` |
| 16-26 | `historical_numeric__attr_history_entry_count`, `attr_avg_improvement_90d`, `attr_avg_improvement_180d`, `attr_avg_improvement_365d`, `attr_avg_trend_per_day`, `attr_weighted_improvement_90d`, `attr_weighted_improvement_180d`, `attr_weighted_improvement_365d`, `attr_weighted_trend_per_day`, `attr_weighted_volatility`, `attr_current_vs_recent_gap` |
| 27-36 | `historical_numeric__match_entry_count`, `match_avg_final_score`, `match_recent_final_score`, `match_avg_minutes`, `match_minutes_volatility`, `match_start_rate`, `match_avg_opponent_level`, `match_high_difficulty_rate`, `match_high_difficulty_score`, `match_natural_position_rate` |
| 37-43 | `historical_numeric__scout_report_count`, `scout_avg_decision_making`, `scout_avg_tactical_reading`, `scout_avg_mental_profile`, `scout_avg_adaptability`, `scout_latest_projection_score`, `scout_projection_trend` |
| 44-54 | `historical_numeric__phys_assessment_count`, `phys_recent_height_cm`, `phys_recent_weight_kg`, `phys_recent_speed_score`, `phys_recent_endurance_score`, `phys_recent_bmi`, `phys_height_growth_365d`, `phys_weight_change_180d`, `phys_growth_spurt_rate`, `phys_left_footed_rate`, `phys_two_footed_rate` |
| 55-63 | `historical_numeric__avail_record_count`, `avail_avg_pct`, `avail_recent_pct`, `avail_avg_fatigue`, `avail_recent_fatigue`, `avail_avg_training_load`, `avail_injury_rate`, `avail_missed_days_avg`, `avail_availability_trend` |
| 64-68 | `categorical__position_Portero`, `categorical__position_Defensa`, `categorical__position_Lateral`, `categorical__position_Mediocampista`, `categorical__position_Delantero` |

## 3. Configuracion del entrenamiento

| Parametro | Valor verificado | Fuente |
|---|---|---|
| Fecha registrada | 19/05/2026 22:51:17 (timestamp sin zona) | `scouting_app/training_metadata.json:2` |
| Seed | 42 | `training_metadata.json:3` |
| Dataset | 20,000 jugadores sinteticos, edades 12-17 observadas | `training_metadata.json:370-386`; `generate_data.py` |
| Split | 14,000 train; 3,000 validation; 3,000 test | `training_metadata.json:30-36`; `train_model.py:525-540` |
| Input | 68 | `training_metadata.json:27`; `model.pt` |
| Arquitectura | rama lineal 68->1 + residual 68->128->64->1, BatchNorm, GELU y dropout | `train_model.py:75-100`; `model.pt` |
| Parametros entrenables | 17,994 | estados y shapes de `model.pt` |
| Batch size | 256 | `training_metadata.json:9` |
| Learning rate inicial | 0.0005 | `training_metadata.json:7` |
| Loss | BCEWithLogitsLoss | `training_metadata.json:14`; `train_model.py:581-586` |
| Balanceo | `pos_weight=11.5223613596`; shuffle | `training_metadata.json:15-18` |
| Optimizador | AdamW, weight decay 0.0005 | `training_metadata.json:10-12`; `train_model.py:587` |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 3, modo max | `train_model.py:588` |
| Epocas solicitadas | 45 | `training_metadata.json:5` |
| Epocas ejecutadas | 15 | `training_metadata.json:6` |
| Mejor epoca | 5 | `training_metadata.json:39` |
| Early stopping | Si; patience 10; corte luego de epocas 6-15 sin superar el mejor PR-AUC | `training_metadata.json:8,39,122-258`; `train_model.py:602-664` |
| Metrica monitorizada | PR-AUC calibrado de validation; F1 como desempate | `train_model.py:624-653` |
| Loss train, mejor epoca | 0.6431135205 | `training_metadata.json`, historial epoca 5 |
| Loss train, ultima epoca | 0.5179418206 | `training_metadata.json`, historial epoca 15 |
| Validation loss | No registrada | `train_model.py:630-639`; ausencia en metadata |
| Duracion | No registrada | ausencia en `training_metadata.json` |
| Calibracion | Isotonic; threshold 0.25 | `training_metadata.json:39-41,260-265` |
| Checkpoint | formato v1; `model_state`, `input_dim`, `seed`, `model_class` | `train_model.py:119-126`; `model.pt` |

## 4. Metricas y baselines en test

| Salida/modelo | Threshold | Accuracy | ROC-AUC | PR-AUC | F1 | Precision | Recall | Matriz de confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PlayerNet + calibracion isotonic | 0.250 | 0.9303 | 0.9174 | 0.5241 | 0.5282 | 0.5764 | 0.4875 | `[[2674,86],[123,117]]` |
| PlayerNet sigmoid crudo | 0.825 | 0.9300 | 0.9203 | 0.5461 | 0.5291 | 0.5728 | 0.4917 | `[[2672,88],[122,118]]` |
| LogisticRegression balanced | 0.850 | 0.9310 | 0.9205 | 0.5378 | 0.5327 | 0.5813 | 0.4917 | `[[2675,85],[122,118]]` |
| Promedio simple de atributos | 0.625 | 0.8960 | 0.8390 | 0.3513 | 0.4201 | 0.3792 | 0.4708 | `[[2575,185],[127,113]]` |

Fuente: `scouting_app/training_metadata.json`, `pytorch.test`, `pytorch.raw_test`, `baselines`, aprox. lineas 38-120 y 275-369.

Nota obligatoria para la tesis: el score combinado que muestra la app (`0.35*modelo + 0.35*rating + 0.30*fit`, con renormalizacion si faltan datos) no fue evaluado por estas metricas. Fuente: `scouting_app/app.py:1225-1257`; `training_metadata.json:266-273`.

## 5. Pruebas y cobertura

| Evidencia | Resultado actual | Fuente |
|---|---|---|
| Suite local | 83 passed, 1 skipped, 4 warnings, 48.68 s | ejecucion 2026-07-10 |
| Test omitido | `tests/test_visual_smoke.py:73`; opt-in mediante `RUN_PLAYWRIGHT=1` | `tests/test_visual_smoke.py:9-14,73` |
| Warnings | 4 `All-NaN slice encountered` de scikit-learn en dos tests de preprocesamiento | salida pytest; sklearn `_array_api.py:794,814` |
| Cobertura total | 79%; 5082 statements; 1062 missed | pytest-cov 2026-07-10 |
| CI configurado | Python 3.11/3.12; pytest-cov; XML | `.github/workflows/ci.yml:1-39` |
| Artefacto CI | `coverage-python-${python-version}` con `coverage.xml` | `.github/workflows/ci.yml:35-39` |
| Ejecucion remota actual | No verificable desde evidencia local | no existe resultado persistido local |

| Archivo | Statements | Missed | Coverage |
|---|---:|---:|---:|
| `app.py` | 799 | 178 | 78% |
| `create_admin.py` | 49 | 49 | 0% |
| `db_utils.py` | 63 | 6 | 90% |
| `evaluate_saved_model.py` | 97 | 32 | 67% |
| `generate_data.py` | 561 | 15 | 97% |
| `ml/runtime.py` | 63 | 27 | 57% |
| `models.py` | 199 | 11 | 94% |
| `player_logic.py` | 61 | 8 | 87% |
| `preprocessing.py` | 724 | 53 | 93% |
| `routes/auth.py` | 58 | 4 | 93% |
| `routes/compare.py` | 175 | 110 | 37% |
| `routes/dashboard.py` | 204 | 48 | 76% |
| `routes/players.py` | 982 | 231 | 76% |
| `routes/settings.py` | 62 | 24 | 61% |
| `routes/staff.py` | 114 | 55 | 52% |
| `seed_demo_data.py` | 34 | 34 | 0% |
| `services/cache.py` | 35 | 5 | 86% |
| `services/locks.py` | 49 | 10 | 80% |
| `services/operational_data.py` | 87 | 21 | 76% |
| `services/security.py` | 52 | 2 | 96% |
| `sync_shortlist.py` | 154 | 40 | 74% |
| `train_model.py` | 453 | 99 | 78% |
| **TOTAL** | **5082** | **1062** | **79%** |

## 6. Reproducibilidad

| Etapa | Comando/configuracion real | Fuente |
|---|---|---|
| Python | 3.11 local; 3.11/3.12 en CI | `python --version`; `.github/workflows/ci.yml:10-12` |
| Entorno | `py -3.11 -m venv .venv` | convencion compatible con `README.md:71-76` |
| Dependencias | `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` y `requirements-dev.txt` | `README.md:74-75`; `RUNBOOK.md:55-56` |
| Variables minimas | `APP_SECRET_KEY`, `APP_DB_URL`; `TRAINING_DB_URL` para pipeline | `RUNBOOK.md:11-28`; `render.yaml:18-25` |
| Admin/inicializacion | `.\.venv\Scripts\python.exe .\scouting_app\create_admin.py` | `README.md:92-95`; `RUNBOOK.md:138-141` |
| App local | `.\.venv\Scripts\python.exe .\scouting_app\app.py` | `README.md:76`; `RUNBOOK.md:58` |
| Tests | `.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing` | `README.md:103`; `RUNBOOK.md:57` |
| Datos de training | `generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset` | `RUNBOOK.md:113` |
| Entrenamiento | `train_model.py ... --epochs 45 --lr 5e-4 --patience 10` | `RUNBOOK.md:114` |
| Evaluacion | `evaluate_saved_model.py --db-url sqlite:///players_training.db --metadata-path training_metadata.json` | `RUNBOOK.md:115` |
| Demo operativa | `sync_shortlist.py ... --limit 100 --min-age 12 --max-age 18 --replace` | `RUNBOOK.md:116` |
| Smoke deploy | `.\.venv\Scripts\python.exe scripts\smoke_render.py --base-url <URL>` | `scripts/smoke_render.py:20-93`; `RUNBOOK.md:84-90` |

## 7. Problemas tecnicos y soluciones

Solo se incluyen problemas documentados como reales en el repositorio.

| Problema | Causa documentada | Solucion implementada | Archivos afectados/evidencia | Limitacion residual |
|---|---|---|---|---|
| SQLite no persistente en Render | filesystem efimero | PostgreSQL administrado mediante `APP_DB_URL`/`TRAINING_DB_URL`; guardrail productivo | `render.yaml:18-25`; `docs/comparacion_falencias_codigo_fuente_2026-04-27.md:31-39`; `RUNBOOK.md:167-208` | Render Free puede expirar/dormir; una sola base en demo |
| Pipeline concurrente | operaciones administrativas largas podian solaparse | lock de thread + archivo y Gunicorn con un worker | `scouting_app/services/locks.py:12-58`; `render.yaml:16`; `docs/auditoria_pendientes_2026-05-17.md:31-35` | no es lock distribuido ni apto para multi-instancia |
| Cache sin limites/latencia de demo | consultas/render pesados y cache local inicialmente acotada de forma insuficiente | TTL, max entries, invalidacion y paginacion 20 en Render | `scouting_app/services/cache.py:1-43`; `render.yaml:38-41`; `docs/cierre_pre_entrega_word_render_2026-05-18.md:187-208` | cache por proceso; no Redis; cold start permanece |
| Rate limit de login basico | proteccion ausente/solo local | rate limiter en memoria con lock y tests | `scouting_app/services/security.py:1-58`; `docs/comparacion_falencias_codigo_fuente_2026-04-27.md:42` | no distribuido ni persistente |
| `app.py` monolitico | rutas, seguridad, cache, datos y ML concentrados | blueprints por familia y servicios parciales | `docs/refactor_arquitectura_2026-04-28.md:13-29,45-97,245-267`; `scouting_app/routes/`; `scouting_app/services/` | `app.py` aun tiene 799 statements y helpers compartidos |
| Target sintetico demasiado duro | una foto fija y reglas sintenticas poco representativas de trayectorias | rediseño longitudinal, arquetipos, atributos/partidos/scout/fisico/disponibilidad y target temporal | `docs/session_2026-04-22_synthetic_redesign.md:9-18,38-99`; `scouting_app/generate_data.py`; `preprocessing.py:1264-1915` | datos y validacion siguen siendo sinteticos; target global antes del split |
| Desbalance de clase | aproximadamente 8% positivos | BCEWithLogitsLoss con `pos_weight`, stratified split, PR-AUC y baseline balanced | `train_model.py:525-588`; `training_metadata.json:14-18,380-386` | F1/recall moderados; baseline sigue competitivo |
| Desalineacion entrenamiento/inferencia | riesgo de transformar distinto o cargar checkpoint incompatible | preprocesador persistido, checkpoint con version/input_dim y runtime con validacion | `train_model.py:119-126,816-889`; `scouting_app/ml/runtime.py`; `app.py:752-769` | artefactos deben versionarse y desplegarse juntos |
| Fechas de nacimiento ausentes en datos demo heredados | base previa guardaba edad pero no `birth_date` | backfill deterministico y edad derivada, con backup previo | `docs/ux_ui_crud_polish_next_step_2026-05-01.md:178-228`; `models.py`; `sync_shortlist.py` | fechas demo no son datos reales |
| Evidencia de cobertura/visual incompleta | smoke visual costoso y cobertura desigual | pytest-cov en CI, artifact XML, Playwright opt-in | `.github/workflows/ci.yml:28-39`; `tests/test_visual_smoke.py:9-14`; `docs/auditoria_pendientes_2026-05-17.md:96-124` | cobertura actual 79%; comparadores 37%; smoke visual omitido por defecto |
