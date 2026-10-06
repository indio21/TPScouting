# Auditoria tecnica de evidencia - Bloque 2

Fecha de inspeccion: 2026-07-10

Repositorio auditado: `C:\Tesis\TPScouting`

Commit local: `7d7680e04dd15b4c4f01d9e0aee8711aeea06f0d` (`main`, un commit por delante de `origin/main`)

Alcance: inspeccion estatica, lectura de artefactos, consultas SQLite de solo lectura, pruebas locales y smoke HTTP sin credenciales. No se modificaron codigo, bases, checkpoints, datasets, Word ni configuracion; no se reentreno el modelo.

## Resumen ejecutivo

- **HECHO VERIFICADO.** El target usado por la corrida oficial no es `Player.potential_label`: es `temporal_target_label`. El entrenamiento lo selecciona en `scouting_app/train_model.py`, funcion `train_model`, aprox. lineas 194-211; la constante se define en `scouting_app/preprocessing.py:106` y el target se construye en `_temporal_target_dataframe`, lineas 1264-1799.
- **HECHO VERIFICADO.** `PlayerNet` recibe 68 entradas: 11 variables base numericas, 52 features historicas y 5 columnas one-hot de posicion. Evidencia: `scouting_app/preprocessing.py`, constantes de columnas y `build_preprocessor`, aprox. lineas 30-148; artefacto `scouting_app/preprocessor.joblib`, `get_feature_names_out()`; `scouting_app/training_metadata.json:27`.
- **HECHO VERIFICADO.** La corrida registrada solicito 45 epocas y ejecuto 15 porque el mejor resultado fue la epoca 5 y luego se acumularon 10 epocas sin mejora, igual a `patience=10`. Evidencia: `scouting_app/training_metadata.json:5-9,39` e historial aprox. lineas 122-258; regla de corte en `scouting_app/train_model.py`, `train_model`, lineas 602-664.
- **HECHO VERIFICADO.** Las metricas rotuladas en el Word como "PlayerNet test" (`accuracy=0.9303`, `PR-AUC=0.5241`) corresponden a la salida **calibrada por isotonic regression**, no al sigmoid crudo de PlayerNet ni al score combinado mostrado por la aplicacion. Evidencia: `scouting_app/train_model.py:670-682`; `scouting_app/training_metadata.json`, secciones `pytorch.test`, `pytorch.raw_test` y `scoring_policy` (aprox. lineas 38-120 y 266-273).
- **HECHO VERIFICADO.** La suite local actual arrojo `83 passed, 1 skipped, 4 warnings`; la cobertura total medida fue `79%` (5082 statements, 1062 missed), no 80%. Comando y detalle en `comandos_ejecutados.txt`.
- **NO VERIFICABLE.** No se puede atribuir la corrida a un commit exacto: `training_metadata.json` no guarda SHA Git. Tampoco registra duracion ni validation loss.

## A. Variable objetivo

### A.1 Dos etiquetas distintas

| Estado | Variable | Evidencia | Uso real |
|---|---|---|---|
| **HECHO VERIFICADO** | `potential_label` | `scouting_app/models.py`, clase `Player`, aprox. lineas 31-63; columna booleana en linea 63. `scouting_app/generate_data.py`, `label_probability` y `generate_player`, lineas 122-193. | Etiqueta sintetica almacenada en la tabla `players`; se genera por Bernoulli desde una probabilidad sintetica. No es el `y` de la corrida oficial actual. |
| **HECHO VERIFICADO** | `temporal_target_label` | `scouting_app/preprocessing.py:106`, `_temporal_target_dataframe` lineas 1264-1799; `scouting_app/train_model.py`, carga del dataframe y asignacion de `y`, aprox. lineas 194-211. | Target binario efectivamente usado para entrenar, validar y testear PlayerNet y baselines. |

### A.2 Como se genera `potential_label`

**HECHO VERIFICADO.** `scouting_app/generate_data.py`, funcion `label_probability`, lineas 122-139, calcula un score latente con:

- score ponderado de atributos segun posicion;
- bonus por coincidencia entre posicion declarada y posicion recomendada;
- componente mental;
- bonus por juventud;
- diferencia respecto del score recomendado;
- ruido gaussiano.

La funcion aplica una sigmoide al score latente. `generate_player`, lineas 142-193, sortea la clase con `random.random() < probability`. Por lo tanto, positiva significa que el sorteo sintetico resulto verdadero; negativa, falso. Las variables originales involucradas son edad, posicion y los diez atributos de `ATTRIBUTE_FIELDS` (`scouting_app/player_logic.py:13-24`): pace, shooting, passing, dribbling, defending, physical, vision, tackling, determination y technique.

### A.3 Como se genera el target temporal real

**HECHO VERIFICADO.** `scouting_app/preprocessing.py`, `_build_player_cutoff_map`, aprox. lineas 951-989, fija para cada jugador un corte anterior a los eventos futuros. Los builders historicos filtran observaciones `<= cutoff`; la construccion del target usa eventos posteriores al corte (`_temporal_target_dataframe`, aprox. lineas 992-1261 y 1264-1799).

**HECHO VERIFICADO.** El score temporal combina, en `_temporal_target_dataframe`, aprox. lineas 1577-1641:

`0.15*growth + 0.12*future_level + 0.13*performance + 0.13*pressure + 0.10*consistency + 0.09*role + 0.09*availability + 0.09*recovery + 0.10*scout + 0.14*breakout - 0.08*stability_penalty`.

Los componentes derivan de atributos futuros ponderados, score final futuro, minutos y titularidad, dificultad de rival, posicion natural, disponibilidad, fatiga, lesiones y dias perdidos, y reportes scout de decision, lectura tactica, perfil mental, adaptabilidad y proyeccion. La seleccion final aplica cuantiles globales, gates de calidad, reglas de consolidacion/breakout, cuota por cohorte posicion:edad cercana al 8% y limites globales de 5%-12% (`scouting_app/preprocessing.py`, `_temporal_target_dataframe`, aprox. lineas 1700-1799). Positiva es la fila seleccionada por `selected_mask`; negativa, toda fila no seleccionada.

### A.4 Distribucion verificada

Fuente: cache temporal ignorada asociada a la base de entrenamiento y split persistido; consistencia contrastada con `scouting_app/training_metadata.json:30-36,380-386` y `scouting_app/training_splits.json`.

| Particion | Total | Positivas | Negativas | Tasa positiva |
|---|---:|---:|---:|---:|
| Total | 20,000 | 1,597 | 18,403 | 7.985% |
| Train | 14,000 | 1,118 | 12,882 | 7.986% |
| Validation | 3,000 | 239 | 2,761 | 7.967% |
| Test | 3,000 | 240 | 2,760 | 8.000% |

**HECHO VERIFICADO.** En la base sintetica de entrenamiento respaldada, `potential_label` tiene 3,967 positivos y 16,033 negativos; no coincide con los 1,597/18,403 del target temporal. Evidencia: consulta SQLite de solo lectura sobre `scouting_app/players_training.db.before_attr_scale_1_20_20260517_225514`; construccion del target en `scouting_app/preprocessing.py:1264-1799`.

### A.5 Leakage y circularidad

- **HECHO VERIFICADO.** No se encontro leakage directo por incluir una columna futura o `potential_label` en el vector de entrada: `MODEL_FEATURE_COLUMNS` excluye ambas etiquetas (`scouting_app/preprocessing.py:30-106`) y los historiales de entrada se filtran hasta el cutoff (`build_temporal_training_dataframe`, lineas 1802-1915).
- **HECHO VERIFICADO - RIESGO METODOLOGICO.** El target completo se construye antes del split (`scouting_app/train_model.py:194-211` frente al split de lineas 525-540) y sus cuantiles/cuotas se calculan sobre los 20,000 jugadores (`scouting_app/preprocessing.py:1700-1799`). Esto contamina la definicion de las etiquetas de validation/test con la distribucion global. No es leakage de una feature futura, pero si dependencia del conjunto de test durante la construccion del target.
- **INFERENCIA.** Al ser datos sinteticos generados por un mismo proceso, features observadas y desenlace futuro comparten reglas latentes. Esto puede facilitar el aprendizaje y limita la validez externa. Evidencia de origen sintetico: `scouting_app/generate_data.py`, modulo y generadores, aprox. lineas 1-1254; limitacion reconocida en `docs/prediction_improvement_progress.md:80-85`.

## B. Features e `input_dim`

| Grupo | Variables originales | Transformacion | Cantidad | Acumulado | Evidencia |
|---|---|---|---:|---:|---|
| Base numericas | edad + 10 atributos | imputacion mediana + `MinMaxScaler` | 11 | 11 | `scouting_app/player_logic.py:13-24`; `scouting_app/preprocessing.py:30-40,114-148` |
| Historicas | estadisticas (4), atributos longitudinales (11), partidos (10), scout (7), fisico (11), disponibilidad (9) | imputacion constante 0 con columnas vacias preservadas + `MinMaxScaler` | 52 | 63 | `scouting_app/preprocessing.py:41-105,114-148` |
| Posicion | Portero, Defensa, Lateral, Mediocampista, Delantero | imputacion por moda + one-hot con categorias fijas y unknown ignorado | 5 | 68 | `scouting_app/player_logic.py:42-48`; `scouting_app/preprocessing.py:114-148` |

**HECHO VERIFICADO.** La suma es `11 + 52 + 5 = 68`. Coincide con `scouting_app/training_metadata.json:27`, con el checkpoint `scouting_app/model.pt` (`input_dim=68`) y con las 68 salidas de `scouting_app/preprocessor.joblib`.

Los nombres exactos emitidos por el preprocesador se listan por grupo en `tablas_para_tesis.md`. Los artefactos asociados son `preprocessor.joblib`, `model.pt`, `probability_calibrator.joblib`, `training_metadata.json` y `training_splits.json` (`scouting_app/train_model.py:816-889`).

## C. Entrenamiento

| Elemento | Valor verificado | Evidencia |
|---|---|---|
| Fecha registrada | `2026-05-19T22:51:17.072099` | `scouting_app/training_metadata.json:2` |
| Seed | 42 | `training_metadata.json:3`; checkpoint `model.pt`, clave `seed` |
| Dataset/split | 20,000; 14,000/3,000/3,000 (70/15/15) | `training_metadata.json:30-36`; split en `train_model.py:525-540` |
| Batch size | 256 | `training_metadata.json:9`; `train_model.py`, `train_model`, aprox. lineas 517-552 |
| Learning rate | 0.0005 | `training_metadata.json:7` |
| Arquitectura | rama lineal 68->1 + rama residual 68->128->64->1, BatchNorm, GELU, dropout 0.15 y escala/bias aprendibles | `scouting_app/train_model.py`, clase `PlayerNet`, lineas 75-100; shapes del `model_state` leido de `model.pt` |
| Loss | `BCEWithLogitsLoss`, `pos_weight=11.522361...` | `training_metadata.json:10-17`; `train_model.py:581-586` |
| Optimizador | AdamW; weight decay 0.0005 | `training_metadata.json:10-12`; `train_model.py:587` |
| Scheduler | ReduceLROnPlateau sobre PR-AUC, factor 0.5, patience 3 | `train_model.py:588,602-664`; historial de LR en metadata |
| Epocas | 45 solicitadas; 15 ejecutadas | `training_metadata.json:5-6` |
| Best epoch | 5 | `training_metadata.json:39`; historial: max PR-AUC calibrado en validacion |
| Early stopping | Si, probado; 10 epocas sin mejora despues de epoca 5 | `training_metadata.json:8,39` e historial; `train_model.py:602-664` |
| Metrica monitorizada | PR-AUC calibrado de validation; F1 como desempate | `train_model.py:624-653` |
| Duracion | **NO VERIFICABLE** | No existe campo en `training_metadata.json` ni log persistido |
| Checkpoint | `scouting_app/model.pt`, formato v1, `PlayerNet`, input 68, 17,994 parametros | `train_model.py:119-126`; lectura de checkpoint |

**HECHO VERIFICADO.** La explicacion de 45/15 no es una suposicion: `best_epoch=5`, las epocas 6-15 no superan el PR-AUC de validacion de epoca 5 y el contador alcanza `patience=10`; entonces se ejecuta el `break` de `train_model.py:662-663`.

**HECHO VERIFICADO.** El historial guarda training loss, PR-AUC/F1 de validation, threshold, calibracion y LR, pero no validation loss (`train_model.py:630-639`). Training loss final: `0.5179418206` en epoca 15; training loss de la mejor epoca: `0.6431135205`.

## D. Resultado de inferencia

### D.1 Secuencia completa

1. **HECHO VERIFICADO.** Se leen jugador e historiales y se consolidan las columnas del modelo en `scouting_app/app.py`, `players_to_model_tensor`, lineas 1019-1047.
2. **HECHO VERIFICADO.** `preprocessor.transform` produce el tensor de 68 columnas; PlayerNet devuelve logits; `torch.sigmoid` produce `raw_base_probs` (`scouting_app/app.py:1050-1086`).
3. **HECHO VERIFICADO.** Si existe calibrador, se calcula una probabilidad isotonic secundaria (`app.py:1086-1089`; `scouting_app/train_model.py:457-511`).
4. **HECHO VERIFICADO.** Se calcula `fit_score` posicional y el promedio historico de `final_score`; luego `combine_probability` genera el score primario mostrado (`app.py:1094-1120,1225-1257`).
5. **HECHO VERIFICADO.** La formula usa `rating=avg_final_score/10`, `fit=fit_score/20` y pesos por defecto modelo 0.35, rating 0.35, fit 0.30. Los componentes ausentes se omiten y los pesos presentes se renormalizan. La salida se limita a `[0, 0.99]` (`app.py:1225-1257`).
6. **HECHO VERIFICADO.** Las categorias operativas son bajo `<0.60`, medio `0.60-<0.80`, alto `>=0.80`, configurables por entorno y validadas para que medio sea menor que alto (`app.py:1314-1358`).
7. **HECHO VERIFICADO.** La vista recibe `combined_prob` como score principal y tambien `base_prob`, `calibrated_base_prob` y delta (`scouting_app/routes/players.py`, flujo de prediccion, aprox. lineas 1439-1501).

### D.2 Que salida evaluan las metricas

| Salida | Se evalua en test | Campo | Evidencia |
|---|---|---|---|
| Sigmoid crudo de PlayerNet | Si | `pytorch.raw_test` | `train_model.py:670-680`; `training_metadata.json:103-120` |
| Probabilidad calibrada isotonic | Si | `pytorch.test` | `train_model.py:674-682`; `training_metadata.json:38-68` |
| Score combinado de la app | No | solo politica de scoring | `training_metadata.json:266-273`; `app.py:1225-1257` |

**CONTRADICCION DOCUMENTAL.** El Word, Tabla 6-3/Tabla 16 del texto extraido, llama "Metricas de PlayerNet en test" a `0.9303/0.9174/0.5241/0.5282`; esos valores son los de `pytorch.test`, es decir, PlayerNet **mas calibracion isotonic**. Debe rotularse asi o reemplazarse por las metricas crudas.

## E. Datasets

| Conjunto | Estado/tamano | Clasificacion | Origen y finalidad | Evidencia |
|---|---|---|---|---|
| Entrenamiento oficial | Base activa ausente; backup local de 20,000 jugadores, 232,960 partidos, 116,448 stats, 180,212 historiales de atributos, 80,062 reportes scout, 180,212 fisicos y 180,212 disponibilidades | Sintetico | Generado para entrenamiento reproducible | `scouting_app/generate_data.py`, modulo y `main`, lineas 1-1254; `RUNBOOK.md:109-117`; consulta SQLite read-only al backup |
| Cache temporal de entrenamiento | 20,000 filas, 107 columnas; 1,597 positivos | Derivado sintetico | Evita reconstruir features/target temporal | `scouting_app/preprocessing.py:1802-1915`; metadata interna del cache leida sin modificarlo |
| Demo | Script default 100 jugadores | Sintetico | Poblar una base operativa vacia para demostracion | `scouting_app/seed_demo_data.py:1-57`; `render.yaml:16`; `docs/cierre_pre_entrega_word_render_2026-05-18.md:128` |
| Operativa actual local | `players_updated_v2.db` no esta presente | **NO VERIFICABLE** | No se puede contar ni clasificar su contenido actual | ausencia comprobada por inventario de archivos |
| `players.db` versionada | 1,000 jugadores legacy, edades 16-22, 143 labels positivos | Origen exacto **NO VERIFICABLE** | Base legacy, no coincide con entrenamiento oficial ni rango juvenil vigente | consulta SQLite read-only; esquema en `models.py` |
| Backups operativos ignorados | 96-100 jugadores segun backup | Sintetico/demo historico segun documentos; no son base activa | Respaldo de migraciones/sincronizacion | nombres y conteos SQLite; `docs/ux_ui_crud_polish_next_step_2026-05-01.md:178-228` |

**HECHO VERIFICADO.** El repositorio no aporta evidencia de datos reales de jugadores para entrenamiento o evaluacion. La afirmacion mas defendible es "datos sinteticos". "Semisinteticos" no esta demostrada para la corrida oficial, aunque el Word usa esa expresion en algunas secciones.

## F. Metricas y artefactos

| Modelo/salida | Accuracy | ROC-AUC | PR-AUC | F1 | Precision | Recall | Matriz de confusion |
|---|---:|---:|---:|---:|---:|---:|---|
| PlayerNet + isotonic | 0.9303 | 0.9174 | 0.5241 | 0.5282 | 0.5764 | 0.4875 | `[[2674,86],[123,117]]` |
| PlayerNet sigmoid crudo | 0.9300 | 0.9203 | 0.5461 | 0.5291 | 0.5728 | 0.4917 | `[[2672,88],[122,118]]` |
| LogisticRegression balanced | 0.9310 | 0.9205 | 0.5378 | 0.5327 | 0.5813 | 0.4917 | `[[2675,85],[122,118]]` |
| Promedio simple de atributos | 0.8960 | 0.8390 | 0.3513 | 0.4201 | 0.3792 | 0.4708 | `[[2575,185],[127,113]]` |

Fuente de todas las filas: `scouting_app/training_metadata.json`, secciones `pytorch.test`, `pytorch.raw_test` y `baselines`, aprox. lineas 38-120 y 275-369; calculo en `scouting_app/train_model.py:324-397,557-575,670-699`.

**CONTRADICCION DOCUMENTAL.** El Word afirma que LogisticRegression supera levemente a PlayerNet en ROC-AUC, PR-AUC y F1. Es correcto si se compara contra la salida calibrada rotulada como PlayerNet, pero no contra el sigmoid crudo: el crudo tiene PR-AUC 0.5461, mayor que 0.5378. Debe explicitarse la variante comparada.

Hashes SHA-256 leidos:

| Artefacto | SHA-256 |
|---|---|
| `scouting_app/model.pt` | `D9FAC23BDFC294C91CB8E7A10AAAA3E4BF90F4E6A3CDA14C022880B2AB12BB19` |
| `scouting_app/preprocessor.joblib` | `9D5286C52A16A7E004C76A39DCA0A970D6746098866FEEB15F22F503B8E61576` |
| `scouting_app/probability_calibrator.joblib` | `D9EE7893D62034200BE5402254757A0B9790020F558A4E7CED1A8AC12299325D` |
| `scouting_app/training_metadata.json` | `043AC307E433DF717B62045FB05BB63E560A2E029D18F809DBF067C7EA5F05B0` |
| `scouting_app/training_splits.json` | `AF6EC6B5E9222AB7847E52F303CF7E0EEFD84003F2601C7AB0E91092A5E26F82` |

**NO VERIFICABLE.** El metadata no incluye commit SHA. `git log` ubica el ultimo commit que introdujo los artefactos ML en `f58fc6bd469c70d0ad60cd7ff114f64c1ec26573`, del 2026-05-19, temporalmente compatible con el timestamp; esa relacion es una **INFERENCIA**, no prueba de que el entrenamiento se haya ejecutado exactamente sobre ese commit.

## G. Pruebas y CI

**HECHO VERIFICADO.** Suite ejecutada en el repositorio de entrega sincronizado, para evitar escribir caches o archivos generados en el repositorio fuente:

`C:\Tesis\TPScouting\.venv\Scripts\python.exe -m pytest -q -rs -p no:cacheprovider`

Resultado: `83 passed, 1 skipped, 4 warnings in 48.68s`.

- Skip: `tests/test_visual_smoke.py:73`; el modulo es opt-in y requiere `RUN_PLAYWRIGHT=1` (`tests/test_visual_smoke.py:9-14`).
- Warnings: cuatro `RuntimeWarning: All-NaN slice encountered`, dos por `nanmin` y dos por `nanmax` de scikit-learn, disparados por `test_prepare_input_matches_batch_transformation` y `test_prepare_input_includes_historical_features`. Son fixtures con columnas historicas completamente NaN, no fallos de test.
- Cobertura: 79% total, 5082 statements, 1062 missed. Menores coberturas: `create_admin.py` 0%, `seed_demo_data.py` 0%, `routes/compare.py` 37%, `routes/staff.py` 52%, `ml/runtime.py` 57%, `routes/settings.py` 61%.

**CONTRADICCION DOCUMENTAL.** `README.md`, `RUNBOOK.md` y documentos de cierre registran 80%; la ejecucion actual redondeada por pytest-cov informa 79%. La tesis debe indicar fecha/commit si conserva 80%, o actualizar a 79% con la corrida presente.

**HECHO VERIFICADO.** `.github/workflows/ci.yml:1-39` define workflow `CI`, en push y pull request, matriz Python 3.11/3.12, instala `requirements.txt` y `requirements-dev.txt`, ejecuta `pytest -q --cov=scouting_app --cov-report=term-missing --cov-report=xml` y sube `coverage.xml` como `coverage-python-${python-version}`.

**NO VERIFICABLE.** No hay evidencia local de que una ejecucion remota actual de GitHub Actions haya pasado. Solo se verifica la configuracion del workflow y la suite local.

## H. Reproducibilidad minima

**HECHO VERIFICADO.** Python local: 3.11.9; CI: 3.11 y 3.12 (`.github/workflows/ci.yml:10-12`). Dependencias: `requirements.txt` y `requirements-dev.txt`.

Desde la raiz del repo, PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:APP_SECRET_KEY = "<valor-seguro>"
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
.\.venv\Scripts\python.exe .\scouting_app\app.py
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Fuentes: `README.md:71-110`, `RUNBOOK.md:11-28,53-58,138-141`. No se reproduce ningun secreto real.

Pipeline oficial documentado, ejecutado desde `scouting_app`:

```powershell
..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///players_training.db --model-out model.pt --preprocessor-out preprocessor.joblib --calibrator-out probability_calibrator.joblib --metadata-out training_metadata.json --splits-out training_splits.json --epochs 45 --lr 5e-4 --patience 10
..\.venv\Scripts\python.exe evaluate_saved_model.py --db-url sqlite:///players_training.db --metadata-path training_metadata.json
..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18 --replace
```

Fuente exacta: `RUNBOOK.md:109-117`. Estos comandos se documentan; **no se ejecutaron** durante la auditoria.

Smoke externo real previsto por el repo:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_render.py --base-url https://tpscouting-mvp.onrender.com
```

`scripts/smoke_render.py:20-93` verifica `/health`, `/login` y, con credenciales opcionales, login/dashboard. La ejecucion publica del 2026-07-10 termino por timeout; no prueba indisponibilidad definitiva, pero impide afirmar disponibilidad vigente. La evidencia historica del 20/05/2026 esta en `docs/cierre_word_corregido_2026-05-19.md:107-123`.

## I. Problemas tecnicos y soluciones verificadas

La tabla completa trasladable esta en `tablas_para_tesis.md`. Fuentes principales: `docs/comparacion_falencias_codigo_fuente_2026-04-27.md:31-52`, `docs/refactor_arquitectura_2026-04-28.md:13-29,95-134,245-267`, `docs/auditoria_pendientes_2026-05-17.md:25-147`, `docs/cierre_pre_entrega_word_render_2026-05-18.md:116-215` y los archivos de codigo citados en cada fila.

## Contradicciones y riesgos academicos prioritarios

1. **Target ambiguo:** no describir `potential_label` como el target de la corrida oficial; el target real es `temporal_target_label`.
2. **Leakage metodologico:** reconocer que el target se construye globalmente antes del split. Esto afecta la pureza de validation/test aunque no haya una feature futura directa.
3. **Metrica mal rotulada:** `pytorch.test` es PlayerNet calibrado; el score combinado visible no tiene metricas de test propias.
4. **Comparacion de baseline:** LogisticRegression no supera al sigmoid crudo en PR-AUC; si se sostiene la frase actual, debe decir que se compara contra la variante calibrada.
5. **Datos:** no afirmar uso de datos reales ni semisinteticos para la corrida oficial; la evidencia disponible es sintetica.
6. **Cobertura:** la evidencia local actual es 79%, aunque documentos previos registren 80%.
7. **Trazabilidad:** no atribuir el entrenamiento al commit actual ni informar duracion/validation loss sin nueva evidencia.
8. **Deploy:** el smoke historico es verificable como registro versionado; la URL no respondio dentro del timeout el 2026-07-10, por lo que no debe presentarse como validacion vigente sin fecha.
