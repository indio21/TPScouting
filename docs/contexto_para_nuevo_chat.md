# Contexto Para Nuevo Chat

Fecha: 2026-04-30

Este archivo sirve como contexto semilla para continuar el proyecto `TPScouting` en un chat nuevo sin arrastrar toda la conversacion anterior.

## Instrucciones Clave Del Usuario

- No inventar y no vender humo.
- Si falta informacion o hay riesgo, preguntar antes.
- Trabajar por etapas chicas y verificables.
- No avanzar a otra etapa sin permiso cuando el usuario lo pida.
- Prioridad metodologica: primero corregir MVP/codigo, despues documento Word de tesis.
- Alcance fijo del MVP: scouting juvenil `12-18` para clubes formativos con bajo presupuesto.
- App multirol, con niveles de seguridad y visibilidad.
- Ser honesto con metricas. No modificar datos solo para maquillar resultados.

## Repo

- Repo remoto: `https://github.com/indio21/TPScouting.git`
- Carpeta local: `C:\Tesis\TPScouting`
- Rama estable cerrada del MVP corregido: `training`
- Rama cerrada para reformas livianas: `reformas-finales`
- Rama activa para UX/UI: `ux-crud-polish`
- Rama base de cambios complejos cerrados: `reformas-complejas`
- Ultimo commit comun al crear `reformas-finales`: `b6c21ea docs: explain app prediction indicators`
- Ultimos bloques tecnicos publicados en `reformas-finales`: cierre de revision de codigo, documentacion, CSRF/inputs invalidos, constantes/cache/conftest, normalizacion puntual `db_session` y type hints quirurgicos.
- Estado al cierre tecnico: `reformas-finales` limpia y sincronizada con `origin/reformas-finales` en `7763bb0`.
- `reformas-complejas` fue creada desde `7763bb0` para trabajos mas grandes.
- Ultimos bloques tecnicos publicados en `reformas-complejas`:
- `5ca89e1 refactor: extract app services and ml runtime`
- `4f93808 refactor: move auth routes to blueprint`
- `a3f1cea refactor: extract staff and player blueprints`
- `60f94b3 refactor: extract compare and settings blueprints`
- `9f02b3c refactor: extract dashboard blueprint`
- Estado de arquitectura verificado: `reformas-complejas` limpia y sincronizada con `origin/reformas-complejas` en `a6d5a36`.
- Rama `ux-crud-polish` creada desde `a6d5a36` para pulido UX/UI y CRUDs.
- Entorno Python local: `C:\Tesis\TPScouting\.venv`

Usar siempre la `.venv`; no usar el Python global.

## Activacion Local

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea scripts:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Decision De Artefactos

El repo versiona codigo, tests y documentacion.

Estos archivos son artefactos generados localmente y no deben versionarse:

- `scouting_app/players_training.db`
- `scouting_app/players_updated_v2.db`
- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`
- `scouting_app/training_metadata.json`
- `scouting_app/training_splits.json`
- `scouting_app/experiments.csv`
- `scouting_app/temporal_training_dataframe.joblib`

No fueron borrados de la maquina local; solo dejaron de estar trackeados por Git.

Archivo principal del flujo reproducible:

- `docs/flujo_reproducible_mvp.md`

## Corrida Oficial Del MVP

Generar datos sinteticos:

```powershell
cd C:\Tesis\TPScouting\scouting_app
..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
```

Entrenar:

```powershell
cd C:\Tesis\TPScouting\scouting_app
..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///players_training.db --model-out model.pt --preprocessor-out preprocessor.joblib --calibrator-out probability_calibrator.joblib --metadata-out training_metadata.json --splits-out training_splits.json --epochs 45 --lr 5e-4 --patience 10
```

Evaluar:

```powershell
cd C:\Tesis\TPScouting\scouting_app
..\.venv\Scripts\python.exe evaluate_saved_model.py --db-url sqlite:///players_training.db --metadata-path training_metadata.json
```

Sincronizar base operativa demo:

```powershell
cd C:\Tesis\TPScouting\scouting_app
..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18 --replace
```

`--replace` reconstruye los jugadores de demo y copia datos ricos sin borrar usuarios.

Crear admin:

```powershell
cd C:\Tesis\TPScouting
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

Levantar app:

```powershell
cd C:\Tesis\TPScouting\scouting_app
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:TRAINING_DB_URL = "sqlite:///players_training.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "AdminDemo123"
..\.venv\Scripts\python.exe app.py
```

URL local:

- `http://127.0.0.1:5000/`

## Metricas Oficiales Ultimas

Dataset:

- `20000` jugadores
- edades `12-17`
- positivos `1597`
- tasa positiva `0.0799`

Evaluador rapido:

- `load_data_seconds` aprox. `0.52s`

PyTorch crudo test:

- `ROC-AUC=0.9102`
- `PR-AUC=0.4826`
- `F1=0.5088`
- `precision=0.4417`
- `recall=0.6000`

PyTorch calibrado test:

- `ROC-AUC=0.9084`
- `PR-AUC=0.4617`
- `F1=0.5162`
- `precision=0.4377`
- `recall=0.6292`

Baseline `LogisticRegression(class_weight="balanced")` test:

- `ROC-AUC=0.9086`
- `PR-AUC=0.4728`
- `F1=0.4875`
- `precision=0.4520`
- `recall=0.5292`

Decision: usar PyTorch crudo como score principal del MVP porque prioriza mejor candidatos por `PR-AUC`. La calibrada queda como referencia secundaria documentada.

## Cambios Tecnicos Importantes Ya Hechos

- Soporte SQLite/PostgreSQL mediante `db_utils.py`.
- `.venv` y dependencias instaladas.
- Multirol real: administrador, scout/ojeador, director.
- Seguridad minima: secret key, passwords fuertes, rate limiting basico.
- Pipeline real con `pandas` + `scikit-learn`:
- `pd.read_sql`
- `ColumnTransformer`
- `SimpleImputer`
- `MinMaxScaler`
- `OneHotEncoder`
- `joblib`
- Preprocesador compartido entrenamiento/inferencia.
- PyTorch con `BCEWithLogitsLoss`, logits, `pos_weight`, split `train/validation/test`, threshold por validacion, early stopping.
- Target temporal de progresion juvenil.
- Generacion sintetica longitudinal con trayectorias juveniles.
- Uso de `PlayerAttributeHistory`, `Match`, `PlayerMatchParticipation`, `ScoutReport`, `PhysicalAssessment`, `PlayerAvailability`.
- Cache de dataframe temporal.
- Persistencia de splits.
- Script `evaluate_saved_model.py`.
- App usa salida cruda `raw_pytorch_sigmoid` como score principal.
- Calibracion queda como referencia secundaria.
- App no reentrena automaticamente al arrancar salvo `AUTO_TRAIN_ON_STARTUP=true`.
- `_PIPELINE_LOCK` incluye comentario explicito sobre dependencia de Gunicorn `--workers 1`.
- Se reemplazo `globals().get("sync_attribute_history_baseline")` por llamada directa.
- CI ejecuta cobertura con `pytest-cov`.
- `requirements-dev.txt` incluye `pytest-cov`.
- `requirements-lock.txt` guarda el snapshot exacto de dependencias instalado en `.venv`.
- `RUNBOOK.md` documenta DiceBear, cache in-memory con limite configurable y `app.py` monolitico como limitaciones reales del MVP.
- `db_utils.ensure_player_columns()` migra timestamps tambien en `physical_assessments` y `player_availability`.
- `train_model.py` guarda checkpoints con `input_dim`, version y `model_state`.
- `app.py` y `evaluate_saved_model.py` cargan checkpoints nuevos y mantienen compatibilidad con `state_dict` legacy.
- `classification_metrics()` devuelve `warnings` cuando ROC-AUC, PR-AUC o F1/precision/recall no pueden calcularse.
- `docs/explicacion_cambios_revision_codigo_2026-04-27.md` explica estos cambios en lenguaje simple.
- Se agrego test matriz de CSRF para POST mutantes criticos.
- Se agregaron tests de alta de jugador con edad invalida y campos obligatorios vacios.
- `base.html` ahora muestra todos los mensajes flash para no ocultar errores multiples de validacion.
- Valores visibles de paginacion, comparadores y rating de estadisticas pasaron a constantes nombradas.
- El cache in-memory del dashboard tiene limite `CACHE_MAX_ENTRIES` configurable, por default `128`.
- `tests/conftest.py` usa nombre estable de modulo de test (`scouting_app_app_test`) y limpia `sys.modules`.
- Helpers operativos y script `create_admin.py` usan `db_session`; la sesion temporal de entrenamiento se llama `training_session`.
- Se agregaron type hints puntuales en helpers compartidos de `app.py`, `db_utils.py`, `sync_shortlist.py` y `evaluate_saved_model.py`.
- En `reformas-complejas` se aplico refactor de arquitectura fase 1: `services/cache.py`, `services/security.py`, `services/operational_data.py` y `ml/runtime.py`.
- Arquitectura fase 2 iniciada: `login`, `logout` y `register` se movieron a `scouting_app/routes/auth.py` como blueprint `auth`.
- Para reducir riesgo se conserva compatibilidad con endpoints historicos mediante aliases `login`, `logout` y `register`; asi siguen funcionando `url_for(...)`, redirects, templates y tests existentes.
- Arquitectura fase 2 continuo con `staff`: `coaches` y `directors` se movieron a `scouting_app/routes/staff.py`.
- Arquitectura fase 2 continuo con `players`: listado, ficha, historial de rendimiento, historial de atributos, edicion/baja, proyeccion y carga manual/masiva se movieron a `scouting_app/routes/players.py`.
- Para reducir riesgo se conservan aliases historicos de staff y players (`index`, `manage_players`, `player_detail`, `player_stats`, `player_attributes`, `predict_player`, `edit_player`, `delete_player`, `list_coaches`, `new_coach`, `edit_coach`, `delete_coach`, `list_directors`, `new_director`, `edit_director`, `delete_director`).
- Arquitectura fase 2 continuo con `compare`: `scouting_app/routes/compare.py` contiene comparador simple y comparador multiple.
- Arquitectura fase 2 continuo con `settings`: `scouting_app/routes/settings.py` contiene configuracion, pipeline y limpieza operativa.
- Para reducir riesgo se conservan aliases historicos tambien en compare/settings (`compare_players`, `compare_multi`, `settings`).
- Arquitectura fase 2 cerro con `dashboard`: `scouting_app/routes/dashboard.py` contiene el dashboard principal con cache, filtros de periodo, agregados por posicion, potencial y evolucion.
- Para reducir riesgo se conserva tambien el alias historico `dashboard`.
- Las rutas Flask que siguen en `app.py` son landing, health, error handlers y helpers compartidos. Ya no quedan familias de blueprints pendientes.
- Validacion posterior al refactor de arquitectura fase 2 dashboard: `52 passed` con cobertura total `77%`, con `4 warnings` conocidos de scikit-learn.
- UX/UI primera etapa iniciada en `ux-crud-polish`: menu agrupado con links activos, `Panel general`, dropdowns `Jugadores`, `Comparadores`, `Staff` y `Administracion`.
- Se separo visualmente `Jugadores > Listado` y `Jugadores > Gestion`.
- Se agruparon comparadores en un solo menu con `Comparar 2 jugadores` y `Ranking multiple`.
- Se agruparon `Cuerpo tecnico` y `Dirigentes` bajo `Staff`.
- Los mensajes flash ahora tienen icono, boton de cierre y estilos diferenciados.
- Listados CRUD de jugadores, cuerpo tecnico y dirigentes usan acciones compactas y consistentes.
- Formularios activos de jugadores y staff separan datos personales, deportivos/institucionales y atributos.
- `Dashboard` se renombro visualmente a `Panel general`.
- Validacion posterior a UX/UI etapa 1: `52 passed`, cobertura total `77%`, con `4 warnings` conocidos de scikit-learn.
- Durante el smoke del servidor local se corrigio un bug de orden de inicializacion: `init_admin_user()` se ejecutaba antes de que existiera `is_strong_password()` cuando `ADMIN_PASSWORD` estaba configurado.
- Smoke local posterior: `GET /health`, `GET /` y `GET /login` respondieron `200` en `http://127.0.0.1:5000/`.

## Performance

Mediciones documentadas:

- Antes: `load_data` reconstruyendo todo aprox. `246.6s`.
- Despues sin cache: aprox. `138.3s`.
- Primer build con cache invalida/inexistente: aprox. `137.2s`.
- Cache ya construida: `0.48s - 0.59s`.

## Validacion

Ultima validacion completa:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest -q
```

Resultado:

- `52 passed`
- `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN

Ultima validacion completa con cobertura:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Resultado:

- `52 passed`
- cobertura total reportada: `77%`
- `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN

Validacion UX/UI etapa 1:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest tests\test_pages.py -q
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Resultado:

- focal: `7 passed`
- completa: `52 passed`
- cobertura total reportada: `77%`
- `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN

Validacion focal de arquitectura fase 2 auth:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest tests\test_auth.py tests\test_mvp_regressions.py::test_register_rejects_weak_password tests\test_mvp_regressions.py::test_register_creates_user_with_valid_role tests\test_mvp_regressions.py::test_register_requires_csrf_token tests\test_mvp_regressions.py::test_mutating_post_routes_reject_missing_csrf tests\test_pages.py -q
```

Resultado:

- `14 passed`

Validacion focal de arquitectura fase 2 staff/players:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest tests\test_pages.py tests\test_auth.py tests\test_mvp_regressions.py -k "manage_players or edit_player or delete_player or player_stats or player_attributes or director_cannot or scout_can_access or mutating_post_routes or admin_can_create" -q
```

Resultado:

- `13 passed`

Validacion focal de arquitectura fase 2 compare/settings:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest tests\test_pages.py tests\test_mvp_regressions.py::test_mutating_post_routes_reject_missing_csrf -q
```

Resultado:

- `8 passed`

Validacion focal de arquitectura fase 2 dashboard:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest tests\test_pages.py -q
```

Resultado:

- `7 passed`

## Comparacion Con Informe Del Profesor 2026-04-27

Documento analizado:

- `C:\Users\Usuario\Desktop\Evaluacion y punto a corregir del Codigo Fuente.docx`

Archivo generado en el repo:

- `docs/comparacion_falencias_codigo_fuente_2026-04-27.md`
- `docs/comparacion_falencias_codigo_fuente_2026-04-28.md`

Resultado general del analisis:

- El informe del profesor contenia `31` observaciones.
- Corregidos o practicamente corregidos para el MVP: `24`.
- Parciales, aceptados o documentados como limitacion tecnica: `7`, con `CALIDAD-03` y `CALIDAD-04` mejorados parcialmente.
- Pendientes criticos: `0`.
- Los tres criticos cambiaron mucho respecto del informe original:
- `CRITICO-01`: Render ya apunta a PostgreSQL administrado.
- `CRITICO-02`: el umbral de potencial esta centralizado y testeado.
- `CRITICO-03`: esta mitigado por `--workers 1 --threads 2` y comentario explicito junto a `_PIPELINE_LOCK`; lock DB/file robusto queda fuera del alcance actual.

Bloques tecnicos ya cerrados en `reformas-finales`:

- Agregar comentario explicito junto a `_PIPELINE_LOCK` sobre dependencia de single-worker.
- Reemplazar `globals().get(...)` por llamada directa a `sync_attribute_history_baseline`.
- Agregar `pytest-cov` y reporte de cobertura en CI.
- Documentar DiceBear, cache in-memory y `app.py` monolitico como limitaciones reales del MVP.
- Guardar `input_dim` y metadata minima en checkpoint del modelo.
- Crear `requirements-lock.txt`.
- Cubrir CSRF en POST mutantes e inputs invalidos de alta de jugador.
- Convertir magic numbers visibles de paginacion/comparadores/rating en constantes nombradas.
- Agregar limite simple al cache in-memory con `CACHE_MAX_ENTRIES`.
- Simplificar `tests/conftest.py` para evitar nombres de modulo con UUID.
- Normalizar puntualmente nombres de sesiones SQLAlchemy en helpers y scripts.
- Agregar type hints quirurgicos en funciones compartidas de mayor uso.

Bloques tecnicos ya iniciados en `reformas-complejas`:

- Refactor de arquitectura fase 1: servicios y runtime ML separados.
- Refactor de arquitectura fase 2 auth: `routes/auth.py` contiene login, logout y register con aliases legacy.
- Refactor de arquitectura fase 2 staff: `routes/staff.py` contiene coaches y directors con aliases legacy.
- Refactor de arquitectura fase 2 players: `routes/players.py` contiene listado, ficha, historial, atributos, edicion/baja, proyeccion y carga de jugadores con aliases legacy.
- Refactor de arquitectura fase 2 compare: `routes/compare.py` contiene comparador simple y multiple con aliases legacy.
- Refactor de arquitectura fase 2 settings: `routes/settings.py` contiene configuracion, pipeline y limpieza operativa con alias legacy.
- Refactor de arquitectura fase 2 dashboard: `routes/dashboard.py` contiene el dashboard con alias legacy `dashboard`.
- UX/UI etapa 1: menu agrupado, links activos, panel general, listados CRUD compactos, flash mejorado y formularios principales ordenados por secciones.

Prioridad recomendada para el proximo bloque de codigo:

- Arquitectura fase 2 de blueprints por familia ya quedo cerrada.
- Evaluar herramientas dev opcionales (`ruff`, `black`, `mypy`) solo si no abre un bloque grande.
- Considerar pasar al documento Word de tesis para alinear la narrativa con el MVP real.

## Documentos Relevantes

- `PROGRESO_MVP.md`
- `README.md`
- `RUNBOOK.md`
- `REVISION_FINAL_MVP.md`
- `docs/comparacion_falencias_codigo_fuente_2026-04-27.md`
- `docs/comparacion_falencias_codigo_fuente_2026-04-28.md`
- `docs/explicacion_cambios_revision_codigo_2026-04-27.md`
- `docs/flujo_reproducible_mvp.md`
- `docs/guia_indicadores_app.md`
- `docs/refactor_arquitectura_2026-04-28.md`
- `docs/model_training_evidence.md`
- `docs/model_training_evidence.docx`
- `docs/model_training_plan.md`
- `docs/model_training_plan.docx`
- `docs/prediction_improvement_progress.md`
- `docs/prediction_improvement_progress.docx`
- `docs/session_2026-04-22_synthetic_redesign.md`
- `docs/session_2026-04-22_synthetic_redesign.docx`

## Proximo Paso Probable

La rama `training` queda como base estable de las correcciones del MVP. La rama `reformas-finales` queda cerrada como bloque liviano. La arquitectura grande quedo cerrada en `reformas-complejas`; el pulido visual continua en `ux-crud-polish`.

Hay cuatro caminos razonables:

- Continuar pulido UX/UI y CRUDs: revisar detalle de jugador, historial, atributos, comparadores y settings para llevar la misma consistencia visual.
- Rendimiento dashboard a escala: dejar de depender de `.all()` y pasar a agregaciones SQL o resumen cacheado/persistido. Recomendado si la prioridad vuelve a ser tecnica.
- Tooling dev opcional: `ruff`, `black`, `mypy` o una combinacion minima. Solo si se acepta ruido de formato/config.
- Documento Word de tesis: alinear narrativa, capturas y afirmaciones con el MVP real ya corregido. Recomendado si la prioridad pasa a entrega academica.

Recomendacion de arranque para el proximo chat:

- Si la prioridad es mostrar mejor el producto: continuar desde `ux-crud-polish` con detalle de jugador, comparadores y settings.
- Si la prioridad es cerrar evidencia academica: pasar al Word y corregirlo contra el repo real.

Antes de tocar codigo en el proximo chat, revisar:

```powershell
cd C:\Tesis\TPScouting
git status -sb
git log --oneline --decorate -5
```
