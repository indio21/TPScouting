# Contexto Para Nuevo Chat

Fecha: 2026-04-26

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
- Rama de trabajo: `training`
- Ultimo commit subido al cierre: `87e9eb3 chore: make MVP artifacts reproducible`
- Estado al cierre: `training` limpio y sincronizado con `origin/training`
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
..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18
```

Crear admin:

```powershell
cd C:\Tesis\TPScouting
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "admin123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

Levantar app:

```powershell
cd C:\Tesis\TPScouting\scouting_app
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:TRAINING_DB_URL = "sqlite:///players_training.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "admin123"
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

- `39 passed`

## Documentos Relevantes

- `PROGRESO_MVP.md`
- `README.md`
- `RUNBOOK.md`
- `docs/flujo_reproducible_mvp.md`
- `docs/session_2026-04-22_synthetic_redesign.md`
- `docs/session_2026-04-22_synthetic_redesign.docx`

## Proximo Paso Probable

Despues de cerrar la estrategia de artefactos, hay dos caminos razonables:

- Seguir con desarrollo/modelo: mejorar generacion sintetica y target temporal, o robustecer evaluacion contra baseline.
- Pasar a documento de tesis: alinear Word con el MVP real y eliminar afirmaciones que no esten respaldadas por el repo.

Antes de tocar codigo en el proximo chat, revisar:

```powershell
cd C:\Tesis\TPScouting
git status -sb
git log --oneline --decorate -5
```

