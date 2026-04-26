# Flujo Reproducible Del MVP

Fecha de referencia: 2026-04-26

Este documento define como regenerar localmente los artefactos del MVP sin versionar bases de datos ni modelos entrenados en Git.

## Decision De Versionado

El repositorio debe versionar codigo, tests y documentacion.

Los siguientes archivos son artefactos generados y no deben ser la fuente principal del repo:

- `scouting_app/players_training.db`
- `scouting_app/players_updated_v2.db`
- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`
- `scouting_app/training_metadata.json`
- `scouting_app/training_splits.json`
- `scouting_app/experiments.csv`
- `scouting_app/temporal_training_dataframe.joblib`

La base de entrenamiento, el modelo, el preprocesador, el calibrador, la metadata y los splits se regeneran con comandos reproducibles.

## Preparar Entorno

Desde PowerShell:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Corrida Oficial Del MVP

Generar la base sintetica de entrenamiento:

```powershell
cd C:\Tesis\TPScouting\scouting_app
..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
```

Entrenar el modelo oficial, guardar preprocesador, calibrador, metadata y splits:

```powershell
cd C:\Tesis\TPScouting\scouting_app
..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///players_training.db --model-out model.pt --preprocessor-out preprocessor.joblib --calibrator-out probability_calibrator.joblib --metadata-out training_metadata.json --splits-out training_splits.json --epochs 45 --lr 5e-4 --patience 10
```

Evaluar los artefactos generados usando cache y splits persistidos:

```powershell
cd C:\Tesis\TPScouting\scouting_app
..\.venv\Scripts\python.exe evaluate_saved_model.py --db-url sqlite:///players_training.db --metadata-path training_metadata.json
```

Resultado esperado de la ultima corrida documentada:

- Dataset: `20000` jugadores, edades `12-17`, tasa positiva aproximada `0.0799`.
- PyTorch crudo test: `PR-AUC=0.4826`, `F1=0.5088`.
- PyTorch calibrado test: `PR-AUC=0.4617`, `F1=0.5162`.
- Baseline logistic test: `PR-AUC=0.4728`, `F1=0.4875`.

## Base Operativa Para Demo Local

Sincronizar una shortlist de hasta `100` jugadores juveniles desde la base de entrenamiento:

```powershell
cd C:\Tesis\TPScouting\scouting_app
..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18
```

Crear o asegurar usuario administrador:

```powershell
cd C:\Tesis\TPScouting
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "admin123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

Levantar la app:

```powershell
cd C:\Tesis\TPScouting\scouting_app
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:TRAINING_DB_URL = "sqlite:///players_training.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "admin123"
..\.venv\Scripts\python.exe app.py
```

Abrir:

- `http://127.0.0.1:5000/`

## Politica De Score Del MVP

El MVP usa la salida cruda de PyTorch como score principal de ranking y priorizacion.

La probabilidad calibrada queda como referencia secundaria documentada porque en la corrida oficial actual mejora levemente `F1` y `recall`, pero reduce `PR-AUC`.

## Validacion Rapida

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest -q
```

