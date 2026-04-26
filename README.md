# TPScouting - Scouting Inteligente con IA (MVP)

Trabajo final orientado al scouting de futbol juvenil, con una app web para:

- registrar jugadores y sus atributos
- cargar historial de rendimiento y evaluaciones
- comparar jugadores (1 vs 1 y multiple)
- estimar potencial con un modelo MLP (PyTorch)
- visualizar datos en dashboard y fichas

## Stack

- Python + Flask
- SQLAlchemy + SQLite/PostgreSQL
- pandas + scikit-learn para preprocesamiento
- PyTorch + scikit-learn (pipeline y metricas)
- Bootstrap + Chart.js

## Estructura del proyecto

- `scouting_app/`: aplicacion principal (backend, templates, logica y scripts operativos)
- `tests/`: pruebas automatizadas (auth, paginas, permisos)
- `docs/flujo_reproducible_mvp.md`: corrida oficial para regenerar datos, modelo y evaluacion
- `docs/guia_indicadores_app.md`: explicacion de los indicadores visibles de proyeccion
- `docs/contexto_para_nuevo_chat.md`: resumen compacto para continuar el proyecto sin perder contexto
- `docs/model_training_evidence.md`: evidencia tecnica del entrenamiento y comparacion con baseline
- `docs/model_training_plan.md`: plan tecnico vigente del modelo
- `render.yaml`: configuracion de deploy en Render
- `RUNBOOK.md`: guia operativa (healthcheck, backup/restore, admin, incidentes)

## Ramas de trabajo

- `training`: rama estable cerrada con las correcciones del MVP.
- `reformas-finales`: rama activa para nuevas reformas y ajustes posteriores.

## Bases de datos del MVP

El proyecto usa dos bases separadas. En local pueden ser SQLite y en despliegue pueden ser PostgreSQL:

- `scouting_app/players_updated_v2.db`: base operativa (maximo 100 jugadores evaluables)
- `scouting_app/players_training.db`: base de entrenamiento (dataset sintetico para el modelo)

En PostgreSQL, la app acepta URLs `postgresql://...` y `postgres://...`; internamente las normaliza para SQLAlchemy con `psycopg`.

Las bases, el modelo entrenado y los artefactos de preprocesamiento son generados localmente. No son la fuente principal del repo. Para regenerarlos, seguir `docs/flujo_reproducible_mvp.md`.

## Ejecucion local

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scouting_app\app.py
```

Abrir en navegador:

- `http://127.0.0.1:5000/`

## Usuario local

El usuario administrador se crea por variable de entorno o con `create_admin.py`.

Ejemplo local:

```powershell
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "admin123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

Nota: en deploy (Render) la clave de admin se configura por variable de entorno (`ADMIN_PASSWORD`).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests --import-mode=importlib
```

## Healthcheck

- Endpoint: `GET /health`
- Esperado: `200` con JSON `status=ok`, conectividad DB y bloque `data_quality`

## Mantenimiento operativo

- `Configuracion` incluye una auditoria de calidad para la base operativa
- La limpieza operativa elimina registros legacy inconsistentes (por ejemplo, jugadores sin identificador) sin inventar datos faltantes

## Deploy (Render)

El repositorio incluye `render.yaml` con:

- `APP_DB_URL` (base operativa)
- `TRAINING_DB_URL` (base de entrenamiento)
- `EVAL_POOL_MAX=100`
- variables de seguridad y logging

El blueprint actual deja preparado el deploy con dos bases PostgreSQL administradas por Render: una operativa y una de entrenamiento.

## Alcance del MVP

Este proyecto es un MVP academico. La prediccion de potencial se valida con dataset sintetico y se usa como soporte a la toma de decisiones, no como reemplazo del criterio del cuerpo tecnico.
