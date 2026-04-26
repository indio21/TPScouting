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
- `render.yaml`: configuracion de deploy en Render
- `RUNBOOK.md`: guia operativa (healthcheck, backup/restore, admin, incidentes)

## Bases de datos del MVP

El proyecto usa dos bases separadas. En local pueden ser SQLite y en despliegue pueden ser PostgreSQL:

- `scouting_app/players_updated_v2.db`: base operativa (maximo 100 jugadores evaluables)
- `scouting_app/players_training.db`: base de entrenamiento (dataset sintetico para el modelo)

En PostgreSQL, la app acepta URLs `postgresql://...` y `postgres://...`; internamente las normaliza para SQLAlchemy con `psycopg`.

Las bases, el modelo entrenado y los artefactos de preprocesamiento son generados localmente. No son la fuente principal del repo. Para regenerarlos, seguir `docs/flujo_reproducible_mvp.md`.

## Ejecucion local

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scouting_app/app.py
```

Abrir en navegador:

- `http://127.0.0.1:5000/`

## Usuario local

El usuario administrador se crea por variable de entorno o con `create_admin.py`.

Ejemplo local:

```bash
APP_DB_URL="sqlite:///players_updated_v2.db" ADMIN_USERNAME="admin" ADMIN_PASSWORD="admin123" python scouting_app/create_admin.py
```

Nota: en deploy (Render) la clave de admin se configura por variable de entorno (`ADMIN_PASSWORD`).

## Tests

```bash
pytest -q tests --import-mode=importlib
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
