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

- `scouting_app/`: aplicacion principal (backend, templates, logica, bases demo)
- `tests/`: pruebas automatizadas (auth, paginas, permisos)
- `render.yaml`: configuracion de deploy en Render
- `RUNBOOK.md`: guia operativa (healthcheck, backup/restore, admin, incidentes)

## Bases de datos del MVP

El proyecto usa dos bases separadas. En local pueden ser SQLite y en despliegue pueden ser PostgreSQL:

- `scouting_app/players_updated_v2.db`: base operativa (maximo 100 jugadores evaluables)
- `scouting_app/players_training.db`: base de entrenamiento (dataset sintetico para el modelo)

En PostgreSQL, la app acepta URLs `postgresql://...` y `postgres://...`; internamente las normaliza para SQLAlchemy con `psycopg`.

## Ejecucion local

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scouting_app/app.py
```

Abrir en navegador:

- `http://127.0.0.1:5000/`

## Usuario de prueba (base demo incluida)

- Usuario: `admin`
- Password: `admin`

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
