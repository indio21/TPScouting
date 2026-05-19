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
- `docs/comparacion_falencias_codigo_fuente_2026-04-27.md`: estado punto por punto frente al informe de codigo fuente
- `docs/explicacion_cambios_revision_codigo_2026-04-27.md`: explicacion simple de los ultimos cambios de hardening
- `docs/model_training_evidence.md`: evidencia tecnica del entrenamiento y comparacion con baseline
- `docs/model_training_plan.md`: plan tecnico vigente del modelo
- `docs/auditoria_pendientes_2026-05-17.md`: riesgos vivos y cierre por fases de auditoria
- `docs/cierre_pre_entrega_word_render_2026-05-18.md`: cierre previo a entrega con Word final y deploy Render
- `scripts/smoke_render.py`: smoke HTTP contra la URL real de Render
- `render.yaml`: configuracion de deploy en Render
- `RUNBOOK.md`: guia operativa (healthcheck, backup/restore, admin, incidentes)

## Ramas de trabajo

- `training`: rama estable cerrada con las correcciones del MVP.
- `ux-crud-polish`: rama actual de pulido UX/UI y cierre de auditoria, sincronizada con GitHub.
- `auditoria-correcciones-mvp`: rama de correcciones de auditoria ya mergeada en `ux-crud-polish`.

## Estado actual

- Fecha de referencia: 2026-05-18.
- Escala de atributos tecnicos, fisicos en escala y reportes scout: `1-20`.
- Potencial bajo: menor a `60%`; medio: `60%` a `79%`; alto: `80%` o mas.
- La edad y categoria juvenil se derivan de `birth_date`; `Player.age` queda como compatibilidad operativa.
- Tests al ultimo cierre: `83 passed, 1 skipped`, cobertura total `80%`.
- Cierre pre-entrega: faltan alinear el Word final de tesis con el MVP real y ejecutar smoke real en Render con URL publica.

## Bases de datos del MVP

El proyecto usa dos bases separadas. En local pueden ser SQLite y en despliegue pueden ser PostgreSQL:

- `scouting_app/players_updated_v2.db`: base operativa (maximo 100 jugadores evaluables)
- `scouting_app/players_training.db`: base de entrenamiento (dataset sintetico para el modelo)

En PostgreSQL, la app acepta URLs `postgresql://...` y `postgres://...`; internamente las normaliza para SQLAlchemy con `psycopg`.

Las bases son generadas localmente y no son la fuente principal del repo. Para regenerarlas, seguir `docs/flujo_reproducible_mvp.md`.

Para que Render pueda ejecutar inferencia sin entrenar en produccion, el repo incluye
solo los artefactos chicos de runtime:

- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`

Las bases SQLite, metadata de entrenamiento y splits siguen fuera de Git.

## Ejecucion local

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scouting_app\app.py
```

Para reproducibilidad exacta de versiones existe `requirements-lock.txt`.

Abrir en navegador:

- `http://127.0.0.1:5000/`

## Usuario local

El usuario administrador se crea por variable de entorno o con `create_admin.py`.

Ejemplo local:

```powershell
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

Nota: en deploy (Render) la clave de admin se configura por variable de entorno (`ADMIN_PASSWORD`).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Con reporte XML de cobertura:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing --cov-report=xml
```

Smoke visual opcional con Playwright:

```powershell
$env:RUN_PLAYWRIGHT = "1"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m pytest -q tests\test_visual_smoke.py
Remove-Item Env:\RUN_PLAYWRIGHT
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

El blueprint actual deja preparado el deploy en Render con PostgreSQL administrado.
En modalidad gratuita se usa una sola base PostgreSQL Free (`scouting-mvp-db`),
porque Render limita las bases Free activas por workspace. En ese modo,
`APP_DB_URL` y `TRAINING_DB_URL` apuntan a la misma base, `AUTO_TRAIN_ON_STARTUP`
queda desactivado y el deploy ejecuta `seed_demo_data.py` para cargar 100 jugadores
demo solo si la base esta vacia. No ejecutar el pipeline de entrenamiento desde la
web en este modo gratuito.

Smoke real de Render:

```powershell
$env:RENDER_SMOKE_BASE_URL = "https://TU_SERVICIO.onrender.com"
$env:SMOKE_USERNAME = "admin"
$env:SMOKE_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe scripts\smoke_render.py
```

Para cerrar la entrega academica, el deploy real debe verificarse contra la URL publicada.
El Word final debe afirmar el alcance real: MVP academico con dataset sintetico, cache y
rate limiting in-memory, migraciones manuales y validacion externa pendiente.

Seguridad MVP:

- `APP_SECRET_KEY` es obligatoria en producción/Render.
- Los formularios POST mutantes usan CSRF.
- El logout se ejecuta por POST con CSRF.
- El login tiene rate limiting en memoria por IP + usuario. Es suficiente para MVP académico, pero no es un limitador distribuido para producción multi-instancia.

## Alcance del MVP

Este proyecto es un MVP academico. La prediccion de potencial se valida con dataset sintetico y se usa como soporte a la toma de decisiones, no como reemplazo del criterio del cuerpo tecnico.
