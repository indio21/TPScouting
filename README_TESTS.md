# Scouting IA - Tests

## Instalacion

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Ejecutar

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Con cobertura:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing --cov-report=xml
```

Smoke visual con navegador real:

```powershell
$env:RUN_PLAYWRIGHT = "1"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m pytest -q tests\test_visual_smoke.py
Remove-Item Env:\RUN_PLAYWRIGHT
```

Smoke del deploy real en Render:

```powershell
$env:RENDER_SMOKE_BASE_URL = "https://TU_SERVICIO.onrender.com"
$env:SMOKE_USERNAME = "admin"
$env:SMOKE_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe scripts\smoke_render.py
```

## Estado Actual

- Ultima validacion documentada: `83 passed, 1 skipped`, cobertura total `80%` al cierre de este bloque.
- Los tests usan bases temporales y no tocan las `.db` reales del MVP.
- `/settings` queda restringido a rol `administrador`.
- CI ejecuta la suite con `pytest-cov` y publica `coverage.xml` como artefacto por version de Python.
- `tests/test_visual_smoke.py` es opt-in con `RUN_PLAYWRIGHT=1`: abre Chromium real y valida login, dashboard y responsive basico.
- Cobertura funcional actual: login/logout, rutas protegidas, CSRF, CRUD de jugadores, inputs invalidos, atributos fuera de rango, dashboard, comparadores, potencial, pipeline ML, checkpoints y comportamiento sin modelo cargado.

## Nota de alcance

La regla actual del codigo para atributos tecnicos, fisicos de escala y reportes scout es `1-20`. La validacion de formularios, importacion CSV y migracion de bases heredadas deben respetar ese rango antes de entregar.
