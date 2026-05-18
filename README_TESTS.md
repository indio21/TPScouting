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

## Estado Actual

- Ultima validacion documentada: `79 passed`, cobertura total `80%` al cierre de Fase 3.
- Los tests usan bases temporales y no tocan las `.db` reales del MVP.
- `/settings` queda restringido a rol `administrador`.
- CI ejecuta la suite con `pytest-cov` y publica `coverage.xml` como artefacto por version de Python.
- Cobertura funcional actual: login/logout, rutas protegidas, CSRF, CRUD de jugadores, inputs invalidos, atributos fuera de rango, dashboard, comparadores, potencial, pipeline ML, checkpoints y comportamiento sin modelo cargado.

## Nota de alcance

La regla actual del codigo para atributos tecnicos es `0-20`. Si el documento academico final afirma `1-20`, hay que alinear documento o regla funcional antes de entregar.
