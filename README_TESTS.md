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

## Estado Actual

- Ultima validacion documentada: `40 passed`.
- Los tests usan bases temporales y no tocan las `.db` reales del MVP.
- `/settings` queda restringido a rol `administrador`.
