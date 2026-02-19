# Scouting IA - Test Patch (pytest)

## Instalación
pip install -r requirements.txt
pip install -r requirements-dev.txt

## Ejecutar
pytest -q

## Notas
- Los tests usan DBs temporales (no tocan tus .db reales).
- /settings queda admin-only con el hotfix aplicado (testea 403 para no-admin).
