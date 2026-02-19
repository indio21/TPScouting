# RUNBOOK – Scouting IA (MVP)

Este runbook cubre operación mínima, backup/restore de SQLite, healthcheck y bootstrap de admin.

## 1) Variables de entorno (producción)
- `APP_SECRET_KEY` (obligatoria; no usar el default del código).
- `APP_DB_URL` (recomendado; controla la BD operativa).
- `TRAINING_DB_URL` (recomendado; controla la BD de entrenamiento).
- `LOG_LEVEL` (`INFO`/`DEBUG`).

## 2) Arranque local
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
python scouting_app/app.py
```

## 3) Healthcheck
- Endpoint: `GET /health`
- Esperado:
  - HTTP 200 y `{"status":"ok"}` si hay conectividad a DB.
  - HTTP 500 si falla conectividad.

## 4) Bootstrap de administrador (si no existe)
Este proyecto restringe `/settings` y `/register` a rol `administrador`.

Crear admin (BD operativa):
```bash
APP_DB_URL="sqlite:///players_updated_v2.db" \
ADMIN_USERNAME="admin" \
ADMIN_PASSWORD="admin123" \
python scouting_app/create_admin.py
```

## 5) Backup de SQLite (operativa y entrenamiento)

### 5.1 Ubicación típica
- Operativa: `scouting_app/players_updated_v2.db` (default)
- Entrenamiento: `scouting_app/players_training.db` (default)

Si usás env vars, el path puede variar (`APP_DB_URL`, `TRAINING_DB_URL`).

### 5.2 Backup “en frío” (recomendado)
1) Detener la app.
2) Copiar el archivo .db:
```bash
cp scouting_app/players_updated_v2.db backups/players_updated_v2_YYYYMMDD.db
cp scouting_app/players_training.db backups/players_training_YYYYMMDD.db
```

### 5.3 Backup “en caliente” (si la app corre)
Si WAL está activo, también copiar `-wal` y `-shm`:
```bash
cp scouting_app/players_updated_v2.db backups/
cp scouting_app/players_updated_v2.db-wal backups/  || true
cp scouting_app/players_updated_v2.db-shm backups/  || true
```

## 6) Restore
1) Detener la app.
2) Reemplazar el `.db` por el backup.
3) Si existían `-wal`/`-shm` y el backup los incluye, restaurarlos también.
4) Levantar la app y validar `/health`.

## 7) Deploy (Render)
- `render.yaml` define build/start.
- Validación post-deploy:
  - `/health`
  - login
  - `/players`
  - `/dashboard`
  - escritura (stats/attributes) con 2 sesiones

## 8) Incidentes comunes

### 8.1 `database is locked`
Acciones:
- Confirmar `journal_mode=WAL` + `busy_timeout`.
- Mantener `--workers 1 --threads 2` con SQLite.
- Evitar transacciones largas.
- Considerar migrar a Postgres si hay concurrencia real.

### 8.2 403 inesperado
- Verificar rol en sesión (`session['role']`).
- Verificar usuario admin creado con `create_admin.py`.

### 8.3 /health devuelve 500
- Verificar que el path del `.db` exista y sea accesible.
- Verificar env vars `APP_DB_URL` y permisos en filesystem.
