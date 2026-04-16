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
  - HTTP 200 con `status=ok`, conectividad a DB y bloque `data_quality`.
  - HTTP 500 si falla conectividad.

## 3.1) Auditoria y limpieza operativa
- `Configuracion` incluye una accion para auditar y limpiar la base operativa.
- La limpieza elimina registros legacy inconsistentes con las reglas actuales del MVP.
- No inventa DNIs ni completa identificadores faltantes con datos ficticios.
- Despues de ejecutarla, conviene validar de nuevo `GET /health`.

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

### 8.4 PostgreSQL en despliegue
- La app ya soporta `APP_DB_URL` y `TRAINING_DB_URL` apuntando a PostgreSQL.
- En Render, conviene usar las URLs internas de las bases administradas por la plataforma.
- Si se despliega con dos bases separadas, validar conectividad a ambas antes de correr entrenamiento o sincronizacion.
- En PostgreSQL, backup y restore deben hacerse con herramientas del motor (`pg_dump`, snapshots del proveedor o backups administrados), no copiando archivos `.db`.

### 8.5 Cache en memoria del dashboard
- El dashboard usa cache in-memory con TTL (`CACHE_TTL_SECONDS`, por default 60s).
- El cache se invalida cuando se cargan o editan jugadores, atributos, historial o cuando corre el pipeline.
- En despliegues con mas de una instancia o proceso, el cache no se comparte entre workers.
- Para este MVP se recomienda mantener una sola instancia de aplicacion si se quiere consistencia inmediata del dashboard.
