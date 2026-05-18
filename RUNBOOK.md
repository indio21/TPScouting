# RUNBOOK – Scouting IA (MVP)

Este runbook cubre operación mínima, backup/restore de SQLite, healthcheck y bootstrap de admin.

## Estado de ramas
- `training`: base estable cerrada del MVP corregido.
- `reformas-finales`: rama activa para nuevas reformas.

## 1) Variables de entorno (producción)
- `APP_SECRET_KEY` (obligatoria; no usar el default del código).
- `APP_DB_URL` (recomendado; controla la BD operativa).
- `TRAINING_DB_URL` (recomendado; controla la BD de entrenamiento).
- `LOG_LEVEL` (`INFO`/`DEBUG`).
- `CACHE_TTL_SECONDS` (opcional; TTL del cache del dashboard, default `60`).
- `CACHE_MAX_ENTRIES` (opcional; limite del cache del dashboard, default `128`).
- `PLAYER_LIST_PER_PAGE` (opcional; paginacion del listado, default `50`).
- `MAX_COMPARE_PLAYERS` (opcional; limite de jugadores cargados en comparadores, default `2000`).
- `LOGIN_RATE_LIMIT_WINDOW_SECONDS` (opcional; ventana del rate limiting de login, default `300`).
- `LOGIN_RATE_LIMIT_MAX_ATTEMPTS` (opcional; intentos fallidos por IP + usuario, default `5`).
- `ALLOW_SQLITE_IN_PRODUCTION` (solo emergencia; mantener apagado en Render para evitar filesystem efimero).
- `RENDER_SMOKE_BASE_URL` (solo validacion externa; URL publica real del servicio Render).
- `SMOKE_USERNAME` / `SMOKE_PASSWORD` (opcional para smoke autenticado del deploy).

## 1.1) Seguridad MVP

- En produccion/Render, `APP_SECRET_KEY` es obligatoria. Si falta, la app no arranca.
- SQLite queda bloqueado en produccion salvo opt-in explicito con `ALLOW_SQLITE_IN_PRODUCTION=1`.
- Los formularios POST que modifican datos validan CSRF.
- El cierre de sesion usa POST con CSRF.
- El rate limiting de login es in-memory y por proceso. Mitiga fuerza bruta basica en MVP, pero para produccion real deberia moverse a Redis, DB o servicio externo.
- Se agregan headers basicos: `X-Content-Type-Options`, `X-Frame-Options` y `Referrer-Policy`.

## 1.2) Datos y ML

- Las migraciones manuales de columnas usan transacciones con `engine.begin()` en `ensure_player_columns`.
- `sync_shortlist.py` valida `limit`, `min_age` y `max_age` antes de copiar jugadores.
- El cache del dashboard es in-memory, con TTL y cantidad maxima; no se comparte entre procesos.
- Los checkpoints del modelo guardan `input_dim` y `seed`, y la carga valida compatibilidad con el preprocesador.
- El entrenamiento fija seed para `random`, `numpy`, `torch` y el orden de `DataLoader`/sampler.
- Las metricas y calibradores registran warnings cuando no pueden calcularse o aplicarse, sin ocultar el fallback.
- Los atributos tecnicos, campos fisicos en escala y reportes scout se validan como escala `1-20`. `ensure_player_columns` normaliza valores heredados fuera de rango.

## 2) Arranque local
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
.\.venv\Scripts\python.exe scouting_app\app.py
```

Para reproducibilidad exacta de dependencias existe `requirements-lock.txt`. Usarlo cuando se necesite recrear el entorno con las mismas versiones instaladas al cierre de esta rama.

La explicacion breve de los cambios agregados para cerrar la revision de codigo fuente esta en `docs/explicacion_cambios_revision_codigo_2026-04-27.md`.

Los artefactos de datos y modelo no son fuente principal del repo. Para regenerar una corrida local completa, usar `docs/flujo_reproducible_mvp.md`.

Por defecto la app no reentrena automaticamente al iniciar si faltan `model.pt` o `preprocessor.joblib`. Si se quiere permitir esa conducta en desarrollo, setear `AUTO_TRAIN_ON_STARTUP=true`.

## 2.0.1) Smoke visual y smoke Render

Playwright es una prueba visual opcional: levanta un navegador Chromium real, entra al login, valida dashboard y revisa que no haya overflow horizontal basico en desktop/mobile.

```powershell
$env:RUN_PLAYWRIGHT = "1"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m pytest -q tests\test_visual_smoke.py
Remove-Item Env:\RUN_PLAYWRIGHT
```

Smoke real de Render significa pegarle a la URL publicada, no al servidor local. El repo no contiene una URL publica fija; usar la URL real del servicio:

```powershell
$env:RENDER_SMOKE_BASE_URL = "https://TU_SERVICIO.onrender.com"
$env:SMOKE_USERNAME = "admin"
$env:SMOKE_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe scripts\smoke_render.py
```

## 2.1) Corrida oficial reproducible
Desde `scouting_app/`:

```powershell
..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///players_training.db --model-out model.pt --preprocessor-out preprocessor.joblib --calibrator-out probability_calibrator.joblib --metadata-out training_metadata.json --splits-out training_splits.json --epochs 45 --lr 5e-4 --patience 10
..\.venv\Scripts\python.exe evaluate_saved_model.py --db-url sqlite:///players_training.db --metadata-path training_metadata.json
..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18 --replace
```

`--replace` refresca los jugadores de la base operativa para demo y copia historiales deportivos completos sin borrar usuarios.

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
```powershell
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
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
- El cache tiene limite configurable de entradas (`CACHE_MAX_ENTRIES`, por default 128).
- El cache se invalida cuando se cargan o editan jugadores, atributos, historial o cuando corre el pipeline.
- Si se supera el limite, se descarta la entrada con vencimiento mas cercano.
- En despliegues con mas de una instancia o proceso, el cache no se comparte entre workers.
- Para este MVP se recomienda mantener una sola instancia de aplicacion si se quiere consistencia inmediata del dashboard.

### 8.6 Dependencia externa DiceBear
- Si un jugador no tiene `photo_url`, la app genera una URL publica de DiceBear (`api.dicebear.com`) para que el navegador cargue un avatar.
- El servidor no descarga esa imagen, por lo que no hay timeout server-side asociado.
- Si DiceBear no responde o el usuario no tiene acceso externo, la ficha sigue funcionando pero el avatar puede no mostrarse.

### 8.7 Limite de arquitectura del MVP
- `scouting_app/app.py` sigue concentrando configuracion, rutas, seguridad, cache, pipeline e inferencia.
- Para el MVP academico se mantiene asi para reducir cambios de alcance.
- Una version productiva deberia separar rutas en blueprints y mover logica de negocio a servicios o modulos especificos.

### 8.8 Categorias juveniles y fechas demo
- La categoria juvenil visible (`Cat. YYYY`) sale de `Player.birth_date.year`.
- En altas, ediciones e importacion CSV, la fecha de nacimiento es obligatoria y la edad se calcula desde esa fecha.
- La base demo local puede contener fechas de nacimiento generadas para MVP cuando el origen legacy no tiene ese dato real.
- `sync_shortlist.py` genera fechas demo deterministicas si el origen no trae `birth_date`, para evitar que una nueva sincronizacion vuelva a mostrar `Cat. N/D`.
- Antes del backfill local del 2026-05-12 se creo backup: `scouting_app/players_updated_v2.before_birthdate_backfill_20260512_201239.db`.

### 8.9 Umbrales de potencial
- `Bajo potencial`: menor a `60%`.
- `Potencial medio`: desde `60%` hasta `79%`.
- `Alto potencial`: `80%` o mas.
- La etiqueta interna `potential_label` usa el mismo umbral alto (`>= 0.80`).
