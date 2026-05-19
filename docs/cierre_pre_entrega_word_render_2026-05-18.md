# Cierre Pre Entrega - Word y Render

Fecha: 2026-05-18

Este documento deja preparado el estado para continuar manana con los documentos Word
de la primera entrega y de observaciones del profesor.

## Conclusion Operativa

El proyecto esta en condicion de MVP academico avanzado. Para dejarlo entregable de
forma seria faltan dos cierres verificables:

1. Alinear el documento Word final de tesis/trabajo con el MVP real.
2. Hacer un deploy real en Render y ejecutar smoke contra la URL publicada.

Con esas dos tareas cerradas, mas la suite local y CI ya existentes, el proyecto queda
defendible como demo academica/MVP. No queda presentado como producto listo para
produccion.

## Que Se Debe Revisar En El Word

Cuando el usuario entregue el Word de primera entrega y el Word con observaciones del
profesor, revisar punto por punto contra el codigo actual:

- Arquitectura real: Flask con blueprints por familia y servicios parciales.
- Base de datos: SQLite local para MVP y PostgreSQL administrado previsto en Render.
- Modelo ML real: PyTorch tipo MLP/residual, preprocesador scikit-learn y baseline
  logistico documentado como comparacion.
- Dataset: sintetico, reproducible y acotado a juveniles de 12 a 18 anos.
- Atributos y reportes en escala `1-20`.
- Potencial: bajo menor a `60%`, medio `60-79%`, alto desde `80%`.
- Edad y categoria juvenil derivadas de `birth_date`.
- Seguridad MVP: secret key obligatoria en produccion, CSRF manual, logout POST,
  rate limiting in-memory y roles.
- Testing: pytest, pytest-cov, CI, cobertura documentada y smoke visual opt-in.
- Despliegue: Render + Gunicorn + PostgreSQL; smoke real pendiente hasta tener URL.
- Limitaciones reales: cache in-memory, rate limiting no distribuido, lock local,
  migraciones manuales sin Alembic, datos sinteticos y validacion no externa.

Si el Word afirma LSTM, Random Forest como modelo principal, datos reales validados,
produccion completa, alta concurrencia o seguridad productiva, debe corregirse la
narrativa. El codigo actual respalda un MVP academico, no una solucion productiva.

## Que Se Debe Probar En Render

El deploy real no es opcional si se quiere cerrar la evidencia productiva. Hay cosas
que solo se verifican en cloud:

- Variables de entorno reales.
- `APP_SECRET_KEY` obligatoria.
- Conexion a PostgreSQL administrado.
- Comando Gunicorn de `render.yaml`.
- HTTPS/cookies en entorno productivo.
- Build con Torch CPU.
- Healthcheck externo.
- Login y rutas protegidas.
- Dashboard y consultas sobre base operativa.

Comando de smoke previsto:

```powershell
$env:RENDER_SMOKE_BASE_URL = "https://TU_SERVICIO.onrender.com"
$env:SMOKE_USERNAME = "admin"
$env:SMOKE_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe scripts\smoke_render.py
```

Checklist minimo de smoke manual:

- `GET /health` responde `200`.
- Login admin correcto.
- `/dashboard` carga sin error.
- `/players` lista jugadores.
- Ficha de jugador abre correctamente.
- Comparador simple y multiple cargan.
- Una accion CRUD minima funciona con CSRF.
- Logout funciona por POST.

## Estado De Riesgos Para Entrega

Corregible con Word:

- Documento academico final no alineado con el MVP real.
- Dataset sintetico y alcance academico.
- Arquitectura parcial, no totalmente separada.
- Limitaciones de cache, rate limiting, lock y migraciones manuales.

Requiere deploy/verificacion:

- Smoke real de Render.
- Variables reales de produccion.
- Conexion PostgreSQL administrada.
- Funcionamiento con Gunicorn y HTTPS.

No conviene abrir antes de entregar:

- Refactor grande de `app.py` o `routes/players.py`.
- Alembic completo.
- Redis/rate limiting distribuido.
- Optimizacion profunda para 10.000 jugadores.
- Tooling obligatorio con formato masivo.

## Punto De Retome

Rama de trabajo actual: `ux-crud-polish`.

Siguiente paso recomendado:

1. Recibir los dos Word del usuario.
2. Comparar documento vs codigo actual sin inventar.
3. Corregir el Word o marcar incoherencias.
4. Hacer deploy en Render.
5. Ejecutar smoke real y documentar evidencia.
6. Si Render falla, corregir puntualmente y volver a probar.

## Ajuste Para Render Free

Render Free permite probar la app, pero las bases PostgreSQL Free expiran y hay
limites de cantidad por workspace. Para no pagar, el despliegue de MVP usa una
sola base nueva:

- `tpscouting-mvp-db`
- `APP_DB_URL` apunta a esa base.
- `TRAINING_DB_URL` apunta a esa misma base solo para evitar SQLite en produccion.
- `AUTO_TRAIN_ON_STARTUP=0`.
- `SYNC_SHORTLIST_ENABLED=0`.
- `DEMO_SEED_ON_STARTUP=1`.
- `seed_demo_data.py` carga 100 jugadores sinteticos si la base esta vacia.

En este modo, no ejecutar el pipeline de entrenamiento desde la pantalla de
configuracion. La prediccion usa artefactos pequenos versionados para el deploy.

## Evidencia De Deploy Render

Fecha: 2026-05-19

Rama desplegada:

- `render-free-deploy`

URL publica:

- `https://tpscouting-mvp.onrender.com`

Commit base de deploy:

- `c5735e2 deploy: avoid Render service name collision`

Smoke publico ejecutado:

- `/` respondio `200`.
- `/health` respondio `200`.
- `/login` respondio `200`.

Smoke autenticado ejecutado con usuario `admin`:

- `scripts/smoke_render.py` confirmo `/health OK`, `/login OK`, login correcto y `/dashboard OK`.
- `/dashboard` respondio `200`.
- `/players` respondio `200`.
- `/compare` respondio `200`.
- `/compare/multi` respondio `200`.
- `/settings` respondio `200`.
- `/players/import` respondio `200`.

Resumen de `/health`:

- `status`: `ok`
- `database`: `ok`
- `players_total`: `100`
- `missing_birth_date`: `0`
- `missing_national_id`: `0`
- `missing_photo_url`: `0`
- `invalid_age`: `0`
- `over_limit_players`: `0`

Pendiente antes de cerrar la entrega:

- Prueba manual de una accion CRUD minima en Render.
- Revisión del Word final y del Word con observaciones del profesor.
