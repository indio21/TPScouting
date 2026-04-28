# Revision Final Del MVP

Fecha: 2026-04-26

Adenda tecnica: 2026-04-27

Este archivo resume la revision final del MVP real de `TPScouting`, apoyada en evidencia ejecutada sobre el proyecto y sobre la demo incluida en el repositorio.

## Alcance Revisado

- scouting juvenil entre 12 y 18 anos
- uso por clubes formativos sin grandes presupuestos
- MVP multirol con:
- `administrador`
- `scout` / `ojeador`
- `director`

## Evidencia Ejecutada

### 1. Suite automatizada

- Estado final validado original: `40 passed`
- Estado tecnico actualizado 2026-04-28: `48 passed` con `pytest-cov`, cobertura total reportada `76%`
- Cobertura reforzada sobre:
- autenticacion
- permisos por rol
- CSRF en rutas criticas
- alta, edicion y baja de jugadores
- validaciones de inputs invalidos
- historial de rendimiento
- historial de atributos
- ABM de staff
- persistencia del preprocesador del modelo
- consistencia entre transformacion individual y batch
- inferencia con features historicas faltantes sin propagar `NaN`
- sincronizacion de demo rica desde la base de entrenamiento
- migracion legacy de timestamps para fisico/disponibilidad
- checkpoint del modelo con `input_dim`
- warnings explicitos para metricas no calculables
- matriz CSRF para POST mutantes criticos
- validacion de alta de jugador con edad invalida y campos obligatorios vacios
- limite configurable para cache in-memory del dashboard
- constantes nombradas en paginacion, comparadores y rating de estadisticas
- fixture de app en tests con nombre de modulo estable
- nomenclatura de sesiones SQLAlchemy mejorada en helpers y scripts

### 1.1. Cierre De Observaciones De Codigo Fuente 2026-04-27

- CI mide cobertura con `pytest-cov`.
- `requirements-lock.txt` registra versiones exactas instaladas.
- `RUNBOOK.md` documenta DiceBear, cache in-memory con limite configurable y `app.py` monolitico como limitaciones reales del MVP.
- `_PIPELINE_LOCK` documenta la dependencia de `--workers 1`.
- Se elimino el uso de `globals().get(...)` para sincronizar historial tecnico.
- El modelo se guarda como checkpoint con `input_dim`, version y `model_state`.
- La carga de modelo mantiene compatibilidad con `state_dict` legacy.
- Las metricas devuelven `warnings` cuando ROC-AUC, PR-AUC o F1/precision/recall no pueden calcularse.
- Se agregaron pruebas para que los POST mutantes criticos rechacen requests sin CSRF.
- Se agregaron pruebas para edad invalida y campos obligatorios vacios.
- La plantilla base muestra todos los mensajes flash para no ocultar errores multiples.
- Se agregaron constantes nombradas para reemplazar literales visibles de paginacion, comparadores y rating.
- El cache in-memory del dashboard ahora tiene limite `CACHE_MAX_ENTRIES`, por default `128`.
- `tests/conftest.py` evita nombres de modulo con UUID para facilitar debugging.
- Helpers operativos y `create_admin.py` usan `db_session`; la sesion temporal de entrenamiento usa `training_session`.

### 2. Smoke funcional sobre la app real del repo

Se ejecuto una verificacion con `Flask test_client()` sobre la app apuntando a la demo actual del repositorio.

Rutas verificadas con respuesta `200`:

- `GET /`
- `GET /health`
- `GET /players`
- `GET /dashboard`
- `GET /compare`
- `GET /compare/multi`
- `GET /coaches`
- `GET /directors`
- `GET /settings`
- detalle de jugador
- proyeccion de jugador
- historial de rendimiento
- historial de atributos

### 3. Estado real de la base demo

Metricas medidas al cierre actualizado:

- usuarios totales: `1`
- roles de usuario: `administrador=1`
- jugadores operativos: `100`
- registros de rendimiento: `713`
- registros de historial de atributos: `908`
- partidos sinteticos copiados a demo: `1419`
- participaciones por partido copiadas a demo: `1419`
- reportes de scout: `406`
- evaluaciones fisicas: `908`
- registros de disponibilidad: `908`

### 4. Calidad operativa de datos

Se ejecuto limpieza controlada sobre la base demo en la revision inicial:

- antes: `100` jugadores
- antes: `4` jugadores sin `national_id`
- despues: `96` jugadores
- despues: `0` jugadores sin `national_id`

La limpieza no invento DNIs faltantes. En su lugar elimino registros legacy inconsistentes con las reglas activas del MVP.

Luego, para la demo final, `sync_shortlist.py --replace` reconstruyo los jugadores juveniles desde la base de entrenamiento y copio informacion longitudinal completa sin borrar usuarios.

## Mejoras Cerradas En Esta Revision

- validacion de stats para impedir porcentajes fuera de `0-100`
- validacion de valoracion final para impedir rangos fuera de `1-10`
- validacion de historial de atributos para impedir rangos fuera de `0-20`
- rechazo de guardado vacio en historial de atributos
- limpieza de warnings relevantes de SQLAlchemy 2.x
- dataset de entrenamiento construido con `pandas`
- preprocesamiento real con `SimpleImputer`, `MinMaxScaler` y `OneHotEncoder`
- `preprocessor.joblib` compartido entre entrenamiento e inferencia
- target temporal de progresion juvenil
- base demo con historial, partidos, reportes scout, fisico y disponibilidad
- explicacion de indicadores visible documentada en `docs/guia_indicadores_app.md`

## Riesgos O Pendientes Reales

Estos puntos siguen siendo reales y no se deben ocultar:

- despliegue final en Render todavia requiere cierre operativo completo
- la app sigue siendo un MVP; no esta pensada para alta concurrencia
- persisten deudas de calidad no criticas: `app.py` monolitico, convenciones parciales en endpoints Flask y type hints parciales
- el documento Word todavia no fue alineado con el estado corregido del MVP en esta fase
- la evidencia del modelo sigue basada en datos sinteticos; no hay validacion externa con datos reales

## Conclusion

Con la evidencia actual, el MVP queda funcional, coherente con su alcance acotado y bastante mas defendible que al inicio de la revision. La rama `training` queda como base estable de estas correcciones y las reformas nuevas continuan en `reformas-finales`. La siguiente etapa natural es revisar visualmente la demo y luego corregir el documento Word para que refleje fielmente este estado real.
