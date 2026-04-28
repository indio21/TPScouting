# Refactor De Arquitectura - Evaluacion Y Fase 1

Fecha: 2026-04-28

Rama: `reformas-complejas`

## Evaluacion De Factibilidad

El refactor de arquitectura es factible, pero no conviene hacerlo como un unico movimiento grande.

`scouting_app/app.py` concentraba rutas Flask, seguridad, cache, mantenimiento de datos, carga de artefactos ML, helpers de prediccion y handlers de error. Separar todo de golpe hacia blueprints y servicios puede romper:

- nombres de endpoints usados por `url_for(...)` en templates;
- decoradores de permisos (`login_required`, `roles_required`);
- sesion Flask y token CSRF;
- estado global del modelo/preprocesador/calibrador;
- cache del dashboard y rate limit de login;
- tests que importan `app.py` dinamicamente;
- rutas que comparten helpers internos.

Por eso el corte mas seguro fue hacer una **fase 1 de servicios**, manteniendo `app.py` como entrypoint y conservando los nombres publicos que ya usan rutas, templates y tests.

## Que Se Toco

Se agregaron modulos nuevos:

- `scouting_app/services/cache.py`: cache TTL con limite de entradas.
- `scouting_app/services/security.py`: rate limit de login, CSRF y helper de IP.
- `scouting_app/services/operational_data.py`: mantenimiento de base operativa, limpieza, calidad de datos y sincronizacion de historial tecnico.
- `scouting_app/ml/runtime.py`: carga de modelo, preprocesador y calibrador para inferencia.

Se modifico:

- `scouting_app/app.py`: ahora delega esas responsabilidades en servicios, pero conserva rutas y endpoints.

## Que No Se Toco A Proposito

No se movieron todavia las rutas a blueprints.

Motivo: si se usan blueprints de forma directa, endpoints como `login`, `index`, `dashboard` o `player_detail` suelen pasar a nombres prefijados como `auth.login` o `players.player_detail`. Eso obliga a actualizar templates, redirects, tests y links internos. Es posible, pero es un bloque mas riesgoso.

Tampoco se modificaron:

- esquema de base de datos;
- templates;
- nombres de rutas;
- logica de prediccion;
- formato de artefactos del modelo;
- comandos reproducibles del MVP.

## Compatibilidad Conservada

Para reducir riesgo se mantuvieron wrappers en `app.py`:

- `_cache_get`, `_cache_set`, `invalidate_dashboard_cache`;
- `is_login_rate_limited`, `register_failed_login`, `clear_failed_logins`;
- `trim_operational_player_pool`, `compute_operational_data_quality`, `cleanup_operational_data`;
- `sync_player_attribute_history`, `sync_attribute_history_baseline`;
- `load_runtime_artifacts`, `apply_probability_calibrator`.

Esto permite que rutas y tests sigan usando los mismos nombres aunque la implementacion viva fuera de `app.py`.

## Resultado Arquitectonico

Antes, `app.py` mezclaba responsabilidades de infraestructura, servicios y rutas.

Despues de esta fase:

- cache y seguridad liviana estan en `services`;
- mantenimiento de base operativa esta en `services`;
- runtime de ML esta en `ml`;
- `app.py` sigue siendo grande, pero ya no contiene toda la implementacion interna de esas piezas.

Medicion aproximada tras el corte:

- `app.py`: `2728` lineas;
- `services/cache.py`: `40` lineas;
- `services/security.py`: `55` lineas;
- `services/operational_data.py`: `206` lineas;
- `ml/runtime.py`: `74` lineas.

## Validacion

Se ejecuto:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Resultado:

- `48 passed`
- cobertura total `77%`
- `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN

## Riesgos Que Se Redujeron

- Menos logica de infraestructura mezclada con rutas.
- Menos acoplamiento entre carga de modelo y Flask.
- Cache y rate limit quedan testeables como servicios separados.
- Mantenimiento operativo deja de estar enterrado dentro del archivo principal.

## Riesgos Que Siguen

- Las rutas siguen en `app.py`.
- Los nombres de endpoints siguen dependiendo del archivo principal.
- El dashboard todavia tiene deuda de rendimiento a escala.
- Una fase de blueprints todavia requeriria revisar templates y tests.

## Siguiente Paso Si Se Continua Arquitectura

El siguiente corte razonable seria mover rutas por dominio, de a una familia:

1. `routes/auth.py`: login, logout, register.
2. `routes/staff.py`: coaches/directors.
3. `routes/players.py`: listado, alta/edicion/baja, ficha, stats, atributos.
4. `routes/dashboard.py`: dashboard.
5. `routes/compare.py`: comparadores.
6. `routes/settings.py`: configuracion y pipeline.

Para hacerlo con seguridad habria que decidir antes si se aceptan endpoints prefijados por blueprint o si se implementa una capa de compatibilidad para no romper `url_for(...)`.
