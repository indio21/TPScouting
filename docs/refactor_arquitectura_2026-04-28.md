# Refactor De Arquitectura - Evaluacion, Fase 1 Y Fase 2

Fecha: 2026-04-28

Ultima actualizacion: 2026-04-29

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

## Fase 2 Auth

El 2026-04-29 empezo la fase 2 con el bloque mas acotado: autenticacion.

Se agregaron:

- `scouting_app/routes/__init__.py`: utilidad para registrar aliases historicos de endpoints.
- `scouting_app/routes/auth.py`: blueprint `auth` con `/login`, `/logout` y `/register`.

Se modifico:

- `scouting_app/app.py`: registra el blueprint `auth` y conserva aliases legacy para `login`, `logout` y `register`.

La decision clave fue mantener compatibilidad con los endpoints historicos. Flask normalmente nombraria estas vistas como `auth.login`, `auth.logout` y `auth.register`; en esta fase tambien se registran los nombres `login`, `logout` y `register` para que `url_for('login')`, redirects, templates y tests sigan funcionando mientras se migra por familias.

## Fase 2 Staff Y Players

Luego se movieron dos familias mas:

- `scouting_app/routes/staff.py`: coaches y directors.
- `scouting_app/routes/players.py`: listado de jugadores, ficha, historial de rendimiento, historial de atributos, edicion, baja, proyeccion y carga manual/masiva.

Se conservaron aliases legacy para los endpoints usados por templates, redirects y tests:

- staff: `list_coaches`, `new_coach`, `edit_coach`, `delete_coach`, `list_directors`, `new_director`, `edit_director`, `delete_director`;
- players: `index`, `manage_players`, `player_detail`, `player_stats`, `player_attributes`, `predict_player`, `edit_player`, `delete_player`.

La decision tecnica fue registrar `players` al final de `app.py`, despues de definir helpers de validacion, prediccion, cache y permisos. Esto evita depender de funciones que aun no existen durante el import de Flask.

## Que No Se Toco A Proposito

No se movieron todavia todas las rutas a blueprints.

Motivo: si se usan blueprints de forma directa, endpoints como `dashboard`, `compare_players` o `settings` suelen pasar a nombres prefijados como `dashboard.dashboard`, `compare.compare_players` o `settings.settings`. Eso obliga a actualizar templates, redirects, tests y links internos. Es posible, pero se sigue haciendo por familia.

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

- `app.py`: `2145` lineas;
- `services/cache.py`: `40` lineas;
- `services/security.py`: `55` lineas;
- `services/operational_data.py`: `206` lineas;
- `ml/runtime.py`: `74` lineas.
- `routes/auth.py`: rutas de login/logout/registro.
- `routes/staff.py`: `156` lineas.
- `routes/players.py`: `854` lineas.

## Validacion

Se ejecuto:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Resultado:

- `48 passed`
- cobertura total `77%`
- `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN

Validacion focal posterior a fase 2 auth:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_auth.py tests\test_mvp_regressions.py::test_register_rejects_weak_password tests\test_mvp_regressions.py::test_register_creates_user_with_valid_role tests\test_mvp_regressions.py::test_register_requires_csrf_token tests\test_mvp_regressions.py::test_mutating_post_routes_reject_missing_csrf tests\test_pages.py -q
```

Resultado:

- `14 passed`

Validacion completa posterior a fase 2 auth:

- `49 passed`
- cobertura total `77%`
- `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN

Validacion focal posterior a fase 2 staff/players:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pages.py tests\test_auth.py tests\test_mvp_regressions.py -k "manage_players or edit_player or delete_player or player_stats or player_attributes or director_cannot or scout_can_access or mutating_post_routes or admin_can_create" -q
```

Resultado:

- `13 passed`

Validacion completa con cobertura posterior a fase 2 staff/players:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Resultado:

- `51 passed`
- cobertura total `77%`
- `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN

## Riesgos Que Se Redujeron

- Menos logica de infraestructura mezclada con rutas.
- Menos acoplamiento entre carga de modelo y Flask.
- Cache y rate limit quedan testeables como servicios separados.
- Mantenimiento operativo deja de estar enterrado dentro del archivo principal.

## Riesgos Que Siguen

- Todavia quedan rutas en `app.py`: landing, health, dashboard, comparadores y settings.
- Ya estan separados `auth`, `staff` y `players`.
- Los nombres de endpoints siguen dependiendo de una capa de compatibilidad mientras se migra por familias.
- El dashboard todavia tiene deuda de rendimiento a escala.
- Las siguientes familias de blueprints todavia requieren revisar templates, redirects y tests.

## Siguiente Paso Si Se Continua Arquitectura

El siguiente corte razonable es seguir moviendo rutas por dominio, de a una familia:

1. `routes/auth.py`: login, logout, register. Hecho.
2. `routes/staff.py`: coaches/directors. Hecho.
3. `routes/players.py`: listado, alta/edicion/baja, ficha, stats, atributos y proyeccion. Hecho.
4. `routes/dashboard.py`: dashboard. Pendiente.
5. `routes/compare.py`: comparadores. Pendiente.
6. `routes/settings.py`: configuracion y pipeline. Pendiente.

Para hacerlo con seguridad habria que decidir antes si se aceptan endpoints prefijados por blueprint o si se implementa una capa de compatibilidad para no romper `url_for(...)`.
