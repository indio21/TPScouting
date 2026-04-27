# Comparacion De Falencias Del Codigo Fuente

Fecha: 2026-04-27

Documento comparado: `C:\Users\Usuario\Desktop\Evaluacion y punto a corregir del Codigo Fuente.docx`

Rama analizada: `reformas-finales`

Commit base del analisis: `50efc4d docs: refresh MVP handoff and training evidence`

## Criterio De Analisis

Este informe compara las observaciones del profesor contra el estado real del codigo del MVP. No asume correcciones no verificadas y no inventa evidencia. La validacion tecnica ejecutada antes de cerrar el analisis fue:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Resultado: `40 passed`, con 4 warnings conocidos de scikit-learn por fixtures con columnas all-NaN en tests.

## Resultado General

El documento del profesor lista 31 observaciones agrupadas en problemas criticos, seguridad, consistencia de datos, rendimiento, calidad de codigo, testing, machine learning y dependencias.

Estado general despues de las correcciones ya implementadas en el MVP:

- Corregidos o practicamente corregidos: 15.
- Parciales o mejorados pero no cerrados del todo: 11.
- Pendientes claros: 5.

Conclusion honesta: el MVP actual esta bastante mas avanzado que el estado descripto en el informe original. Los tres problemas criticos ya no estan en el mismo estado: Render ya apunta a PostgreSQL, el umbral de potencial fue unificado y el lock del pipeline esta mitigado por single-worker, aunque este ultimo todavia merece un comentario explicito en codigo. Lo pendiente se concentra sobre todo en hardening, trazabilidad tecnica y deuda de calidad.

## Tabla Punto Por Punto

| Punto | Estado actual | Nivel | Observacion / recomendacion |
|---|---:|---:|---|
| CRITICO-01 Render con SQLite efimero | Corregido | 100% | `render.yaml` ya define dos PostgreSQL administradas y usa `APP_DB_URL` / `TRAINING_DB_URL`. |
| CRITICO-02 umbral inconsistente de potencial | Corregido | 100% | `app.py` centraliza umbrales; `refresh_player_potential` usa `is_high_potential_probability`. Hay test especifico. |
| CRITICO-03 lock multi-worker | Parcial | 70% | Render fuerza `--workers 1 --threads 2`; `RUNBOOK.md` lo documenta. Falta comentario explicito junto a `_PIPELINE_LOCK` o lock DB/file robusto. |
| SEG-01 CSRF | Bastante avanzado | 80% | Muchos POST ya llaman `_require_csrf`; hay tests para registro/CRUD. Recomiendo auditoria final ruta por ruta y test para cada POST mutante critico. |
| SEG-02 secret key insegura | Corregido | 100% | `app.py` usa `secrets.token_urlsafe(32)` si no hay secret y exige secret en produccion. |
| SEG-03 rate limiting login | Corregido basico | 90% | Existe rate limit en memoria y test. Para produccion real faltaria persistente/Redis, pero para MVP esta bien. |
| SEG-04 contrasena debil en create_admin | Corregido | 100% | `create_admin.py` exige minimo 8 caracteres, letras y numeros. |
| SEG-05 DiceBear externo | Parcial | 60% | Es solo URL generada en navegador, no server-side. Falta documentarlo explicitamente como dependencia externa/limitacion. |
| DATOS-01 migraciones sin transaccion | Corregido | 100% | `db_utils.py` usa `engine.begin()`, por lo que hay transaccion/commit. |
| DATOS-02 limite de sync_shortlist | Corregido | 100% | `sync_shortlist.py` incrementa `existing_total` al insertar. |
| DATOS-03 cache dashboard no invalidada | Corregido mayormente | 90% | Existe `invalidate_dashboard_cache()` y se llama tras mutaciones. RUNBOOK documenta TTL y limite multi-worker. |
| DATOS-04 timestamps | Corregido mayormente | 90% | `TimestampMixin` esta en modelos principales. `ensure_player_columns` cubre tablas existentes principales, aunque conviene agregar `physical_assessments` y `player_availability` al helper de migracion. |
| REND-01 N+1 listado jugadores | Corregido | 95% | El listado usa `batch_project_players` y mapas agregados, no `compute_projection` por jugador. |
| REND-02 dashboard carga todos los jugadores | Parcial aceptable | 75% | Sigue haciendo `.all()` en dashboard, pero existe `EVAL_POOL_MAX=100` y guardrails. Para MVP sirve; si crece, paginar/agregar consultas SQL. |
| REND-03 cache sin limite de tamano | Parcial | 60% | TTL existe, pero no maximo de entradas. Bajo riesgo en MVP; agregar `CACHE_MAX_ENTRIES` seria simple. |
| CALIDAD-01 app.py monolitico | Pendiente | 25% | `app.py` tiene aproximadamente 3276 lineas. Se movio logica a modulos, pero sigue monolitico. Documentar o refactorizar en blueprints. |
| CALIDAD-02 magic numbers | Parcial | 55% | Algunos pasaron a env vars, pero quedan `50`, `2000`, `90.0`, pesos de score, etc. Conviene constantes nombradas. |
| CALIDAD-03 nomenclatura db/db_session | Parcial | 50% | Sigue mezclado. No rompe funcionalidad, pero conviene estandarizar gradualmente a `db_session` en helpers y `db` en endpoints. |
| CALIDAD-04 type hints | Mejorado parcial | 70% | Hay mas hints, por ejemplo `Dict[str, Optional[float]]`; no esta completo en toda la app. |
| CALIDAD-05 uso de globals() | Pendiente | 20% | Sigue el patron `globals().get(...)` en `app.py`. Conviene reemplazar por llamada directa a `sync_attribute_history_baseline`. |
| TEST-01 cobertura muy baja | Mejorado fuerte | 85% | Ya hay 40 tests y `test_mvp_regressions.py` cubre mucho mas. Falta medicion formal de cobertura. |
| TEST-02 CRUD | Corregido | 100% | Hay tests de crear/editar/eliminar jugador. |
| TEST-03 inputs invalidos | Corregido mayormente | 90% | Hay tests de DNI duplicado, CSRF, stats invalidos y atributos fuera de rango. Podrian sumarse edad invalida y campos vacios. |
| TEST-04 conftest carga dinamica | Pendiente/parcial | 40% | Sigue usando modulo con UUID. Funciona, pero complica debugging. No es urgente. |
| TEST-05 CI sin cobertura | Pendiente | 20% | CI sigue ejecutando solo `pytest -q`; no hay `pytest-cov`. |
| ML-01 dimension modelo/inferencia | Parcial alto | 80% | Ya no calcula dimension manual fija: usa `preprocessor_input_dim`. Pero `model.pt` aun guarda solo `state_dict`, no checkpoint con `input_dim`. |
| ML-02 etiqueta sintetica no reproducible | Corregido | 100% | `generate_data.py` usa `DEFAULT_SEED`, `--seed`, `--reset` y target temporal mas serio. |
| ML-03 SEED no propagada split | Corregido | 100% | `train_model.py` usa `SEED` en `safe_train_test_split`. |
| ML-04 metricas fallan silenciosamente | Parcial | 60% | Ya se guardan metricas y metadata, pero `classification_metrics` todavia captura excepciones y devuelve `""` sin log detallado. |
| DEP-01 requirements no pinadas | Pendiente | 20% | Sigue usando `>=` en casi todas salvo PyTorch. Para defensa de tesis, pinnear versiones seria mejor. |
| DEP-02 dev requirements incompletas | Parcial | 40% | Se agrego `python-docx`, pero faltan `pytest-cov`, `ruff/black`, etc. |

## Prioridad Recomendada Para Cerrar Pendientes

1. Agregar comentario explicito junto a `_PIPELINE_LOCK` sobre `--workers 1`.
2. Quitar `globals().get(...)` y llamar directo a `sync_attribute_history_baseline`.
3. Agregar `pytest-cov` y reporte de cobertura en CI.
4. Guardar `input_dim`/metadata en checkpoint del modelo o validar explicitamente contra metadata.
5. Pinnear dependencias o crear `requirements-lock.txt`.
6. Documentar DiceBear, cache en memoria y `app.py` monolitico como limitaciones reales del MVP.

## Proximo Bloque Sugerido

Si se decide continuar con codigo, el bloque mas razonable y acotado seria:

- Comentario de single-worker junto a `_PIPELINE_LOCK`.
- Reemplazo de `globals().get(...)`.
- Agregado de `pytest-cov` y CI con cobertura.
- Documentacion breve de DiceBear y cache.

Eso cerraria varios puntos del informe sin modificar el alcance del MVP ni abrir una refactorizacion grande.
