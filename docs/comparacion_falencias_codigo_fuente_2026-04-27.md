# Comparacion De Falencias Del Codigo Fuente

Fecha: 2026-04-27

Documento comparado: `C:\Users\Usuario\Desktop\Evaluacion y punto a corregir del Codigo Fuente.docx`

Rama analizada: `reformas-finales`

Commit base del analisis: `d60ad6d chore: close source review follow-ups`

## Criterio De Analisis

Este informe compara las observaciones del profesor contra el estado real del codigo del MVP. No asume correcciones no verificadas y no inventa evidencia. La validacion tecnica ejecutada antes de cerrar el analisis fue:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Resultado actualizado tras el bloque CSRF/inputs: `45 passed`, cobertura total reportada `76%`, con 4 warnings conocidos de scikit-learn por fixtures con columnas all-NaN en tests.

## Resultado General

El documento del profesor lista 31 observaciones agrupadas en problemas criticos, seguridad, consistencia de datos, rendimiento, calidad de codigo, testing, machine learning y dependencias.

Estado general despues de las correcciones ya implementadas en el MVP:

- Corregidos o practicamente corregidos para el alcance del MVP: 22.
- Parciales, aceptados o documentados como limitacion tecnica: 9.
- Pendientes criticos: 0.

Conclusion honesta: el MVP actual esta bastante mas avanzado que el estado descripto en el informe original. Los tres problemas criticos ya no estan en el mismo estado: Render ya apunta a PostgreSQL, el umbral de potencial fue unificado y el lock del pipeline quedo mitigado por single-worker con comentario explicito en codigo. Lo pendiente se concentra en hardening incremental, deuda de calidad y optimizaciones que no bloquean el MVP academico.

## Tabla Punto Por Punto

| Punto | Estado actual | Nivel | Observacion / recomendacion |
|---|---:|---:|---|
| CRITICO-01 Render con SQLite efimero | Corregido | 100% | `render.yaml` ya define dos PostgreSQL administradas y usa `APP_DB_URL` / `TRAINING_DB_URL`. |
| CRITICO-02 umbral inconsistente de potencial | Corregido | 100% | `app.py` centraliza umbrales; `refresh_player_potential` usa `is_high_potential_probability`. Hay test especifico. |
| CRITICO-03 lock multi-worker | Cerrado para MVP | 90% | Render fuerza `--workers 1 --threads 2`; `RUNBOOK.md` lo documenta y `app.py` incluye comentario junto a `_PIPELINE_LOCK`. Un lock DB/file robusto queda fuera del alcance actual. |
| SEG-01 CSRF | Corregido mayormente | 95% | Los POST mutantes llaman `_require_csrf`; se agrego un test matriz que verifica rechazo `400` sin token para login, registro, jugadores, stats, atributos, staff y settings. |
| SEG-02 secret key insegura | Corregido | 100% | `app.py` usa `secrets.token_urlsafe(32)` si no hay secret y exige secret en produccion. |
| SEG-03 rate limiting login | Corregido basico | 90% | Existe rate limit en memoria y test. Para produccion real faltaria persistente/Redis, pero para MVP esta bien. |
| SEG-04 contrasena debil en create_admin | Corregido | 100% | `create_admin.py` exige minimo 8 caracteres, letras y numeros; los ejemplos documentales usan una clave demo mas fuerte. |
| SEG-05 DiceBear externo | Documentado | 85% | Es solo URL generada para el navegador, no descarga server-side. `RUNBOOK.md` y `docs/explicacion_cambios_revision_codigo_2026-04-27.md` documentan la dependencia externa. |
| DATOS-01 migraciones sin transaccion | Corregido | 100% | `db_utils.py` usa `engine.begin()`, por lo que hay transaccion/commit. |
| DATOS-02 limite de sync_shortlist | Corregido | 100% | `sync_shortlist.py` incrementa `existing_total` al insertar. |
| DATOS-03 cache dashboard no invalidada | Corregido mayormente | 90% | Existe `invalidate_dashboard_cache()` y se llama tras mutaciones. RUNBOOK documenta TTL y limite multi-worker. |
| DATOS-04 timestamps | Corregido | 100% | `TimestampMixin` esta en modelos principales y `ensure_player_columns` cubre tambien `physical_assessments` y `player_availability`, con test de migracion legacy. |
| REND-01 N+1 listado jugadores | Corregido | 95% | El listado usa `batch_project_players` y mapas agregados, no `compute_projection` por jugador. |
| REND-02 dashboard carga todos los jugadores | Parcial aceptable | 75% | Sigue haciendo `.all()` en dashboard, pero existe `EVAL_POOL_MAX=100` y guardrails. Para MVP sirve; si crece, paginar/agregar consultas SQL. |
| REND-03 cache sin limite de tamano | Documentado parcial | 70% | TTL existe y la limitacion de no tener maximo de entradas quedo documentada. Agregar `CACHE_MAX_ENTRIES` sigue siendo una mejora simple. |
| CALIDAD-01 app.py monolitico | Documentado parcial | 45% | `app.py` sigue siendo monolitico. La limitacion quedo documentada; una separacion en blueprints queda como mejora productiva posterior. |
| CALIDAD-02 magic numbers | Parcial | 55% | Algunos pasaron a env vars, pero quedan `50`, `2000`, `90.0`, pesos de score, etc. Conviene constantes nombradas. |
| CALIDAD-03 nomenclatura db/db_session | Parcial | 50% | Sigue mezclado. No rompe funcionalidad, pero conviene estandarizar gradualmente a `db_session` en helpers y `db` en endpoints. |
| CALIDAD-04 type hints | Mejorado parcial | 70% | Hay mas hints, por ejemplo `Dict[str, Optional[float]]`; no esta completo en toda la app. |
| CALIDAD-05 uso de globals() | Corregido | 100% | Se reemplazo `globals().get(...)` por llamada directa a `sync_attribute_history_baseline`, moviendo la llamada de startup para respetar el orden de definicion. |
| TEST-01 cobertura muy baja | Medido formalmente | 90% | La suite tiene `45` tests y CI mide cobertura con `pytest-cov`; la corrida local reporto `76%` total. |
| TEST-02 CRUD | Corregido | 100% | Hay tests de crear/editar/eliminar jugador. |
| TEST-03 inputs invalidos | Corregido | 100% | Hay tests de DNI duplicado, CSRF, stats invalidos, atributos fuera de rango, edad invalida y campos obligatorios vacios. |
| TEST-04 conftest carga dinamica | Pendiente/parcial | 40% | Sigue usando modulo con UUID. Funciona, pero complica debugging. No es urgente. |
| TEST-05 CI sin cobertura | Corregido | 100% | `.github/workflows/ci.yml` ejecuta `pytest -q --cov=scouting_app --cov-report=term-missing` y `requirements-dev.txt` incluye `pytest-cov`. |
| ML-01 dimension modelo/inferencia | Corregido | 100% | El checkpoint nuevo guarda `input_dim`, version y `model_state`; `app.py` y `evaluate_saved_model.py` validan contra el preprocesador y mantienen compatibilidad con `state_dict` legacy. |
| ML-02 etiqueta sintetica no reproducible | Corregido | 100% | `generate_data.py` usa `DEFAULT_SEED`, `--seed`, `--reset` y target temporal mas serio. |
| ML-03 SEED no propagada split | Corregido | 100% | `train_model.py` usa `SEED` en `safe_train_test_split`. |
| ML-04 metricas fallan silenciosamente | Corregido mayormente | 90% | `classification_metrics` ahora devuelve una lista `warnings` cuando ROC-AUC, PR-AUC o F1/precision/recall no pueden calcularse. |
| DEP-01 requirements no pinadas | Mitigado | 75% | `requirements.txt` mantiene rangos flexibles, pero se agrego `requirements-lock.txt` con versiones exactas validadas localmente. |
| DEP-02 dev requirements incompletas | Mejorado parcial | 65% | Se agrego `pytest-cov`; siguen siendo opcionales herramientas como ruff/black/mypy. |

## Cambios Cerrados En El Bloque `d60ad6d`

- Comentario explicito junto a `_PIPELINE_LOCK` sobre dependencia de `--workers 1`.
- Reemplazo de `globals().get(...)` por llamada directa a `sync_attribute_history_baseline`.
- Agregado de `pytest-cov` y reporte de cobertura en CI.
- Documentacion de DiceBear, cache in-memory sin limite, monolito `app.py` y lock de dependencias.
- Migracion de timestamps para `physical_assessments` y `player_availability`.
- Checkpoint del modelo con `input_dim`, version y `model_state`, manteniendo compatibilidad con `state_dict` legacy.
- Warnings explicitos cuando metricas como ROC-AUC o PR-AUC no se pueden calcular.
- `requirements-lock.txt` con snapshot exacto de dependencias.
- Test matriz de CSRF para POST mutantes criticos.
- Tests de edad invalida y campos obligatorios vacios en alta de jugador.
- Renderizado de todos los mensajes flash para mostrar errores multiples de validacion.

## Proximo Bloque Sugerido

Si se decide continuar con codigo, el bloque mas razonable y acotado seria:

- Convertir magic numbers visibles (`50`, `2000`, pesos de score) en constantes nombradas.
- Agregar limite simple al cache in-memory (`CACHE_MAX_ENTRIES`) si se quiere cerrar `REND-03` con codigo.
- Evaluar simplificar `tests/conftest.py` para evitar nombres de modulo con UUID.

Eso seguiria cerrando puntos del informe sin abrir todavia la refactorizacion grande de `app.py` en blueprints.
