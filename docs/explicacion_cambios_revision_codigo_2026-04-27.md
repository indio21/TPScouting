# Explicacion De Cambios De Revision De Codigo

Fecha: 2026-04-27

Este documento explica, en lenguaje simple, los cambios tecnicos agregados para cerrar puntos observados en la revision del codigo fuente. La idea es que sirva para entender que problema resolvia cada cambio, que se modifico y por que eso mejora el MVP.

## 1. CI con cobertura usando pytest-cov

Antes, el workflow de GitHub Actions solo ejecutaba:

```bash
pytest -q
```

Eso confirmaba si los tests pasaban, pero no mostraba que partes del codigo estaban cubiertas por pruebas. Ahora el CI ejecuta:

```bash
pytest -q --cov=scouting_app --cov-report=term-missing
```

Esto hace dos cosas:

- Corre la suite de tests igual que antes.
- Muestra un reporte de cobertura indicando que porcentaje de cada archivo esta probado y que lineas faltan cubrir.

No significa que la cobertura sea perfecta. Significa que ahora se mide formalmente y queda visible en cada push o pull request. Eso responde al punto del profesor sobre "CI sin reporte de cobertura".

## 2. pytest-cov en dependencias de desarrollo y lock de dependencias

`pytest-cov` es el plugin que permite a `pytest` medir cobertura. Por eso se agrego a `requirements-dev.txt`, que contiene herramientas usadas para desarrollo y testing.

Tambien se agrego `requirements-lock.txt`. Ese archivo guarda las versiones exactas que estaban instaladas en el entorno local al momento del cierre de esta rama. Sirve para reproducibilidad: si en otro momento una dependencia cambia y rompe algo, se puede comparar contra este lock o recrear un entorno mas parecido al validado.

Diferencia entre archivos:

- `requirements.txt`: dependencias principales de la app.
- `requirements-dev.txt`: herramientas para desarrollo y tests.
- `requirements-lock.txt`: snapshot exacto de versiones instaladas.

## 3. Documentacion de DiceBear, cache in-memory y app.py monolitico

Se documento en `RUNBOOK.md` porque son limitaciones reales del MVP, no bugs ocultos.

### DiceBear

Cuando un jugador no tiene foto, la app genera una URL de DiceBear para mostrar un avatar. El servidor no descarga esa imagen: solo guarda o renderiza una URL, y el navegador del usuario la carga.

Riesgo real:

- Si DiceBear no responde o no hay internet, puede no verse el avatar.
- La ficha del jugador y el resto del sistema siguen funcionando.

Por eso no se agrego timeout server-side: no hay llamada HTTP desde el backend.

### Cache in-memory

El dashboard usa una cache en memoria para no recalcular todo en cada request. Esto mejora rendimiento, pero tiene limites:

- Vive dentro del proceso Python.
- No se comparte entre varios workers o instancias.
- Tiene limite maximo configurable con `CACHE_MAX_ENTRIES`, por default `128`.

Si se supera el limite, la app descarta la entrada que vence antes. Para el volumen del MVP y `EVAL_POOL_MAX=100`, es aceptable. Para una version productiva, convendria usar una cache externa o politicas mas completas de expiracion/eviccion.

### app.py monolitico

`scouting_app/app.py` sigue concentrando muchas responsabilidades: rutas, seguridad, cache, pipeline, inferencia y parte de la logica de negocio.

Para el MVP academico se acepta para no abrir una refactorizacion grande. Para una version productiva, lo correcto seria separar:

- Rutas Flask en blueprints.
- Logica de negocio en servicios.
- Logica de ML/inferencia en modulos dedicados.
- Configuracion y seguridad en componentes separados.

## 4. Migracion de timestamps en physical_assessments y player_availability

Los modelos principales usan `TimestampMixin`, que agrega:

- `created_at`
- `updated_at`

El helper `ensure_player_columns()` ya podia agregar esas columnas a varias tablas legacy existentes. Ahora tambien las agrega a:

- `physical_assessments`
- `player_availability`

Esto importa porque esas tablas guardan informacion longitudinal del jugador: evaluaciones fisicas y disponibilidad. Si una base vieja no tenia esas columnas, al iniciar o preparar la base el helper puede completarlas sin borrar datos.

Tambien se agrego un test que crea tablas legacy sin timestamps y verifica que `ensure_player_columns()` agregue las columnas y complete valores iniciales.

## 5. Checkpoints del modelo con input_dim, version y model_state

Antes, `model.pt` guardaba directamente el `state_dict` de PyTorch. Eso guardaba los pesos del modelo, pero no incluia metadata minima como la dimension de entrada.

El problema es que el modelo depende de cuantas columnas produce el preprocesador. Si cambian las features, el modelo entrenado puede no coincidir con la inferencia.

Ahora el checkpoint nuevo guarda una estructura con:

- `version`: version del formato de checkpoint.
- `model_class`: clase del modelo, actualmente `PlayerNet`.
- `input_dim`: cantidad de features esperadas por el modelo.
- `model_state`: pesos reales del modelo.

Esto permite validar que el modelo cargado corresponde al preprocesador actual. Si el preprocesador produce otra cantidad de features, la carga falla con un mensaje claro en lugar de fallar mas tarde o comportarse de forma confusa.

## 6. Compatibilidad con checkpoints viejos

Como ya podia existir un `model.pt` viejo que fuera solo un `state_dict`, la carga mantiene compatibilidad hacia atras.

Eso significa:

- Si el archivo tiene el formato nuevo, se lee `input_dim` y se valida.
- Si el archivo tiene formato viejo, se trata como `state_dict` legacy y se intenta cargar igual.

Esto evita romper artefactos locales antiguos mientras se mejora el formato nuevo.

La compatibilidad se aplica tanto en:

- `app.py`, para inferencia dentro de la aplicacion.
- `evaluate_saved_model.py`, para evaluar artefactos entrenados desde script.

## 7. Warnings en metricas cuando no se pueden calcular

Algunas metricas de clasificacion no se pueden calcular en cualquier situacion. Por ejemplo:

- ROC-AUC necesita que existan dos clases reales.
- PR-AUC tambien necesita positivos y negativos.
- F1/precision/recall pueden fallar si los arrays vienen mal formados.

Antes, si algo fallaba, el codigo devolvia valores vacios sin explicar el motivo. Ahora `classification_metrics()` agrega una lista `warnings` con mensajes como:

```text
ROC-AUC no calculado: Se requieren al menos dos clases reales para ROC-AUC.
```

Esto no cambia las metricas cuando se pueden calcular. Solo hace mas transparente el caso en que no se pueden calcular.

## 8. Tests nuevos

Se agregaron pruebas para cubrir los cambios anteriores:

- Migracion de timestamps en `physical_assessments` y `player_availability`.
- Persistencia del checkpoint nuevo con `input_dim`.
- Warnings cuando ROC-AUC o PR-AUC no pueden calcularse.
- Matriz CSRF para POST mutantes criticos.
- Inputs invalidos en alta de jugador.
- Limite de entradas del cache in-memory.
- Constantes nombradas del rating de estadisticas.
- Fixture de app con nombre de modulo estable en `tests/conftest.py`.

Con estos tests, los cambios no quedan solo documentados: tambien quedan verificados automaticamente.

## 9. Constantes, cache limitado y conftest estable

En un bloque posterior se cerraron tres mejoras livianas del informe del profesor:

- Los literales visibles de paginacion (`50`), comparadores (`2000`) y pesos/rangos del rating de estadisticas pasaron a constantes nombradas.
- `PLAYER_LIST_PER_PAGE`, `MAX_COMPARE_PLAYERS`, `CACHE_TTL_SECONDS` y `CACHE_MAX_ENTRIES` pueden ajustarse por variables de entorno.
- `tests/conftest.py` ya no genera un nombre de modulo con UUID; usa `scouting_app_app_test` y limpia `sys.modules` antes de cargar la app.

Esto no cambia la regla funcional del MVP. Mejora lectura, configurabilidad y debugging.

## Resultado De Validacion

La validacion ejecutada despues de los cambios fue:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Resultado:

- `48 passed`
- Cobertura total reportada: `76%`
- `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN en tests

## Lectura General

Estos cambios no transforman el MVP en un sistema productivo completo. Lo que hacen es cerrar deuda concreta marcada por la revision:

- medir cobertura,
- documentar limitaciones reales,
- mejorar trazabilidad de migraciones,
- hacer mas seguro el artefacto del modelo,
- mantener compatibilidad con artefactos anteriores,
- hacer mas explicables los casos donde una metrica no aplica,
- y cerrar deuda chica de legibilidad/configuracion sin abrir una refactorizacion grande.
