# Cierre del bloque 6: Ruff y logging graduales

Fecha: 2026-10-14.

## Línea base y alcance

La medición completa de Ruff produjo 751 hallazgos. La mayoría correspondía a
modernización masiva de anotaciones: 341 `UP006`, 202 `UP045` y 48 `UP035`.
Aplicarlas habría generado un diff amplio sin mejorar el comportamiento del MVP.

Se agregó a CI un segundo control gradual sobre `scouting_app` y `tests` con las
reglas `F401`, `F841`, `RUF059`, `S110` y `S112`. El control crítico anterior
continúa sobre `scouting_app`, `tests` y `scripts`.

## Cambios

- Se eliminaron imports, asignaciones y desempaquetados sin uso en la aplicación
  y las pruebas.
- Los fallos esperados al resolver endpoints de navegación ahora capturan
  `BuildError` y se registran en nivel debug.
- Los fallos opcionales al consultar metadata de caché temporal se registran en
  nivel debug.
- Se retiró un `try/except` que ocultaba fallos del logger en settings.
- Dos advertencias operativas de entrenamiento pasaron de `print` a
  `logger.warning`.
- Se conservaron los `print` que constituyen la salida intencional de comandos
  CLI, demos y exportadores.
- Se corrigió `temporal_breakout_path`: ahora exige las condiciones breakout
  ajustadas y no clasifica automáticamente como breakout cualquier positivo que
  no cumpla la ruta de consolidación.
- La prueba del calibrador importa directamente `ml.runtime`, en lugar de depender
  de una reexportación accidental desde `app.py`.

## Verificación

- Control gradual Ruff: aprobado sin hallazgos.
- Ruff crítico: aprobado.
- `compileall`: aprobado.
- Prueba nueva de la ruta breakout: aprobada.
- Suite integral: 123 aprobadas, 1 omitida, 4 advertencias conocidas.
- Cobertura total: 83,29 %.

La medición completa queda en 732 avisos, principalmente modernización de tipos e
imports ordenados. No se los ocultó ni se redujo el alcance para declarar Ruff
completo. Cualquier ampliación futura debe hacerse por regla o módulo, con pruebas.

## Publicación y verificación remota

- Repositorio propio: commit `1804540bb056e053e29b03943e60ce95fb6d5d94`.
- CI Linux/Python 3.11 y 3.12 y auditoría de dependencias: ejecución
  `38648685553`, aprobada.
- macOS Apple Silicon/Python 3.11: ejecución `38648685581`, aprobada.
- Repositorio de entrega: sincronización funcional en
  `76390c31ad854c1ec0cd319092e2f508f18fb826` y cierre de calidad en
  `fb9de4be474faf610759461bd19376aae4e5c315`.
- CI de entrega: ejecución `38648888954`, aprobada.
- macOS de entrega: ejecución `38648888903`, aprobada.

Las páginas de comprobación de ambos commits mostraron estado global exitoso y
cero trabajos fallidos. Los artefactos internos, candidatos documentales y la
corrida ML versionada no se copiaron al repositorio de entrega.
