# Cierre del bloque 7: evaluación de application factory

Fecha: 2026-10-14.

## Veredicto

**NO IMPLEMENTAR EN EL MVP ACTUAL.** La migración a una application factory es
técnicamente posible, pero su alcance y acoplamiento actuales implican un riesgo
alto de regresión frente al beneficio inmediato. La aplicación ya dispone de
blueprints por dominio y supera sus verificaciones funcionales. Para la entrega,
conviene conservar la arquitectura ejecutable y tratar el refactor como una
evolución posterior, con etapas y pruebas de caracterización propias.

Esta decisión resuelve la evaluación solicitada en CAL-01. No afirma que la
arquitectura actual sea ideal ni marca el refactor como implementado.

## Evidencia verificada

- `scouting_app/app.py` tiene 1.755 líneas, 85 funciones y 72 asignaciones en el
  nivel de módulo.
- En la importación se ejecutan 17 llamadas con efectos: configuración de
  logging, creación/ajuste de tablas, bootstrap de administrador, limpieza del
  pool operativo y registro de blueprints y alias.
- El proceso registra 85 reglas Flask y seis blueprints: `auth`, `compare`,
  `dashboard`, `players`, `settings` y `staff`.
- La base, `Session`, cache, rate limiter, lock, modelo, preprocesador y
  calibrador viven como estado global del módulo.
- Cuatro blueprints reciben dependencias mediante `SimpleNamespace`. Esa
  inyección reduce el acoplamiento interno de las rutas, pero las dependencias se
  construyen actualmente desde los objetos globales de `app.py`.
- Las pruebas contienen 218 referencias a `app_module.*`; varias sustituyen
  directamente `Session`, `engine`, cache o artefactos ML.
- Una importación caracterizada con bases temporales tardó 3,108 s, creó la base
  durante el import, cargó los tres artefactos ML y respondió `/health` con 200.
- El motor SQLite debió cerrarse con `engine.dispose()` antes de retirar la base
  temporal. Esto confirma que el ciclo de vida de recursos debe diseñarse antes
  de introducir múltiples instancias de aplicación.

Los tiempos son una observación local de esta ejecución, no un benchmark de
rendimiento ni una garantía para otras plataformas.

## Riesgos de una migración directa

1. Crear dos aplicaciones en el mismo proceso podría compartir cache, rate
   limiter, modelo, `Session` o locks sin aislamiento real.
2. Mover la inicialización de la base podría cambiar el arranque de Gunicorn, la
   demo, los scripts y las pruebas.
3. Diferir la carga ML podría alterar errores de inicio y rutas de predicción.
4. Reubicar bootstrap y mantenimiento operativo podría ejecutarlos cero veces o
   múltiples veces por worker.
5. Los alias de endpoints conservan compatibilidad con templates y pruebas; una
   reorganización puede romper `url_for` aunque las vistas sigan existiendo.
6. Un cambio transversal dificultaría atribuir una regresión y ampliaría el diff
   de una entrega que ya tiene evidencia reproducible.

## Diseño propuesto para una evolución futura

El refactor debe autorizarse como trabajo independiente y realizarse en etapas:

1. **Caracterizar interfaces:** guardar un snapshot de reglas/endpoints, headers,
   handlers, context processors y configuración de desarrollo/producción.
2. **Separar configuración pura:** hacer que los lectores de entorno no dependan
   de `app.logger` ni creen recursos.
3. **Definir un contenedor de runtime:** agrupar engine, `Session`, cache, limiter,
   lock y artefactos ML con `close()`/`dispose()` explícitos.
4. **Crear `create_app(config=None, runtime=None)`:** construir Flask y registrar
   hooks, handlers, blueprints y alias sin bootstrap implícito.
5. **Mover tareas de arranque:** exponer inicialización de esquema, administrador
   y mantenimiento como comandos o funciones idempotentes con política clara por
   proceso/worker.
6. **Adaptar puntos de entrada:** conservar temporalmente `app = create_app()`
   para Gunicorn y la demo, y migrar pruebas a fixtures de aplicaciones aisladas.
7. **Retirar compatibilidad global:** hacerlo solo después de aprobar dos
   instancias simultáneas con bases, caches y artefactos independientes.

## Criterios previos para autorizar la implementación

- snapshot aprobado de las 85 reglas y nombres de endpoint;
- pruebas de dos aplicaciones simultáneas sin estado compartido;
- bases temporales y cierre verificable de engines/sesiones;
- comportamiento definido de carga y fallo de artefactos ML;
- bootstrap y mantenimiento separados del import;
- demo, Gunicorn, CI Linux y macOS cubiertos;
- migración dividida en commits reversibles, sin mezclar cambios funcionales.

## Alcance aplicado

No se modificó código, configuración, base, modelo, Word ni repositorio de
entrega. El resultado del bloque es una decisión arquitectónica sustentada y un
plan verificable. El siguiente bloque es el diseño de controles legales y
operativos previos a cualquier uso de datos reales de menores.

## Verificación de cierre

- Ruff crítico (`E9,F63,F7,F82`): aprobado.
- Ruff gradual (`F401,F841,RUF059,S110,S112`): aprobado.
- `compileall` de aplicación, pruebas y scripts: aprobado.
- Import y arranque con bases SQLite temporales: aprobado; 85 reglas, seis
  blueprints, artefactos ML cargados y `/health` con estado 200.
- Liberación explícita del engine y eliminación de la base temporal: aprobadas.
- Suite integral: 123 aprobadas, 1 omitida y 4 advertencias conocidas de
  scikit-learn por columnas completamente vacías en dos escenarios sintéticos.
- Cobertura total: 83,29 %, por encima del umbral de 80 %.

No hay evidencia de una regresión funcional en este bloque. La prueba confirma el
estado actual; no reduce el riesgo estimado de una futura migración transversal.
