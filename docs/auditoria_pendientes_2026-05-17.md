# Pendientes de auditoria tecnica - actualizado 2026-05-18

Este archivo registra riesgos que quedaron vivos durante las fases de correccion,
para atacarlos en pasos controlados sin perder contexto.

## Estado de avance

- Rama de trabajo original: `auditoria-correcciones-mvp`.
- Rama actual mergeada: `ux-crud-polish`.
- Fase 1 completada: guardrail contra SQLite en produccion, umbrales de potencial
  robustos y lock de pipeline con archivo.
- Fase 2 completada: logout por POST + CSRF, headers basicos de seguridad,
  tests de secret key/admin password/CSRF y documentacion de rate limiting.
- Tests al cierre de Fase 2: `76 passed`, cobertura total `80%`.
- Tests al cierre de Fase 3: `79 passed`, cobertura total `80%`.
- Fase 4 completada: CI publica `coverage.xml` como artefacto y se refuerzan tests
  de rutas protegidas, inputs invalidos y comportamiento sin modelo cargado.
- Tests al cierre de Fase 4: `82 passed`, cobertura total `80%`.
- Tests al cierre del bloque escala/smoke: `83 passed, 1 skipped`, cobertura total `80%`.
- Fase 5 completada en alcance MVP: documentacion actualizada, constantes nombradas adicionales y type hint puntual sin refactor grande.
- Cierre pre-entrega 2026-05-18: queda explicitado que Word final alineado y smoke Render real son los dos cierres necesarios para entrega academica.

## Riesgos restantes despues de Fase 1

1. Smoke real de Render con datos cargados.
   - Estado: pendiente.
   - Motivo: la app usa PostgreSQL en `render.yaml`, pero falta evidencia de una
     prueba completa en cloud con base operativa poblada.
   - Accion sugerida: desplegar en Render, documentar URL, fecha, commit, `/health`, login, dashboard, jugadores, comparadores y una accion CRUD minima.

2. Lock de pipeline apto para MVP, no distribuido.
   - Estado: mitigado para MVP.
   - Motivo: se agrego lock por archivo, suficiente para una instancia Render, pero
     no reemplaza una cola o lock distribuido para produccion real multi-instancia.
   - Accion sugerida: mantener `--workers 1` en Render y dejar lock distribuido
     para version futura.

3. Documentacion final pendiente.
   - Estado: README/RUNBOOK/contexto/auditoria actualizados en Fase 5.
   - Motivo: las limitaciones de despliegue, cache, lock, dataset sintetico y score
     combinado deben quedar reflejadas en README/RUNBOOK/documento academico.
   - Accion sugerida: alinear el documento final de tesis con el MVP real.

## Riesgos/limitaciones registrados durante Fase 2

1. Rate limiting de login no distribuido.
   - Estado: documentado.
   - Motivo: `LoginRateLimiter` vive en memoria del proceso Flask/Gunicorn.
   - Accion sugerida: si el proyecto pasa de MVP a produccion, usar Redis, DB o
     un servicio externo de rate limiting.

2. CSRF manual por ruta.
   - Estado: mitigado con tests.
   - Motivo: los POST mutantes llaman `require_csrf()` manualmente.
   - Accion sugerida: mantener tests de matriz CSRF; post-MVP evaluar proteccion
     global tipo Flask-WTF.

## Riesgos pendientes para continuar desde Fase 3

1. Migraciones manuales sin Alembic.
   - Estado: aceptado para MVP y documentado.
   - Accion aplicada Fase 3: se verifico que `ensure_player_columns` usa
     `engine.begin()` y queda documentado como migracion manual transaccional.
   - Riesgo restante: no reemplaza un sistema formal de migraciones como Alembic.

2. `sync_shortlist` y base operativa.
   - Estado: mitigado para MVP.
   - Accion aplicada Fase 3: `sync_shortlist.py` valida `limit`, `min_age` y
     `max_age`, y devuelve un resumen verificable de sincronizacion.
   - Riesgo restante: si se usa sin `replace`, no borra jugadores operativos
     existentes que excedan el limite; el recorte final sigue en el pipeline/app.

3. Cache in-memory.
   - Estado: documentado.
   - Accion aplicada Fase 3: se explicito en RUNBOOK que el cache tiene TTL/max
     entradas y no se comparte entre procesos.
   - Riesgo restante: para produccion multi-instancia se requiere Redis/cache externo.

4. Modelo ML y metricas.
   - Estado: mitigado para MVP.
   - Accion aplicada Fase 3: checkpoints guardan `input_dim` y `seed`; la carga
     valida `input_dim`; el DataLoader/sampler usa seed explicito; metricas,
     calibradores y runtime registran warnings cuando caen a fallback.
   - Riesgo restante: la validacion predictiva sigue dependiendo de dataset sintetico.

## Riesgos/decisiones registrados durante Fase 4

1. Rango de atributos.
   - Estado: corregido en rama `auditoria-correcciones-mvp`.
   - Evidencia: el codigo valida atributos tecnicos, campos fisicos en escala y
     reportes scout con rango `1-20`; `ensure_player_columns` normaliza valores
     heredados por debajo de 1 o por encima de 20.
   - Accion restante: verificar que el documento academico final editable diga
     siempre `1-20` para atributos y reportes en escala.

2. Cobertura CI.
   - Estado: mejorado.
   - Accion aplicada Fase 4: GitHub Actions genera `coverage.xml` y lo sube como
     artefacto por version de Python.

## Ajustes posteriores a Fase 4

1. Escala de atributos `1-20`.
   - Estado: corregido.
   - Accion aplicada: `player_logic.py` define rango unico `1-20`; formularios,
     importacion CSV, generador sintetico, reportes scout, campos fisicos en escala,
     tests y documentacion fueron alineados.
   - Accion aplicada: `ensure_player_columns` normaliza bases heredadas con valores
     menores a 1 o mayores a 20 sin tocar `NULL` en campos opcionales.
   - Accion local: se hicieron backups `*.db.before_attr_scale_1_20_YYYYMMDD_HHMMSS`
     y se normalizaron `players_updated_v2.db` y `players_training.db` de la demo local.

2. Smoke visual Playwright.
   - Estado: incorporado como prueba opt-in.
   - Accion aplicada: `tests/test_visual_smoke.py` abre Chromium real cuando
     `RUN_PLAYWRIGHT=1` y valida login, dashboard y responsive basico.
   - Riesgo restante: no queda activo por defecto en CI para evitar peso/tiempo de
     navegadores en la suite comun.

3. Smoke real de Render.
   - Estado: incorporado como script ejecutable.
   - Accion aplicada: `scripts/smoke_render.py` valida `/health`, `/login` y
     opcionalmente `/dashboard` autenticado usando `RENDER_SMOKE_BASE_URL`.
   - Riesgo restante: no se encontro una URL publica fija de Render en el repo; se
     debe ejecutar con la URL real del servicio desplegado.

## Cierre de Fase 5 - Calidad y documentacion

1. Documentacion operativa.
   - Estado: actualizada.
   - Accion aplicada: `README.md`, `RUNBOOK.md`, `README_TESTS.md`,
     `REVISION_FINAL_MVP.md`, `PROGRESO_MVP.md`, `docs/contexto_para_nuevo_chat.md`
     y documentos tecnicos de entrenamiento fueron alineados con `ux-crud-polish`,
     escala `1-20`, tests actuales y smoke Playwright/Render.

2. Calidad de codigo sin refactor grande.
   - Estado: aplicado.
   - Accion aplicada: se agregaron constantes nombradas para cupo operativo,
     cantidad default de dataset, vencimiento del lock de pipeline, umbral de
     sugerencias y bandas de score; se agrego type hint puntual a `sort_key`.

3. Riesgos que quedan vivos.
   - Smoke real de Render pendiente hasta contar con URL real del servicio.
   - Rate limiting, cache y lock siguen siendo in-memory/locales, aceptables para
     MVP pero no para produccion multi-instancia.
   - No hay Alembic; las migraciones manuales transaccionales alcanzan para MVP.
   - Dataset y evaluacion del modelo siguen siendo sinteticos; falta validacion
     externa con datos reales.
   - El documento final de tesis debe revisar narrativa, capturas y afirmaciones
     para que coincidan con el MVP real.

## Cierre pre-entrega Word/Render - 2026-05-18

1. Documento Word final.
   - Estado: pendiente hasta que el usuario entregue el Word de primera entrega y
     el Word con observaciones del profesor.
   - Accion sugerida: comparar cada afirmacion contra codigo, tests, docs y
     evidencia real. Corregir especialmente modelo ML, dataset sintetico,
     arquitectura parcial, seguridad MVP, despliegue y limitaciones.
   - Efecto: corrige coherencia academica, pero no reemplaza validaciones tecnicas.

2. Deploy real en Render.
   - Estado: smoke real ejecutado el 2026-05-19; queda pendiente prueba manual CRUD minima.
   - Accion sugerida: publicar la rama final, crear/validar admin, crear una nueva
     base Free si la anterior expiro y ejecutar `scripts/smoke_render.py` con
     `RENDER_SMOKE_BASE_URL`.
   - Ajuste aplicado: para Render Free se prepara una sola base PostgreSQL y seed
     demo idempotente, porque no conviene depender de dos bases gratuitas.
   - Efecto: cierra variables reales, PostgreSQL administrado, Gunicorn, HTTPS,
     healthcheck y rutas protegidas.
   - Evidencia: `https://tpscouting-mvp.onrender.com`, `/health` con `database=ok`
     y `players_total=100`; login admin y `/dashboard` OK; rutas principales y
     comparadores respondieron `200`.

3. Condicion de entrega.
   - Con suite local/CI vigente, Word final alineado y smoke Render documentado,
     el proyecto queda defendible como MVP academico entregable.
   - No se debe presentar como producto listo para produccion.
