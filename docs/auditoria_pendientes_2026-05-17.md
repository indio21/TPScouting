# Pendientes de auditoria tecnica - actualizado 2026-05-18

Este archivo registra riesgos que quedaron vivos durante las fases de correccion,
para atacarlos en pasos controlados sin perder contexto.

## Estado de avance

- Rama de trabajo: `auditoria-correcciones-mvp`.
- Fase 1 completada: guardrail contra SQLite en produccion, umbrales de potencial
  robustos y lock de pipeline con archivo.
- Fase 2 completada: logout por POST + CSRF, headers basicos de seguridad,
  tests de secret key/admin password/CSRF y documentacion de rate limiting.
- Tests al cierre de Fase 2: `76 passed`, cobertura total `80%`.
- Tests al cierre de Fase 3: `79 passed`, cobertura total `80%`.
- Fase 4 en curso: CI publica `coverage.xml` como artefacto y se refuerzan tests
  de rutas protegidas, inputs invalidos y comportamiento sin modelo cargado.
- Tests al cierre de Fase 4: `82 passed`, cobertura total `80%`.

## Riesgos restantes despues de Fase 1

1. Smoke real de Render con datos cargados.
   - Estado: pendiente.
   - Motivo: la app usa PostgreSQL en `render.yaml`, pero falta evidencia de una
     prueba completa en cloud con base operativa poblada.
   - Accion sugerida: documentar URL, fecha, `/health`, login y dashboard con datos.

2. Lock de pipeline apto para MVP, no distribuido.
   - Estado: mitigado para MVP.
   - Motivo: se agrego lock por archivo, suficiente para una instancia Render, pero
     no reemplaza una cola o lock distribuido para produccion real multi-instancia.
   - Accion sugerida: mantener `--workers 1` en Render y dejar lock distribuido
     para version futura.

3. Documentacion final pendiente.
   - Estado: parcialmente actualizada; cierre integral pendiente para Fase 5.
   - Motivo: las limitaciones de despliegue, cache, lock, dataset sintetico y score
     combinado deben quedar reflejadas en README/RUNBOOK/documento academico.
   - Accion sugerida: actualizar documentacion al cerrar las fases tecnicas.

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
   - Estado: documentado para confirmar con el documento academico final.
   - Evidencia: el codigo actual valida atributos `0-20` en `player_logic.py` y
     mensajes de `routes/players.py`; la fase de testing refuerza ese contrato.
   - Accion sugerida: si el documento final afirma `1-20`, alinear documento o
     cambiar regla funcional en una fase separada porque impacta generacion de
     datos sinteticos, validaciones y textos de UI.

2. Cobertura CI.
   - Estado: mejorado.
   - Accion aplicada Fase 4: GitHub Actions genera `coverage.xml` y lo sube como
     artefacto por version de Python.
