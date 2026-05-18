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
   - Estado: aceptado para MVP.
   - Accion sugerida Fase 3: verificar transacciones/commits y dejar documentada la
     limitacion de migraciones manuales.

2. `sync_shortlist` y base operativa.
   - Estado: pendiente de revision Fase 3.
   - Accion sugerida: revisar limites de sincronizacion, edad/categoria y no superar
     `EVAL_POOL_MAX`.

3. Cache in-memory.
   - Estado: limitado por TTL y cantidad maxima.
   - Accion sugerida: revisar invalidaciones principales y documentar que no es
     compartido entre procesos.

4. Modelo ML y metricas.
   - Estado: checkpoints con `input_dim` y warnings de metricas ya existen.
   - Accion sugerida: revisar consistencia de dimension entrenamiento/inferencia,
     uso de `SEED` y logging de errores/fallbacks.
