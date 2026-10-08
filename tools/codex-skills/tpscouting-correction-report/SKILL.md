---
name: tpscouting-correction-report
description: "Genera o actualiza informes de corrección integral de TPScouting, similares a correccion-octubre.md, mediante auditoría cruzada del Word/PDF, código, Git, pruebas, dependencias, datos y ML. Úsala cuando se necesite un nuevo diagnóstico independiente y trazable; no para aplicar automáticamente las correcciones."
metadata:
  version: "1.0.0"
  language: "es"
  domain: "thesis-audit-report"
---

# TPScouting Correction Report

Produce un diagnóstico técnico-académico reproducible. El resultado debe permitir
que otra persona confirme cada hallazgo y convierta luego el informe en un plan de
corrección.

## Límite de la tarea

Trabaja en modo de solo lectura salvo que el usuario pida expresamente aplicar
correcciones. Auditar no autoriza a editar el documento, modificar código, entrenar,
alterar bases, hacer commit, publicar ni desplegar.

No supongas que el informe anterior sigue vigente. Úsalo como lista de riesgos e
hipótesis; vuelve a comprobar sus ubicaciones, cifras, versiones y estados sobre
los artefactos actuales.

## Contrato de evidencia

- No inventes resultados, citas, líneas, páginas, vulnerabilidades, métricas ni
  comandos ejecutados.
- Identifica documento, cantidad de páginas, hash, repositorio, rama, commit y
  fecha de revisión antes de emitir el veredicto.
- Etiqueta internamente cada afirmación como `REPRODUCIDO`, `INSPECCIONADO`,
  `DOCUMENTADO`, `INFERIDO`, `NO VERIFICADO` o `CONTRADICCIÓN`.
- En el informe, expresa el método cuando importe: “verificado ejecutando”, “por
  inspección”, “según el documento” o “no comprobado”.
- Separa hallazgos del documento, del código y de la coherencia entre ambos.
- Una prueba aprobada demuestra solo el comportamiento cubierto por esa prueba.
- Para leyes, servicios, dependencias, vulnerabilidades o documentación que pueda
  haber cambiado, consulta fuentes oficiales vigentes y fecha la comprobación.
- Si no está disponible el artefacto rector, detén únicamente esa parte y señala
  qué falta. Continúa con lo que sí pueda demostrarse.

## Procedimiento

Lee [references/audit-method.md](references/audit-method.md) antes de una auditoría
integral. Ejecuta primero la fase de identidad y después las verificaciones que
correspondan al alcance real.

Lee [references/report-template.md](references/report-template.md) al redactar el
entregable. Conserva la estructura A–F cuando facilite comparar revisiones. Asigna
un ID estable a cada hallazgo; nunca escondas varios problemas bajo un único estado.

Prioriza, en este orden:

1. defectos explotables o pérdida/corrupción de datos;
2. afirmaciones centrales falsas o métricas no reproducibles;
3. divergencias tesis–código y falta de trazabilidad;
4. instalación, demo, pruebas, CI y despliegue;
5. validez de datos y ML;
6. objetivos, metodología, ética y conclusiones;
7. estructura, citas, redacción y formato.

## Ejecución segura

- Usa bases temporales para pruebas y demos; no abras con escritura las bases del
  usuario si no es imprescindible.
- No reentrenes para “rellenar” metadata histórica. Propón la nueva corrida con
  costo, seed, artefactos y separación de resultados históricos.
- No instales ni actualices dependencias para hacer pasar una auditoría sin dejar
  constancia. Inspecciona primero el entorno disponible.
- Inspecciona imágenes solo para presencia, caption, referencia y legibilidad
  evidente, salvo pedido de auditoría visual. No rasterices todas las páginas.
- Resume salidas extensas y conserva el comando, fecha, versión y resultado útil.

## Veredicto

Usa exactamente uno:

- `NO LISTO PARA ENTREGA`
- `REQUIERE CORRECCIONES IMPORTANTES`
- `REQUIERE CORRECCIONES MENORES`
- `LISTO PARA REVISIÓN FINAL HUMANA`

No uses el último si existe un crítico abierto, una métrica central sin respaldo,
una contradicción material entre tesis y código, una fuente esencial dudosa o una
corrección docente crítica pendiente. Un informe puede aprobar el código y exigir
correcciones al documento por separado.

## Cierre

Entrega el informe `.md` y una síntesis breve con artefactos revisados, comandos
realmente ejecutados, límites, bloqueantes y próximos pasos. Si el usuario pidió
solo el informe, no empieces a corregirlo.
