# TPScouting — contexto completo para reanudar

> **Actualización final, 2026-10-07:** el Bloque 8 quedó cerrado localmente. El
> documento autoritativo es
> `word/TRABAJO_FINAL_TPScouting_FINAL_BLOQUE8_2026-10-07.docx` y su PDF homónimo
> está en `pdf/`. Los hashes, pruebas, límites y el conjunto propuesto para una
> sincronización selectiva están en `cierre_bloque8_2026-10-07.md`. Permanecen
> pendientes el legajo, la sincronización/publicación autorizada y la CI del commit
> realmente entregado. Este aviso reemplaza las instrucciones antiguas que indican
> ejecutar el Bloque 8.

> **Regla documental permanente indicada por el usuario:** la portada, el índice
> general, el resumen, el abstract y la bibliografía deben comenzar en hojas
> separadas. No deben disponerse como contenido continuo en una misma página. Toda
> regeneración futura del DOCX/PDF debe conservar esta separación y comprobarla
> visualmente antes de considerarse final.

Fecha de corte: 2026-10-06 (America/Buenos_Aires).

> **Pausa más reciente:** antes de continuar, leer
> `docs/revision_octubre_2026/PAUSA_BLOQUE8_2026-10-06.md`. Allí están el
> candidato v4, sus hashes, lo verificado, lo pendiente y la estrategia indicada
> por el usuario: revisar primero una copia sin imágenes y regenerarlas al final.

## Instrucción de reanudación

Continuar con el **Bloque 8 — Cierre y entrega verificable** del plan de
`C:\Users\Usuario\Desktop\correccion-octubre.md`. No repetir los bloques 0–7.
No sincronizar el repositorio de entrega sin autorización específica.

Antes de editar, leer este archivo, el registro acumulado
`docs/revision_octubre_2026/registro_bloques.md`, la corrección de octubre y el
contexto histórico `docs/contexto_para_nuevo_chat.md`.

## Repositorios y alcance

- Proyecto de trabajo: `C:\Tesis\TPScouting`.
- Aplicación: `C:\Tesis\TPScouting\scouting_app`.
- Entrega del profesor: `C:\Tesis\TPScouting-entrega`.
- Python: `C:\Tesis\TPScouting\.venv\Scripts\python.exe`.
- Rama principal: `main`.
- Commit base previo: `49aa51c0167fdb24c5f2f3a6ab6e3f397830b462`.
- Commit de respaldo completo de los bloques 1–6: `1782dc9`
  (`checkpoint: complete October review blocks 1-6`).
- HEAD de entrega: `ffdefdf8035c994ae285a270de0a4ff4e9f336a8`.
- El repositorio de entrega está limpio y no fue sincronizado.

No hacer commit, push, deploy ni sincronización de entrega sin indicación del
usuario. No borrar ni revertir cambios preexistentes. Todos los cambios se hacen
primero en el proyecto principal. El MVP se limita a scouting juvenil de 12–18
años, datos sintéticos y ninguna validación con jugadores reales.

## Estado de los bloques

| Bloque | Estado al corte | Resultado principal |
|---|---|---|
| 0 | Completado | Inventario, contraste de repositorios y matriz inicial |
| 1 | Completado y probado | Seguridad de autenticación, `next`, sesión, rol, CSRF y rate limiting |
| 2 | Completado y probado | Health, CSP/HSTS, `photo_url`, artefactos confiables y operaciones productivas |
| 3 | Completado y probado | Dependencias, locks, auditoría, CI gradual y política Render |
| 4 | Completado y probado | Calidad puntual y pruebas; 116 passed, 1 skipped, 4 warnings; cobertura 83,74 % |
| 5 | Completado con D-16 parcial | Auditoría ML reproducible; no hubo reentrenamiento |
| 6 | Completado como borrador | Copia documental corregida; original aprobado intacto |
| 7 | Completado el 2026-10-06 | Redacción, fuentes y estructura corregidas en una nueva copia |
| 8 | Pendiente | Verificación final, índices, PDF, CI y eventual sincronización autorizada |

El detalle individual de IDs, evidencia, límites y pendientes está en
`docs/revision_octubre_2026/registro_bloques.md`. Ese archivo es el registro
interno acumulativo y debe seguir actualizándose al cerrar cada bloque.

## Evidencia técnica vigente

- Suite completa tras Bloque 4: **116 passed, 1 skipped, 4 warnings**.
- Cobertura: **83,74 %**.
- Ruff gradual y compilación: correctos.
- Parámetros entrenables de PlayerNet: **17.608**.
- Buffers: **386**.
- Elementos totales de `state_dict`: **17.994**.
- Features: 64 antes de encoding y 68 después de transformación.
- Splits persistidos: 14.000 train, 3.000 validation y 3.000 test, sin IDs
  compartidos.
- La prevalencia positiva 7,985 % fue fijada por el generador sintético; no es
  una observación poblacional.
- Bootstrap pareado, 2.000 remuestras, semilla de auditoría 20261005:
  - crudo menos regresión logística, ROC-AUC: media −0,00015071; IC95 %
    [−0,00269305; 0,00247069];
  - PR-AUC: media 0,00819354; IC95 % [−0,00250809; 0,01954283].
  Ambos intervalos incluyen cero: no hay evidencia suficiente de superioridad
  y el resultado tampoco prueba equivalencia.
- No se modificaron modelos, joblib, metadata, splits, bases ni experimentos.

Archivos de evidencia ML:

- `scripts/audit_ml_block5.py`.
- `docs/revision_octubre_2026/evidencia_ml_bloque5.json`.
- `docs/revision_octubre_2026/evidencia_ml_bloque5.md`.
- `docs/revision_octubre_2026/plan_reentrenamiento_pendiente.md`.

## Documento de trabajo vigente

Fuente aprobada, que no se debe sobrescribir:

`C:\Users\Usuario\Desktop\TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_REVISADA_26-08-2026_v2.docx`

SHA-256 de la fuente aprobada:

`77261990098b6fd0761c7e1d27fcaf8cb91bef8fe13346c39836075e6049955d`

Backup exacto:

`docs/revision_octubre_2026/word/ORIGINAL_APROBADO_26-08-2026_v2_BACKUP_SHA77261990.docx`

Borrador del Bloque 6, conservado como punto de recuperación:

`docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_CORREGIDO_OCTUBRE_2026_BORRADOR.docx`

Documento vigente para continuar con el Bloque 8:

`docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_CORREGIDO_BLOQUE7_2026-10-06.docx`

Script reproducible que genera el borrador del Bloque 6:

`scripts/correct_word_block6_october_2026.py`

El documento corregido conserva 22 tablas, 10 secciones, 27 objetos gráficos y
20 archivos multimedia. El ZIP interno es válido y se reabre con `python-docx`.
La automatización Word COM no respondió en 30 segundos; el proceso oculto creado
para esa prueba fue cerrado sin guardar. La exportación y revisión visual del PDF
corresponden al Bloque 8.

El abstract en inglés fue incorporado porque la autorización actual incluyó el
Bloque 6, aunque anteriormente había sido omitido por decisión del usuario. El
legajo no fue suministrado y figura como `[PENDIENTE DE INFORMAR]`.

## Bloque 7 completado

La redacción, las fuentes y la estructura fueron revisadas el 2026-10-06. El
detalle y la evidencia se encuentran en `registro_bloques.md` y
`auditoria_fuentes_bloque7.md`. El DOCX conserva 10 secciones, 27 objetos
gráficos y los mismos 20 archivos multimedia; contiene 19 tablas después de
eliminar tres comparativas de tecnologías no utilizadas.

## Alcance exacto del próximo bloque

El próximo paso es el **Bloque 8 — Cierre y entrega verificable**. Debe ejecutarse
sólo cuando el usuario lo indique. Incluye:

1. Ejecutar suite y cobertura finales con fecha, commit, versiones y límites.
2. Probar la demo desde entorno y base vacíos.
3. Actualizar de forma controlada índices, lista de figuras y lista de tablas.
4. Exportar y revisar visualmente el PDF completo.
5. Preparar el diff selectivo para la entrega; sincronizar sólo con autorización.
6. Usar CI del commit efectivamente entregado o mantener el punto abierto.
7. Cerrar la matriz sólo cuando cada hallazgo tenga evidencia.

## Pendientes que no deben perderse

- Legajo del alumno: falta información del usuario; no inventarlo.
- D-16: duración, validation loss, versiones históricas y SHA del entrenamiento
  original no pueden reconstruirse. Existe plan de nueva corrida, no ejecutado.
- El score combinado no tiene evaluación propia en test.
- Application factory: evaluada y diferida; requiere autorización específica
  por ser un refactor grande.
- Dockerfile y `.env.example`: opcionales.
- LICENSE: no seleccionar sin decisión del usuario.
- CI del árbol corregido: pendiente de publicación.
- Commit final, sincronización selectiva, demo en entorno vacío, actualización
  de índices y listas, exportación/revisión visual del PDF: Bloque 8.
- No usar la CI histórica del desarrollo para certificar la entrega.
- No presentar el uso de datos reales de menores como realizado. Cualquier uso
  futuro exige consentimiento, política de retención/baja, evaluación ética y
  análisis de sesgos.

## Significado del veredicto

El veredicto de revisión actual es **REQUIERE CORRECCIONES IMPORTANTES** porque
el trabajo completo todavía no está listo para entregar: faltan los bloques 7 y
8 y permanecen pendientes verificables como el legajo, la CI final, los índices,
la revisión visual del PDF y la sincronización autorizada. No significa que las
correcciones no se estén realizando. Los bloques 1–6 ya fueron corregidos y
verificados dentro de su alcance. El veredicto debe reevaluarse al cerrar la
matriz completa en el Bloque 8; no se cambia antes sólo por avance parcial.

## Comando inicial recomendado para la próxima sesión

Solicitar: **“Continuá desde `docs/revision_octubre_2026/CONTINUAR_AQUI_2026-10-05.md` y ejecutá únicamente el Bloque 8.”**

Al terminar el Bloque 8, informar IDs resueltos, archivos afectados, pruebas o
evidencia, límites, pendientes y próximo bloque. Actualizar el registro interno y
detenerse.
