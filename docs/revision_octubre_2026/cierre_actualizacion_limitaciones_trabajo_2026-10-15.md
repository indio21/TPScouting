# Actualización de limitaciones y trabajo futuro

Fecha: 2026-10-15.

## Documento base

La actualización se realizó sobre
`TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_VERIFICADA_2026-10-10.docx`.
Su SHA-256 es
`D7FFBCE018C9FD438012083E48F053618C9E14E67A30AB249412A02522D4C946`,
idéntico al del documento presente en el repositorio de entrega con el nombre
`TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-10.docx`.

Esta versión es posterior a `TRABAJO_FINAL_TPScouting_FINAL_POSTAUDITORIA_2026-10-10.docx`.
La copia anterior todavía conservaba referencias previas a la incorporación de
PyTorch 2.14.1 y otros identificadores anteriores de entrega y CI. Por ese motivo
no se utilizó como base.

## Cambios incorporados

Se creó una copia nueva, sin sobrescribir el documento verificado. En las
secciones 7.3 y 7.5 se registraron como limitaciones o trabajo futuro:

- los 732 avisos de Ruff que permanecen como deuda técnica controlada;
- la evaluación y postergación del refactor hacia una application factory;
- la falta de soporte para macOS Intel con la versión fijada de PyTorch;
- los controles todavía no implementados para datos reales de menores;
- los metadatos irrecuperables de la corrida histórica y su registro prospectivo.

También se actualizaron cuatro párrafos que habían quedado desfasados respecto
de la evidencia posterior: la corrida candidata sin fuga por cuantiles, la
evaluación independiente del score combinado, la trazabilidad de la nueva
corrida y la discusión de resultados. La probabilidad cruda continúa como salida
principal y el score combinado queda descrito como heurística secundaria.

## Artefactos resultantes

- Word: `word/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ACTUALIZADA_2026-10-15.docx`.
- PDF: `pdf/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ACTUALIZADA_2026-10-15.pdf`.
- Evidencia automática: `verificacion_documento_actualizado_2026-10-15.json` y
  `verificacion_estructura_actualizada_2026-10-15.json`.
- Script reproducible: `scripts/update_final_word_future_work_2026_10_15.py`.

## Verificación

- exportación de Word completada: 95 páginas;
- 28 imágenes, 19 tablas y 9 secciones conservadas;
- 45 entradas de listas verificadas, sin diferencias de página;
- sin errores de campos, elementos editoriales pendientes ni páginas con texto
  anormalmente escaso;
- portada, índice, resumen, abstract y bibliografía comienzan en páginas
  independientes;
- estilo normal Arial 14 conservado, también en los párrafos agregados;
- sin dobles espacios, espacios antes de puntuación, primera o segunda persona;
- sin palabras consecutivas repetidas;
- el único candidato de 45 palabras o más corresponde al bloque reproducible de
  comandos por plataforma, no a una oración académica;
- las imágenes no se regeneraron ni se volvieron a analizar en esta actualización.

No se volvió a ejecutar la suite de la aplicación porque no se modificó código de
runtime. El repositorio de entrega no fue modificado. La nueva copia debe pasar
primero por revisión del titular antes de una eventual sincronización selectiva.
