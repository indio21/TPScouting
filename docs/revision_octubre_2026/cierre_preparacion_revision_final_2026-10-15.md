# Preparación para la revisión final del documento

Fecha: 2026-10-15.

## Decisión de derechos

El titular decidió no incorporar un archivo `LICENSE` en esta entrega. Todos los
derechos se encuentran reservados. README, guía de evaluación y Trabajo Final
aclaran que la disponibilidad del repositorio permite su revisión académica, pero
no concede una autorización general de copia, modificación o redistribución.

El proyecto no se presenta como código abierto y no se generó una licencia en
nombre del titular.

## Evidencia actualizada en el Trabajo Final

La nueva copia reemplaza referencias técnicas que habían quedado desactualizadas:

- suite local del 15/10/2026: 123 pruebas aprobadas, 1 omitida y 4 advertencias
  conocidas;
- cobertura total: 83,29 %;
- commit funcional del repositorio propio: `754ccb3601a489ed1249a078ad700d1eb546a494`;
- CI propia: `38655727560`; smoke de contenedor propio: `38655727631`;
- commit funcional de entrega: `8093fe616c9c50b72078ff4f241d9ab3427e763d`;
- CI de entrega: `38656130065`; smoke de contenedor de entrega: `38656129892`;
- soporte verificado para Windows, Linux y macOS Apple Silicon; macOS Intel
  permanece fuera del soporte con PyTorch 2.14.1;
- inclusión comprobada de `Dockerfile`, `.dockerignore` y `.env.example`;
- decisión definitiva de no incorporar `LICENSE` y reservar todos los derechos.

Las menciones anteriores de 116 pruebas, 83,74 % de cobertura, el commit
`72314c2` y la CI `38068749412` ya no aparecen en la nueva copia.

## Documento preparado

- Word: `word/TRABAJO_FINAL_TPScouting_PRE_REVISION_FINAL_2026-10-15.docx`.
- PDF: `pdf/TRABAJO_FINAL_TPScouting_PRE_REVISION_FINAL_2026-10-15.pdf`.
- Verificación documental: `verificacion_documento_pre_revision_final_2026-10-15.json`.
- Evidencia integral: `evidencia_pre_revision_final_2026-10-15.json`.
- Cobertura: `evidencia_revision_final_coverage.xml`.
- Generación reproducible: `scripts/prepare_thesis_final_review_2026_10_15.py`.

## Verificación

- 96 páginas, 28 imágenes, 19 tablas y 9 secciones;
- 45 entradas de listas sin diferencias de página;
- sin errores de campos ni elementos editoriales pendientes;
- portada, índice, resumen, abstract y bibliografía en páginas independientes;
- sin dobles espacios, espacios antes de puntuación, primera o segunda persona;
- sin palabras consecutivas repetidas;
- el único candidato de oración extensa es el bloque de comandos por plataforma;
- Ruff crítico y gradual, `pip check` y `compileall`: aprobados;
- suite completa y cobertura: aprobadas;
- imágenes conservadas sin volver a analizarlas.

No se cambió el comportamiento de la aplicación, no se reemplazaron artefactos
de runtime y no se modificó el repositorio de entrega.

## Próxima tarea

Ejecutar el punto 4: revisión final integral del documento actualizado. Esa etapa
debe revisar redacción, coherencia interna, evidencia, paginación y presentación
final antes de decidir una sincronización selectiva con el repositorio de entrega.
