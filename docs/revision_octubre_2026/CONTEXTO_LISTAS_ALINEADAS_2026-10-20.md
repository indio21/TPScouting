# Continuación vigente — listas alineadas, 20/10/2026

## Documento final vigente

Carpeta interna `docs/revision_octubre_2026/listas_alineadas_2026-10-20/`:

- `TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_LISTAS_ALINEADAS_2026-10-20.docx`
- PDF con el mismo nombre.
- Copias idénticas y únicas en `C:\Tesis\TPScouting-entrega\documento`.
- Word SHA-256: `09E39E79BBFADB3154396FF91FCC606F53B9BD2D552ABC833FB263B8C9958EAF`.
- PDF SHA-256: `836082639D4A00EE4A315FA9E8D7F3ABDAB2161253E528F795E8AF5216B590FE`.

El usuario aprobó casi todo el documento anterior y pidió alinear los números
de figuras/diagramas y asegurar que la entrega esté actualizada.

## Corrección realizada

Las listas estáticas heredaban tabulaciones predeterminadas: los números de
página quedaban después del título, en posiciones distintas. Las 45 entradas
de figuras y tablas ahora tienen una tabulación derecha común con puntos guía
y reserva de espacio para evitar números aislados al ajustar títulos largos.

Se conservó el original aprobado, sin modificar imágenes ni texto del cuerpo.
No se modificó código funcional, modelos, métricas, requisitos ni tests.

## Verificaciones nuevas

- Las 45 coordenadas derechas de los números de página coinciden en el PDF.
- Control visual dirigido de las tres páginas de listas: 6, 7 y 8 físicas.
- 88 entradas del índice y 45 de figuras/tablas coinciden con páginas reales.
- 97 páginas, 28 imágenes, 19 tablas y 9 secciones: estructura conservada.
- Texto desde Resumen, contenido de tablas y hashes de imágenes idénticos.
- Fuentes textuales del PDF: únicamente Arial, Arial negrita y Arial cursiva.
- Sin tamaños Word menores de 10 pt, campos rotos, páginas vacías, contenido
  fuera de página ni leyendas de figuras sin imagen.
- Portada sin número, numeración continua y preliminares/bibliografía separados.
- Ruff crítico del nuevo script documental aprobado.
- Paridad de 85 archivos de aplicación/tests/dependencias/infraestructura entre
  principal y entrega. Única diferencia prevista: comentario de derechos
  reservados al inicio del Dockerfile; instrucciones ejecutables idénticas.
- Código de entrega idéntico al commit funcional verificado `2a27d53`.

## Publicación y evidencia

- Código funcional: `2a27d53ae4f8a4849bee7d1bea362fd9146886b5`.
- CI funcional `39509728373`, Container `39509728476` y macOS Apple Silicon
  `39509728473`: aprobados, según evidencia conservada del cierre anterior.
- CI de publicación anterior `39510545376`: éxito reconfirmado en esta revisión.
- Suite local anterior del mismo código: 131 passed, 1 skipped, 4 warnings,
  cobertura 83,31 %. No presentar ese registro como una nueva ejecución local.
- Publicación completa vigente de entrega:
  `0f7696b2b002ea149d81092aae921bed450ef6bb`.
- Se necesitaron dos commits documentales: `dcad76d` publicó el PDF y enlaces;
  `0f7696b` incorporó el Word, inicialmente excluido por `*.docx`, y una excepción
  exacta en `.gitignore`. El HEAD final incluye ambos archivos, comprobados.
- CI nueva del HEAD: `39513994230`, completada con resultado `success`.
  Pruebas Python 3.11, pruebas Python 3.12 y auditoría de dependencias aprobadas.
  Se comprobó el SHA completo del run contra el HEAD publicado.

Evidencia en `listas_alineadas_2026-10-20/`: `alineacion_e_integridad.json`,
`control_formato.json`, `verificacion_documento.json`, `paridad_codigo.json`,
imágenes de las listas y copia de `VERIFICACION_ENTREGA.md` público.
El registro final de publicación está guardado en `publicacion.json`.

El Word sigue citando el commit funcional probado. Su propia publicación queda
identificada externamente para evitar un hash circular dentro del documento.
README y el registro público apuntan a la nueva versión y sus hashes reales.

## Para retomar

Revisar nuevas observaciones del usuario/profesor. Las limitaciones sustantivas
previas siguen declaradas; ver `CONTEXTO_FINAL_ARIAL_2026-10-20.md` para detalle.
Preservar Arial y mínimo 10 pt, originales, hojas independientes y legajo omitido.
No regenerar imágenes ni reentrenar por iniciativa propia. No publicar evidencia
interna ni candidatos y no agregar masivamente archivos antiguos no seguidos.
