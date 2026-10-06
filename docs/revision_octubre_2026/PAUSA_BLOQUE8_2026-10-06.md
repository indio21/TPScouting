# Punto de reanudación — Bloque 8

Fecha de pausa: 2026-10-06.

## Estado seguro

- Repositorio principal: `C:\Tesis\TPScouting`.
- HEAD al iniciar el bloque: `979325ae9c4cc5119f191673a7ca03016d260aa5`.
- No se realizó commit, push, despliegue ni sincronización con la entrega.
- `C:\Tesis\TPScouting-entrega` no fue modificado.
- No quedó ningún proceso de Microsoft Word abierto.

## Último candidato comprobado

- DOCX: `docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_CANDIDATO_FINAL_BLOQUE8_v4_2026-10-06.docx`
  - 2.756.294 bytes.
  - SHA-256: `F66EE0C475C33F7D0A4880254F6510E590CA7531EC8F972941468E7FF78A21EE`.
- PDF: `docs/revision_octubre_2026/pdf/TRABAJO_FINAL_TPScouting_CANDIDATO_FINAL_BLOQUE8_v4_2026-10-06.pdf`
  - 2.623.202 bytes.
  - SHA-256: `B46372C29CD638123A8D13EF6232FC241DAC6F9F5FFB291801E5C72AB39DF0E2`.
- El PDF tiene 84 páginas.
- Se verificaron 25 entradas de figuras y 20 de tablas contra sus páginas reales: 45 de 45 coincidieron.
- Ya no aparece el error `No se encontraron entradas de tabla de contenido`.
- El legajo continúa como `[PENDIENTE DE INFORMAR]`; no se inventó.

Es un punto de trabajo, no una entrega final. La revisión del Bloque 8 quedó interrumpida por pedido del usuario.

## Evidencia funcional obtenida

- `scripts/iniciar_demo.py` se ejecutó con una base temporal vacía en `%TEMP%`, 60 jugadores sintéticos y semilla 42.
- Pasaron nacimiento, edad, categoría, credenciales y presencia de todos los historiales requeridos.
- No se utilizó ni modificó la base persistente del usuario.
- Faltan la navegación integral, silueta, predicción y la suite final con cobertura fechada.

## Hallazgos y limitaciones

- Microsoft Word 16 está disponible.
- LanguageTool no está instalado localmente. No se automatizó su API pública.
- La enumeración gramatical por COM de Word no terminó en 120 segundos y se abandonó.
- Se creó una auditoría textual reproducible; contiene falsos positivos que exigen revisión humana.
- Se dividieron cuatro oraciones extensas y se corrigieron una viñeta y la numeración de tablas del capítulo 4.
- La página de inicio de Bibliografía contiene solo el título por un salto estructural; queda por decidir si se compacta.
- Existe un archivo temporal de bloqueo de Word no versionado (`~$...docx`). Al retomar, comprobar que Word esté cerrado y eliminar solo ese temporal.

## Regla nueva indicada por el usuario

Para acelerar la revisión académica:

1. Crear una copia de trabajo sin imágenes y revisar allí texto, gramática, tablas, referencias, numeración e índices.
2. Inventariar cada imagen retirada con leyenda, ubicación, tipo y fuente.
3. Regenerar e insertar imágenes una sola vez, al cerrar el contenido.
4. No inventar capturas ni evidencia. Distinguir diagramas regenerables, capturas actuales de la aplicación y evidencia histórica.
5. No inspeccionar individualmente imágenes antiguas antes de decidir si siguen siendo necesarias.

## Archivos creados en el bloque

- `scripts/correct_word_block8_academic.py`
- `scripts/finalize_word_block8.ps1`
- `scripts/fill_word_lists_block8.py`
- `scripts/audit_document_block8.py`
- `docs/revision_octubre_2026/auditoria_redaccion_bloque8.md`
- varios DOCX/PDF intermedios no versionados.

Al retomar, conservar el candidato v4 y clasificar los intermedios. No usar como final archivos con `PREFIELDS`, `LISTAS_TEMP`, `VERIFICADO...FINAL` ni candidatos anteriores a v4.

## Secuencia para retomar

1. Comprobar este archivo, hashes y estado Git.
2. Crear desde v4 una copia sin imágenes y un inventario gráfico.
3. Terminar la revisión académica y gramatical sobre esa copia.
4. Resolver el salto de Bibliografía y verificar índices/listas.
5. Ejecutar la prueba integral y la suite final con cobertura/versiones.
6. Regenerar las imágenes necesarias desde fuentes comprobables e insertarlas al final.
7. Exportar un único PDF final y hacer inspección visual selectiva.
8. Actualizar la matriz y preparar el diff sin modificar la entrega.
