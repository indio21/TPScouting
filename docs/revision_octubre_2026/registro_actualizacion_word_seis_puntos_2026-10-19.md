# Actualización del Word: seis puntos pendientes

Fecha de trabajo: 19/10/2026.

## Fuente preservada

- Archivo: `C:\Tesis\TPScouting-entrega\documento\TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-15.docx`
- SHA-256: `8385153268405E1069496E3E39D774C97DB4C47B35F295C4B064324171CC97AE`
- La fuente no fue modificada.

## Copias generadas

- Copia limpia: `docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_ACTUALIZADO_SEIS_PUNTOS_2026-10-19.docx`
  - SHA-256: `6B001A1F83E9BB1F1CDC899701171DFC648AB1B138B461E087165697575F2833`
- Copia para revisión: `docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_ACTUALIZADO_SEIS_PUNTOS_CAMBIOS_ROJO_2026-10-19.docx`
  - SHA-256: `39568B4DBF045808C4D7E6211765651D3E1DDE8FFFAF6FC255286D333FA8C524`
  - Los párrafos y celdas con contenido actualizado se muestran en rojo.

## Alcance aplicado

1. Se distinguió el commit funcional del código del commit del paquete de entrega y se incorporaron las ejecuciones CI y de contenedor correspondientes.
2. Se aclaró que `combined_prob` es el valor principal mostrado por la interfaz, pero una heurística secundaria desde el punto de vista de la evidencia predictiva.
3. Se corrigió la descripción del rate limiting para reflejar la clave por usuario normalizado y se redujeron afirmaciones introductorias no medidas.
4. Se agregaron equivalentes para Linux/macOS en pruebas y pipeline opcional; la corrida candidata usa nombres de artefactos separados.
5. Se documentaron los límites actuales de cobertura, CSP/HSTS, Ruff y application factory sin introducir refactors.
6. Se mantuvieron explícitas las limitaciones de la corrida histórica, la separación de la candidata y los controles aún no implementados para datos reales de menores.

## Formato e integridad

- Se elevaron a 10 puntos todos los tamaños Word explícitos inferiores a ese valor. No se redujeron tamaños existentes.
- Estructura conservada: 611 párrafos del cuerpo, 19 tablas, 9 secciones y 28 imágenes.
- Los 28 archivos multimedia son idénticos a los de la fuente.
- Los 89 campos del documento, incluidos los de índices, se conservaron sin actualización.
- Ambas copias abrieron correctamente en Microsoft Word 16.0 y Word calculó 100 páginas.
- La revisión automática de lógica de los párrafos modificados no detectó problemas.

## Verificación del proyecto

- `pytest`: 123 aprobadas, 1 omitida y 4 advertencias conocidas.
- Cobertura: 83,29 %; `compare.py` 70 % y `settings.py` 65 %.
- Ruff crítico: aprobado.
- Ruff gradual: aprobado.
- `pip check`: sin dependencias rotas.

## Pendiente deliberado para el cierre visual

- Regenerar o sustituir imágenes con la información final.
- Actualizar índice general, lista de figuras, lista de tablas y referencias de página después de cerrar las imágenes.
- Exportar el PDF definitivo y hacer la revisión visual de paginación, cortes, tablas y figuras.
- Revisar la copia con cambios en rojo y, si se aprueba, usar la copia limpia como base del cierre visual.

El repositorio de entrega no fue modificado durante esta actualización.
