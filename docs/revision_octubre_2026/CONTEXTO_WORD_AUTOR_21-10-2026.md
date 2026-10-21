# Continuación vigente — Word final del autor, nombre 21/10/2026

> Cierre de conversación: el usuario planea enviar esta versión y esperar
> feedback. Leer primero `PUNTO_GUARDADO_ENVIO_FEEDBACK_2026-10-20.md`.
> No hay nuevas correcciones ni modificaciones pendientes autorizadas.

## Versión que rige la entrega

El usuario designó como Word final su archivo del Escritorio:
`C:\Users\Usuario\Desktop\Trabajo Final_Solari Pablo_21-10-2026.docx`.
Se preservó byte a byte, sin guardar cambios automáticos ni reformatearlo.

Documento publicado en `C:\Tesis\TPScouting-entrega\documento`:

- `Trabajo Final_Solari Pablo_21-10-2026.docx`
- `Trabajo Final_Solari Pablo_21-10-2026.pdf`

Copias y evidencias en el repositorio propio:
`docs/revision_octubre_2026/entrega_word_usuario_21-10-2026/`.

Word SHA-256: `81A743DF38FB32820EA7DC3BEC247F57CCE6F3F826510DFEA2ED9CF3D63A131C`.
PDF SHA-256: `587FB8393C6ED864A2FE6399548E458F8E2654943DD939649ED01856C31DBCC2`.

El nombre 21/10 pertenece al archivo del usuario. La revisión se ejecutó el
20/10/2026, hora de Buenos Aires; sus registros UTC cruzan al 21/10.

## Cambios del autor y control documental

Respecto del Word anterior: nuevo título «Scouting inteligente aplicado al fútbol
juvenil mediante Machine Learning», punto en la fecha de portada y reemplazo de
«historizar» por «almacenar el histórico» en un párrafo. Tablas e imágenes no
cambiaron; las diferencias completas se contrastaron con python-docx.

PDF exportado mediante Microsoft Word, abriendo la copia en solo lectura.
97 páginas, 28 imágenes, 19 tablas y 9 secciones. Se verificaron 88 entradas de
índice y 45 de figuras/tablas; los 45 números tienen la misma coordenada derecha.
Fuentes textuales exclusivamente Arial; no hay tamaños explícitos menores de
10 pt, errores de campo, páginas vacías, desbordamientos ni saltos de numeración.
Portada sin número y preliminares/bibliografía con comienzo independiente.
Control visual dirigido de portada y listas; no auditoría exhaustiva de imágenes.

## Revisión de todos los .md y .txt públicos

Se leyeron completos los nueve Markdown y cinco manifiestos de texto de entrega.

- README: título, documento vigente, enlaces públicos y rama `main`; se quitaron
  referencias a documentos internos ausentes y a ramas históricas como vigentes.
- Guía del profesor / puesta en marcha / pruebas: Windows y Linux CPU con lock
  y wheel CPU; Apple Silicon con requirements.txt; Intel no soportado.
- RUNBOOK: resultados actuales, ventana de login de 900 s por usuario, macOS
  probado, blueprints existentes y factory diferida; Render como histórico.
- Backup: se sustituyó la copia insegura de sidecars activos por la API SQLite,
  con conexiones cerradas explícitamente. Receta probada sobre WAL temporal.
- Entrenamiento: ejemplos guardan una corrida nueva en un directorio aparte;
  no reemplazan modelos ni bases operativas y no se presentan como la candidata.
- Modelo/indicadores: métricas históricas por variante, cruda/calibrada/combinada,
  parámetros vs buffers, límites sintéticos y candidata que no reemplaza runtime.
- REAL_DATA_READINESS: revisado y conservado, con fuentes oficiales consultadas;
  sus controles siguen propuestos, no implementados.
- VERIFICACION_ENTREGA: Word del autor, PDF/hashes, pruebas y trazabilidad.
- Requirements: Gunicorn excluido en Windows también en el lock; certifi fijado
  a 2026.10.19 en herramientas de desarrollo. Los otros tres .txt se revisaron
  y conservaron. PyTorch y scikit-learn no se actualizaron.

## Seguridad del entorno y pruebas nuevas

El primer pip-audit encontró nueve avisos en dos paquetes adicionales del entorno
local: certifi 2026.7.22 y Pillow 12.3.0. Se consultaron fuentes oficiales y se
actualizaron a certifi 2026.10.19 / Pillow 13.0.0. Pillow no se añadió al runtime.
Pip check y auditorías posteriores del entorno y manifiestos quedaron limpios.
La revisión canónica torch==2.14.1 complementa el wheel con sufijo +cpu que el
escaneo local puede omitir. No se afirma seguridad absoluta ni futura.

- Suite local nueva: 131 passed, 1 skipped, 4 warnings en 85,05 s.
- Cobertura: 83,31 %; 5337 sentencias, 891 sin cubrir.
- Ruff crítico y gradual: aprobados; scripts nuevos: crítico aprobado.
- Demo nueva con SQLite temporal: 60 jugadores, seed42, login, nacimiento,
  edad/categoría, historiales, silueta y predicción correctos; rutas comprobadas200.
- Auditoría automática de 14 archivos públicos: enlaces y referencias presentes,
  versiones instaladas compatibles, lock/directas alineados, backup íntegro.
- Comprobación de Git: blobs Word/PDF coinciden con sus copias y con el Word
  del Escritorio. Ningún candidato, reporte interno o contexto se publicó.

## Commits y verificaciones remotas

Código de aplicación/tests/modelos idéntico al funcional
`2a27d53ae4f8a4849bee7d1bea362fd9146886b5`, que sigue citado en el Word.
Los cambios nuevos son documentación y los dos ajustes de manifiestos.

Entrega publicada: `705d5c65696144eec18249b8f03ccf06db6cac81`.
Se conserva `0f7696b` como entrega anterior, con originales también en el repo propio.

Runs del commit actual comprobados como `completed / success`:

- CI: `39523230353`.
- Container: `39523230238`.
- macOS compatibility: `39523230354`.

Estado final y SHA exacto registrados en `entrega_word_usuario_21-10-2026/publicacion.json`.
La CI, el contenedor y macOS quedaron aprobados. El Word se mantiene idéntico al del
autor; la publicación se identifica externamente, evitando hashes circulares.

## Para retomar

El documento vigente es el Word del autor y su PDF con nombre 21-10-2026.
Ver `CONTEXTO_LISTAS_ALINEADAS_2026-10-20.md` y `CONTEXTO_FINAL_ARIAL_2026-10-20.md`
solo para historial. Se mantienen límites: datos sintéticos, sin validación
deportiva real, deuda Ruff gradual, factory diferida, Intel no soportado y
controles de datos reales pendientes. Legajo omitido por decisión del usuario.
No regenerar imágenes ni reentrenar por iniciativa propia. Preservar originales.
