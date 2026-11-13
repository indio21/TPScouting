# Punto de guardado — noviembre, 12/11/2026

## Estado al cerrar

El usuario aprobó las correcciones menores del profesor y las dos comprobaciones
posteriores, pidió subir el Word y PDF a GitHub, guardar contexto y hacer commit.
No pidió más cambios y terminó la jornada. No se enviaron mensajes al profesor.

Entrega vigente publicada, rama main de TPScouting-entrega:
`84b6e0a30a269c6f6cb2ac6c1c1ceef148347793`.

## Archivos definitivos

En `C:\Tesis\TPScouting-entrega\documento\`:

- `Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_12-11-2026.docx`
- `Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_12-11-2026.pdf`

SHA-256 Word: `C6BEB90582E818D371AC9CF96EBAB0C1A1DA025C6DE689DC2383EB606DBFDC79`.
SHA-256 PDF: `E43883FC89601B5E14B7144BA885BD6454C84C2FBCF0CC139B9DBC5621EAE5F0`.

Copias idénticas, informes, muestras visuales y respaldo del Word original en:
`C:\Tesis\TPScouting\docs\revision_noviembre_2026\correcciones_menores_2026-11-12\`.
El original aprobado del 21/10 sigue también en su carpeta de octubre y en Git.

## Cuatro correcciones aplicadas, sin ampliar el alcance

1. NOV-01: retirada la leyenda huérfana de comparación de motores y su entrada
   de lista, sin crear una tabla. Luego se renumeraron 13 apariciones de títulos
   y referencias del capítulo 4. Ahora hay tablas 4-1 a 4-7, sin saltos.
2. NOV-02: /health describe disponibilidad mínima y consulta de conectividad,
   no calidad de datos. La tabla de endpoints antes 4-6 es ahora 4-5.
3. NOV-03: eliminada la URL concatenada del comando Linux/macOS en §9.4,
   conservando el texto negro y una URL por plataforma.
4. NOV-04: ampliado el segundo párrafo de §4.2.4 (página impresa35, física36):
   cinco fallos en ventana móvil de 15 minutos, siguientes intentos HTTP429
   aun con credenciales correctas, posible restricción provocada por terceros
   y registro de intentos en memoria por proceso. Describe el código existente.

El usuario pidió conservar todo lo demás: no cambiaron métricas, resultados,
bibliografía, imágenes, portada, tipografía, tamaños, títulos de capítulos,
estilos, encabezados, pies ni geometría. Styles/theme/numbering y encabezados/pies
son byte a byte iguales al original. Correcciones en el color original, sin rojo.

## Comprobaciones realizadas

- 97 páginas, 19 tablas, 25 figuras, 28 imágenes y nueve secciones.
- 19 tablas con sus 19 leyendas asociadas; ninguna huérfana.
- Todas las figuras tienen imágenes; Figura6-5 conserva a), b), c).
- 84 títulos numerados: secuencias completas, sin números repetidos.
- 88 entradas del índice y 44 entradas de listas con páginas actuales.
- 107 referencias a tablas/figuras con leyendas existentes.
- 88 enlaces internos del índice con sus destinos presentes.
- 23 apariciones de URL sin concatenaciones o duplicaciones accidentales.
  No se comprobó disponibilidad actual de todos los sitios externos.
- Comparación íntegra del texto: solo las correcciones y renumeración acordadas.
- Única celda cambiada: propósito de /health. Imágenes idénticas por hash.
- Control visual dirigido de las páginas modificadas, sin analizar ni regenerar
  los diagramas existentes. Índices alineados y sin errores de campo.
- Ruff crítico del script documental aprobado.
- Archivos publicados: blobs de Git coinciden con los Word/PDF verificados.

Evidencia: `CONTEXTO_Y_CIERRE.md`, `COMPARACION_ANTES_DESPUES.md`,
`verificacion_integridad.json`, `control_formato.json`,
`control_formato_conservado.json`, `comparacion_independiente_word_pdf.json`,
`auditoria_numeracion_leyendas_enlaces.json` y `archivos_publicados.json`.

## Publicación y aplicación

README y VERIFICACION_ENTREGA identifican los nuevos archivos, hashes y las
44 entradas de listas. El .gitignore público permite exclusivamente el Word
vigente. Solo se sincronizaron esos metadatos y el Word/PDF; ningún contexto,
backup, candidato, reporte interno o script de edición fue copiado al profesor.

No cambió el código, modelos, tests, dependencias ni configuración del runtime.
El Word conserva el commit funcional probado `2a27d53` y evidencia histórica.
La última suite local sigue siendo la del 20/10: 131 passed, 1 skipped,
4 warnings y 83,31 % de cobertura. No presentarla como ejecución local nueva.

La publicación disparó la CI `43339674664`. La auditoría de dependencias falló:
informó diez avisos en filelock 3.28.0 (nueve CVE) y setuptools 83.0.0 (un GHSA).
El run terminó `completed/failure`: pruebas de Python 3.11 y 3.12 aprobadas;
job dependency-audit fallido. No es una aprobación global de CI. Los estados
finales se registran en `correcciones_menores_2026-11-12/publicacion.json`.

Detalle exacto observado: `correcciones_menores_2026-11-12/alertas_dependencias_ci.json`.
No se evaluaron todavía severidad, vías explotables en esta app ni compatibilidad
de las versiones sugeridas por la herramienta. No se cambiaron dependencias:
el usuario autorizó exclusivamente las correcciones escritas y su publicación.
Retomar estos avisos como tarea separada: contrastar fuentes oficiales y alcance
real antes de proponer un ajuste mínimo con sus pruebas. No actualizar a ciegas
ni suprimir avisos para que el pipeline aparezca aprobado.

El commit de respaldo previo a noviembre es `c101539`. El nuevo commit que
incorpora este contexto se consulta con `git log -1 --format="%H %s" -- docs/revision_noviembre_2026/PUNTO_GUARDADO_NOVIEMBRE_2026-11-12.md`.

## Para retomar

Esperar nuevas observaciones del usuario/profesor. Usar este Word de noviembre,
conservar los originales y trabajar primero en una copia dentro del repo propio.
Usar las skills de revisión locales según corresponda, contrastar observaciones
individualmente y mantener evidencia. No modificar código ni reentrenar por
iniciativa propia. Solo actualizar entrega selectivamente cuando se autorice.

Mantener Arial y formato aprobado, preliminares/bibliografía separados y legajo
omitido. No regenerar imágenes ni ampliar el alcance de redacción sin pedido.
Persisten los límites declarados del MVP: datos sintéticos sin validación real,
deuda Ruff gradual, factory diferida, Intel no soportado y controles para datos
reales pendientes. La auditoría automática de noviembre sí detectó los avisos
descritos arriba; la evaluación de seguridad y su eventual corrección quedan
pendientes y separadas de las cuatro observaciones documentales resueltas.
Preservar archivos históricos no seguidos; no hacer limpieza ni git add masivo.
