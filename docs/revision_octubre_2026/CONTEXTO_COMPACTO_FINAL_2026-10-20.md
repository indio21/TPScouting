# Contexto para continuar TPScouting — 20/10/2026

Este punto fue sustituido por `CONTEXTO_FINAL_ARIAL_2026-10-20.md`, que registra
las correcciones autorizadas y verificadas posteriores a la revisión.

## Revisión posterior al cierre — leer primero

El usuario pidió volver a verificar código, commits/resultados del Word,
tipografía y escritura. El informe vigente de esta revisión es
`revision_posterior_codigo_word_2026-10-20.md`: **REQUIERE CORRECCIONES MENORES**.
Se reprodujeron dos HTTP 500 (contraseña ausente y CSRF no ASCII), sin sesión
autenticada. Faltan incorporar al Word el commit `d3e62fe` y su CI aprobada,
actualizar la fecha de pruebas, ajustar la descripción de HSTS y unificar
diferencias puntuales de títulos/captions/fuentes de comandos.
No se corrigieron aún la aplicación ni el Word/PDF; la entrega publicada sigue
intacta. Las notas de cierre siguientes describen el estado anterior a esta
revisión. Próximo bloque recomendado: dos correcciones de autenticación,
regresiones/suite y, después, actualización documental conjunta.

## Estado y archivo vigente

El usuario aprobó la copia limpia del 19/10 y autorizó regenerar imágenes,
actualizar índices/listas, revisar formato, exportar PDF y reemplazar
selectivamente Word/PDF del repositorio de entrega.

La versión vigente es:

- `docs/revision_octubre_2026/final/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20.docx`
- `docs/revision_octubre_2026/final/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20.pdf`
- Copias idénticas en `C:\Tesis\TPScouting-entrega\documento`.
- Word SHA-256: `2EA4A6EA4C99617CB3EDA36F8C9218E28D4D9B479D812F1240F81347569AAD16`.
- PDF SHA-256: `4774BD07C32DD393BB56FC7F3DD06E8D121AA19DBF7064A66F797481D9AE55AD`.
- 97 páginas físicas; portada sin número; páginas siguientes numeradas de 1 a 96.

La copia aprobada y la copia con cambios rojos del 19/10 son antecedentes de
revisión. No son la versión final para enviar. El original del 15/10 se preservó
en el historial y en `final/backup_entrega_anterior`.

## Trabajo completado

1. Los seis puntos documentales pendientes se incorporaron en la copia aprobada.
2. Ocho diagramas regenerados desde las fuentes PlantUML. Las fuentes de
   inferencia, componentes y despliegue se alinearon con la aplicación vigente.
   El diagrama de componentes se redistribuyó para aprovechar la página.
3. Capturas nuevas del 19/10: aplicación local, SQLite temporal, 60 jugadores
   sintéticos, seed 42. Incluye las subfiguras a), b) y c). La base fue eliminada.
4. Curva histórica regenerada exclusivamente desde `training_metadata.json`.
   No se inventaron validation loss ni datos faltantes.
5. Evidencia CI reemplazada por captura real del run `38815440730`, del paquete
   `4c5ef2c88494e257a3bd8232bccba7f33f79b66a`. Su alcance está explicitado.
6. Índice general actualizado por Microsoft Word. Listas de figuras/tablas
   conservadas como listas estáticas, reconstruidas por páginas reales del PDF.
7. Formato: tabla de endpoints compactada solo en espaciado; ubicación de portada
   alineada a la izquierda; comandos alineados a la izquierda; bibliografía con
   2 pt entre entradas y referencias indivisibles, manteniendo Arial 14.
8. Se evitó que textos introductorios de figuras quedaran solos en otra página.
9. Tres referencias del anexo a documentos internos se sustituyeron por archivos
   que sí están incluidos en la entrega: `docs/modelo_ml.md`,
   `GUIA_DEMO_PROFESOR.md` y `docs/REAL_DATA_READINESS.md`.
10. README de entrega enlaza Word/PDF y el badge apunta a su propia CI.

## Comprobaciones efectivamente ejecutadas

- Suite en `C:\Tesis\TPScouting-entrega` con la `.venv` del proyecto principal:
  `123 passed, 1 skipped, 4 warnings`, cobertura exacta `83,29 %`, 20/10/2026.
- Advertencias conocidas: All-NaN en dos pruebas de preprocesamiento.
- Ruff crítico y gradual: aprobados. `pip check`: sin dependencias rotas.
- No hubo cambios de contenido en los 88 archivos operativos comprobados:
  código, pruebas, scripts públicos, workflows, dependencias, Docker/Render y
  artefactos ML. La comparación textual normaliza CRLF/LF; los modelos se
  comparan como binarios.
- DOCX íntegro: 28 imágenes, 19 tablas, 9 secciones.
- 88 entradas del índice general y 45 de listas verificadas contra el PDF.
- Ningún tamaño Word explícito inferior a 10 pt; esto no mide texto rasterizado
  dentro de las capturas/diagramas.
- Portada, resumen, abstract y bibliografía comienzan en hojas separadas.
- Sin errores de campo, páginas vacías, captions de figuras sin imagen ni
  contenido fuera de la página según los controles automatizados.
- Revisión visual dirigida de portada, preliminares, bibliografía, muestras del
  cuerpo y anexos. No se afirma una inspección detallada de cada píxel.

## Publicación

- Commit de entrega publicado:
  `d3e62fe720e24d89b2715ad50a0c100a66be43df` (`main`, `origin/main`).
- Cambia únicamente cuatro archivos finales: Word, PDF, README y README_TESTS.
- No se publicaron auditorías internas, candidatos ML, documentos rojos,
  respaldos ni scripts de edición documental.
- CI del commit final: run `39507542398`, finalizada con `success`.
  Los jobs `test (3.11)`, `test (3.12)` y `dependency-audit` aprobaron.
  Evidencia: `final/evidencia_publicacion_2026-10-20.json`.
- El workflow Docker no se dispara por cambios solo documentales. Se mantiene
  la evidencia del contenedor del paquete anterior; el contenido operativo no
  cambió. No se afirma haber ejecutado un contenedor nuevo en esta sesión.

## Reglas que deben conservarse

- No inventar verificaciones, datos, citas ni resultados históricos.
- Preservar el contenido aprobado; cambios mínimos, sobre copias nuevas.
- Portada, índice, resumen, abstract y bibliografía en hojas independientes.
- Legajo omitido por decisión explícita del usuario.
- Los diagramas ampliados deben aprovechar la hoja vertical; los horizontales
  se rotan. No perder tiempo analizando versiones antiguas de imágenes.
- Mantener el mínimo solicitado de 10 pt para texto editable.
- MVP con datos sintéticos, jugadores de 12–18 años, sin validación deportiva real.
- Conservar la distinción: score combinado principal en interfaz, heurística
  secundaria desde la evidencia predictiva; modelos históricos sin reemplazar.
- Las limitaciones históricas, Ruff gradual, application factory diferida,
  macOS Intel y controles de menores siguen explícitos; no presentarlos como
  implementaciones o validaciones resueltas.
- Todos los derechos reservados, sin LICENSE abierta.

## Próximo trabajo autorizado o posible

CI de entrega comprobada. Este contexto forma parte del guardado de respaldo
del cierre. La siguiente sesión debe revisar comentarios del usuario/profesor
y auditar únicamente observaciones nuevas o reabiertas.
No volver a empezar la auditoría completa ni reentrenar por iniciativa propia.
El repositorio principal contiene archivos viejos no seguidos por Git:
preservarlos; no hacer `git add .` ni limpieza masiva.
