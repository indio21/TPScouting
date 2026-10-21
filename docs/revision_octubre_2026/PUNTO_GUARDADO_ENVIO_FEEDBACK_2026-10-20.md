# Punto de guardado — preparado para envío y feedback

Registro: 20/10/2026, Buenos Aires; 21/10/2026 UTC.

## Última decisión del usuario

El usuario confirmó que enviará el trabajo mañana y esperará el feedback y las
correcciones. Después pidió guardar todo el contexto. **No se ha enviado el
trabajo al profesor desde esta sesión**. No hay nuevas correcciones docentes.

La tarea actual está cerrada. No realizar más cambios en código, documentación,
Word, imágenes o entrega hasta recibir una nueva indicación.

## Archivos definitivos

Entrega: `C:\Tesis\TPScouting-entrega\documento\`.

- `Trabajo Final_Solari Pablo_21-10-2026.docx`
- `Trabajo Final_Solari Pablo_21-10-2026.pdf`

Son los únicos Word/PDF vigentes de ese repositorio. El Word es idéntico al
adjuntado por el autor en el Escritorio; no se reescribió ni se guardaron cambios
automáticos. El PDF se exportó con Microsoft Word.

SHA-256 Word: `81A743DF38FB32820EA7DC3BEC247F57CCE6F3F826510DFEA2ED9CF3D63A131C`.
SHA-256 PDF: `587FB8393C6ED864A2FE6399548E458F8E2654943DD939649ED01856C31DBCC2`.

Título vigente: «Scouting inteligente aplicado al fútbol juvenil mediante
Machine Learning». Subtítulo conservado: «Evaluación y predicción del potencial
de desarrollo mediante análisis de datos».

## Repositorios y trazabilidad

- Principal/respaldo: `C:\Tesis\TPScouting`, rama `main`.
- Último respaldo de la revisión integral, previo a este guardado:
  `a519aab86bdafa4b43dfea6bc535e674be41539f`.
- Entrega: `C:\Tesis\TPScouting-entrega`, rama `main`.
- Commit entregable publicado, reconfirmado al guardar este contexto:
  `705d5c65696144eec18249b8f03ccf06db6cac81`.
- Código funcional y modelos citados en el Word:
  `2a27d53ae4f8a4849bee7d1bea362fd9146886b5`.

El commit publicado posterior al funcional incluye el Word del autor, su PDF,
documentación y dos ajustes de requirements. No modificó código de aplicación,
tests ni modelos. Los hashes de la publicación documental se registran fuera
del Word para evitar un hash circular dentro de su propio archivo.

El repositorio de entrega está limpio y su HEAD coincide con origin/main.
Este guardado añade únicamente contexto interno al repositorio de respaldo;
no corresponde copiarlo al profesor. El hash de este nuevo guardado se consulta
con `git log -1 --format="%H %s" -- docs/revision_octubre_2026/PUNTO_GUARDADO_ENVIO_FEEDBACK_2026-10-20.md`.

## Verificaciones efectivamente realizadas

- Suite local Windows/Python 3.11.9: 131 passed, 1 skipped, 4 warnings,
  83,31 % de cobertura; 85,05 segundos.
- Prueba omitida: Playwright opt-in. Advertencias: cuatro All-NaN conocidas.
- Ruff crítico y gradual, pip check y demo aislada aprobados.
- Demo: 60 jugadores sintéticos, seed42, base SQLite temporal eliminada;
  login, nacimiento/edad/categoría, historiales, silueta y predicción correctos.
- CI `39523230353`, Container `39523230238` y macOS `39523230354`:
  completed/success sobre el commit público `705d5c6`.
- Auditoría de los nueve .md y cinco .txt públicos: enlaces/referencias,
  instrucciones por plataforma, métricas, ramas, operación y versiones revisados.
- Requirements: versiones de runtime conservadas; Gunicorn excluido en Windows
  también en lock; certifi 2026.10.19 fijado en herramientas de desarrollo.
- Entorno local: certifi y Pillow actualizados por avisos nuevos; auditorías
  posteriores limpias. Pillow no se añadió al runtime. torch se auditó también
  mediante su versión canónica sin +cpu.
- Word/PDF: 97 páginas, 28 imágenes, 19 tablas y 9 secciones. Arial, mínimo
  explícito 10 pt, 88 entradas de índice y 45 de figuras/tablas comprobadas;
  números de las listas alineados; sin errores de campo ni numeración.

No se repitieron estas pruebas para el guardado de contexto: este registro
conserva la evidencia de la revisión anterior. No afirmar ausencia absoluta de
errores ni una auditoría exhaustiva de píxeles o validación deportiva externa.

## Contexto completo y evidencia

Leer después `CONTEXTO_WORD_AUTOR_21-10-2026.md`.
Evidencia detallada en `entrega_word_usuario_21-10-2026/`: `CIERRE.md`,
`publicacion.json`, `pytest.txt`, `coverage.json`, auditorías, demo, hashes,
controles de formato y copias exactas de los entregables.
Los contextos anteriores permanecen como historia, no como versión vigente.

## Cómo retomar con las correcciones del profesor

1. Identificar el feedback recibido y contrastarlo con esta entrega exacta.
2. Usar las skills locales `scouting-thesis-reviewer` y, si se solicita un nuevo
   diagnóstico integral, `tpscouting-correction-report`. Leer sus SKILL.md.
3. Crear una matriz por observación con fuente, estado, evidencia y acción.
   No asumir vigentes los números de página, commits o métricas de informes viejos.
4. Mantener intactos los originales; corregir sobre una copia nueva. Trabajar
   primero en el repositorio propio y validar cada bloque antes de continuar.
5. Actualizar código, texto y evidencia cuando la corrección lo requiera, sin
   cambiar código solo para justificar afirmaciones erróneas del informe.
6. Actualizar selectivamente la entrega cuando corresponda y esté autorizado;
   excluir contextos, candidatos y reportes internos. No enviar mensajes al
   profesor desde herramientas sin instrucciones expresas.

## Preferencias y límites que deben conservarse

- Mínimos cambios al texto aprobado; no inventar hechos, citas ni verificaciones.
- Arial en texto editable; las imágenes conservan tipografía rasterizada.
- Portada, índice, resumen, abstract y bibliografía comienzan por separado.
- Legajo omitido por decisión del usuario. Todos los derechos reservados,
  sin LICENSE; no elegir ni agregar una licencia.
- No invertir tiempo en analizar todas las imágenes ni regenerarlas sin pedido.
- Usar la .venv del proyecto y bases temporales para pruebas; no alterar bases
  del usuario. No reentrenar ni reemplazar artefactos históricos por iniciativa.
- No hacer limpieza ni git add masivo de los archivos antiguos no seguidos.
- Mantener límites declarados: MVP juvenil 12–18 con datos sintéticos, sin
  validación deportiva real; factory diferida, deuda Ruff gradual, macOS Intel
  no soportado, metadatos históricos irrecuperables y controles para datos
  reales de menores diseñados pero pendientes de implementar.

Estado al cierre: LISTO PARA REVISIÓN FINAL HUMANA. Próximo paso del usuario:
enviar el trabajo y posteriormente aportar feedback; no se confirma envío.
