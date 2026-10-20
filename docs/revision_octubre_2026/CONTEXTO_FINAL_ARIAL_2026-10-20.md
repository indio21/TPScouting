# Punto de continuación vigente — Arial y autenticación, 20/10/2026

## Versión final vigente

- `cierre_correcciones_2026-10-20/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20.docx`
- PDF con el mismo nombre.
- Copias idénticas en `C:\Tesis\TPScouting-entrega\documento`.
- Word SHA-256: `29ACEE4C6117C143FA7086E7F15A2FDB27F6716ADE3261BD23DC32F960BCB03E`.
- PDF SHA-256: `E97B4902D8A02AEC2C883CF0A02A96E8782ADBD48C1FCF861BEF755786B2944B`.
- 97 páginas, 28 imágenes, 19 tablas y 9 secciones.

Sustituye como versión vigente a la entrega del 20/10 sin ARIAL en el nombre.
Los originales y las copias aprobadas del 19/10 siguen preservados.

## Autorización y cambios aplicados

El usuario pidió aplicar los hallazgos de la revisión posterior, usar Arial,
repetir pruebas y actualizar commits, Word y lo necesario.

- SEG-09: credenciales ausentes/vacías devuelven HTTP 400, sin autenticar.
- La sesión de base se cierra en `finally`, incluido un error del verificador.
- SEG-10: tokens CSRF no ASCII se rechazan con HTTP 400 sin TypeError.
- Ocho casos nuevos de regresión. El subconjunto auth/seguridad obtuvo 37 passed.
- Fuente Arial en todo el texto editable; títulos de 14 pt, caption de Tabla 6-4
  unificado, comandos Arial 10 pt. Tamaño Word explícito mínimo de 10 pt.
- PDF: solo ArialMT, Arial-BoldMT y Arial-ItalicMT en sus fuentes textuales.
  Las capturas y diagramas son imágenes y conservan su tipografía propia.
- Actualizados §6.2, Tabla 6-2, §6.2.4, Tabla 7-1, §9.4 y §10.2 con resultados y
  commits comprobados. Descripción de HSTS alineada con modo de producción.
- Capturas CI anteriores conservadas explícitamente como evidencia histórica.
- Sin cambios de métricas ML, artefactos históricos ni imágenes.
- Índice general y listas reconstruidos, PDF exportado y revisado.

## Pruebas y publicación

- Suite local nueva: 131 passed, 1 skipped, 4 warnings; 83,31 % de cobertura,
  Python 3.11.9/Windows. Las advertencias son las cuatro All-NaN ya conocidas.
- La prueba omitida es Playwright opt-in. Demo funcional adicional ejecutada
  desde SQLite vacía: 60 jugadores sintéticos, seed 42; login, edad/categoría,
  historiales, silueta y predicción verificados. Base temporal eliminada.
- Ruff crítico/gradual y pip check aprobados.
- Commit funcional de respaldo:
  `8809760cd20d804927d3604c160aee2459506b3d`.
- Commit funcional de entrega:
  `2a27d53ae4f8a4849bee7d1bea362fd9146886b5`.
- CI `39509728373`: success, Python 3.11/3.12, Ruff y auditoría de dependencias.
- Container `39509728476`: success, construcción y smoke.
- macOS `39509728473`: success, Apple Silicon.
- Publicación de Word/PDF/README/VERIFICACION_ENTREGA.md:
  `fdf2805975efbf1de78dfe7d350f0ebb30f1c1bd`, publicada en `origin/main` de entrega.
- CI de la publicación `fdf2805`: ejecución `39510545376`, completada con
  resultado `success`, comprobada contra el SHA completo del commit publicado.
  Evidencia: `cierre_correcciones_2026-10-20/publication_evidence.json`.

## Evidencias y método

Carpeta `docs/revision_octubre_2026/cierre_correcciones_2026-10-20/`:

- `pytest.txt` y `coverage.json`: ejecución completa nueva.
- `demo.json`/`demo.md`: demo aislada nueva.
- `entradas_auth_corregidas.json`: reproducción anterior ahora con HTTP 400.
- `functional_evidence.json`: commits y ejecuciones remotas comprobadas.
- `verificacion_documento.json` y `control_formato.json`: 88 entradas del índice,
  45 de figuras/tablas, numeración continua, sin errores de campo ni contenido
  fuera de página; preliminares y bibliografía comienzan por separado.
- `fuentes_pdf.json`: fuentes reales del PDF.
- `integridad_contenido.json`: imágenes y tablas ML intactas, referencias nuevas
  presentes y cifra anterior de pytest ausente en el documento actualizado.
- `VERIFICACION_ENTREGA.md`: copia del registro público de hashes y validación.

El Word cita un commit funcional ya evaluado. El commit que incorpora el Word y
PDF queda fuera del propio documento, registrado por hashes y por el historial
Git; no intentar un ciclo imposible de insertar el hash futuro del propio Word.

## Para continuar

Revisar únicamente nuevas observaciones del usuario/profesor. No hay que repetir
todo el trabajo desde agosto. No reentrenar ni refactorizar por iniciativa propia.
Las limitaciones declaradas siguen vigentes: datos sintéticos, ausencia de
validación deportiva real, corrida histórica con limitaciones irreparables,
deuda Ruff gradual, application factory diferida, macOS Intel no soportado y
controles para datos reales de menores sin implementar.

Conservar Arial, mínimo 10 pt, legajo omitido y hojas independientes para
portada, índice, resumen, abstract y bibliografía. No sobrescribir originales ni
agregar documentos internos/candidatos al repositorio del profesor. Preservar
archivos viejos no seguidos por Git; no hacer limpieza ni `git add .`.
