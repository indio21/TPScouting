# Cierre del Bloque 8 — 2026-10-07

> **Revisión lingüística posterior:** la copia autoritativa para revisión del usuario
> pasó a ser `word/TRABAJO_FINAL_TPScouting_REVISION_GRAMATICAL_2026-10-07.docx`
> y su PDF homónimo. Se corrigieron gramática, cohesión, mayúsculas de títulos,
> anglicismos evitables y referencias al estado local del repositorio, sin alterar
> resultados ni conclusiones. DOCX SHA-256:
> `AA197775FBA8441260F9068B9350F1F533249B03C7E73AD5D4F194E336AF66E8`;
> PDF SHA-256:
> `9F102768CA0561C11F2771F63E1DBF91C1198AEE8ED12B5EBA4D150931A0CAAC`.

## Veredicto

**REQUIERE CORRECCIONES MENORES**.

El contenido, la aplicación y el documento final quedaron verificados localmente. La
entrega aún requiere el legajo del alumno y, cuando el usuario lo autorice, la
sincronización/publicación del árbol corregido y una CI correspondiente al commit
efectivamente entregado. La CI histórica #73 se conserva únicamente como evidencia
histórica y no certifica estos cambios locales.

## IDs del bloque

| ID | Resultado | Estado |
|---|---|---|
| A4 | Suite final: 116 passed, 1 skipped, 4 warnings; cobertura 83,74 % | Resuelta localmente |
| A5 | Evidencia fechada, versiones y límites registrados | Resuelta localmente |
| A6 | Se distinguió CI #73 histórica de la futura CI del commit entregado | Parcial: publicación no autorizada |
| A8 | Demo reproducida desde una base temporal vacía y documento final revisado | Resuelta localmente |

## Documento final vigente

- DOCX: `word/TRABAJO_FINAL_TPScouting_FINAL_BLOQUE8_2026-10-07.docx`
  - SHA-256: `5EAA742A6CD83BC91C0C406844A74348B3711D25846FA785AFB24B9F8C3DF36C`.
- PDF: `pdf/TRABAJO_FINAL_TPScouting_FINAL_BLOQUE8_2026-10-07.pdf`
  - SHA-256: `4AD71AB97E66A4C5C481E59CE81DAA0D5728722E7005A2118CC03CA23A36DA5E`.
- Extensión: 81 páginas; 26 imágenes incorporadas, 19 tablas y 9 secciones.
- Índice y listas: 45 entradas de figuras/tablas contrastadas con el número visible
  de la página; 0 discrepancias.
- Campos: no se detectaron mensajes de error.
- Páginas de texto mínimo: una, correspondiente a un diagrama de página completa.
- Revisión visual selectiva: portada, índice, listas, anexos técnicos, glosario y
  últimas páginas. La portada queda en una sola página y la Tabla 9-3 conserva su
  rótulo junto al glosario.
- Regla de maquetación del usuario: portada, índice general, resumen, abstract y
  bibliografía deben comenzar en hojas separadas y conservarse así en toda versión
  regenerada.
- Pendiente editorial: una sola marca, `Legajo: [PENDIENTE DE INFORMAR]`.

La auditoría de la revisión lingüística informa 0 espacios dobles, 0 espacios
indebidos antes de puntuación, 0 usos de primera persona plural, 0 usos de segunda
persona, 0 palabras consecutivas repetidas y 0 oraciones de 45 palabras o más. Las
cuatro coincidencias del control promocional corresponden a formulaciones negativas
(`no garantiza`) y no constituyen lenguaje publicitario.

La fuente aprobada del 26/08/2026 y su backup permanecen intactos. Las imágenes se
retiraron para la revisión textual, se inventariaron y se regeneraron/integraron una
sola vez al final. Las capturas actuales proceden de la demo local reproducible; las
dos imágenes de CI están rotuladas como históricas.

## Evidencia funcional

La demo se ejecutó con SQLite temporal, semilla 42 y 60 jugadores sintéticos. La
base temporal se eliminó al terminar. Se verificaron credenciales, login, edades
derivadas de fecha de nacimiento, categorías, historiales, ficha, silueta local,
predicción y rutas principales. El jugador de control poseía los seis tipos de
historial requeridos.

Evidencia: `evidencia_demo_bloque8.json` y `evidencia_demo_bloque8.md`.

La validación final de código produjo:

- Ruff crítico: correcto.
- pytest: 116 passed, 1 skipped, 4 warnings, en 65,84 s.
- Cobertura total exacta: 83,74 %; umbral requerido: 80 %.
- Las cuatro advertencias son los avisos conocidos de scikit-learn ante columnas
  completamente NaN en dos pruebas de regresión del MVP.

Evidencia: `evidencia_bloque8/ruff_final.txt`, `pytest_final.txt` y `coverage.xml`.

## Pendientes preservados

- **Legajo:** debe informarlo el usuario; no se inventó.
- **A6 / CI final:** exige publicar el commit entregado. No se hizo push.
- **Sincronización:** `C:\Tesis\TPScouting-entrega` continúa limpio en
  `ffdefdf8035c994ae285a270de0a4ff4e9f336a8`.
- **D-16:** la duración, validation loss y versiones de la corrida histórica no
  pueden recuperarse. No se ejecutó el plan opcional de cinco seeds ni se
  sobrescribieron artefactos.
- **Score combinado:** carece de evaluación persistida propia en test.
- **Opcionales diferidos:** application factory, Dockerfile, `.env.example` y
  elección de LICENSE.

## Preparación de sincronización, sin ejecutarla

El conjunto público candidato debe limitarse a los cambios funcionales y sus pruebas:

- `.github/workflows/ci.yml`, `.python-version`, `README.md`, `RUNBOOK.md`,
  `render.yaml` y archivos de requisitos;
- módulos modificados bajo `scouting_app/`;
- pruebas modificadas o nuevas bajo `tests/`;
- el DOCX/PDF final únicamente si se decide que el repositorio del profesor debe
  contener la memoria académica.

Quedan excluidos de la entrega los backups, candidatos, documentos sin imágenes,
capturas de trabajo, auditorías internas, matrices, reportes de pausa y scripts de
edición documental. Antes de copiar se debe comparar el contenido equivalente del
repositorio de entrega, revisar el diff resultante y ejecutar allí la suite. Esta
preparación no alteró la entrega.
