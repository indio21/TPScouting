# Cierre de correcciones posteriores a la revalidación — 2026-10-10

## Resultado

Las correcciones accionables detectadas en
`revalidacion_correccion_octubre_2026-10-07.md` quedaron aplicadas en el proyecto
principal y en una copia nueva del documento. No se modificó el repositorio de
entrega.

## Código y dependencias

- PyTorch se actualizó de `2.9.1` a `2.14.1` en `requirements.txt` y
  `requirements-torch-cpu.txt`.
- La CI conserva la auditoría del entorno instalado y agrega una auditoría directa
  de `requirements.txt`, necesaria porque el build `+cpu` puede ser omitido por
  `pip-audit --local`.
- El checkpoint existente cargó con PyTorch `2.14.1+cpu`, `input_dim=68`, formato
  no legacy y 20 entradas de estado.
- Suite final: 116 passed, 1 skipped, 4 warnings conocidos; cobertura 83,74 %.
- Ruff crítico, compilación, `pip check`, auditoría local y auditoría del manifest:
  aprobados.
- Commit funcional de respaldo: `6f83e4066db7ccb9900554568a0389cdf1d12d20`.
- CI del commit funcional: `37709526994`, conclusión `success`.

## Documento vigente

- DOCX:
  `word/TRABAJO_FINAL_TPScouting_FINAL_POSTAUDITORIA_2026-10-10.docx`
  - SHA-256: `1326866ABBA0E04672072AD9078AF3B65D9110BEAFB0ACC577C116723A6F8C8A`.
- PDF:
  `pdf/TRABAJO_FINAL_TPScouting_FINAL_POSTAUDITORIA_2026-10-10.pdf`
  - SHA-256: `0B2872B2CEE20808596FEE41E09D2C469840E807CEA2AA0E71D2C2422300C03C`.

Correcciones documentales:

- commit de entrega actualizado a `7a47bd3f164e865677f2c70075cbedc9fa63427d`;
- CI de entrega actualizada a `37707576905`;
- commit de respaldo y CI posterior diferenciados de la entrega;
- guía de instalación alineada con lock, manifest CPU, desarrollo y macOS;
- evidencia histórica CI #73 conservada y rotulada como histórica.

Validación del documento:

- 94 páginas, 28 imágenes, 19 tablas y 9 secciones;
- 45 entradas de índices/listas y 0 discrepancias;
- 0 errores de campos, 0 páginas de texto mínimo y 0 marcas editoriales;
- portada, índice, resumen, abstract y bibliografía en páginas separadas;
- 0 dobles espacios, 0 espacios indebidos antes de puntuación, 0 primera persona
  plural, 0 segunda persona y 0 palabras consecutivas repetidas;
- la única detección de longitud corresponde al bloque técnico de comandos por
  plataforma, no a una oración de la tesis.

## Estado de los hallazgos

Quedaron cerrados A5, A6, DEP-03 y D-15. Continúan declarados, sin ocultarlos:

- CAL-01: application factory y reducción de módulos grandes, deuda opcional;
- CAL-02: lint completo y reemplazo gradual de prints, deuda no bloqueante;
- CAL-03: división adicional de `test_mvp_regressions.py`, deuda no bloqueante;
- ML-01: cuantiles/cuotas del target calculados antes del split;
- ML-04: un uso futuro con menores reales requiere implementar consentimiento,
  retención, baja y controles adicionales;
- D-16: la corrida histórica no contiene validation loss, duración ni SHA exacto;
  no se inventaron ni reconstruyeron esos datos;
- Dockerfile, `.env.example` y LICENSE permanecen opcionales.

## Veredicto

**LISTO PARA REVISIÓN FINAL HUMANA**.

Este estado no afirma ausencia absoluta de defectos ni validez deportiva externa.
La entrega sigue limitada a un MVP con datos sintéticos y sin validación con
jugadores reales.
