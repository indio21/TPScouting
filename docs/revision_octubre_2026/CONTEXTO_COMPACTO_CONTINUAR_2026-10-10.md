# Contexto compacto para continuar TPScouting

Actualizado: 2026-10-10, America/Buenos_Aires.

## Objetivo y reglas vigentes

- Proyecto principal y respaldo: `C:\Tesis\TPScouting`.
- Entrega del profesor: `C:\Tesis\TPScouting-entrega`.
- Trabajar primero en el principal. No modificar ni sincronizar la entrega sin una
  instrucción expresa nueva.
- Usar `C:\Tesis\TPScouting\.venv`.
- No inventar datos, resultados, citas, capturas ni metadata histórica.
- MVP de scouting juvenil de 12–18 años, datos sintéticos y sin validación con
  jugadores reales.
- Portada, índice, resumen, abstract y bibliografía comienzan en páginas separadas.
- No volver a analizar imágenes en detalle; conservarlas y revisar solo estructura,
  caption y legibilidad evidente salvo pedido expreso.

## Estado Git

- Principal `main`: `85278db6b9c55c3fe10fcfbe9cbf49971bf70dfc` (`release: record verified academic delivery`), con CI `38069261221` aprobada.
- Bloque funcional anterior: `6f83e40` (`security: update PyTorch and close audit gap`).
- `origin/main` recibió ambos commits.
- Entrega `main`: `bf2de2f5872e9fdd05b5175bba97bca5469267bf`, publicada y certificada por la CI `38069047550`.
- La entrega contiene PyTorch 2.14.1, auditoría doble, guía reproducible y el documento académico final en Word/PDF.

## Documento autoritativo

- DOCX:
  `docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_FINAL_POSTAUDITORIA_2026-10-10.docx`
  - SHA-256: `1326866ABBA0E04672072AD9078AF3B65D9110BEAFB0ACC577C116723A6F8C8A`.
- PDF:
  `docs/revision_octubre_2026/pdf/TRABAJO_FINAL_TPScouting_FINAL_POSTAUDITORIA_2026-10-10.pdf`
  - SHA-256: `0B2872B2CEE20808596FEE41E09D2C469840E807CEA2AA0E71D2C2422300C03C`.
- 94 páginas, 28 imágenes, 19 tablas, 9 secciones y 45 entradas de listas.
- Verificación: 0 discrepancias de listas, 0 errores de campos, 0 marcas editoriales
  y 0 páginas de texto mínimo.
- Informe: `docs/revision_octubre_2026/cierre_postauditoria_2026-10-10.md`.

La copia exacta vinculada a la entrega publicada es:

- DOCX: `docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_VERIFICADA_2026-10-10.docx`.
- PDF: `docs/revision_octubre_2026/pdf/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_VERIFICADA_2026-10-10.pdf`.
- Conserva 94 páginas, 28 imágenes, 19 tablas, 9 secciones y 45 entradas; la verificación automatizada aprobó.
- Identifica el commit funcional de entrega `72314c2072729c524c0eb4ca57e5e241feb433b6` y su CI aprobada `38068749412`.

## Código verificado

- PyTorch actualizado a `2.14.1` / `2.14.1+cpu`.
- CI audita el entorno instalado y también `requirements.txt --no-deps`, para que
  el sufijo `+cpu` no oculte avisos de la versión canónica.
- Checkpoint existente compatible: `input_dim=68`, formato no legacy, 20 entradas
  de estado.
- Suite Windows/Python 3.11.9: 116 passed, 1 skipped, 4 warnings conocidos.
- Cobertura: 83,74 %; umbral: 80 %.
- Ruff crítico, compilación, `pip check` y ambas auditorías: aprobados.
- CI de `6f83e40`: ejecución `37709526994`, `success`.
- CI del cierre documental `f128261`: ejecución `38067969112`, `success`.
- CI de la entrega `7a47bd3`: ejecución `37707576905`, `success`.
- CI funcional actual de la entrega `72314c2`: ejecución `38068749412`, `success`.
- CI del commit final completo `bf2de2f`: ejecución `38069047550`, `success`.

## Correcciones cerradas

- A1–A9: cerradas en el documento actual; legajo omitido por decisión del usuario.
- SEG-03–08, DEP-01–03, DATOS-01–02, TEST-01 y CI-01: cerradas para el alcance.
- D-07–15 y observaciones de forma/fuentes: cerradas.
- La guía 9.4 usa lock + manifest CPU en Windows/Linux y requirements directo en
  macOS; identifica el commit/CI de entrega y el ajuste posterior del respaldo.

## Límites preservados

- D-16: la corrida histórica no guardó validation loss, duración ni SHA exacto.
  No reentrenar ni reconstruir esos valores como históricos.
- ML-01: cuantiles/cuotas del target se calcularon antes del split; está declarado.
- ML-04: un uso real con menores requiere consentimiento, retención/baja y controles
  adicionales; el MVP no usa datos reales.
- Score combinado sin evaluación persistida independiente en test.
- Application factory, lint completo, división adicional de regresiones,
  Dockerfile, `.env.example` y LICENSE siguen como opciones no bloqueantes.
- macOS no fue probado localmente; Linux se cubre mediante CI.

## Próximo paso

No hay correcciones obligatorias abiertas para evaluar el MVP. El estado es
`LISTO PARA REVISIÓN FINAL HUMANA`. Las mejoras posteriores se enumeran en
`LIMITACIONES_PENDIENTES_2026-10-10.md` y deben abordarse una por una, empezando
por la evaluación independiente del score combinado.

Si el usuario pide actualizar la entrega:

1. comparar `TPScouting` con `TPScouting-entrega`;
2. copiar selectivamente los cambios posteriores a `7a47bd3`;
3. excluir registros internos, candidatos, bases y scripts documentales salvo
   autorización expresa;
4. ejecutar suite, auditorías y demo dentro del árbol de entrega;
5. revisar diff antes de commit/push;
6. publicar solo con autorización y verificar la CI del SHA entregado.

No recorrer otra vez todos los candidatos ni regenerar imágenes. Empezar leyendo
este archivo y `cierre_postauditoria_2026-10-10.md`.
