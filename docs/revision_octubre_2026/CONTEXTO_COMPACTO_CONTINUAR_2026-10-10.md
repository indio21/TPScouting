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

## Avance local posterior — limitaciones 1, 2 y 3

Trabajo realizado solo en `TPScouting`; todavía sin commit y sin modificar la entrega:

- score combinado evaluado independientemente sobre test histórico y nueva corrida; no mostró mejora defendible frente a la probabilidad cruda y empeoró Brier/log loss;
- nueva política temporal ajustada únicamente con train después de un split independiente del target;
- corrida candidata autoritativa: `artifacts/runs/20261010T_local_seed42_leakage_safe_v2/`;
- repetición exacta aprobada para política, splits, modelo, preprocesador, calibrador, historial y métricas;
- artefactos operativos existentes no reemplazados;
- suite: 122 passed, 1 skipped, 4 warnings; cobertura 83,23 %;
- cierre detallado: `docs/revision_octubre_2026/cierre_bloques_1_2_3_limitaciones_2026-10-10.md`.

El Word no fue modificado. Sus cambios se difieren hasta el cierre integral. El próximo bloque es la verificación real en macOS.

Decisión confirmada: mantener la probabilidad cruda como score principal y tratar
el score combinado únicamente como heurística secundaria. No presentar el
combinado como probabilidad ni como mejora demostrada.

## Bloque 4: verificación de macOS

- Se eligió una verificación remota real mediante GitHub Actions, sin requerir una Mac del usuario.
- Workflow preparado: `.github/workflows/macos.yml`.
- Apple Silicon `macos-15` arm64 con Python 3.11 aprobó la ejecución real sobre el commit `39bda92`.
- Intel x64 falló durante instalación porque PyPI no publica `torch==2.14.1` para esa arquitectura; quedó fuera del soporte documentado y no se rebajó PyTorch.
- Alcance: instalación limpia, carga de artefactos, Ruff crítico, suite con cobertura y demo en una base temporal.
- Evidencia inicial: https://github.com/indio21/TPScouting/actions/runs/38643910674.
- El repositorio `TPScouting-entrega` permanece fuera de alcance.

## Bloque 5: pruebas divididas por dominio

- `test_mvp_regressions.py` fue dividido en cuatro módulos: runtime,
  autenticación/seguridad, jugadores y ML/datos.
- Los 10 helpers comunes quedaron en `tests/mvp_regression_helpers.py`.
- Se conservaron los 76 nombres, el orden de colección y cuerpos AST idénticos.
- Verificación integral: 122 aprobadas, 1 omitida, 4 advertencias conocidas y
  83,23 % de cobertura.
- El commit funcional `8e6d82a` aprobó CI Linux 3.11/3.12, auditoría y macOS
  Apple Silicon 3.11.
- Informe: `cierre_bloque_5_pruebas_por_dominio_2026-10-14.md`.
- Próximo bloque: ampliación gradual de Ruff y logging.

## Bloque 6: Ruff y logging graduales

- Línea base Ruff completa: 751 hallazgos; 591 eran modernización masiva de tipos.
- CI ahora agrega `F401,F841,RUF059,S110,S112` para aplicación y pruebas, además
  del control crítico existente.
- Se mejoró logging de fallos recuperables y se conservaron salidas CLI.
- Se corrigió la ruta `temporal_breakout_path` y se agregó una regresión específica.
- Suite: 123 aprobadas, 1 omitida, 4 advertencias y 83,29 % de cobertura.
- Ruff completo restante: 732 avisos declarados, sin ocultarlos.
- Informe: `cierre_bloque_6_ruff_logging_2026-10-14.md`.
- Próximo bloque: evaluar application factory; no implementarla sin una decisión específica.

## Publicación posterior a los bloques 1–6

- Repositorio propio: `1804540bb056e053e29b03943e60ce95fb6d5d94`.
- Repositorio de entrega: `fb9de4be474faf610759461bd19376aae4e5c315`.
- Ambos repositorios aprobaron CI Linux/Python 3.11 y 3.12, auditoría de
  dependencias y macOS Apple Silicon/Python 3.11.
- Ejecuciones propias: `38648685553` y `38648685581`.
- Ejecuciones de entrega: `38648888954` y `38648888903`.
- Verificación local de la entrega: 123 aprobadas, 1 omitida, 4 advertencias y
  83,29 % de cobertura; Ruff gradual/crítico, `compileall`, `pip check` y demo
  temporal aprobados.
- La entrega recibió solo archivos necesarios para ejecutar y verificar el MVP;
  no recibió informes internos, candidatos Word/PDF ni artefactos de corridas.

Próximo bloque: **7, evaluación de application factory**. Preparar un diagnóstico,
mapa de dependencias, riesgos, pruebas de caracterización y una propuesta por
etapas. No implementar el refactor sin una autorización específica.

## Bloque 7: evaluación de application factory

- Decisión: no implementar en el MVP actual por riesgo alto de regresión y bajo
  beneficio inmediato para una entrega ya validada.
- Evidencia: `app.py` tiene 1.755 líneas, 85 funciones, 72 asignaciones globales
  y 17 llamadas de nivel de módulo con efectos; registra 85 reglas y seis
  blueprints.
- Las pruebas contienen 218 referencias directas a `app_module.*`.
- El import temporal creó la base, cargó modelo/preprocesador/calibrador y exigió
  `engine.dispose()` para liberar correctamente SQLite.
- Se definió un plan en siete etapas con snapshot de rutas, contenedor de runtime,
  tareas de arranque explícitas, compatibilidad de entrada y pruebas de dos apps.
- No se cambió código, Word, artefactos ni repositorio de entrega.
- Verificación final: Ruff crítico/gradual y `compileall` aprobados; 123 pruebas
  aprobadas, 1 omitida, 4 advertencias conocidas y 83,29 % de cobertura.
- Informe: `cierre_bloque_7_application_factory_2026-10-14.md`.
- Commit: `ad331344d81d52ac82065cb569687e6b174f9d11`; CI Linux y auditoría
  aprobadas en `38649994032`. macOS no se relanzó por ser un cambio solo
  documental; la última validación ejecutable aprobada sigue siendo `38648685581`.
- Próximo bloque: controles legales y operativos antes de usar datos reales de
  menores.

## Bloque 8: preparación para datos reales de menores

- Estado: diseño completado; uso real expresamente no habilitado.
- La aplicación actual no registra consentimiento/representación, avisos
  versionados, auditoría de accesos, solicitudes de derechos, retención ni
  incidentes; tampoco tiene validación externa del modelo.
- Se contrastaron Ley 25.326, derechos publicados por AAIP, Ley 26.061 y
  Resolución AAIP 47/2018 en fuentes oficiales vigentes al 2026-10-14.
- Documento público creado: `docs/REAL_DATA_READINESS.md`.
- Informe interno: `cierre_bloque_8_datos_reales_menores_2026-10-14.md`.
- README y guía de demo advierten que no se deben cargar datos reales.
- No se agregaron tablas, pantallas ni controles ficticios; las decisiones
  institucionales quedan abiertas.
- Verificación: fuentes oficiales y enlaces locales accesibles; Ruff y
  `compileall` aprobados; 123 pruebas aprobadas, 1 omitida, 4 advertencias
  conocidas y 83,29 % de cobertura.
- Commit propio: `0da89594e685076cdf3ecbd00c96d6e9d50fd718`; CI aprobada en
  `38651555664`.
- Entrega: `916c7bc0ff47aadaf76cca6b9831d34e85840028`; CI aprobada en
  `38651775522`. Solo se publicaron el documento público y las advertencias de
  README/guía; no se publicaron registros internos.
- Próximo bloque: decidir Dockerfile, `.env.example` y LICENSE.
