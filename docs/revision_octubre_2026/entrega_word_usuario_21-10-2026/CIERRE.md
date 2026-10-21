# Cierre: Word final del autor y revisión de textos públicos

## Resultado

Word del Escritorio conservado exactamente y publicado con su PDF de 97 páginas.
Entrega: `705d5c65696144eec18249b8f03ccf06db6cac81`.
CI `39523230353`, Container `39523230238` y macOS `39523230354`: aprobados.

| Archivo público revisado | Estado y acción |
|---|---|
| README.md | Actualizado: título, enlaces finales, estructura real, rama main, instalación y cobertura |
| README_TESTS.md | Actualizado: dependencias por plataforma y comandos/resultados vigentes |
| GUIA_DEMO_PROFESOR.md | Actualizado: Windows/Linux CPU con lock, Apple Silicon con manifiesto directo |
| RUNBOOK.md | Actualizado: seguridad, validación, macOS, blueprints, backup probado y nuevas corridas aisladas |
| VERIFICACION_ENTREGA.md | Actualizado: Word del autor, PDF, hashes y evidencia de pruebas |
| docs/puesta_en_marcha.md | Actualizado: guía portable por plataforma y demo sin entrenar |
| docs/modelo_ml.md | Actualizado: métricas por variante, parámetros, limitaciones y comparación candidata |
| docs/guia_indicadores_app.md | Actualizado: fórmula real, calibrada/combinada y bandas visuales |
| docs/REAL_DATA_READINESS.md | Revisado y conservado: controles futuros, sin habilitar datos reales |
| requirements.txt | Revisado y conservado: versiones directas y compatibilidad con el runtime |
| requirements-lock.txt | Actualizado: Gunicorn excluido en Windows, mismas versiones del runtime |
| requirements-torch-cpu.txt | Revisado y conservado: wheel oficial CPU, misma versión que manifiesto directo |
| requirements-dev.txt | Actualizado: certifi 2026.10.19 para herramientas de auditoría |
| requirements-docs.txt | Revisado y conservado: dependencias documentales separadas del runtime |

## Evidencia nueva

- 131 passed, 1 skipped, 4 warnings; cobertura 83,31 %. Bases temporales.
- Ruff crítico/gradual y pip check aprobados.
- Auditorías del entorno y manifiestos sin vulnerabilidades conocidas.
- Demo aislada: 60 jugadores, seed42, login, historiales, silueta y predicción.
- 88 entradas del índice y 45 de listas coinciden con páginas reales.
- 45 números de listas alineados; Arial, mínimo 10 pt y numeración conservados.
- Links/referencias de los 14 textos públicos y versiones de dependencias verificados.
- Receta de backup probada con WAL y SQLite temporal, sin usar bases del usuario.
- Blobs de Git publicados idénticos a Word/PDF verificados; Word idéntico al adjunto.

Las nueve vulnerabilidades iniciales estaban en certifi y Pillow adicionales del
entorno local; se actualizaron y se repitieron auditorías y pruebas. Pillow no
se añadió a los requirements públicos. No se actualizó PyTorch ni scikit-learn.
La revisión directa de torch cubre la omisión del sufijo +cpu en el scan local.

## Límites

Se conservan las cuatro advertencias conocidas y la prueba visual opt-in omitida.
Los resultados no garantizan ausencia absoluta de errores ni vulnerabilidades.
Render no se verificó como servicio disponible actualmente. Se mantienen los
límites declarados del MVP sintético y de la corrida histórica.
No se publicó documentación interna ni candidatos experimentales.

LISTO PARA REVISIÓN FINAL HUMANA
