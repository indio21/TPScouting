# Registro interno de correcciones de octubre de 2026

Este registro pertenece al repositorio principal y no debe copiarse al
repositorio público `TPScouting-entrega`.

## Bloque 1 — Seguridad de autenticación

Estado: cerrado localmente el 2026-10-05. Pendiente de futura sincronización al
repositorio de entrega, únicamente cuando el usuario la autorice.

| ID | Cambio | Evidencia requerida | Estado |
|---|---|---|---|
| A1 | Validar `next` como ruta interna segura | Regresiones de destinos válidos e inválidos | Resuelta localmente |
| A2 | Evitar evasión del límite mediante headers/IP rotatorios | Bloqueo tras intentos con headers diferentes | Resuelta localmente |
| SEG-03 | Revalidar usuario y rol; limpiar sesión al autenticar | Usuario eliminado, cambio de rol y rotación de sesión | Resuelta localmente con limitación documentada |
| SEG-07 | Comparar CSRF con `secrets.compare_digest` | Tokens válidos, ausentes y de tipo inválido | Resuelta localmente |

Decisión de proxy: no se instala `ProxyFix`. No existe evidencia local que
establezca una cantidad fija de proxies confiables. El límite se aplica por
cuenta y se ignora `X-Forwarded-For`; `remote_addr` queda como dato auxiliar.

Limitación del modelo de usuarios: no existe una columna de estado activo. En
este bloque se comprueban existencia y rol válido contra la base en cada ruta
protegida; agregar estado requeriría una migración de esquema fuera del alcance.

Evidencia ejecutada con `C:\Tesis\TPScouting\.venv`:

- Regresiones focales del bloque: `14 passed, 71 deselected`.
- Archivo completo de autenticación: `15 passed`.
- Suite completa: `94 passed, 1 skipped, 4 warnings` en `46.16s`.
- Las cuatro advertencias son los `RuntimeWarning: All-NaN slice encountered`
  ya conocidos en dos pruebas de preprocesamiento.

Archivos funcionales y de prueba modificados:

- `scouting_app/services/security.py`
- `scouting_app/routes/auth.py`
- `scouting_app/app.py`
- `tests/test_auth.py`
- `tests/test_mvp_regressions.py`
- `README.md`

No se modificaron bases, el Word, artefactos ML ni `TPScouting-entrega`.

## Bloque 2 — Seguridad restante y operaciones

Estado: cerrado localmente el 2026-10-05. Pendiente de futura sincronización al
repositorio de entrega, únicamente cuando el usuario la autorice.

| ID | Cambio | Evidencia requerida | Estado |
|---|---|---|---|
| SEG-04 | Reducir `/health` a estado público mínimo y registrar el diagnóstico interno | Respuesta de éxito mínima y error sin excepción | Resuelta localmente |
| SEG-05 | Aplicar CSP compatible con la interfaz y HSTS en producción HTTPS | Headers, nonce, gráficos, formularios y modal en navegador real | Resuelta localmente con limitaciones documentadas |
| SEG-06 | Centralizar la política de `photo_url` en alta, edición e importación | Casos válidos, esquemas peligrosos, traversal y ausencia de mutaciones | Resuelta localmente |
| SEG-08 | Documentar el origen confiable de artefactos joblib/PyTorch | README, runbook y revisión de la carga existente | Resuelta localmente |
| DATOS-01 | Impedir generación y entrenamiento síncronos mediante requests de producción | Regresión que confirma que el pipeline no se ejecuta | Resuelta localmente |

Decisiones aplicadas:

- Las fotos personalizadas admiten recursos propios bajo `/static/` y URLs HTTPS
  sin credenciales. Se rechazan HTTP, esquemas peligrosos, rutas de red, barras
  invertidas y traversal, incluso codificado varias veces. La silueta por defecto
  sigue siendo local.
- No se incorporó una cola de tareas. La acción web de generación/entrenamiento
  queda desactivada en producción y el runbook indica el flujo fuera del request.
- La CSP usa nonce para scripts propios inline y permite los CDN que usa la
  interfaz. `style-src` conserva temporalmente `'unsafe-inline'` porque las
  plantillas actuales contienen estilos inline y posiciones calculadas.
- HSTS se emite sólo cuando la aplicación detecta el entorno de producción. No
  se modificó la confianza en headers de proxy ni se instaló `ProxyFix`, porque
  la cantidad de proxies confiables sigue sin estar documentada.

Evidencia ejecutada con `C:\Tesis\TPScouting\.venv`:

- Regresiones focales iniciales del bloque: `11 passed, 91 deselected`.
- Política de fotos, incluido traversal doblemente codificado: `6 passed, 70 deselected`.
- Suite completa posterior a los cambios funcionales: `102 passed, 1 skipped, 4 warnings` en `61.56s`.
- Repetición final después de ampliar el smoke visual y la documentación:
  `102 passed, 1 skipped, 4 warnings` en `107.60s`.
- Smoke Playwright ampliado: `1 passed` en `9.89s`; verificó login, formulario,
  creación real de Chart.js, apertura de un modal Bootstrap y ausencia de
  rechazos CSP en consola.
- Las cuatro advertencias siguen siendo `RuntimeWarning: All-NaN slice encountered`
  en dos pruebas de preprocesamiento; no fueron introducidas por este bloque.
- Se comprobó que un scout recibe `403` al intentar una operación administrativa
  y que el registro de jugador permanece sin cambios.

Archivos del Bloque 2:

- `scouting_app/app.py`
- `scouting_app/player_logic.py`
- `scouting_app/routes/players.py`
- `scouting_app/routes/settings.py`
- `scouting_app/templates/base.html`
- `scouting_app/templates/coaches.html`
- `scouting_app/templates/compare.html`
- `scouting_app/templates/compare_multi.html`
- `scouting_app/templates/dashboard.html`
- `scouting_app/templates/directors.html`
- `scouting_app/templates/player_attributes.html`
- `scouting_app/templates/player_detail.html`
- `scouting_app/templates/player_stats.html`
- `scouting_app/templates/prediction.html`
- `scouting_app/templates/settings.html`
- `tests/test_auth.py`
- `tests/test_mvp_regressions.py`
- `tests/test_pages.py`
- `tests/test_visual_smoke.py`
- `README.md`
- `RUNBOOK.md`

Pendientes registrados para bloques posteriores:

- Probar HSTS y el comportamiento completo detrás del HTTPS/proxy del despliegue
  real cuando exista una publicación autorizada (Bloque 8).
- Reducir o eliminar `'unsafe-inline'` de estilos exigiría migrar estilos dinámicos
  de las plantillas; no se amplió ese refactor en este bloque.
- Registrar hashes y procedencia reproducible de artefactos ML junto con corrida,
  configuración y dependencias (Bloques 5 y 8).
- Revisar y actualizar las métricas históricas de tests que todavía figuran en
  otras secciones de README/RUNBOOK como parte del cierre documental, sin tratarlas
  como evidencia vigente (Bloques 6 y 8).

No se modificaron bases persistentes, el Word, artefactos ML ni
`TPScouting-entrega`.

## Bloque 3 — Dependencias, portabilidad y CI

Estado: cerrado localmente el 2026-10-05. El workflow modificado todavía no fue
publicado ni ejecutado en GitHub Actions. Pendiente de futura sincronización al
repositorio de entrega sólo cuando el usuario la autorice.

| ID | Cambio | Evidencia requerida | Estado |
|---|---|---|---|
| DEP-01 | Definir instalación de PyTorch por plataforma y Python soportado | Guía Windows/Linux/macOS y resolución local del wheel CPU | Resuelta localmente con plataformas no ejecutadas identificadas |
| DEP-02 | Separar runtime, desarrollo y documentación; usar un lock real | CI y Render alineados con el snapshot de runtime | Resuelta localmente |
| DEP-03 | Reproducir auditoría y corregir vulnerabilidades vigentes compatibles | `pip-audit` antes/después y suite completa | Resuelta localmente con exclusión informada de Torch |
| DATOS-02 | Verificar compatibilidad de serialización sin actualizar ML a ciegas | Carga de los tres artefactos existentes | Resuelta localmente con limitación histórica |
| CI-01 | Agregar cache, auditoría, lint gradual y umbral medido | YAML válido, controles locales y futura corrida remota | Resuelta localmente; evidencia CI remota pendiente |

Cambios y decisiones:

- Python soportado se mantiene en 3.11 y 3.12. Render queda fijado a 3.11.9,
  coincidente con la `.venv` verificada, mediante `.python-version` y
  `PYTHON_VERSION`.
- Windows y Linux CPU instalan primero `requirements-lock.txt` y después el
  wheel oficial de PyTorch desde `requirements-torch-cpu.txt`. macOS usa
  `requirements.txt`, donde PyTorch se obtiene de PyPI sin sufijo `+cpu`.
- `requirements-lock.txt` es el snapshot de runtime usado por CI y Render.
  `requirements-dev.txt` contiene pruebas, Playwright, Ruff y pip-audit;
  `requirements-docs.txt` contiene python-docx/lxml.
- Se actualizaron `click 8.3.2 -> 8.3.3`, `fsspec 2026.3.0 -> 2026.6.0`,
  `Werkzeug 3.1.8 -> 3.1.9`, `lxml 6.0.4 -> 6.1.0`, `pip 26.0.1 -> 26.2.1`
  y `setuptools 82.0.1 -> 83.0.0`, que eran las correcciones indicadas por la
  auditoría vigente. PyTorch y scikit-learn no se cambiaron.
- CI usa cache de pip, Ruff limitado a errores críticos (`E9,F63,F7,F82`),
  auditoría como job separado y cobertura mínima de 80%, basada en el 80.03%
  medido. No se aplicó reformateo masivo.
- La política oficial vigente de Render Free queda documentada: una base por
  workspace, 1 GB, expiración a los 30 días y sin backups. No se cambió ni se
  contrató un plan.

Evidencia ejecutada con `C:\Tesis\TPScouting\.venv`:

- Auditoría inicial: `15` registros de vulnerabilidad en `6` paquetes, con
  duplicados del proveedor para algunos IDs, además de `pip` y `setuptools`.
- Auditoría posterior: `No known vulnerabilities found`. `torch 2.9.1+cpu`
  quedó informado como no auditable por pip-audit porque ese identificador no
  está disponible en PyPI; no se ocultó ni ignoró el resultado.
- `pip check`: sin requisitos rotos.
- Resolución `--dry-run --ignore-installed`: exitosa para el runtime fijado y
  para el wheel PyTorch CPU oficial.
- Ruff gradual: todos los controles configurados pasan.
- YAML de CI y Render: parseo correcto.
- Suite con umbral: `102 passed, 1 skipped, 4 warnings`; cobertura `80.03%`,
  umbral `80%`, en `69.32s`.
- Compatibilidad actual: cargan `model.pt` (checkpoint 1, entrada 68),
  `preprocessor.joblib` y `probability_calibrator.joblib` con PyTorch
  `2.9.1+cpu`, scikit-learn `1.8.0` y joblib `1.5.3`.

Limitaciones y pendientes registrados:

- Sólo se ejecutó Windows 11/Python 3.11.9. Linux 3.11/3.12 queda configurado en
  CI, pero requiere publicación para producir evidencia. macOS está documentado
  conforme a PyTorch oficial y no fue probado.
- El metadata histórico de entrenamiento no contiene versiones de PyTorch,
  scikit-learn ni joblib. La carga actual demuestra compatibilidad presente, no
  reconstruye las versiones originales (Bloque 5).
- pip-audit no audita el build local `torch 2.9.1+cpu`; se conserva el warning
  explícito y la procedencia desde el índice oficial de PyTorch.
- El umbral global no corrige módulos con cobertura baja. Compare, staff,
  runtime ML, settings, creación de admin y seed quedan priorizados para el
  Bloque 4.
- La política Free de Render implica pérdida/expiración y ausencia de backups;
  cualquier cambio de plan o servicio requiere decisión del usuario.

No se ejecutó CI remota, deploy, commit, push, sincronización de entrega,
reentrenamiento ni modificación de bases, Word o artefactos ML.

## Bloque 4 — Calidad y pruebas

Estado: cerrado localmente el 2026-10-05.

| ID | Cambio | Evidencia requerida | Estado |
|---|---|---|---|
| CAL-01 | Centralizar lectura de enteros de entorno y evaluar application factory | Regresiones de configuración y decisión de alcance | Mejora puntual resuelta; factory diferida |
| CAL-02 | Incorporar lint gradual, logging y UTC compatible | Ruff, compilación y pruebas de timestamps | Resuelta localmente en alcance puntual |
| CAL-03 | Dividir nuevas pruebas por dominio | Archivos de prueba separados | Resuelta incrementalmente |
| TEST-01 | Mejorar pruebas en compare, staff, runtime ML, settings, admin y seed | Cobertura por módulo y suite completa | Resuelta localmente |
| INFRA-01 | Evaluar Dockerfile, `.env.example` y LICENSE | Decisión registrada sin elegir licencia | Registrada como opcional |

Cambios principales:

- Se agregó `env_int` y se reemplazaron bloques repetidos de configuración para
  cache, paginación, comparación, rate limiting, payload y lock del pipeline.
- Los timestamps nuevos usan UTC mediante `datetime.now(timezone.utc)` y se
  guardan sin `tzinfo` para conservar compatibilidad con las columnas y fechas
  existentes. Nuevas corridas de entrenamiento escriben ISO 8601 con zona UTC.
- El runtime ML usa logging en lugar de `print`; los `print` de utilidades CLI
  se conservaron porque constituyen su salida operativa.
- Se agregaron pruebas separadas en `test_ml_runtime.py`, `test_admin_seed.py`,
  `test_compare_staff.py` y `test_quality_config.py`.
- No se implementó application factory: exigiría reorganizar el módulo global,
  blueprints, carga ML y fixtures, y el bloque pedía sólo evaluar sin autorización
  específica. Tampoco se incorporaron Dockerfile, `.env.example` o LICENSE; la
  licencia requiere una decisión explícita del usuario.

Evidencia:

- Pruebas nuevas focales: `14 passed`.
- Suite con cobertura: `116 passed, 1 skipped, 4 warnings` en `113.96s`.
- Cobertura total: `83.74%` (antes `80.03%`).
- Módulos señalados: compare `70%`, staff `82%`, runtime ML `86%`, settings
  `63%`, create_admin `78%` y seed_demo_data `91%`.
- Ruff gradual y `compileall`: correctos.
- Las cuatro advertencias All-NaN conocidas permanecen y no aumentaron.

Pendientes:

- Settings conserva `63%`; sus rutas críticas de excepción, bloqueo, producción
  y autorización ya están cubiertas. Ampliar casos de presentación puede hacerse
  después sin bajar el umbral.
- Application factory y reducción estructural de `app.py`/`players.py` requieren
  autorización específica para un refactor mayor.
- Dockerfile y `.env.example` siguen opcionales; LICENSE queda sin elegir.

No se modificaron bases persistentes, Word, artefactos ML ni la entrega pública.

## Bloque 5 — ML, métricas y trazabilidad

Estado: cerrado localmente el 2026-10-05, con D-16 pendiente de una nueva corrida
autorizada. No se reentrenó ni se sobrescribió ningún artefacto.

| ID | Cambio | Evidencia requerida | Estado |
|---|---|---|---|
| A3 | Contar parámetros entrenables y distinguir buffers/state_dict | Conteo ejecutable sobre el checkpoint | Resuelta localmente |
| D-08 | Cuantificar incertidumbre de la comparación | Bootstrap pareado sobre test persistido | Resuelta para la corrida histórica, con límites |
| D-09 | Separar umbral de evaluación y bandas visuales | Trazabilidad código/metadata | Resuelta localmente |
| D-10 | Explicar prevalencia sintética fijada por diseño | Trazabilidad del target temporal | Resuelta localmente |
| D-16 | Registrar duración, validation loss, SHA y metadata | Hashes actuales y plan previo de nueva corrida | Parcial: duración/validation loss históricos no recuperables |
| ML-D | Verificar circularidad, features, etiquetas y splits | Script y JSON reproducibles | Resuelta como auditoría; limitación metodológica permanece |

Resultados:

- Parámetros entrenables: 17.608; buffers: 386; elementos del state_dict: 17.994.
- Features antes de encoding: 64; dimensión transformada: 68.
- `potential_label` y `temporal_target_label` existen en el dataframe, pero no
  integran `MODEL_FEATURE_COLUMNS`. Tampoco entran scores o predicciones.
- Splits seed 42: 14.000/3.000/3.000, sin IDs compartidos.
- La prevalencia cercana al 8% y los cuantiles son decisiones del generador
  sintético calculadas antes del split; no describen una población real.
- Las probabilidades cruda/calibrada, el score combinado y las bandas 0,60/0,80
  quedaron diferenciados. Las métricas binarias usan umbrales seleccionados en
  validación: 0,825 crudo, 0,25 calibrado y 0,85 baseline logístico.
- Métricas de test reproducidas: crudo ROC-AUC 0,920341 / PR-AUC 0,546127;
  calibrado 0,917431 / 0,524116; logística 0,920506 / 0,537776.
- Bootstrap pareado de 2.000 remuestreos: diferencia crudo menos logística,
  ROC-AUC IC95% [-0,002693; 0,002471] y PR-AUC [-0,002508; 0,019543]. Ambos
  incluyen cero: no hay evidencia suficiente de superioridad; tampoco constituye
  una prueba de equivalencia.

Evidencia:

- `scripts/audit_ml_block5.py`
- `docs/revision_octubre_2026/evidencia_ml_bloque5.json`
- `docs/revision_octubre_2026/evidencia_ml_bloque5.md`
- `docs/revision_octubre_2026/plan_reentrenamiento_pendiente.md`
- Ruff y compilación del script: correctos.
- Hashes externos con PowerShell coinciden con el JSON generado.
- Git confirma que modelo, joblib, metadata, splits, cache y experiments no se
  modificaron.

Pendientes:

- D-16 no puede cerrarse por completo con la corrida histórica: no se guardaron
  duración, versiones de bibliotecas ni validation loss. No se inventaron.
- Una corrida de cinco seeds 42–46, en directorio nuevo y con artefactos separados,
  quedó planificada pero no ejecutada. El costo deberá medirse en la primera
  corrida; no se estimó desde datos inexistentes.
- El score combinado carece de predicciones de test persistidas y no tiene
  evaluación propia. Las métricas actuales no validan sus bandas visuales.
- La circularidad del generador y los cuantiles previos al split son limitaciones
  metodológicas que deben permanecer explícitas en el documento.

## Bloque 6 — Documento: contenido y coherencia

Estado: cerrado localmente el 2026-10-05 como borrador verificable. El Word
aprobado no fue sobrescrito y el repositorio de entrega no fue modificado.

| ID | Cambio documental | Evidencia / estado |
|---|---|---|
| A3 | Corregir conteo del modelo | 17.608 parámetros entrenables, 386 buffers y 17.994 elementos de `state_dict`; resuelta en borrador |
| A4–A5 | Actualizar pruebas y cobertura | 116 passed, 1 skipped, 4 warnings y 83,74 % con fecha; contenido actualizado, cierre final en Bloque 8 |
| A6 | Relacionar CI con el código documentado | CI histórica separada del árbol local y entrega; pendiente CI del commit finalmente entregado |
| A7 | Identificar repositorios y commits | Desarrollo `49aa51c…` con cambios locales; entrega `ffdefdf…`; resuelta en borrador |
| A9 | Portada y abstract | Datos conocidos completados; abstract incorporado por autorización actual; legajo pendiente |
| D-07 | Explicar objetivo y target reales | `temporal_target_label`, entradas y límites explicitados; resuelta en borrador |
| D-08 | Incorporar incertidumbre | Bootstrap pareado y límites incorporados; resuelta para la corrida histórica |
| D-09 | Diferenciar probabilidades, score y bandas | Cruda, calibrada, combinada, umbrales de validación y bandas 0,60/0,80 diferenciados |
| D-10 | Explicar prevalencia sintética | 7,985 % declarada como decisión del generador, no dato poblacional |
| D-11 | Responder la pregunta de investigación | Respuesta técnica explícita y exclusión de impacto no validado |
| D-12 | Justificar tecnologías | Comparativas genéricas reducidas a decisiones sin benchmarks inventados; psycopg 3 corregido |
| D-13 | Alinear entidades con el código | Once entidades del modelo enumeradas individualmente |
| D-14 | Reproducibilidad portable | Windows verificado; comandos Linux/macOS documentados sin afirmar prueba; `scripts/iniciar_demo.py` priorizado |
| D-15 | Marco teórico y limitaciones | Calibración, desbalance, fuga, datos sintéticos y edad relativa incorporados con fuentes verificadas |
| D-16 | Trazabilidad experimental | Hashes y ausencias históricas declarados; nueva corrida sigue pendiente y separada |
| F-metodología | Objetivos, criterios y uso de Scrum | Objetivos verificables, evaluación retrospectiva y uso incremental —no Scrum formal— explicitados |
| F-ética | Menores, consentimiento, retención, baja y etiquetado | Sección incorporada; medidas propuestas diferenciadas de funciones implementadas |

Archivos:

- Fuente aprobada, sin cambios: `C:\Users\Usuario\Desktop\TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_REVISADA_26-08-2026_v2.docx`.
- Backup byte a byte: `docs/revision_octubre_2026/word/ORIGINAL_APROBADO_26-08-2026_v2_BACKUP_SHA77261990.docx`.
- Borrador corregido: `docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_CORREGIDO_OCTUBRE_2026_BORRADOR.docx`.
- Transformación reproducible: `scripts/correct_word_block6_october_2026.py`.

Evidencia:

- SHA-256 de fuente y backup: `77261990098b6fd0761c7e1d27fcaf8cb91bef8fe13346c39836075e6049955d`.
- El DOCX corregido se reabrió con `python-docx`; el ZIP interno no presenta
  errores y conserva 22 tablas, 10 secciones, 27 objetos gráficos y los mismos
  20 archivos multimedia del original.
- Se comprobaron por contenido el abstract, la sección ética, el legajo marcado
  como pendiente, el conteo 17.608 y la ausencia de la frase incorrecta
  “17.994 parámetros entrenables”.
- Ruff y `py_compile` verifican el script de transformación.

Decisiones y pendientes:

- La autorización actual de ejecutar el Bloque 6 se tomó como confirmación para
  incorporar el abstract que anteriormente se había omitido por decisión del
  usuario. Quedó registrado para revisión.
- Falta informar el legajo; el borrador muestra `[PENDIENTE DE INFORMAR]` y no
  inventa un valor.
- Los índices, listas y campos no se regeneraron. La actualización y revisión
  visual del PDF corresponden al Bloque 8.
- La automatización de Word mediante COM no respondió dentro de 30 segundos y
  se cerró el proceso oculto creado por esta verificación. No se guardó ni se
  actualizó ningún campo. La apertura estructural del DOCX sí quedó validada.
- CI del árbol corregido, commit final, sincronización de entrega y PDF final
  permanecen pendientes del Bloque 8.
- Las correcciones generales de estilo, citas, numeración y estructura de todas
  las observaciones F permanecen para el Bloque 7; este bloque sólo cubrió los
  puntos de contenido, metodología y ética que le correspondían.

## Punto de reanudación — 2026-10-05

Se consolidó el estado completo, las restricciones, la evidencia y el próximo
paso en `docs/revision_octubre_2026/CONTINUAR_AQUI_2026-10-05.md`. La próxima
acción es ejecutar únicamente el Bloque 7 sobre una copia del borrador del
Bloque 6. El repositorio de entrega continúa limpio y sin sincronizar.

Por autorización posterior del usuario, el estado completo de los bloques 1–6
se guardó en el proyecto principal mediante el commit de respaldo `1782dc9`
(`checkpoint: complete October review blocks 1-6`). No se hizo push y el
repositorio `TPScouting-entrega` no fue modificado.

## Bloque 7 — Redacción, fuentes y estructura

Estado: cerrado localmente el 2026-10-06. No se actualizaron campos, índices ni
listas de Word y no se exportó PDF; esas verificaciones corresponden al Bloque 8.

| ID interno | Observación de F | Cambio | Estado |
|---|---|---|---|
| F-01 | Artefactos de edición y enumeraciones | Se eliminaron `Esta versi{on`, fragmentos y pseudoencabezados; se condensaron las secciones 2.1.1–2.1.5 | Resuelta en borrador |
| F-02 | Estructura, numeración, Discusión y diagramas repetidos | Tabla 3-1 renombrada como secuencia; comparativas no usadas eliminadas; Discusión 6.4 incorporada; capítulo 7 en plural; anexos ampliados justificados | Resuelta; índices pendientes del Bloque 8 |
| F-03 | Voz inconsistente | Segunda y primera persona reemplazadas por redacción impersonal | Resuelta en los hallazgos auditados |
| F-04 | Lenguaje promocional | Se retiraron afirmaciones de superioridad, facilidad, escalabilidad y rendimiento sin evidencia | Resuelta en los hallazgos auditados |
| F-05 | Puntuación | Se corrigieron comas entre sujeto y verbo y frases señaladas de resumen, introducción y conclusiones | Resuelta en los hallazgos auditados |
| F-06 | Terminología y tiempos verbales | Se normalizó “panel general”; probabilidad, calibración y score permanecen diferenciados; pasado/presente/futuro se usan según evidencia | Resuelta; rutas y archivos conservan nombres técnicos como `/dashboard` |
| F-07 | Referencias sin cita y APA 7 | 21 referencias ordenadas, citadas y con sangría francesa; se eliminaron recuperaciones mensuales no demostrables | Resuelta en borrador |
| F-08 | Fuentes faltantes | Se incorporaron PyTorch, scikit-learn, Scrum y fuga de datos; las fuentes de calibración, PR-AUC, edad relativa y ley ya añadidas se conservaron | Resuelta en borrador |
| F-09 | Marco teórico desconectado | Se condensó y alineó con calibración, desbalance, fuga, datos sintéticos y sesgos realmente usados | Resuelta en borrador |
| F-10 | Criterios e hipótesis a priori | Se mantuvo la declaración de evaluación retrospectiva sin inventar hipótesis | Resuelta desde Bloque 6 y verificada |
| F-11 | Ética y datos de menores | Se citó la Ley 25.326 y se conservaron límites sobre consentimiento, retención, baja y etiquetado | Resuelta desde Bloque 6 y revisada |
| F-12 | Scrum/Sprint en metodología y glosario | Se aclaró que sólo fue referencia; se eliminaron ambos términos del glosario como prácticas aplicadas | Resuelta en borrador |

Cambios documentales principales:

- Se creó `scripts/correct_word_block7_october_2026.py`, ligado al SHA-256 del
  borrador verificado del Bloque 6.
- Se generó
  `docs/revision_octubre_2026/word/TRABAJO_FINAL_TPScouting_CORREGIDO_BLOQUE7_2026-10-06.docx`.
- Se eliminaron tres tablas comparativas sobre tecnologías no implementadas y
  se renumeraron las siete tablas restantes del capítulo 4 en el contenido.
- Se agregó `6.4 Discusión` y la sección ética pasó a `6.5`.
- La bibliografía contiene 21 entradas, todas citadas al menos una vez en el
  cuerpo. La evidencia de contraste quedó en
  `docs/revision_octubre_2026/auditoria_fuentes_bloque7.md`.
- Se completaron la URL del repositorio, las variables de `create_admin.py` y el
  comando `Set-Location ..` que faltaba en la guía reproducible.
- El texto identifica el checkpoint `6e29b45…`; los cambios del Bloque 7 aún no
  se han sincronizado con la entrega.

Evidencia:

- DOCX válido: 10 secciones, 19 tablas, 27 objetos gráficos y los mismos 20
  archivos multimedia; `ZipFile.testzip()` sin errores.
- Auditoría automática: artefactos señalados ausentes, encabezados nuevos
  presentes, 21/21 referencias con cita en el cuerpo.
- Ruff y `py_compile` del script: correctos.
- Suite completa de aplicación: `116 passed, 1 skipped, 4 warnings`.
- Cobertura: `83,74 %`; umbral de 80 % satisfecho.
- Las cuatro advertencias All-NaN conocidas no cambiaron.

Pendientes:

- El legajo continúa sin informarse y no se inventó.
- La lista de tablas aún contiene las tres comparativas eliminadas y la
  numeración anterior porque no se actualizaron campos. El índice tampoco
  refleja Discusión, 6.5 ni el plural de Conclusiones. Debe resolverse mediante
  actualización controlada y revisión visual en el Bloque 8.
- La versión no fue exportada a PDF ni inspeccionada página por página.
- La CI del árbol corregido, el commit finalmente entregado y la sincronización
  selectiva permanecen pendientes.
- D-16, la evaluación del score combinado y las decisiones opcionales de
  infraestructura conservan el estado registrado anteriormente.

No se modificaron código funcional, bases, modelos, artefactos ML ni el
repositorio de entrega durante este bloque.

## Bloque 8 — Cierre y entrega verificable

Estado: cerrado localmente el 2026-10-07. La entrega pública no fue modificada.

| ID | Evidencia de cierre | Estado |
|---|---|---|
| A4 | 116 passed, 1 skipped, 4 warnings; cobertura exacta 83,74 % | Resuelta localmente |
| A5 | Fecha, versiones, logs, hashes y límites conservados | Resuelta localmente |
| A6 | CI #73 rotulada como histórica; CI del commit entregado pendiente | Parcial por publicación no autorizada |
| A8 | Demo temporal completa, listas/campos y PDF final verificados | Resuelta localmente |

Documento autoritativo:

- `word/TRABAJO_FINAL_TPScouting_FINAL_BLOQUE8_2026-10-07.docx`, SHA-256
  `5EAA742A6CD83BC91C0C406844A74348B3711D25846FA785AFB24B9F8C3DF36C`.
- `pdf/TRABAJO_FINAL_TPScouting_FINAL_BLOQUE8_2026-10-07.pdf`, SHA-256
  `4AD71AB97E66A4C5C481E59CE81DAA0D5728722E7005A2118CC03CA23A36DA5E`.

El PDF tiene 81 páginas. Se verificaron 45 entradas de listas contra la numeración
visible, 26 imágenes, 19 tablas, 9 secciones, ausencia de errores de campos y una
única marca pendiente: el legajo. La revisión visual selectiva confirmó portada,
índices, glosario y cierre. El detalle está en `cierre_bloque8_2026-10-07.md`.

La demo desde base temporal vacía verificó login, edad/categoría, historiales,
silueta y predicción. No se modificaron bases persistentes. El repositorio de
entrega sigue limpio en `ffdefdf8035c994ae285a270de0a4ff4e9f336a8`.

Pendientes finales: legajo; sincronización/push autorizados; CI del commit realmente
entregado; D-16 histórico y opcionales ya registrados. Veredicto:
**REQUIERE CORRECCIONES MENORES**.

Regla de maquetación permanente comunicada el 2026-10-07: portada, índice general,
resumen, abstract y bibliografía deben comenzar en hojas separadas. Esta condición
debe volver a comprobarse después de cualquier regeneración del Word o PDF.

Revisión lingüística final del 2026-10-07: se generó
`TRABAJO_FINAL_TPScouting_REVISION_GRAMATICAL_2026-10-07` en DOCX y PDF. Se
aplicaron 15 sustituciones verificadas y se retiraron cuatro párrafos vacíos con
estilos de título o leyenda. La revisión preservó evidencia, métricas, referencias,
imágenes y tablas. El control documental volvió a pasar y la separación de portada,
índice, resumen, abstract y bibliografía fue confirmada visualmente. La única marca
editorial restante es el legajo.
