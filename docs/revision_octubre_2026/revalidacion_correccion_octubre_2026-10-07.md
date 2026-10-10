# Revalidación de correcciones de octubre — TPScouting

**Fecha de revisión:** 2026-10-07, America/Buenos_Aires
**Informe base:** `C:\Users\Usuario\Desktop\correccion-octubre.md`
**Documento vigente:** `TRABAJO_FINAL_TPScouting_CORRECCIONES_MENORES_2026-10-07.docx`
**DOCX SHA-256:** `CBED5344342F4E96E0CD996520093AF99A364B851EE24B792A45E171294F474F`
**PDF vigente:** `TRABAJO_FINAL_TPScouting_CORRECCIONES_MENORES_2026-10-07.pdf`
**PDF SHA-256:** `17850B625C63E06282E1FE68CEEAE2DF8A4063604F1645EE5DBC3CBEF1F2D582`
**Extensión verificada:** 94 páginas, 28 imágenes, 19 tablas y 9 secciones
**Repositorio de entrega:** `https://github.com/indio21/TPScouting-entrega`
**Rama y commit revisados:** `main`, `7a47bd3f164e865677f2c70075cbedc9fa63427d`
**CI del commit:** ejecución `37707576905`, concluida con éxito

## Veredicto

- **Código:** funcional y cubierto por pruebas para el alcance revisado. Permanece
  abierta la decisión sobre cuatro avisos de seguridad de `torch==2.9.1` y el hecho
  de que la auditoría de CI omite el build `2.9.1+cpu`.
- **Documento:** las correcciones académicas y de forma están mayormente resueltas,
  pero la trazabilidad de entrega y los comandos de instalación quedaron
  desactualizados después de publicar `7a47bd3`.
- **Veredicto general:** **REQUIERE CORRECCIONES MENORES**.

Estados usados: `RESUELTO`, `PARCIAL`, `ABIERTO` y `NO APLICA`.

## A. Prioridad 1

| ID | Observación de octubre | Evidencia actual | Estado |
|---|---|---|---|
| A1 | Open redirect mediante `next` | `safe_internal_redirect_target` rechaza esquemas, `//`, barras invertidas, controles y variantes codificadas; existen regresiones en `test_auth.py`. | RESUELTO |
| A2 | Rate limiting evadible rotando `X-Forwarded-For` | La clave se basa en usuario normalizado y no confía en ese header; la regresión rota el header y verifica el bloqueo. | RESUELTO |
| A3 | Conteo incorrecto de parámetros entrenables | El documento distingue 17.608 parámetros de `model.parameters()`, 386 buffers y 17.994 elementos de `state_dict`. | RESUELTO |
| A4 | Métricas de pruebas desactualizadas | El documento informa 116 passed, 1 skipped, 4 warnings y 83,74 %, coincidentes con la ejecución sobre el árbol entregado. | RESUELTO |
| A5 | Evidencia CI de otro repositorio/commit | El documento todavía presenta CI #73 como histórica y deja pendiente la CI final. Ya existe la CI exitosa `37707576905` para `7a47bd3`; el texto quedó desactualizado. | ABIERTO |
| A6 | Falta URL, commit y comando de demo | La URL y `scripts/iniciar_demo.py` están incorporados, pero el commit indicado sigue siendo `ffdefdf`, no el entregado `7a47bd3`. | PARCIAL |
| A7 | Portada institucional incompleta | Incluye UCSE, Departamento Académico Rafaela, carrera, alumno, directores, ciudad y fecha. El legajo fue omitido por decisión expresa del usuario. | RESUELTO; legajo NO APLICA |
| A8 | Índices y listas desfasados | El PDF vigente tiene índice/listas regenerados; el control previo registró 45 entradas sin discrepancias. Las páginas visibles comienzan en índice 1 y llegan a anexos en 93. | RESUELTO |
| A9 | Falta abstract en inglés | `ABSTRACT` y keywords están incorporados en una página propia después del resumen. | RESUELTO |

## B. Código — seguridad

| ID | Observación | Evidencia actual | Estado |
|---|---|---|---|
| SEG-03 | Sesión no revalidada y sin limpieza al iniciar sesión | `_refresh_authenticated_session` consulta usuario y rol; invalida usuarios ausentes/roles inválidos. Login ejecuta `session.clear()`. Hay regresiones de borrado y cambio de rol. | RESUELTO |
| SEG-04 | `/health` exponía excepciones y contadores | La respuesta pública devuelve solo `status`; las excepciones quedan en log. Los contadores se consultan desde configuración administrativa. | RESUELTO |
| SEG-05 | Sin CSP ni HSTS | Se genera nonce CSP por request y HSTS se agrega en runtime productivo; las cabeceras tienen pruebas. Falta únicamente comprobarlas en un despliegue HTTPS/proxy real. | RESUELTO con límite operativo |
| SEG-06 | `photo_url` sin política | `is_valid_player_photo_url` se aplica a alta, edición e importación; permite rutas propias seguras y HTTPS. README documenta silueta local y fotos externas. | RESUELTO |
| SEG-07 | Comparación CSRF no constante | `secrets.compare_digest` se usa con control de tipos y ausencias; existe regresión. | RESUELTO |
| SEG-08 | Riesgo de deserialización joblib/PyTorch | README y RUNBOOK exigen artefactos confiables; la UI no permite cargarlos. PyTorch usa `weights_only=True`. El riesgo inherente de joblib permanece documentado. | RESUELTO |

Los controles positivos de octubre siguen presentes por inspección y pruebas:
CSRF en operaciones mutantes, roles, hash y política mínima de contraseñas, secreto
obligatorio en producción, cookies seguras según entorno y debug desactivado. Bandit
no se reprodujo en esta revalidación porque no está instalado en la `.venv`.

## C. Dependencias, datos, calidad, pruebas, CI e infraestructura

| ID | Observación | Evidencia actual | Estado |
|---|---|---|---|
| DEP-01 | `torch==2.9.1+cpu` incompatible con macOS | `requirements.txt` usa `torch==2.9.1`; Windows/Linux CPU usan `requirements-torch-cpu.txt`; las plataformas están diferenciadas en README/RUNBOOK. macOS no fue ejecutado localmente. | RESUELTO con límite de plataforma |
| DEP-02 | Lock no usado y mezclado con herramientas | CI y Render instalan `requirements-lock.txt` y el manifest CPU. Desarrollo y documentos se separaron en `requirements-dev.txt` y `requirements-docs.txt`. Scikit-learn queda fijado en 1.8.0. | RESUELTO |
| DEP-03 | Vulnerabilidades de click, lxml y torch; falta auditoría CI | `click==8.3.3`, `lxml==6.1.0` y `pip-audit` en CI están corregidos. Sin embargo, la auditoría directa de `requirements.txt` encontró cuatro avisos en `torch==2.9.1`: `PYSEC-2026-139`, `PYSEC-2025-194`, `PYSEC-2025-195` y `PYSEC-2026-2286`. La auditoría del entorno/CI omite `torch 2.9.1+cpu`. | PARCIAL |
| DATOS-01 | Generación/entrenamiento síncronos en producción | La ruta de settings bloquea la operación en producción; el autoentrenamiento está desactivado por defecto y en Render. Existen pruebas de no ejecución. | RESUELTO |
| DATOS-02 | Riesgo de PostgreSQL Free | README/RUNBOOK documentan caducidad, capacidad, ausencia de backups y uso solo demostrativo. No se cambió ni contrató un plan. | RESUELTO |
| CAL-01 | Módulos grandes, contenedor implícito y configuración repetida | `env_int` centraliza enteros y tiene pruebas. No se implementó application factory; `app.py` tiene 1.768 líneas y `routes/players.py`, 1.760. | PARCIAL |
| CAL-02 | Sin lint; prints; UTC deprecado | Ruff crítico está en CI y `datetime.now(timezone.utc)` reemplaza el caso señalado. Persisten deuda de estilo y usos históricos de `print`; Ruff no cubre todo el proyecto con una política completa. | PARCIAL |
| CAL-03 | Archivo de pruebas monolítico | Se agregaron suites por dominio (`auth`, `compare_staff`, `ml_runtime`, `quality_config`, `admin_seed`), pero `test_mvp_regressions.py` conserva 3.328 líneas. | PARCIAL |
| TEST-01 | Cobertura insuficiente en módulos relevantes | Se añadieron pruebas de compare, staff, runtime ML, settings, admin y seed. La suite alcanza 116 passed y 83,74 % total. | RESUELTO |
| CI-01 | Sin cache, lint, auditoría ni umbral | CI incluye cache, Ruff crítico, `pip-audit`, Python 3.11/3.12 y `--cov-fail-under=80`; la ejecución de `7a47bd3` pasó. La limitación de PyTorch se registra en DEP-03. | RESUELTO |
| INFRA-01 | Sin Dockerfile, `.env.example` ni LICENSE | Los tres siguen ausentes. Octubre los declaró opcionales y el usuario no autorizó elegir licencia ni cambiar la estrategia Render nativa. | NO APLICA como requisito de cierre |

## D. Código — validez de ML y trazabilidad

| ID | Observación | Evidencia actual | Estado |
|---|---|---|---|
| ML-01 | Circularidad y cuantiles del target antes del split | El documento declara que la prevalencia es sintética y que cuantiles/cuotas se calculan globalmente. El diseño no fue reentrenado para eliminar ese riesgo metodológico. | PARCIAL |
| ML-02 | `potential_label` persistida podría entrar como feature | El documento y el pipeline distinguen `potential_label`, `temporal_target_label` y las 64 columnas de entrada; las etiquetas/predicciones no ingresan como features del entrenamiento oficial. | RESUELTO |
| ML-03 | Umbral evaluado distinto de bandas de interfaz | El documento diferencia probabilidad cruda, calibrada, umbral de validación 0,25/0,825, score combinado y bandas 0,60/0,80. También declara que el score combinado carece de test propio. | RESUELTO con limitación declarada |
| ML-04 | Datos personales de menores sin política implementada | El MVP usa datos sintéticos y el documento incorpora consentimiento, minimización, retención, supresión y riesgo de etiquetado. Esas medidas se declaran futuras; la aplicación no implementa un flujo completo para uso real. | PARCIAL |

## E. Documento — contenido

| ID | Observación | Evidencia actual | Estado |
|---|---|---|---|
| D-07 | Seguridad presentada como suficiente | Se describen límites de una instancia, memoria de proceso, redirección segura y revalidación de sesión. | RESUELTO |
| D-08 | Una seed, sin incertidumbre; comparación no interpretable | Se incorporó bootstrap pareado de 2.000 remuestras con intervalos y se declara que no demuestra superioridad ni equivalencia. La limitación de un split permanece explícita. | RESUELTO |
| D-09 | Umbral experimental confundido con interfaz | La relación entre umbral elegido en validation, evaluación en test y bandas visuales está explicada. | RESUELTO |
| D-10 | Prevalencia sintética presentada como observación | Se indica expresamente que 7,985 % está fijado por diseño y no describe futbolistas reales. | RESUELTO |
| D-11 | Objetivos aspiracionales y pregunta sin responder | Los objetivos se reformularon, la pregunta se responde en 1.1 y las conclusiones separan logro técnico de impacto no validado. | RESUELTO |
| D-12 | Comparativas extensas de tecnologías no usadas y psycopg incorrecto | Se redujo a una decisión breve y se aclara que .NET, Peewee, MySQL y SQL Server no se implementaron ni compararon. Las dependencias vigentes usan psycopg 3. | RESUELTO |
| D-13 | Tres tablas declaradas frente a once entidades reales | El texto enumera las once entidades implementadas y la tabla correspondiente. | RESUELTO |
| D-14 | Afirmaciones de impacto sin sustento | Introducción, resumen y conclusiones delimitan factibilidad técnica y niegan validación de precisión, ahorro, igualdad o eficacia deportiva. | RESUELTO |
| D-15 | Guía solo Windows, incompleta y sin demo portable | Se añadieron Windows/Linux/macOS, variables de admin, retorno a raíz e `iniciar_demo.py`. Sin embargo, los comandos aún indican `requirements.txt -r requirements-dev.txt` para Windows/Linux y no reflejan el lock/manifiesto CPU actuales; también parten del repositorio principal y del commit anterior. | PARCIAL |
| D-16 | Sin validation loss, duración ni SHA de corrida histórica | El documento no inventa esos datos y declara su ausencia. No se repitió una corrida multiseed ni se reconstruyó metadata histórica. | PARCIAL |

## F. Documento — forma, redacción y fuentes

| ID | Observación | Evidencia actual | Estado |
|---|---|---|---|
| F-01 | Artefactos de edición y enumeraciones incompletas | Las frases señaladas ya no aparecen; las enumeraciones fueron reorganizadas. | RESUELTO |
| F-02 | Numeración, cronograma, conclusión, discusión y diagramas repetidos | Los títulos están numerados; Tabla 3-1 se llama “Secuencia de actividades”; existe 6.4 Discusión; capítulo 7 usa “Conclusiones”. Los diagramas ampliados se conservan con función declarada y por decisión del usuario. | RESUELTO |
| F-03 | Voz en segunda persona y primera plural | La auditoría del DOCX registró cero coincidencias y las frases originales no aparecen. | RESUELTO |
| F-04 | Lenguaje promocional | Las expresiones originales fueron eliminadas; las coincidencias restantes son formulaciones negativas como “no garantiza”. | RESUELTO |
| F-05 | Comas entre sujeto y verbo | Las construcciones señaladas ya no aparecen; el control automático no encontró espacios indebidos antes de puntuación. | RESUELTO |
| F-06 | Terminología y tiempos verbales inconsistentes | Se diferencia etiqueta, probabilidad cruda, calibrada, score combinado y bandas; “panel general” es el término español predominante. | RESUELTO |
| F-07 | Bibliografía sin correspondencia y APA incorrecta | Las 21 referencias tienen uso en el cuerpo, incluidas las seis antes huérfanas; se eliminó la fecha de recuperación impropia y se incorporaron DOI/URL. No se volvió a comprobar manualmente la disponibilidad de cada enlace en esta pasada. | RESUELTO con límite de verificación externa |
| F-08 | Faltan fuentes de calibración, PR-AUC, scikit-learn, PyTorch, leakage, sintéticos y sesgo | Se incorporaron Niculescu-Mizil y Caruana, Saito y Rehmsmeier, Pedregosa et al., Paszke et al., Kaufman et al. y Cobley et al., con citas en el texto. | RESUELTO |
| F-09 | Marco teórico no cubre conceptos usados | Las secciones 2.1.6.4 y 2.1.6.5 desarrollan calibración, desbalance, fuga, datos sintéticos y sesgos de edad/maduración. | RESUELTO |
| F-10 | Metodología sin criterios a priori ni hipótesis | Se aclara que los criterios son retrospectivos y que no existieron umbrales cuantitativos a priori; no se inventaron hipótesis posteriores. | RESUELTO |
| F-11 | Ética insuficiente | Se incorporaron Ley 25.326, consentimiento/representantes, minimización, retención, supresión, supervisión y riesgo de “potencial bajo”, distinguiendo medidas futuras. | RESUELTO |
| F-12 | Scrum/Sprint presentados como aplicados | Se aclara que hubo un enfoque incremental inspirado en iteración, pero no Scrum formal ni sprints documentados. | RESUELTO |

## Evidencia reproducida o comprobada

| Control | Resultado | Limitación |
|---|---|---|
| Identidad Git entrega | `main` en `7a47bd3f...`, árbol limpio y sincronizado | No prueba disponibilidad del despliegue Render |
| CI GitHub | `37707576905`, `success`, SHA exacto `7a47bd3f...` | La auditoría instalada omite el build CPU de PyTorch |
| Suite del árbol entregado | 116 passed, 1 skipped, 4 warnings; cobertura 83,74 % | Cuatro warnings All-NaN conocidos; smoke visual opt-in se ejecutó aparte |
| Smoke visual | 1 passed con Playwright | Verificación local, no del despliegue público |
| Demo temporal | 60 jugadores, edades/categorías/historias/admin verificados | Datos sintéticos y SQLite temporal |
| Ruff crítico/compilación/pip check | Aprobados | Ruff completo conserva deuda de estilo |
| `pip-audit --local` | Sin avisos, PyTorch omitido | Resultado incompleto para torch CPU |
| `pip-audit -r requirements.txt --no-deps` | Cuatro avisos en torch 2.9.1 | No se evaluó compatibilidad de una actualización |
| DOCX | Hash, 598 párrafos, 19 tablas, 9 secciones y 28 imágenes | Sin inspección visual exhaustiva de imágenes |
| PDF | Hash y 94 páginas; secciones principales localizadas | Se reutilizó el control previo de índices/listas; no se regeneraron campos |

## Pendientes que deben corregirse o decidirse

1. Actualizar en el documento el commit de entrega a `7a47bd3f...`, reemplazar el
   estado “CI pendiente” por la ejecución `37707576905` y revisar las figuras que
   muestran únicamente CI histórica.
2. Alinear la sección 9.4 con `requirements-lock.txt`,
   `requirements-torch-cpu.txt`, `requirements-dev.txt` y el repositorio de entrega.
3. Evaluar los cuatro avisos actuales de PyTorch y decidir una actualización
   compatible o una mitigación documentada. La CI debe auditar el paquete de forma
   que no quede omitido silenciosamente.
4. Mantener declarados como deuda no bloqueante: application factory, división
   adicional del archivo de regresiones, lint gradual y opciones Docker/.env/LICENSE.
5. D-16 solo puede cerrarse mediante una nueva corrida separada; no debe alterarse
   la evidencia histórica para simular metadata ausente.

No se modificaron el DOCX, el PDF, el código, las bases, los modelos ni el
repositorio de entrega durante esta revalidación.
