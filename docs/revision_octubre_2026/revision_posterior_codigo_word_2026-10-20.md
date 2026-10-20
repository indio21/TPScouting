# Revisión posterior del código y del Word — 20/10/2026

## Cierre posterior autorizado

Los hallazgos de este informe quedaron corregidos y verificados en la versión
`TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20`.
Código funcional: `2a27d53ae4f8a4849bee7d1bea362fd9146886b5` en entrega;
publicación documental: `fdf2805975efbf1de78dfe7d350f0ebb30f1c1bd`.
Suite: 131 aprobadas, 1 omitida, 4 warnings; cobertura 83,31 %. CI, contenedor y
macOS del código funcional aprobaron. Word/PDF, Arial, índices, contenido y
numeración pasaron los controles registrados en
`cierre_correcciones_2026-10-20/`.

El contenido siguiente conserva el diagnóstico anterior para trazabilidad;
sus estados abiertos no representan el estado después de aplicar las correcciones.

**Veredicto: REQUIERE CORRECCIONES MENORES.**

La revisión encontró dos entradas de autenticación que generan HTTP 500, un
desfase de trazabilidad en el Word y diferencias puntuales de tipografía. No se
modificaron código de aplicación, Word/PDF ni el repositorio de entrega.

## Artefactos y alcance

- Proyecto principal: `C:\Tesis\TPScouting`, `main`,
  `3c6be7078445eab6f9a6ccb0490b47589410732a`.
- Entrega: `C:\Tesis\TPScouting-entrega`, `main`,
  `d3e62fe720e24d89b2715ad50a0c100a66be43df`.
- Word: `final/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20.docx`.
  SHA-256: `2EA4A6EA4C99617CB3EDA36F8C9218E28D4D9B479D812F1240F81347569AAD16`.
- PDF: mismo nombre, extensión `.pdf`, 97 páginas.
  SHA-256: `4774BD07C32DD393BB56FC7F3DD06E8D121AA19DBF7064A66F797481D9AE55AD`.
- Se usó `correccion-octubre.md` como lista histórica de riesgos, contrastando
  autenticación, CI, métricas, metadatos y formato actuales. Esta es una revisión
  dirigida; no vuelve a certificar individualmente todas las observaciones docentes.

## A. Hallazgos reproducidos de código

| ID | Severidad | Ubicación | Evidencia | Acción propuesta y criterio de cierre | Estado |
|---|---|---|---|---|---|
| SEG-09 | Bajo | `scouting_app/routes/auth.py:51`, `login` | Con usuario existente, CSRF válido y campo `password` ausente, `check_password_hash` recibe `None`; la petición devuelve HTTP 500. Base temporal, sesión sin autenticar. | Validar campos antes del hash; garantizar cierre de sesión de base ante errores. Regresión que exija respuesta controlada y sesión sin autenticar. | ABIERTO |
| SEG-10 | Bajo | `scouting_app/services/security.py:76`, `require_csrf` | Con token enviado `é`, `secrets.compare_digest` sobre cadenas no ASCII provoca una excepción y HTTP 500. La sesión no obtiene acceso. | Mantener comparación constante y rechazar tokens no válidos sin excepción. Regresión que exija HTTP 400 y ausencia de cambios. | ABIERTO |

No se observó una autenticación autorizada por esos casos: ambos conservaron
`authenticated=false`. Esto no elimina el defecto de manejo de entradas ni
constituye una auditoría de seguridad exhaustiva.

Reproducción conservada en:

- `scripts/audit_auth_malformed_inputs_2026_10_20.py`.
- `final/reproduccion_entradas_auth_2026-10-20.json`.

## B. Correcciones históricas inspeccionadas

Se mantienen implementados la validación de redirecciones internas, el limitador
por usuario normalizado, la relectura de usuario/rol, la limpieza de sesión,
el healthcheck mínimo, la política de fotos y el bloqueo del pipeline web en
producción. Hay pruebas específicas existentes para esas conductas.

La suite de entrega ejecutada en el cierre de hoy obtuvo `123 passed`,
`1 skipped`, `4 warnings` y `83,29 %` de cobertura. Los dos casos adicionales
reproducidos en esta revisión no forman parte todavía de esa suite. Por eso
la aprobación de las 123 pruebas no demuestra que esos defectos estén corregidos.

Ruff crítico, Ruff gradual y `pip check` se ejecutaron durante esta revisión y
aprobaron. No se instalaron ni actualizaron dependencias. La auditoría de
dependencias de CI del commit de entrega también aprobó; no se afirma haber
repetido aquí un `pip-audit` local nuevo.

## C. Trazabilidad y resultados del Word

| ID | Ubicación | Hallazgo | Evidencia y acción propuesta | Estado |
|---|---|---|---|---|
| DOC-TRAZA-01 | Tabla 6-2, §6.2.4, Tabla 7-1, §9.4 y §10.2 | Falta incorporar el último commit de entrega evaluado y su CI. §10.2 aún presenta la nueva ejecución como requisito pendiente. | El Word cita `8093fe6` y `4c5ef2c`, válidos como antecedentes. El commit publicado actual es `d3e62fe720e24d89b2715ad50a0c100a66be43df`; CI `39507542398`, `success`, Python 3.11/3.12 y auditoría aprobados. Incorporar esta evidencia conservando el carácter histórico de la anterior. | PARCIAL |
| DOC-TRAZA-02 | §6.2 y Tabla 6-2 | La ejecución local citada está fechada el 19/10; existe una más reciente del 20/10 con las mismas cifras. | El dato histórico no es falso. Para presentar el estado más reciente, actualizar fecha y referencia del registro del 20/10. Tras corregir SEG-09/10, volver a medir antes de fijar cifras definitivas. | PARCIAL |
| DOC-SEG-01 | §5.5 | El texto afirma que HSTS se emite cuando la aplicación reconoce HTTPS. | `app.py:281–282` condiciona la cabecera al modo de producción, sin consultar `request.is_secure`. Corregir la descripción al comportamiento real; no cambiar código solo para justificar la frase. | ABIERTO |

La captura CI `38815440730` corresponde realmente al paquete `4c5ef2c`. Puede
conservarse como evidencia fechada. No debe renombrarse como si fuera una captura
del run nuevo; el run `39507542398` se verificó por la API de GitHub.

No es posible introducir en un Word el hash del futuro commit que todavía va a
contenerlo. La solución trazable es citar el commit ejecutable evaluado y su CI,
y registrar el commit exacto de publicación y los hashes del Word/PDF en un
registro externo. Evitar un ciclo de cambios que solo intenta actualizar el hash
del propio documento.

### Métricas que sí coinciden

- Tabla 6-3: umbrales, accuracy, ROC-AUC, PR-AUC, F1, precision y recall coinciden
  con `scouting_app/training_metadata.json`, con el redondeo mostrado.
- Las matrices de confusión coinciden con los metadatos históricos.
- La comparación histórica/candidata conserva valores consistentes con los
  artefactos de `artifacts/runs/20261010T_local_seed42_leakage_safe_v2`:
  crudo histórico ROC-AUC 0,9203, PR-AUC 0,5461, Brier 0,1049, log loss 0,3294,
  F1 0,5291; candidato 0,9056, 0,4888, 0,1175, 0,3673 y 0,4931.
- Carga real sin reentrenar: 17.608 parámetros entrenables, 386 buffers y
  17.994 elementos del `state_dict`; 64 columnas antes de transformar y 68
  variables después. Ninguna de las dos etiquetas integra `MODEL_FEATURE_COLUMNS`.
- No se reentrenaron ni sustituyeron modelos.

## D. Tipografía

| ID | Ubicación actual | Observación | Acción propuesta | Estado |
|---|---|---|---|---|
| FORMA-01 | §3.1.2 | Subtítulo de nivel 3 con 16 pt, frente a numerosos títulos de 14 pt. | Unificar con los subtítulos equivalentes, preservando el texto. | ABIERTO |
| FORMA-02 | §3.3, §3.4, §6.4, §6.5, §7.1, §9.3 y §10.5 | Algunos títulos de nivel 2 tienen 12 pt y otros 14 pt. | Aplicar una jerarquía consistente de títulos. | ABIERTO |
| FORMA-03 | Caption de Tabla 6-4 | Está a 10 pt; los demás captions se presentan mayormente a 14 pt. | Igualar con el estilo de captions acordado. | ABIERTO |
| FORMA-04 | §9.4 | Bloques de comandos mezclan Arial y Courier New. | Unificar la familia monoespaciada y conservar tamaño de al menos 10 pt. | ABIERTO |

El cuerpo ordinario está mayormente en Arial 14. No hay tamaños Word explícitos
por debajo de 10 pt, pero cumplir ese mínimo no equivale a tener una jerarquía
tipográfica uniforme. Este control no mide las letras rasterizadas de imágenes.

## E. Escritura y maquetación

- El control sobre 940 párrafos/celdas no detectó palabras consecutivas repetidas,
  espacios indebidos antes de puntuación, primera persona plural, segunda persona
  ni marcadores editoriales pendientes.
- Las cuatro coincidencias del detector de lenguaje promocional contienen
  «no garantiza»; son falsos positivos, no motivos para reescribir esos párrafos.
- Las cuatro unidades largas detectadas son comandos, no oraciones académicas.
- No se detectó en ese control otro problema sustancial de redacción. Es un
  resultado acotado; no garantiza ausencia absoluta de errores lingüísticos.
- Siguen correctas las 88 entradas del índice y las 45 de figuras/tablas; no hay
  errores de campos ni numeración. Se conservan los inicios independientes de
  portada, índice, resumen, abstract y bibliografía.

## F. Orden recomendado y límites

1. Corregir SEG-09/10 en el proyecto propio y agregar regresiones significativas.
2. Ejecutar suite y cobertura; registrar los resultados nuevos, sin anticipar el
   futuro total de pruebas.
3. Actualizar conjuntamente trazabilidad, fecha de pruebas, descripción HSTS y
   diferencias de tipografía en una copia nueva del Word.
4. Actualizar índices/listas y exportar/verificar PDF después de esos cambios.
5. Mantener intacta la entrega publicada durante ese bloque; sincronizarla
   únicamente cuando corresponda según instrucciones del usuario.

No se repitieron instalación desde cero, macOS, contenedor, entrenamiento,
validación deportiva ni revisión legal/bibliográfica completa en esta pasada.
Las limitaciones ya declaradas del MVP permanecen vigentes.
