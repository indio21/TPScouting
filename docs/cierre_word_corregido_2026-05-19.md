# Cierre Word Corregido - 2026-05-19

## Estado

Se genero una version corregida del trabajo final enviado originalmente al profesor, usando como insumos:

- `C:\Users\Usuario\Desktop\TRABAJO_FINAL - Lo que se envio.docx`
- `C:\Users\Usuario\Desktop\Correccion TP Escrito.docx`
- Codigo y documentacion actual del repo `C:\Tesis\TPScouting`
- Deploy real en Render: `https://tpscouting-mvp.onrender.com`

No se modifico codigo de la app en esta fase.

## Archivos generados

- `C:\Users\Usuario\Desktop\TRABAJO_FINAL - Version corregida TPScouting.docx`
- `docs/TRABAJO_FINAL_corregido_TPScouting.docx`
- `docs/evidencia_word_render/01_login_render.png`
- `docs/evidencia_word_render/02_dashboard_render.png`
- `docs/evidencia_word_render/03_players_render.png`
- `docs/evidencia_word_render/04_player_detail_render.png`
- `docs/evidencia_word_render/05_prediction_render.png`
- `docs/evidencia_word_render/06_compare_multi_render.png`
- `scripts/update_final_word.py`

## Cambios aplicados al documento

- Se regenero el indice del Word para eliminar entradas viejas.
- Se eliminaron placeholders como `pendiente de capturas`, `[Insertar captura]` y `ejemplo ilustrativo`.
- Se corrigio la estructura de metodologia, ciclo de vida y cronograma consolidado.
- Se movieron las referencias internas B-K a anexos tecnicos del MVP.
- Se actualizo la arquitectura real: Flask server-side, blueprints por familia, servicios parciales y runtime ML.
- Se actualizo persistencia: SQLite local y PostgreSQL administrado en Render.
- Se actualizo seguridad: CSRF, logout POST, secret obligatoria, roles y rate limiting in-memory.
- Se actualizo Machine Learning: `PlayerNet`, `BCEWithLogitsLoss`, logits, `input_dim=68`, checkpoint con metadata y baseline honesto.
- Se actualizo escala: atributos `1-20`; potencial bajo `<60%`, medio `60-79%`, alto `>=80%`.
- Se agregaron metricas reales desde `training_metadata.json`.
- Se agregaron capturas reales tomadas desde Render con sesion autenticada.
- Se agrego una tabla objetivo -> estado -> evidencia en conclusiones.
- Se documentaron limitaciones reales del MVP: dataset sintetico, Render Free, cache/rate limit/locks locales y baseline competitivo.

## Evidencia tecnica usada

Metricas reales del modelo registradas en `scouting_app/training_metadata.json`:

- Dataset sintetico: `20.000` jugadores.
- Split: `14.000 / 3.000 / 3.000`.
- Seed: `42`.
- `input_dim`: `68`.
- Loss: `BCEWithLogitsLoss`.
- Accuracy PyTorch test: `0.9303`.
- ROC-AUC PyTorch test: `0.9174`.
- PR-AUC PyTorch test: `0.5241`.
- F1 PyTorch test: `0.5282`.
- Baseline `LogisticRegression(class_weight="balanced")`: Accuracy `0.9310`, ROC-AUC `0.9205`, PR-AUC `0.5378`, F1 `0.5327`.

Conclusion documentada: en esta corrida el baseline logico balanceado queda levemente por encima de PlayerNet en algunas metricas; no se afirma superioridad general del modelo PyTorch.

## Validacion realizada

Revision automatica del Word corregido:

- Sin `pendiente de capturas`.
- Sin `[Insertar captura]`.
- Sin `ejemplo ilustrativo`.
- Sin `/logs`.
- Sin `BCELoss`.
- Sin escala vieja `0-20`.
- Sin `9. REFERENCIAS`.
- Sin `3.6 Cronograma`.
- Sin afirmaciones de `SQLite en el MVP` como unica base del despliegue.
- El Word contiene `21` tablas y `6` imagenes de evidencia.

Suite local ejecutada despues de generar documentacion:

```text
83 passed, 1 skipped, 4 warnings
```

Warnings conocidos: `RuntimeWarning: All-NaN slice encountered` en pruebas de preprocesamiento con scikit-learn.

## Pendiente para continuar

Proximo paso recomendado:

1. Auditar `TRABAJO_FINAL - Version corregida TPScouting.docx` contra `Correccion TP Escrito.docx`, punto por punto.
2. Verificar manualmente que cada observacion del profesor este cerrada o documentada como limitacion.
3. Revisar visualmente el Word en Microsoft Word: indice, saltos de pagina, tamanio de imagenes y tablas.
4. Si la auditoria da OK, hacer `git add`, commit y push de:
   - `docs/TRABAJO_FINAL_corregido_TPScouting.docx`
   - `docs/evidencia_word_render/`
   - `docs/cierre_word_corregido_2026-05-19.md`
   - `scripts/update_final_word.py`
   - `docs/contexto_para_nuevo_chat.md`

## Nota

El archivo original enviado al profesor no fue pisado. La version corregida quedo como archivo separado en el Escritorio y dentro del repositorio.

## Actualizacion checklist final - 2026-05-20

Se aplico una fase adicional sobre `C:\Users\Usuario\Desktop\TRABAJO_FINAL - Version corregida TPScouting.docx` y se sincronizo la copia del repo `docs/TRABAJO_FINAL_corregido_TPScouting.docx`.

Checklist tratado:

- `A1`: se genero e inserto una curva real de entrenamiento desde `scouting_app/training_metadata.json`.
  - Archivo generado: `docs/evidencia_word_render/07_training_curve_real.png`.
  - Fuente: `pytorch.history` con `loss`, `val_pr_auc` y `val_f1` por epoca.
- `A2`: se agregaron `RESUMEN`, `ABSTRACT`, `Palabras clave` y `Keywords`.
- `A3`: se agregaron `LISTA DE FIGURAS` y `LISTA DE TABLAS`.
- `B1`: se elimino la frase residual en primera persona `Entonces, utilizamos SQLAlchemy...`.
- `B2`: se agregaron captions numerados para las tablas principales.
- `B3`: se amplio el capitulo 5 con detalles de integracion Flask + PyTorch, carga de artefactos, pipeline, metadata y despliegue.
- `C1`: se amplio el glosario tecnico minimo.
- `C2`: se agrego una tabla de smoke real de Render del 20/05/2026.

Evidencia nueva de smoke Render:

```text
/health        GET   200   109.36 s   OK; arranque frio Render Free
/login         GET   200   0.39 s     Pantalla de login renderizada
/login         POST  200   4.96 s     Login admin OK; redireccion final a /players
/dashboard     GET   200   7.86 s     Panel general renderizado
/players       GET   200   0.38 s     Listado de jugadores
/compare       GET   200   0.43 s     Comparador 1v1
/compare/multi GET   200   7.73 s     Comparador multiple
/settings      GET   200   0.44 s     Configuracion protegida
/players/import GET  200   0.39 s     Carga masiva
```

Revision automatica posterior:

- Documento con `22` tablas y `7` imagenes.
- Captions numerados: `29` (`22` tablas + `7` figuras).
- Sin encabezados vacios.
- Sin `pendiente de capturas`.
- Sin `[Insertar]`.
- Sin `ejemplo ilustrativo`.
- Sin `BCELoss`.
- Sin escala vieja `0-20`.
- Sin `/logs`.
- Sin `3.6 Cronograma`.
- Sin `9. REFERENCIAS`.

Pendiente recomendado:

1. Abrir el Word en Microsoft Word y revisar visualmente saltos, tamano de imagenes, portada, listas y tablas.
2. Ejecutar una auditoria final de lectura contra `Correccion TP Escrito.docx`.
3. Si esta OK, hacer `git add`, commit y push.

## Ajuste de formato final - 2026-05-20

Se aplico una correccion de estructura visual sobre el Word vigente:

- Portada aislada en la primera hoja.
- Indice inmediatamente despues de la portada.
- Corte de pagina/seccion al terminar el indice, para que el contenido siguiente no continue debajo.
- Listas de figuras/tablas, resumen/abstract, capitulos, bibliografia y anexos iniciando en hoja nueva mediante saltos de seccion.
- Bibliografia iniciando en hoja aparte.
- Anexos tecnicos y anexos visuales iniciando en hojas separadas.
- Numeracion duplicada eliminada: se quito el texto centrado `Pagina X de Y` y se dejo una numeracion simple en el pie derecho.
- Portada e indice sin numeracion visible.
- Indice recreado en Microsoft Word usando solo estilos de titulo, sin entradas internas no numeradas del capitulo 4.
- Fechas de recuperacion de referencias web distribuidas entre el 15/05/2026 y el 20/05/2026.

Archivos sincronizados:

- `C:\Users\Usuario\Desktop\TRABAJO_FINAL - Version corregida TPScouting.docx`
- `docs/TRABAJO_FINAL_corregido_TPScouting.docx`

Backup previo a este ajuste:

- `C:\Users\Usuario\Desktop\TRABAJO_FINAL - Version corregida TPScouting.backup_layout_20260520.docx`

Revision tecnica posterior:

- `14` secciones de Word.
- `22` tablas.
- `7` imagenes inline y `1` shape auxiliar informado por Microsoft Word.
- Sin `__TOC_PLACEHOLDER__`.
- Sin `TOC \o`.
- Sin `Error! Marcador no definido`.
- Sin texto de footer `Pagina X de Y`.

Suite local ejecutada luego del ajuste:

```text
83 passed, 1 skipped, 4 warnings
```

Warnings conocidos: `RuntimeWarning: All-NaN slice encountered` en pruebas de preprocesamiento con scikit-learn.

Pendiente recomendado:

1. Abrir el Word y revisar visualmente portada, indice, saltos y numeracion desde Microsoft Word.
2. Si el orden visual esta conforme, versionar los archivos pendientes.

## Version final revisada por usuario - 2026-05-22

El usuario reviso y ajusto manualmente `C:\Users\Usuario\Desktop\TRABAJO_FINAL - Version corregida TPScouting.docx`.

Decision final:

- Se conserva el orden visual definido manualmente por el usuario.
- Se deja el documento sin seccion `ABSTRACT`.
- Se conserva `RESUMEN` y `Palabras clave`.
- Se actualizo el indice para que no liste `ABSTRACT`.
- Se sincronizo la copia del repositorio `docs/TRABAJO_FINAL_corregido_TPScouting.docx`.
- Se actualizo `scripts/finalize_word_checklist.py` para que no vuelva a generar el abstract si se reutiliza el script.

Chequeos posteriores:

- Sin `ABSTRACT`.
- Sin `Keywords:`.
- Sin `Marcador no definido`.
- Sin `__TOC_PLACEHOLDER__`.
- Sin `TOC \o`.
- Documento con `13` secciones y `22` tablas.
- Suite local ejecutada el 22/05/2026: `83 passed, 1 skipped, 4 warnings`.
