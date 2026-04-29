# Progreso Del MVP

Fecha de actualizacion: 2026-04-27

Este archivo resume, sin inventar nada, las etapas ya trabajadas sobre el MVP real del proyecto `TPScouting`.

## Actualizacion 2026-04-27

- Se cerro un bloque de observaciones del informe del profesor sobre codigo fuente en la rama `reformas-finales`.
- `app.py` ahora documenta junto a `_PIPELINE_LOCK` que el lock es intra-proceso y depende de Gunicorn `--workers 1`.
- Se reemplazo el uso de `globals().get(...)` por llamada directa a `sync_attribute_history_baseline`.
- CI ahora mide cobertura con `pytest-cov` y `requirements-dev.txt` incluye esa dependencia.
- Se agrego `requirements-lock.txt` como snapshot exacto de dependencias instaladas en `.venv`.
- `RUNBOOK.md` y `docs/explicacion_cambios_revision_codigo_2026-04-27.md` documentan DiceBear, cache in-memory con limite configurable, `app.py` monolitico y el lock de dependencias.
- `db_utils.ensure_player_columns()` ahora migra `created_at` y `updated_at` tambien en `physical_assessments` y `player_availability`.
- `train_model.py` guarda checkpoints con `input_dim`, version y `model_state`.
- `app.py` y `evaluate_saved_model.py` cargan checkpoints nuevos y conservan compatibilidad con `state_dict` legacy.
- `classification_metrics()` devuelve `warnings` cuando ROC-AUC, PR-AUC o F1/precision/recall no pueden calcularse.
- Se agregaron tests de migracion legacy, checkpoint con `input_dim` y warnings de metricas.
- Validacion: `42 passed` con `pytest -q --cov=scouting_app --cov-report=term-missing`, cobertura total reportada `75%`.
- Bloque CSRF/inputs invalidos: se agrego un test matriz para POST mutantes sin token, tests de edad invalida y campos obligatorios vacios, y `base.html` ahora muestra todos los mensajes flash de validacion.
- Validacion posterior: `45 passed` con `pytest -q --cov=scouting_app --cov-report=term-missing`, cobertura total reportada `76%`.
- Bloque constantes/cache/conftest: se nombraron constantes de paginacion, comparadores y rating de estadisticas; se agrego `CACHE_MAX_ENTRIES` al cache in-memory; y `tests/conftest.py` usa un nombre estable de modulo de test.
- Bloque nomenclatura de sesiones: helpers operativos y `create_admin.py` usan `db_session`; la sesion temporal de entrenamiento se llama `training_session`; los endpoints Flask conservan `db` como variable local corta.
- Bloque type hints quirurgicos: se agregaron anotaciones en helpers compartidos de cache, CSRF/context processors, carga de artefactos, DB, `sync_shortlist.py` y `evaluate_saved_model.py`.
- Validacion posterior: `48 passed` con `pytest -q --cov=scouting_app --cov-report=term-missing`, cobertura total reportada `76%`.
- Cierre de rama: `reformas-finales` queda cerrada en `7763bb0` como bloque de reformas livianas. Se creo `reformas-complejas` desde ese commit para trabajos de mayor alcance.
- Se agrego una comparacion actualizada punto por punto en `docs/comparacion_falencias_codigo_fuente_2026-04-28.md`.
- Refactor de arquitectura fase 1 en `reformas-complejas`: se extrajeron cache, seguridad liviana, mantenimiento operativo y runtime ML a `services/` y `ml/`, manteniendo rutas/endpoints en `app.py`.
- Se documento la evaluacion de factibilidad y riesgos en `docs/refactor_arquitectura_2026-04-28.md`.
- Validacion posterior al refactor de arquitectura: `48 passed` con `pytest -q --cov=scouting_app --cov-report=term-missing`, cobertura total reportada `77%`.
- Arquitectura fase 2 iniciada: login, logout y registro se movieron a `scouting_app/routes/auth.py` como blueprint `auth`, conservando aliases legacy `login`, `logout` y `register` para no romper `url_for(...)`, redirects ni tests.
- Validacion focal fase 2 auth: `14 passed`. Validacion completa posterior: `49 passed` con `pytest -q --cov=scouting_app --cov-report=term-missing`, cobertura total reportada `77%`.

## Actualizacion 2026-04-23

- Se optimizo la construccion del dataframe temporal de entrenamiento en `scouting_app/preprocessing.py`.
- `_temporal_target_dataframe` ahora preordena y agrupa una sola vez por `player_id`, reusando mapas de observado/futuro en vez de filtrar dataframes grandes en cada iteracion.
- Se agrego cache versionada del dataframe temporal (`temporal_training_dataframe.joblib`) con invalidacion por version, `mtime`, tamano y cantidad de jugadores de `players_training.db`.
- Se agrego persistencia de splits `train/validation/test` en `training_splits.json` desde `scouting_app/train_model.py`.
- Se creo `scouting_app/evaluate_saved_model.py` para reevaluar artefactos entrenados usando cache y splits guardados, sin reconstruir todo manualmente.
- Se agregaron pruebas para cache reutilizable, invalidacion al cambiar la base, persistencia de splits y evaluacion rapida con validacion de `player_id` faltantes.
- Medicion real sobre `scouting_app/players_training.db`:
- `load_data(use_cache=False)`: bajo de aprox. `246.6s` a `138.3s`
- primer `load_data(use_cache=True)` con cache invalida o inexistente: `137.2s`
- reutilizacion de cache ya construida: `0.48s - 0.59s`
- Estado validado: `38 passed` en `pytest -q`
- Corrida oficial 2026-04-26: se reentreno el modelo con `45` epocas, `patience=10` y `lr=5e-4`, generando `training_splits.json`.
- Evaluacion rapida sobre artefactos reales con `scouting_app/evaluate_saved_model.py`: `load_data_seconds=0.5175`, `train=14000`, `validation=3000`, `test=3000`.
- Dataset oficial evaluado: `20000` jugadores, edades `12-17`, `1597` positivos, tasa positiva `0.0799`.
- PyTorch crudo en test: `ROC-AUC=0.9102`, `PR-AUC=0.4826`, `F1=0.5088`, `precision=0.4417`, `recall=0.6000`.
- PyTorch calibrado en test: `ROC-AUC=0.9084`, `PR-AUC=0.4617`, `F1=0.5162`, `precision=0.4377`, `recall=0.6292`.
- Baseline `LogisticRegression(class_weight="balanced")` en test: `ROC-AUC=0.9086`, `PR-AUC=0.4728`, `F1=0.4875`, `precision=0.4520`, `recall=0.5292`.
- Conclusion honesta: PyTorch crudo supera al baseline en `PR-AUC` y `F1`; PyTorch calibrado mejora `F1` y `recall`, pero baja `PR-AUC`. Conviene seguir evaluando crudo vs calibrado antes de decidir la version final del MVP.
- Decision de MVP: usar la salida cruda de PyTorch como score principal de ranking/priorizacion y conservar la probabilidad calibrada como referencia secundaria documentada.
- Decision de repo: mantener codigo, tests y documentacion en GitHub; las bases, modelos, preprocesadores, metadata, splits y experimentos pasan a ser artefactos generados localmente. Flujo documentado en `docs/flujo_reproducible_mvp.md`.
- Correccion demo 2026-04-26: se corrigio la inferencia cuando la base operativa tiene campos historicos faltantes (`None`), evitando `NaN` en PyTorch y probabilidades combinadas `0.0%`. `sync_shortlist.py --replace` ahora copia jugadores con stats, atributos, partidos, reportes scout, evaluaciones fisicas y disponibilidad para presentacion.
- Estado demo local tras la correccion: `100` jugadores, `713` stats, `908` snapshots de atributos, `1419` partidos/participaciones, `406` reportes scout, `908` evaluaciones fisicas y `908` registros de disponibilidad.
- Validacion tras la correccion demo: `40 passed` en `pytest -q`; smoke basico `/`, `/health` y `/login` con respuesta `200`.
- Se agrego `docs/guia_indicadores_app.md` para explicar `Rendimiento en posicion`, `Puntaje de ficha`, `Probabilidad combinada` y `Referencia calibrada`.
- Cambio de rama 2026-04-26: `training` queda cerrada como base estable de las correcciones del MVP. Se creo y publico `reformas-finales` desde `b6c21ea` para continuar nuevas reformas sin mezclar con el cierre anterior.

## Estado Git

- Rama local actual: `reformas-complejas`
- Remoto configurado: `origin -> https://github.com/indio21/TPScouting.git`
- Ultimo commit publicado en `main`: `bd8e4ef`
- Rama estable cerrada del MVP corregido: `training`
- Rama cerrada para reformas livianas: `reformas-finales`
- Rama activa para cambios complejos: `reformas-complejas`
- Ultimo commit comun al crear `reformas-finales`: `b6c21ea`
- Ultimos bloques tecnicos publicados en `reformas-finales`: revision de codigo fuente, documentacion, CSRF/inputs invalidos, constantes/cache/conftest, normalizacion puntual de sesiones y type hints quirurgicos.
- Estado tecnico documentado: `reformas-finales` cerrada en `7763bb0`; `reformas-complejas` continua desde ese punto.

## Etapas Ya Trabajadas

Nota: las etapas numeradas conservan evidencia historica de cada corrida. El estado vigente del modelo y de la demo es el resumido en las actualizaciones superiores de fecha 2026-04-26 y 2026-04-27.

### 1. Preparacion para PostgreSQL

- Se agrego una capa comun de conexion y normalizacion de URLs en `scouting_app/db_utils.py`.
- La app y scripts principales quedaron preparados para usar SQLite o PostgreSQL.
- `render.yaml` quedo orientado a despliegue con PostgreSQL administrado.

### 2. Instalacion de entorno y validacion funcional PostgreSQL

- Se creo `.venv` en la raiz del repo.
- Se instalaron dependencias del proyecto y de testing.
- Se valido el soporte PostgreSQL con pruebas reales de:
- `generate_data.py`
- `train_model.py`
- `sync_shortlist.py`
- app Flask usando `APP_DB_URL` y `TRAINING_DB_URL`

### 3. Correcciones criticas del MVP

- Se unifico la logica de `potential_label`, categoria textual y filtro de alto potencial.
- El dashboard paso a clasificar potencial con la misma logica base de proyeccion.
- Se corrigio la recarga del modelo en memoria despues del pipeline.
- Se agrego invalidacion de cache del dashboard tras cambios de datos.

### 4. Seguridad minima y reproducibilidad

- `APP_SECRET_KEY` insegura por defecto fue reemplazada por clave efimera en desarrollo y obligatoria en produccion.
- `create_admin.py` y registro de usuarios ya no aceptan contraseñas debiles.
- `generate_data.py` ahora acepta semilla reproducible.
- `train_model.py` usa `SEED` tambien en el split y deja de fallar en silencio al calcular metricas.

### 5. Multirol real del MVP

- Se normalizaron roles para:
- `administrador`
- `scout` / `ojeador`
- `director`
- Se implementaron permisos reales en backend.
- Se ocultaron acciones no permitidas en la interfaz.
- `director` quedo en modo lectura para jugadores, historial, atributos y comparadores.

### 6. Seguridad operativa y auditoria

- Se agrego rate limiting basico al login.
- Se agregaron timestamps `created_at` y `updated_at` en entidades clave.
- `db_utils.py` asegura esas columnas tambien en bases existentes.

### 7. Rendimiento inicial

- Se creo `batch_project_players(...)` para calcular proyecciones por lote.
- El listado de jugadores dejo de recalcular proyeccion jugador por jugador con consultas repetidas al historial.
- El dashboard reutiliza logica batch para potencial.
- Los comparadores quedaron mas livianos en calculo de proyeccion.
- Se documento el comportamiento y limite del cache in-memory del dashboard en `RUNBOOK.md`.

### 8. Consistencia final y limpieza de demo

- Se agrego auditoria de calidad operativa en `GET /health`.
- `Configuracion` ahora muestra metricas de consistencia y permite limpiar la base operativa.
- La limpieza elimina registros legacy invalidos sin inventar identificadores faltantes.
- Se ejecuto la limpieza real sobre la base demo incluida en el repo.
- Resultado medido:
- antes: `100` jugadores y `4` sin `national_id`
- despues: `96` jugadores y `0` sin `national_id`

### 9. Fortalecimiento de tests y cobertura funcional

- Se ampliaron pruebas de registro, CSRF, alta/edicion/baja de jugadores, validaciones de historial y ABM de staff.
- Se corrigieron validaciones faltantes en carga de stats y atributos para no aceptar valores fuera de rango.
- Se limpiaron warnings de compatibilidad con SQLAlchemy 2.x en modelos y ABM de staff.

### 10. Revision final integral del MVP

- Se ejecuto una revision final apoyada en evidencia real y no solo en lectura de codigo.
- Smoke real sobre la app/demo actual:
- `GET /`, `GET /health`, `/players`, `/dashboard`, `/compare`, `/compare/multi`, `/coaches`, `/directors`, `/settings`
- vistas de jugador: detalle, proyeccion, historial y atributos
- Resultado: todas esas rutas respondieron `200` en la revision final.
- Se dejo el detalle en `REVISION_FINAL_MVP.md`.

### 11. Preprocesamiento real con pandas y scikit-learn

- Se agrego `scouting_app/preprocessing.py` como capa compartida entre entrenamiento e inferencia.
- El dataset de entrenamiento ahora se construye con `pandas` leyendo `players` hacia `DataFrame`.
- El preprocesamiento ahora usa `scikit-learn` con:
- `SimpleImputer(strategy="median")` + `MinMaxScaler()` para columnas numericas
- `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")` para `position`
- El preprocesador se persiste en `scouting_app/preprocessor.joblib`.
- La app ya no usa normalizacion manual para inferencia: reutiliza el mismo preprocesador persistido que usa entrenamiento.
- Se reentreno el modelo para alinear pesos y preprocesamiento.

### 12. Endurecimiento del entrenamiento y documentacion tecnica

- `train_model.py` ahora usa `BCEWithLogitsLoss`, logits, `pos_weight`, split `train / validation / test`, threshold seleccionado por validacion y early stopping.
- El dataset sintetico de entrenamiento quedo alineado por defecto al rango real del producto: `12-18`.
- La etiqueta sintetica `potential_label` dejo de depender casi solo del promedio simple y ahora combina score ponderado por posicion, ajuste etario y ruido controlado.
- Se formalizo `LogisticRegression(class_weight="balanced")` como baseline obligatorio en la comparacion.
- Se agrega `scouting_app/training_metadata.json` para persistir:
- threshold elegido
- metricas de validacion y test
- tamanos de split
- configuracion y seed
- Se genero documentacion tecnica reproducible en:
- `docs/model_training_evidence.md`
- `docs/model_training_evidence.docx`
- `docs/model_training_plan.md`
- `docs/model_training_plan.docx`
- Se agrego `docs/generate_training_docs.py` para regenerar ambos documentos sin depender de `pandoc`.

### 13. Features historicas agregadas en entrenamiento e inferencia

- Se agregaron features historicas agregadas al pipeline del modelo:
- cantidad de registros por jugador
- promedio historico de `final_score`
- promedio historico de `pass_accuracy`
- ultimo `final_score` registrado
- `preprocessing.py` ahora mergea esas features con `pandas` tanto para entrenamiento como para inferencia.
- `generate_data.py` ahora sintetiza `PlayerStat` en la base de entrenamiento para que esas señales existan de verdad.
- La inferencia de la app usa esas features historicas antes de pasar por el preprocesador persistido.
- Resultado medido en la corrida real actual:
- PyTorch: `ROC-AUC 0.9130`, `PR-AUC 0.7887`, `F1 0.6905`
- Baseline `LogisticRegression(class_weight="balanced")`: `ROC-AUC 0.9621`, `PR-AUC 0.9175`, `F1 0.8536`
- Conclusion honesta: PyTorch mejoro bastante, pero el baseline lineal balanceado sigue siendo superior.

### 14. Trayectoria tecnica con PlayerAttributeHistory

- El pipeline del modelo ahora incorpora senales longitudinales reales a partir de `PlayerAttributeHistory`.
- Se agregaron features como:
- mejora media a 90, 180 y 365 dias
- pendiente de crecimiento
- volatilidad del progreso
- gap entre la ficha actual y la trayectoria reciente
- La inferencia en la app ahora usa tambien esas features tecnicas historicas, no solo `PlayerStat`.
- `generate_data.py` paso de generar una foto fija con stats derivados a generar:
- perfil base del jugador
- trayectoria tecnica mensual sintetica
- historial de rendimiento derivado de esa evolucion
- Resultado medido sobre la corrida real actual:
- base de entrenamiento: `20.000` jugadores
- `PlayerStat`: `162.778` registros
- `PlayerAttributeHistory`: `180.051` snapshots
- PyTorch: `ROC-AUC 0.8468`, `PR-AUC 0.6153`, `F1 0.5751`
- Baseline `LogisticRegression(class_weight="balanced")`: `ROC-AUC 0.9052`, `PR-AUC 0.7360`, `F1 0.6878`
- Conclusion honesta: esta version es metodologicamente mas rica y mas defendible, pero hoy todavia rinde menos que la etapa anterior y sigue por debajo del baseline lineal balanceado.

### 15. Contexto de partido y ScoutReport

- Se agregaron tres entidades nuevas al esquema:
- `Match`
- `PlayerMatchParticipation`
- `ScoutReport`
- `generate_data.py` ahora genera, por cada jugador:
- trayectoria tecnica mensual
- partidos sinteticos con contexto minimo
- participacion puntual por partido
- `PlayerStat` agregado a partir de esas participaciones
- reportes cualitativos sinteticos del scout
- El pipeline del modelo ahora agrega tambien features de:
- cantidad de participaciones y minutos medios
- tasa de titularidad
- nivel medio del rival
- porcentaje de partidos en posicion natural
- cantidad de reportes de scout
- medias de toma de decisiones, lectura tactica, perfil mental y adaptabilidad
- ultima proyeccion observada por scout
- Resultado medido sobre la corrida real actual:
- base de entrenamiento: `20.000` jugadores
- `PlayerAttributeHistory`: `179.377` snapshots
- `PlayerStat`: `162.180` registros
- `Match`: `324.165` partidos sinteticos
- `PlayerMatchParticipation`: `324.165` participaciones
- `ScoutReport`: `79.574` reportes
- PyTorch: `ROC-AUC 0.8628`, `PR-AUC 0.6508`, `F1 0.6042`
- Baseline `LogisticRegression(class_weight="balanced")`: `ROC-AUC 0.8996`, `PR-AUC 0.7373`, `F1 0.6769`
- Conclusion honesta: respecto de la etapa longitudinal anterior, PyTorch mejora y la prediccion queda mas defendible por contexto; aun asi, el baseline lineal balanceado sigue siendo superior.

### 16. Target temporal de progresion

- El entrenamiento ya no usa `potential_label` como target principal.
- Se construyo un dataset temporal con corte observado/futuro por jugador.
- Las features de entrenamiento ahora se calculan solo sobre la parte observada de la trayectoria.
- Los atributos base para entrenar se anclan en el punto de corte temporal para evitar fuga de informacion desde el estado final del jugador.
- El nuevo target `temporal_target_label` marca positivo cuando el tramo futuro del jugador muestra:
- crecimiento tecnico ponderado por posicion
- mejora o consolidacion del rendimiento futuro
- Resultado medido sobre la corrida real actual:
- base de entrenamiento: `20.000` jugadores
- positivos del target temporal: `988`
- tasa positiva: `4.94%`
- PyTorch: `ROC-AUC 0.8374`, `PR-AUC 0.2598`, `F1 0.2528`
- Baseline `LogisticRegression(class_weight="balanced")`: `ROC-AUC 0.9279`, `PR-AUC 0.3645`, `F1 0.3922`
- Conclusion honesta: metodologicamente esta version es mejor porque predice progresion futura y no una etiqueta estatica, pero el problema quedo bastante mas dificil y hoy PyTorch rinde peor que en la etapa anterior. El baseline lineal sigue siendo superior.

### 17. Recalibracion temporal, calibracion y ajuste del entrenamiento

- Se recalibro el target temporal para evitar etiquetas patologicamente raras y dejar dos caminos positivos defendibles:
- `consolidacion`
- `breakout`
- Se incorporo calibracion de probabilidades al pipeline con seleccion automatica entre `none`, `isotonic` y `platt`.
- `generate_data.py` ahora permite regenerar la base sintetica con `--reset`, manteniendo reproducibilidad del experimento.
- En la corrida real actual:
- positivos del target temporal recalibrado: `599`
- tasa positiva: `3.00%`
- casos de consolidacion: `542`
- casos de breakout: `66`
- PyTorch: `ROC-AUC 0.9247`, `PR-AUC 0.3279`, `F1 0.4231`
- Baseline `LogisticRegression(class_weight="balanced")`: `ROC-AUC 0.9431`, `PR-AUC 0.3923`, `F1 0.4409`
- Conclusion honesta: esta etapa mejoro con claridad respecto del target temporal publicado antes (`F1 +0.1703`, `PR-AUC +0.0681`), pero PyTorch todavia no supera al baseline lineal balanceado.

### 18. Availability, PhysicalAssessment y trayectorias mas ricas

- Se agregaron dos nuevas fuentes longitudinales al esquema:
- `PhysicalAssessment`
- `PlayerAvailability`
- La base sintetica de entrenamiento ahora modela:
- evaluaciones fisicas mensuales
- crecimiento corporal
- velocidad y resistencia estimadas
- disponibilidad, fatiga, carga y lesion/inactividad
- `generate_data.py` ahora conecta tecnica, fisico, disponibilidad y rendimiento en una misma trayectoria temporal.
- El pipeline del modelo incorpora esas nuevas features tanto en entrenamiento como en inferencia.
- Resultado medido sobre la corrida real actual:
- positivos del target temporal: `338`
- tasa positiva: `1.69%`
- PyTorch: `ROC-AUC 0.9044`, `PR-AUC 0.2029`, `F1 0.2542`
- Baseline `LogisticRegression(class_weight="balanced")`: `ROC-AUC 0.9425`, `PR-AUC 0.2775`, `F1 0.3659`
- Conclusion honesta: la etapa mejora la riqueza metodologica del sistema, pero en esta corrida concreta empeora frente a la etapa anterior y PyTorch sigue sin superar al baseline lineal balanceado.

### 19. Reentrenamiento residual apoyado en el baseline lineal

- El entrenamiento ya no usa doble rebalanceo por defecto:
- `pos_weight` pasa a usar la razon completa de clases
- `shuffle` queda como estrategia default de batches
- `WeightedRandomSampler` queda como opcion
- `PlayerNet` paso a una arquitectura residual inicializada desde la solucion de `LogisticRegression(class_weight="balanced")`.
- La rama lineal queda dentro del modelo PyTorch y una rama residual aprende correcciones no lineales sobre esa base.
- Resultado medido sobre la corrida real actual:
- PyTorch: `ROC-AUC 0.9306`, `PR-AUC 0.2371`, `F1 0.3768`, `precision 0.2989`, `recall 0.5098`
- Baseline `LogisticRegression(class_weight="balanced")`: `ROC-AUC 0.9425`, `PR-AUC 0.2775`, `F1 0.3659`
- Conclusion honesta de esa etapa: PyTorch superaba al baseline lineal en `F1` y precision al umbral operativo elegido, pero todavia no en `ROC-AUC` ni `PR-AUC`. Esta conclusion quedo superada por la corrida oficial 2026-04-26 resumida al inicio, donde PyTorch crudo pasa a superar al baseline en `PR-AUC` y `F1`.

## Tests Ejecutados

- La suite automatizada actual termina pasando en esta maquina.
- Ultimo estado validado sin cobertura: `40 passed`
- Ultimo estado validado con cobertura: `49 passed`, cobertura total `77%`

## Puntos Mejorados De Forma Clara

- Coherencia del potencial entre modelo, categoria, filtro y dashboard
- Soporte real para PostgreSQL
- Seguridad minima de secretos y credenciales
- Multirol y control de acceso
- Trazabilidad temporal basica
- Reproducibilidad del pipeline
- Mejora inicial de rendimiento en listado, dashboard y comparadores
- Base demo consistente con las reglas activas del MVP
- Healthcheck con visibilidad real de calidad de datos
- Cobertura funcional mas fuerte sobre CRUD, permisos, CSRF e inputs invalidos
- MVP revisado integralmente con evidencia de smoke y datos reales
- Preprocesamiento real y persistido con pandas + scikit-learn, sin desalineacion entre entrenamiento e inferencia
- Entrenamiento PyTorch endurecido y alineado al alcance 12-18
- Evidencia tecnica y plan de entrenamiento guardados en Markdown y Word
- Features historicas agregadas funcionando en entrenamiento e inferencia
- Trayectoria tecnica mensual incorporada al entrenamiento e inferencia
- Contexto de partido y senal cualitativa del scout incorporados al pipeline de prediccion
- Target temporal de progresion incorporado al entrenamiento
- Cobertura formal en CI con `pytest-cov`
- Tests CSRF sobre POST mutantes criticos
- Tests de inputs invalidos para edad y campos obligatorios vacios
- Checkpoint del modelo con `input_dim` validable
- Dependencias exactas registradas en `requirements-lock.txt`
- Limitaciones reales de DiceBear, cache in-memory y monolito documentadas
- Constantes nombradas para paginacion, comparadores y rating de estadisticas
- Limite configurable `CACHE_MAX_ENTRIES` para cache in-memory
- `tests/conftest.py` simplificado con nombre de modulo estable
- Nomenclatura de sesiones SQLAlchemy mejorada en helpers y scripts
- Type hints mejorados en helpers compartidos
- Arquitectura fase 1: servicios y runtime ML separados de `app.py`
- Arquitectura fase 2 iniciada: rutas de autenticacion separadas en blueprint con compatibilidad de endpoints historicos

## Puntos Que Siguen Parciales O Pendientes

- Persistencia/despliegue final en Render
- Optimizaciones adicionales de rendimiento
- Nomenclatura `db` / `db_session` aceptada como parcial en endpoints Flask; no bloquea el MVP
- Herramientas dev opcionales todavia parciales (`ruff`, `black`, `mypy`)
- Rutas Flask todavia mayormente concentradas en `app.py`; fase 2 ya empezo con `auth`, faltan `staff`, `players`, `dashboard`, `compare` y `settings`
- Correccion del documento Word, que todavia no se empezo en esta fase

## Bloques Restantes

Tomando como base la checklist operativa del MVP:

- No quedan bloques principales del MVP real abiertos en esta fase.

- Luego queda 1 bloque grande aparte:
- correccion del documento Word para alinearlo con el MVP ya corregido

## Regla De Trabajo

- No inventar nada
- No vender humo
- Si algo no se puede verificar, dejarlo explicitado
- No avanzar al documento hasta terminar antes el MVP real
