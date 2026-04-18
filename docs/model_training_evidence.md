# Evidencia tecnica del entrenamiento

## Estado actual de la rama
- Rama analizada: `training`
- Cambio estructural ya incorporado: pipeline compartido de preprocesamiento con `pandas`, `ColumnTransformer`, `SimpleImputer`, `MinMaxScaler`, `OneHotEncoder` y persistencia en `preprocessor.joblib`.
- Artefactos actuales del entrenamiento: `model.pt`, `preprocessor.joblib`, `training_metadata.json` y `experiments.csv`.
- El objetivo del producto se mantiene acotado a scouting juvenil de 12 a 18 anos para clubes formativos.

## Diagnostico previo verificado antes de endurecer el entrenamiento
- Dataset original analizado: 20000 jugadores sinteticos.
- Distribucion de clases original: 3038 positivos y 16962 negativos.
- Tasa positiva original: 15.19%.
- Rango etario original del dataset: 12-28.
- Resultado real de la MLP anterior: accuracy 0.8480, ROC-AUC 0.2123, PR-AUC 0.0915, F1 0.0000.
- Interpretacion honesta del diagnostico previo: la red colapsaba a clase negativa y la accuracy era enganosa.
- Baseline `LogisticRegression` anterior: ROC-AUC 0.9072, PR-AUC 0.6258, F1 0.5132.
- Baseline `LogisticRegression(class_weight="balanced")` anterior: ROC-AUC 0.9074, PR-AUC 0.6261, F1 0.5777, recall 0.8257.
- Baseline simple por promedio de atributos anterior: ROC-AUC 0.9162, PR-AUC 0.6434.

## Cambios implementados en esta iteracion
- Se reemplazo `BCELoss` por `BCEWithLogitsLoss`.
- Se elimino la `Sigmoid` final de `PlayerNet` y ahora se trabaja con logits; la probabilidad se recupera con `torch.sigmoid(...)` solo en evaluacion e inferencia.
- Se incorporo manejo explicito del desbalance con `pos_weight`.
- Se cambio el split a `train / validation / test` con seleccion del threshold en validacion.
- Se agrego early stopping sobre `PR-AUC` con desempate por `F1` positiva.
- El entrenamiento ahora se alinea por defecto al alcance real del MVP: edades 12-18.
- La generacion sintetica de `potential_label` ahora usa score ponderado por posicion, ajuste etario, componente mental y ruido controlado.
- Se formalizo `LogisticRegression(class_weight="balanced")` como baseline obligatorio de comparacion.
- Se guarda metadata completa de entrenamiento en `training_metadata.json`.
- Se agregaron features historicas agregadas al pipeline:
- cantidad de registros por jugador
- promedio historico de `final_score`
- promedio historico de `pass_accuracy`
- ultimo `final_score` registrado
- La generacion sintetica ahora crea `PlayerStat` para que esas features existan tambien en la base de entrenamiento.
- `PlayerAttributeHistory` ahora entra al pipeline con senales longitudinales como:
- mejora media en 90, 180 y 365 dias
- pendiente de crecimiento
- volatilidad del progreso
- gap entre la ficha actual y la trayectoria reciente
- La generacion sintetica ahora crea entre 6 y 12 snapshots tecnicos por jugador y deriva `PlayerStat` desde esa evolucion.
- Se agrego contexto minimo de partido con `Match` y `PlayerMatchParticipation`.
- El pipeline ahora agrega senales de partido como:
- cantidad de participaciones
- minutos medios
- tasa de titularidad
- nivel medio del rival
- porcentaje de partidos en posicion natural
- Se agregaron `ScoutReport` sinteticos y el pipeline agrega:
- cantidad de reportes
- medias de toma de decisiones, lectura tactica, perfil mental y adaptabilidad
- ultima proyeccion observada por scout
- El entrenamiento ya no usa `potential_label` como target principal.
- Ahora el target es `temporal_target_label`, derivado de un corte observado/futuro por jugador:
- las features se construyen sobre la parte observada de la trayectoria
- el target se marca positivo cuando el tramo futuro muestra crecimiento tecnico y mejora o consolidacion de rendimiento
- el dataset temporal usa atributos anclados en el punto de corte para evitar fuga de informacion desde el estado final del jugador

## Resultado actual del entrenamiento mejorado
- Fecha de corrida registrada: `2026-04-18T00:10:13.257885`
- Dataset actual: 20000 jugadores dentro del rango 12-17.
- Distribucion actual de clases: 988 positivos y 19012 negativos.
- Tasa positiva actual: 4.94%.
- Split efectivo: train 14000, validation 3000, test 3000.
- `pos_weight` utilizado: 19.2312.
- Early stopping: mejor epoca 30 y threshold elegido 0.525.

## Metricas del modelo PyTorch actual
- Validacion: accuracy 0.8637, ROC-AUC 0.8353, PR-AUC 0.2018, F1 0.2735, precision 0.1855, recall 0.5203.
- Test: accuracy 0.8680, ROC-AUC 0.8374, PR-AUC 0.2598, F1 0.2528, precision 0.1754, recall 0.4527.
- Matriz de confusion PyTorch en test: [[2537, 315], [81, 67]].

## Baselines actuales bajo el mismo split y preprocesamiento
- `LogisticRegression(class_weight="balanced")`: accuracy 0.9380, ROC-AUC 0.9279, PR-AUC 0.3645, F1 0.3922, precision 0.3797, recall 0.4054.
- Baseline simple por promedio de atributos: accuracy 0.1037, ROC-AUC 0.4094, PR-AUC 0.0389, F1 0.0906.

## Hallazgos verificados
- El nuevo preprocesamiento compartido con `pandas` y `scikit-learn` quedo implementado y funcionando tanto en entrenamiento como en inferencia.
- La MLP actual ya no colapsa a todo negativo: paso de F1 0.0000 a F1 0.2528 y de PR-AUC 0.0915 a PR-AUC 0.2598.
- El entrenamiento ya no usa solo foto fija: aprende con rendimiento historico y con evolucion tecnica de `PlayerAttributeHistory`.
- El entrenamiento ahora tambien incorpora contexto de partido y senal cualitativa sintetica del scout.
- El problema de entrenamiento ahora es metodologicamente mas realista porque el target representa progresion futura y no un booleano estatico sintetico.
- El cambio de target volvio el problema mucho mas desbalanceado y exigente: la tasa positiva actual es 4.94%.
- La alineacion del dataset a 12-18, el entrenamiento endurecido y las features longitudinales mejoraron fuerte la defendibilidad metodologica respecto al diagnostico previo.
- Aun asi, el baseline `LogisticRegression(class_weight="balanced")` sigue superando a la MLP en ROC-AUC, PR-AUC y F1.
- El baseline simple por promedio de atributos ya no explica bien el target frente al nuevo pipeline, lo que indica que la etiqueta sintetica quedo menos trivial que antes.
- La senal del dataset existe, pero la red PyTorch todavia no demuestra una ventaja clara sobre el baseline lineal balanceado.

## Limites que todavia no estan resueltos
- Los partidos sinteticos todavia no representan encuentros compartidos entre varios jugadores del mismo plantel.
- Los `ScoutReport` actuales siguen siendo sinteticos, no manuales ni cargados por usuarios reales.
- Aunque el target ya es temporal, sus umbrales todavia son sinteticos y pueden requerir recalibracion.
- No se implemento calibracion de probabilidades.
- La evidencia actual sigue basada en datos sinteticos; no hay una validacion externa con datos reales.
- La MLP mejoro, pero todavia no justifica por rendimiento reemplazar al baseline lineal como referencia formal.

## Pruebas ejecutadas
- `pytest -q`: 33 tests aprobados.
- Cobertura nueva o reforzada:
- persistencia del `preprocessor.joblib`
- consistencia entre inferencia individual y batch
- split `train / validation / test` y metadata de threshold
- filtro de entrenamiento acotado a 12-18
- sensibilidad de la etiqueta sintetica a edad y posicion
- merge de features historicas en el dataset de entrenamiento
- inclusion de features historicas en la inferencia de la app
- features longitudinales de `PlayerAttributeHistory`
- smoke del pipeline completo de entrenamiento

## Procedimiento reproducible
- Regenerar dataset de entrenamiento:
- `python scouting_app/generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42`
- Reentrenar artefactos:
- `python scouting_app/train_model.py --db-url sqlite:///players_training.db --model-out scouting_app/model.pt --preprocessor-out scouting_app/preprocessor.joblib --metadata-out scouting_app/training_metadata.json --epochs 30 --lr 1e-3 --patience 8`
- Ejecutar tests:
- `python -m pytest -q`
