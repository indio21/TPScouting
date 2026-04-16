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

## Resultado actual del entrenamiento mejorado
- Fecha de corrida registrada: `2026-04-16T23:38:49.493944`
- Dataset actual: 20000 jugadores dentro del rango 12-18.
- Distribucion actual de clases: 4058 positivos y 15942 negativos.
- Tasa positiva actual: 20.29%.
- Split efectivo: train 14000, validation 3000, test 3000.
- `pos_weight` utilizado: 3.9296.
- Early stopping: mejor epoca 30 y threshold elegido 0.525.

## Metricas del modelo PyTorch actual
- Validacion: accuracy 0.8460, ROC-AUC 0.9010, PR-AUC 0.7657, F1 0.6705, precision 0.5927, recall 0.7718.
- Test: accuracy 0.8557, ROC-AUC 0.9130, PR-AUC 0.7887, F1 0.6905, precision 0.6114, recall 0.7931.
- Matriz de confusion PyTorch en test: [[2084, 307], [126, 483]].

## Baselines actuales bajo el mismo split y preprocesamiento
- `LogisticRegression(class_weight="balanced")`: accuracy 0.9447, ROC-AUC 0.9621, PR-AUC 0.9175, F1 0.8536, precision 0.9219, recall 0.7947.
- Baseline simple por promedio de atributos: accuracy 0.7557, ROC-AUC 0.8088, PR-AUC 0.5515, F1 0.5292.

## Hallazgos verificados
- El nuevo preprocesamiento compartido con `pandas` y `scikit-learn` quedo implementado y funcionando tanto en entrenamiento como en inferencia.
- La MLP actual ya no colapsa a todo negativo: paso de F1 0.0000 a F1 0.6905 y de PR-AUC 0.0915 a PR-AUC 0.7887.
- La alineacion del dataset a 12-18, el entrenamiento endurecido y las features historicas agregadas mejoraron fuerte la calidad respecto al diagnostico previo.
- Aun asi, el baseline `LogisticRegression(class_weight="balanced")` sigue superando a la MLP en ROC-AUC, PR-AUC y F1.
- El baseline simple por promedio de atributos ya no explica bien el target frente al nuevo pipeline, lo que indica que la etiqueta sintetica quedo menos trivial que antes.
- La senal del dataset existe, pero la red PyTorch todavia no demuestra una ventaja clara sobre el baseline lineal balanceado.

## Limites que todavia no estan resueltos
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
- smoke del pipeline completo de entrenamiento

## Procedimiento reproducible
- Regenerar dataset de entrenamiento:
- `python scouting_app/generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42`
- Reentrenar artefactos:
- `python scouting_app/train_model.py --db-url sqlite:///players_training.db --model-out scouting_app/model.pt --preprocessor-out scouting_app/preprocessor.joblib --metadata-out scouting_app/training_metadata.json --epochs 30 --lr 1e-3 --patience 8`
- Ejecutar tests:
- `python -m pytest -q`
