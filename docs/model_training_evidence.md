# Evidencia tecnica del entrenamiento

## Estado actual de la rama
- Rama analizada: `training`
- Cambio estructural ya incorporado: pipeline compartido de preprocesamiento con `pandas`, `ColumnTransformer`, `SimpleImputer`, `MinMaxScaler`, `OneHotEncoder` y persistencia en `preprocessor.joblib`.
- Artefactos actuales del entrenamiento: `model.pt`, `preprocessor.joblib`, `training_metadata.json` y `experiments.csv`.
- Artefacto adicional actual: `probability_calibrator.joblib`.
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
- El entrenamiento paso a usar `shuffle` por defecto y deja `WeightedRandomSampler` como opcion, evitando el doble rebalanceo por defecto.
- Se cambio el split a `train / validation / test` con seleccion del threshold en validacion.
- Se agrego early stopping sobre `PR-AUC` con desempate por `F1` positiva.
- Se incorporo calibracion de probabilidades con seleccion automatica entre `none`, `isotonic` y `platt`.
- Se adopto `AdamW` como optimizador base.
- `PlayerNet` ya no es una MLP plana: ahora arranca desde la solucion de `LogisticRegression(class_weight="balanced")` y aprende una correccion residual no lineal encima.
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
- Se agregaron `PhysicalAssessment` y `PlayerAvailability` al esquema sintetico.
- El pipeline ahora incorpora senales de:
- altura y peso recientes
- crecimiento corporal
- velocidad y resistencia estimadas
- disponibilidad, fatiga, carga de trabajo y tasa de lesion
- El entrenamiento ya no usa `potential_label` como target principal.
- Ahora el target es `temporal_target_label`, derivado de un corte observado/futuro por jugador:
- las features se construyen sobre la parte observada de la trayectoria
- el target se recalibra con umbrales explicitos de progresion y rendimiento futuro
- el target positivo ahora admite dos caminos: consolidacion y breakout
- el dataset temporal usa atributos anclados en el punto de corte para evitar fuga de informacion desde el estado final del jugador
- `generate_data.py` ahora admite `--reset` para regenerar la base sintetica de entrenamiento desde cero de forma reproducible

## Resultado actual del entrenamiento mejorado
- Fecha de corrida registrada: `2026-04-22T00:48:09.395350`
- Dataset actual: 20000 jugadores dentro del rango 12-17.
- Distribucion actual de clases: 338 positivos y 19662 negativos.
- Tasa positiva actual: 1.69%.
- Split efectivo: train 14000, validation 3000, test 3000.
- `pos_weight` utilizado: 58.3220.
- Estrategia de desbalance activa: `pos_weight_strategy=full_ratio` y `sampler_strategy=shuffle`.
- Early stopping: mejor epoca 3 y threshold elegido 0.150.
- Metodo de calibracion seleccionado: `isotonic`.
- Threshold de progresion del target: 0.3700.
- Threshold de rendimiento futuro del target: 4.6600.
- Casos positivos por via de consolidacion: 310.
- Casos positivos por via de breakout: 28.

## Metricas del modelo PyTorch actual
- Validacion: accuracy 0.9710, ROC-AUC 0.9485, PR-AUC 0.2305, F1 0.3040, precision 0.2568, recall 0.3725.
- Test: accuracy 0.9713, ROC-AUC 0.9306, PR-AUC 0.2371, F1 0.3768, precision 0.2989, recall 0.5098.
- Matriz de confusion PyTorch en test: [[2888, 61], [25, 26]].

## Baselines actuales bajo el mismo split y preprocesamiento
- `LogisticRegression(class_weight="balanced")`: accuracy 0.9653, ROC-AUC 0.9425, PR-AUC 0.2775, F1 0.3659, precision 0.2655, recall 0.5882.
- Baseline simple por promedio de atributos: accuracy 0.9690, ROC-AUC 0.9004, PR-AUC 0.1993, F1 0.2560.

## Hallazgos verificados
- El nuevo preprocesamiento compartido con `pandas` y `scikit-learn` quedo implementado y funcionando tanto en entrenamiento como en inferencia.
- La MLP actual ya no colapsa a todo negativo: paso de F1 0.0000 a F1 0.3768 y de PR-AUC 0.0915 a PR-AUC 0.2371.
- El entrenamiento ya no usa solo foto fija: aprende con rendimiento historico y con evolucion tecnica de `PlayerAttributeHistory`.
- El entrenamiento ahora tambien incorpora contexto de partido y senal cualitativa sintetica del scout.
- El entrenamiento ahora tambien incorpora maduracion fisica y disponibilidad longitudinal.
- El problema de entrenamiento ahora es metodologicamente mas realista porque el target representa progresion futura y no un booleano estatico sintetico.
- El target temporal ya no depende de una sola puerta monotona: mezcla consolidacion y breakout con umbrales explicitados en metadata.
- La calibracion de probabilidades si quedo implementada y en la corrida actual el metodo elegido fue `isotonic`.
- El cambio de target mantiene un problema exigente y todavia desbalanceado: la tasa positiva actual es 1.69%.
- La alineacion del dataset a 12-18, el entrenamiento endurecido y las features longitudinales mejoraron fuerte la defendibilidad metodologica respecto al diagnostico previo.
- PyTorch ahora supera al baseline `LogisticRegression(class_weight="balanced")` en `F1` y precision al threshold operativo seleccionado.
- El baseline `LogisticRegression(class_weight="balanced")` sigue superando a PyTorch en `ROC-AUC` y `PR-AUC`.
- El baseline simple por promedio de atributos ya no explica bien el target frente al nuevo pipeline, lo que indica que la etiqueta sintetica quedo menos trivial que antes.
- La senal del dataset existe y la arquitectura residual acorta la brecha, pero PyTorch todavia no demuestra una ventaja global clara sobre el baseline lineal balanceado.

## Limites que todavia no estan resueltos
- Los partidos sinteticos todavia no representan encuentros compartidos entre varios jugadores del mismo plantel.
- Los `ScoutReport` actuales siguen siendo sinteticos, no manuales ni cargados por usuarios reales.
- Aunque el target ya es temporal y esta mejor calibrado, sus umbrales siguen siendo sinteticos y pueden requerir otro ajuste.
- La calibracion de probabilidades existe, pero por ahora no alcanza para que PyTorch gane tambien en `PR-AUC`.
- La nueva senal de disponibilidad y fisico mejora la riqueza del problema, y la arquitectura residual si mejora el comportamiento operativo, pero no cambia aun el orden global por ranking.
- La evidencia actual sigue basada en datos sinteticos; no hay una validacion externa con datos reales.
- La arquitectura residual mejora, pero todavia no justifica por rendimiento global reemplazar al baseline lineal como referencia formal.

## Pruebas ejecutadas
- `pytest -q`: 35 tests aprobados.
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
- `python scouting_app/generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset`
- Reentrenar artefactos:
- `python scouting_app/train_model.py --db-url sqlite:///players_training.db --model-out scouting_app/model.pt --preprocessor-out scouting_app/preprocessor.joblib --calibrator-out scouting_app/probability_calibrator.joblib --metadata-out scouting_app/training_metadata.json --epochs 45 --lr 5e-4 --patience 10`
- Ejecutar tests:
- `python -m pytest -q`
