# Mejora de prediccion - avance tecnico

## Estado actual
- Rama de trabajo: `training`
- Ultimo commit publicado antes de esta etapa: `1b3cac3`
- HEAD local al generar este documento: `1b3cac3`
- Objetivo de esta iteracion: recalibrar el target temporal, ajustar entrenamiento/calibracion y endurecer la generacion sintetica con reglas futbolisticas y temporales mas coherentes.
- Alcance del producto mantenido: scouting juvenil de 12 a 18 anos para clubes formativos.

## Que se implemento en esta etapa
- Se recalibro el target temporal para evitar una clase positiva patologicamente rara.
- El target positivo deja de depender de una sola puerta monotona y pasa a admitir dos caminos:
- consolidacion futura
- breakout futuro
- Se implemento calibracion de probabilidades con seleccion automatica entre `none`, `isotonic` y `platt`.
- Se reforzo la arquitectura de PyTorch con entrenamiento por mini-batch balanceado, `AdamW` y threshold elegido en validacion.
- La generacion sintetica agrega reglas mas coherentes de progresion, resiliencia, disciplina tactica, adaptabilidad y profesionalismo.
- `generate_data.py` ahora puede regenerar la base sintetica desde cero con `--reset`.
- La app sigue sin cambiar su interfaz; el cambio permanece en el pipeline, el target y los artefactos del modelo.

## Conteos reales de la base de entrenamiento actual
- Jugadores: 20000
- Snapshots de `PlayerAttributeHistory`: 179810
- Registros agregados de `PlayerStat`: 162490
- Partidos sinteticos (`Match`): 324735
- Participaciones por partido: 324735
- Reportes de scout: 80133

## Comparacion con la etapa anterior
### Etapa anterior publicada: target temporal inicial
- PyTorch: ROC-AUC 0.8374, PR-AUC 0.2598, F1 0.2528
- Logistic balanceado: ROC-AUC 0.9279, PR-AUC 0.3645, F1 0.3922

### Etapa actual: target temporal recalibrado + calibracion de probabilidades
- PyTorch: accuracy 0.9600, ROC-AUC 0.9247, PR-AUC 0.3279, F1 0.4231, precision 0.3729, recall 0.4889
- Logistic balanceado: accuracy 0.9527, ROC-AUC 0.9431, PR-AUC 0.3923, F1 0.4409

## Hallazgos verificados
- El target temporal deja el problema mucho mas exigente: la tasa positiva actual es 3.00%.
- PyTorch mejora respecto de la etapa publicada anterior:
- cambio en F1: +0.1703
- cambio en PR-AUC: +0.0681
- La calibracion de probabilidades quedo implementada y en la corrida actual el metodo seleccionado fue `isotonic`.
- El target actual deja 542 positivos por consolidacion y 66 por breakout.
- El baseline `LogisticRegression(class_weight="balanced")` sigue siendo mejor que PyTorch en esta corrida.
- La prediccion es metodologicamente mas defendible porque el modelo ahora intenta anticipar progresion futura en lugar de reproducir una etiqueta estatica.
- El sistema mantiene el pipeline compartido entre entrenamiento e inferencia, pero el entrenamiento ya no mira la trayectoria completa como si fuera toda observable en el momento de decidir.

## Limites honestos de esta etapa
- Los partidos sinteticos siguen siendo por jugador; todavia no representan encuentros compartidos entre varios jugadores del mismo plantel.
- `ScoutReport` sigue siendo sintetico, no manual ni proveniente de observacion real de usuarios.
- El target temporal actual sigue siendo sintetico aunque ya no es tan extremo: deja 599 positivos sobre 20000 jugadores.
- Aunque el target y la calibracion mejoran la validez metodologica, PyTorch todavia no supera al baseline lineal.
- PyTorch sigue sin superar al baseline lineal balanceado.
- La base de entrenamiento SQLite ya es pesada y el repo recibio advertencia de GitHub por tamano de `players_training.db`.

## Validacion ejecutada
- `pytest -q`: 35 tests aprobados.
- Smoke de app con artefactos nuevos:
- `/` -> 200
- `/health` -> 200
- `/login` -> 200
- Reentrenamiento completo ejecutado sobre la nueva base sintetica.

## Proximo paso recomendado
- Evaluar si conviene introducir `Availability` o `PhysicalAssessment`.
- Seguir enriqueciendo la generacion sintetica con senales longitudinales no triviales, sin aumentar volumen por aumentar.
- Replantear si el baseline lineal debe quedar como referencia principal hasta que PyTorch demuestre ventaja clara.
