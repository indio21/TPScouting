# Nota de sesion - rediseño de datos sinteticos

Fecha: 2026-04-22
Rama: `training`

## Contexto
- La ultima corrida experimental dejo solo 6 positivos sobre 20000 jugadores.
- Esa tasa positiva equivale a 0.03%, demasiado extrema para evaluar aprendizaje real.
- El problema detectado no es solo de arquitectura del modelo: el target temporal quedo demasiado duro y la generacion sintetica necesita representar mejor trayectorias juveniles plausibles.

## Regla de honestidad
- No se modifican datos para maquillar metricas.
- Si se rediseña la base sintetica, debe hacerse con criterio futbolistico y temporal.
- PyTorch debe compararse contra el baseline lineal bajo el mismo split, el mismo preprocesamiento y evidencia reproducible.
- No se avanza al siguiente bloque sin permiso.

## Bloque 1 autorizado
Objetivo: mejorar la generacion sintetica para pasar de "foto fija del jugador" a "trayectoria del jugador".

Alcance de este bloque:
- Rediseñar arquetipos de desarrollo juvenil.
- Hacer que esos arquetipos afecten atributos, disponibilidad, fatiga, rendimiento y scout reports.
- Mantener la app y el pipeline existentes funcionando.
- No recalibrar todavia el target temporal.
- No asumir mejora de metricas hasta reentrenar y comparar.

## Estado antes de tocar codigo
- Existen cambios locales de una etapa experimental previa.
- `scouting_app/preprocessing.py` contiene un target temporal mas rico, pero todavia demasiado restrictivo.
- `scouting_app/training_metadata.json` muestra una clase positiva patologicamente baja: 6 positivos sobre 20000.
- El siguiente bloque pendiente sera recalibrar target y prevalencia objetivo.

## Criterio para continuar
- Este bloque debe pasar validaciones tecnicas basicas.
- Si el generador funciona y los tests no rompen la app, se pedira permiso antes de iniciar el Bloque 2.

## Bloque 1 implementado
- Se ampliaron los arquetipos de desarrollo sintetico:
- `steady_builder`
- `late_bloomer`
- `early_burst`
- `setback_rebound`
- `volatile_creator`
- `consistent_ceiling`
- `fatigue_limited`
- Se agrego un estado temporal mensual de trayectoria que no mira el target:
- crecimiento esperado
- rendimiento bajo contexto
- disponibilidad
- fatiga
- confianza
- confianza del entrenador
- lectura del scout
- volatilidad
- Ese estado temporal ahora impacta:
- historial de atributos
- disponibilidad mensual
- rendimiento en partidos
- titularidad y rol
- reportes de scout

## Validacion ejecutada
- Compilacion Python de archivos principales: correcta.
- Generacion sintetica temporal de 250 jugadores: correcta.
- `pytest tests/test_mvp_regressions.py -q`: 26 tests aprobados.
- `pytest -q`: 35 tests aprobados.
- Smoke Flask:
- `/` -> 200
- `/health` -> 200
- `/login` -> 200

## Pendiente detectado al cierre de Bloque 1
- El target temporal sigue sin recalibrar.
- La ultima distribucion medida sigue siendo patologica y no debe tomarse como resultado final.
- El Bloque 2 debe recalibrar prevalencia objetivo y reglas del target antes de regenerar la base principal y reentrenar.

## Bloque 2 implementado
- Se recalibro el target temporal para evitar una clase positiva patologicamente rara.
- Se reemplazo la seleccion basada en muchas condiciones duras simultaneas por:
- score compuesto de progresion
- pisos minimos de calidad futbolistica y temporal
- ranking por cohorte de posicion y edad
- prevalencia objetivo cercana al 8%
- limites de seguridad entre 5% y 12%
- Las cohortes usadas para balancear la seleccion son:
- posicion
- banda etaria `12-14`, `15-16`, `17-18`
- El target sigue dependiendo de patrones temporales:
- mejora sostenida
- rendimiento futuro
- continuidad fisica
- disponibilidad
- scout projection
- consolidacion competitiva
- respuesta en contexto exigente

## Bloque 3 implementado
- Se corrigio la escala sintetica de `final_score` para que represente mejor una calificacion futbolistica 1-10.
- Se corrigio la escala sintetica de `observed_projection_score` en `ScoutReport`.
- Se regenero la base principal `scouting_app/players_training.db` con 20000 jugadores juveniles.
- La base regenerada mantiene el alcance 12-18.

## Medicion real despues de Bloque 2 y Bloque 3
- Jugadores: 20000
- Positivos: 1597
- Negativos: 18403
- Tasa positiva: 7.99%
- Quality gate: 7143 jugadores
- Candidatos al target: 4808 jugadores
- Positivos por consolidacion: 1333
- Positivos por breakout/trayectoria: 264
- Umbral operativo del target: 0.3013
- Umbral de rendimiento futuro: 6.1929

## Distribucion por posicion
- Defensa: 318 positivos sobre 3991 jugadores, 7.97%
- Delantero: 315 positivos sobre 3945 jugadores, 7.98%
- Lateral: 318 positivos sobre 3977 jugadores, 8.00%
- Mediocampista: 325 positivos sobre 4073 jugadores, 7.98%
- Portero: 321 positivos sobre 4014 jugadores, 8.00%

## Distribucion por edad
- 12-14: 678 positivos sobre 8491 jugadores, 7.98%
- 15-16: 458 positivos sobre 5736 jugadores, 7.98%
- 17-18: 461 positivos sobre 5773 jugadores, 7.99%

## Conteos de la base regenerada
- `players`: 20000
- `player_attribute_history`: 180212
- `player_stats`: 116448
- `matches`: 232960
- `player_match_participations`: 232960
- `scout_reports`: 80062
- `physical_assessments`: 180212
- `player_availability`: 180212

## Validacion despues de Bloque 2 y Bloque 3
- Compilacion Python de archivos principales: correcta.
- `pytest -q`: 35 tests aprobados.
- Smoke Flask:
- `/` -> 200
- `/health` -> 200
- `/login` -> 200

## Pendiente para Bloque 4
- Reentrenar PyTorch y baseline lineal usando la nueva base.
- Comparar `PR-AUC`, `ROC-AUC`, `F1`, precision, recall y matriz de confusion.
- Actualizar artefactos de modelo solo si el resultado es defendible.
- Documentar honestamente si PyTorch supera o no supera al baseline.

## Bloque 4 iniciado
- Se confirmo con evidencia que el target recalibrado selecciona por cohorte `posicion:edad`.
- Constantes de target en codigo:
- `TEMPORAL_TARGET_POSITIVE_RATE = 0.08`
- `TEMPORAL_TARGET_MIN_POSITIVE_RATE = 0.05`
- `TEMPORAL_TARGET_MAX_POSITIVE_RATE = 0.12`
- Medicion real sobre `players_training.db`:
- 1597 positivos sobre 20000 jugadores
- tasa positiva 7.99%
- distribucion por posicion cercana al 8%
- distribucion por edad cercana al 8%
- La escala sintetica corregida quedo medida asi:
- `future_final_score`: mediana 5.59, percentil 90 7.38, maximo 10.00
- `future_scout_projection_score`: mediana 5.65, percentil 90 7.00, maximo 9.36

## Entrenamiento oficial ejecutado en Bloque 4
- Configuracion: BCEWithLogitsLoss, AdamW, lr 0.0005, paciencia 10, 45 epocas maximas.
- Early stopping en epoca 15.
- Artefactos oficiales actualizados localmente:
- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`
- `scouting_app/training_metadata.json`
- Estos artefactos todavia no fueron commiteados ni pusheados.

## Metricas reales del entrenamiento oficial
- PyTorch test:
- Accuracy 0.9057
- ROC-AUC 0.9084
- PR-AUC 0.4617
- F1 0.5162
- Precision 0.4377
- Recall 0.6292
- Matriz de confusion `[[2566, 194], [89, 151]]`
- LogisticRegression balanceado test:
- ROC-AUC 0.9086
- PR-AUC 0.4728
- F1 0.4875
- Precision 0.4520
- Recall 0.5292
- Baseline promedio de atributos test:
- ROC-AUC 0.8298
- PR-AUC 0.3298
- F1 0.3792

## Conclusion honesta parcial de Bloque 4
- PyTorch ya no colapsa.
- PyTorch supera al baseline lineal en F1 y recall operativo.
- PyTorch todavia no supera al baseline lineal en PR-AUC.
- La brecha de PR-AUC actual es chica: 0.4617 vs 0.4728.
- No seria honesto afirmar que PyTorch ya gana globalmente.

## Variantes probadas antes de pausar
- Variante focal loss:
- PR-AUC 0.4595
- F1 0.4837
- Resultado: peor que la corrida oficial.
- Variante residual con menor dropout y sin penalizacion al prior lineal:
- PR-AUC 0.4678
- F1 0.4734
- Resultado: se acerca mas en PR-AUC, pero sigue debajo de LogisticRegression y queda peor en F1 que la corrida oficial.
- Las variantes se guardaron en carpetas temporales, no reemplazaron los artefactos oficiales.

## Punto exacto donde se pauso
- Se empezo a verificar si la calibracion isotonic podia estar bajando el ranking crudo de PyTorch en PR-AUC.
- Esa verificacion fue interrumpida manualmente y no dejo resultado valido.
- Se detuvieron los procesos Python que habian quedado vivos por esa verificacion.

## Proximo paso recomendado
- Retomar midiendo `PR-AUC` crudo vs calibrado de PyTorch sobre el mismo split.
- Si el `PR-AUC` crudo supera al calibrado, ajustar la evaluacion para no usar calibracion como metrica de ranking.
- Si el `PR-AUC` crudo tampoco supera a LogisticRegression, documentar que PyTorch mejora F1/recall pero no ranking PR-AUC.
- Recien despues decidir si se commitean los artefactos actuales o si conviene una ultima mejora controlada.
