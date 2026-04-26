# Plan tecnico de entrenamiento

## Resumen
- Este plan se actualiza sobre la rama activa `reformas-finales`.
- `training` queda como rama estable cerrada del MVP corregido; `reformas-finales` queda como rama activa para nuevas reformas.
- PyTorch sigue siendo el modelo principal del MVP.
- `LogisticRegression(class_weight="balanced")` queda formalizada como baseline obligatorio de comparacion.
- La documentacion tecnica y la evidencia se guardan separadas del documento final de tesis.

## Etapas implementadas en esta iteracion
### Etapa 1 completada: endurecer entrenamiento PyTorch
- `BCEWithLogitsLoss` en lugar de `BCELoss`.
- Modelo sin `Sigmoid` final y trabajo con logits.
- `pos_weight` aplicado por defecto para desbalance.
- `shuffle` por defecto en lugar de doble rebalanceo fijo.
- Split `train / validation / test`.
- Seleccion de threshold por validacion.
- Early stopping sobre `PR-AUC` con desempate por `F1`.
- Resultado: PyTorch dejo de colapsar y alcanzo F1 0.5088 en test.

### Etapa 2 completada: alinear dataset al alcance real 12-18
- Generacion sintetica por defecto restringida a 12-18.
- Entrenamiento por defecto restringido a 12-18.
- Resultado: el pipeline ahora refleja el alcance real del producto.

### Etapa 3 completada: hacer mas realista `potential_label`
- Score ponderado por posicion.
- Ajuste etario juvenil.
- Mayor incidencia de `determination`, `technique` y `vision`.
- Ruido controlado en lugar de regla casi lineal por promedio simple.

### Etapa 4 completada: formalizar baseline y comparacion
- Baseline obligatorio: `LogisticRegression(class_weight="balanced")`.
- Comparacion bajo el mismo split y el mismo preprocesamiento.
- Persistencia de metadata en `training_metadata.json`.
- Estado actual: PyTorch crudo supera al baseline en PR-AUC con 0.4826 vs 0.4728 y en F1 con 0.5088 vs 0.4875.

### Etapa 5 completada en nivel MVP
- Se persisten threshold, metricas, tamanos de split, seed y configuracion.
- La evidencia tecnica puede regenerarse desde los artefactos del repo.

### Etapa 6 completada: crecimiento con features historicas
- Se agregaron features historicas agregadas con `pandas`.
- Se sintetizo historial de `PlayerStat` en la base de entrenamiento.
- Se integraron features de `PlayerAttributeHistory` al entrenamiento e inferencia.
- La base de entrenamiento ahora representa trayectoria tecnica del jugador, no solo foto fija.
- Resultado actual: PyTorch queda en F1 0.5088 y PR-AUC 0.4826.
- En la corrida oficial actual, la salida cruda de PyTorch queda por encima del baseline lineal balanceado en PR-AUC y F1.

### Etapa 7 completada: contexto de partido y ScoutReport
- Se agregaron `Match` y `PlayerMatchParticipation` al esquema.
- El generador sintetico ahora crea partidos con contexto y participacion puntual del jugador.
- `PlayerStat` pasa a derivarse de esas participaciones.
- Se agregaron `ScoutReport` sinteticos al esquema y al pipeline.
- Resultado actual: PyTorch queda en F1 0.5088 y PR-AUC 0.4826.
- La comparacion se conserva porque el baseline sigue siendo referencia obligatoria.

### Etapa 8 completada: target temporal de progresion
- El entrenamiento deja de usar `potential_label` como target principal.
- Las features de entrenamiento se construyen en un punto de corte observado de la trayectoria.
- El target positivo se define por crecimiento tecnico futuro y mejora o consolidacion de rendimiento futuro.
- Resultado actual: PyTorch queda en F1 0.5088 y PR-AUC 0.4826 sobre un target con tasa positiva 7.99%.
- La salida calibrada queda como referencia secundaria con PR-AUC 0.4617 y F1 0.5162.

### Etapa 9 completada: senales de disponibilidad y fisico
- Se agregaron `PhysicalAssessment` y `PlayerAvailability`.
- El generador sintetico ahora modela maduracion corporal, fatiga, carga, lesion e inactividad.
- Las features agregadas de disponibilidad y fisico ya entran tanto en entrenamiento como en inferencia.
- Resultado actual: PyTorch queda en F1 0.5088 y PR-AUC 0.4826.
- Conclusion honesta: el dataset queda metodologicamente mas rico y la salida cruda de PyTorch ahora supera el baseline en la corrida oficial actual, pero debe seguir comparandose en cada reentrenamiento.

### Etapa 10 completada: bootstrap lineal y correccion residual en PyTorch
- `PlayerNet` ahora se inicializa desde la solucion de `LogisticRegression(class_weight="balanced")`.
- La rama lineal queda dentro del modelo PyTorch y una rama residual aprende correcciones no lineales.
- El entrenamiento pasa a usar `pos_weight` completo y `shuffle` por defecto.
- Resultado actual: PyTorch crudo supera al baseline en `PR-AUC` y `F1`; la version calibrada queda como evidencia secundaria.

## Siguiente iteracion recomendada
### Etapa 11 recomendada: revisar estabilidad del target y la relacion partido-plantel
- Revisar otra vez los umbrales del target temporal para evitar un positivo demasiado raro o demasiado facil.
- Pasar de partidos por jugador a partidos compartidos por plantel si se quiere ganar realismo extra.
- Verificar que la ventaja actual de PyTorch se mantenga con nuevas semillas o con datos reales cuando existan.

## Criterios de aceptacion para la siguiente iteracion
- La siguiente iteracion debe mejorar o estabilizar al menos una de estas metricas de PyTorch en test sin degradar claramente las demas:
- PR-AUC
- F1 positiva
- Recall positiva
- Y ademas debe volver mas explicable la prediccion desde el punto de vista futbolistico.
- La comparacion debe seguir quedando trazable en `training_metadata.json`.
- Si PyTorch deja de superar al baseline en una futura corrida, el baseline debe quedar reconocido como referencia principal de rendimiento para esa version.

## Pruebas de validacion vigentes
- Tests del preprocesador compartido.
- Tests de consistencia entre inferencia individual y batch.
- Tests del split `train / validation / test`.
- Tests del threshold seleccionado y su persistencia.
- Tests de entrenamiento limitado a 12-18.
- Tests de generacion sintetica sensibles a edad y posicion.
- Tests de merge de features historicas.
- Tests de features longitudinales de `PlayerAttributeHistory`.
- Tests de contexto de partido y `ScoutReport`.
- Smoke del pipeline completo con artefactos reales.

## Decisiones fijas
- No se toca todavia el documento final de tesis.
- `training` queda cerrada como base estable; las nuevas reformas continuan en `reformas-finales` salvo decision explicita en contrario.
- El producto sigue acotado a scouting juvenil 12-18 para clubes formativos sin grandes presupuestos.
