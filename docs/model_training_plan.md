# Plan tecnico de entrenamiento

## Resumen
- Este plan queda versionado para la rama `training`.
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
- Resultado: PyTorch dejo de colapsar y alcanzo F1 0.3768 en test.

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
- Estado actual: PyTorch supera al baseline en F1 operativa con 0.3768 vs 0.3659, pero sigue por debajo en `PR-AUC`.

### Etapa 5 completada en nivel MVP
- Se persisten threshold, metricas, tamanos de split, seed y configuracion.
- La evidencia tecnica puede regenerarse desde los artefactos del repo.

### Etapa 6 completada: crecimiento con features historicas
- Se agregaron features historicas agregadas con `pandas`.
- Se sintetizo historial de `PlayerStat` en la base de entrenamiento.
- Se integraron features de `PlayerAttributeHistory` al entrenamiento e inferencia.
- La base de entrenamiento ahora representa trayectoria tecnica del jugador, no solo foto fija.
- Resultado actual: PyTorch queda en F1 0.3768 y PR-AUC 0.2371.
- Aun asi, el baseline lineal balanceado sigue siendo superior.

### Etapa 7 completada: contexto de partido y ScoutReport
- Se agregaron `Match` y `PlayerMatchParticipation` al esquema.
- El generador sintetico ahora crea partidos con contexto y participacion puntual del jugador.
- `PlayerStat` pasa a derivarse de esas participaciones.
- Se agregaron `ScoutReport` sinteticos al esquema y al pipeline.
- Resultado actual: PyTorch queda en F1 0.3768 y PR-AUC 0.2371.
- El baseline lineal balanceado sigue siendo superior.

### Etapa 8 completada: target temporal de progresion
- El entrenamiento deja de usar `potential_label` como target principal.
- Las features de entrenamiento se construyen en un punto de corte observado de la trayectoria.
- El target positivo se define por crecimiento tecnico futuro y mejora o consolidacion de rendimiento futuro.
- Resultado actual: PyTorch queda en F1 0.3768 y PR-AUC 0.2371 sobre un target con tasa positiva 1.69%.
- El baseline lineal balanceado sigue siendo superior.

### Etapa 9 completada: senales de disponibilidad y fisico
- Se agregaron `PhysicalAssessment` y `PlayerAvailability`.
- El generador sintetico ahora modela maduracion corporal, fatiga, carga, lesion e inactividad.
- Las features agregadas de disponibilidad y fisico ya entran tanto en entrenamiento como en inferencia.
- Resultado actual: PyTorch queda en F1 0.3768 y PR-AUC 0.2371.
- Conclusion honesta: el dataset queda metodologicamente mas rico y la arquitectura residual ya mejora el F1 operativo, pero el baseline lineal balanceado sigue arriba en ranking global.

### Etapa 10 completada: bootstrap lineal y correccion residual en PyTorch
- `PlayerNet` ahora se inicializa desde la solucion de `LogisticRegression(class_weight="balanced")`.
- La rama lineal queda dentro del modelo PyTorch y una rama residual aprende correcciones no lineales.
- El entrenamiento pasa a usar `pos_weight` completo y `shuffle` por defecto.
- Resultado actual: PyTorch supera al baseline en `F1` y precision, pero no en `ROC-AUC` ni `PR-AUC`.

## Siguiente iteracion recomendada
### Etapa 11 recomendada: redisenar aun mas el target y la relacion partido-plantel
- Revisar otra vez los umbrales del target temporal para evitar un positivo demasiado raro o demasiado facil.
- Pasar de partidos por jugador a partidos compartidos por plantel si se quiere ganar realismo extra.
- Revisar si PyTorch puede justificar su complejidad frente al baseline lineal tambien en `PR-AUC`, no solo en `F1`.

## Criterios de aceptacion para la siguiente iteracion
- La siguiente iteracion debe mejorar o estabilizar al menos una de estas metricas de PyTorch en test sin degradar claramente las demas:
- PR-AUC
- F1 positiva
- Recall positiva
- Y ademas debe volver mas explicable la prediccion desde el punto de vista futbolistico.
- La comparacion debe seguir quedando trazable en `training_metadata.json`.
- Si PyTorch no supera al baseline tras esa iteracion, el baseline debe quedar reconocido como referencia principal de rendimiento.

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
- Todo este trabajo se mantiene en la rama `training` hasta que se decida mergearlo.
- El producto sigue acotado a scouting juvenil 12-18 para clubes formativos sin grandes presupuestos.
