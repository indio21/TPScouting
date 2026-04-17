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
- Split `train / validation / test`.
- Seleccion de threshold por validacion.
- Early stopping sobre `PR-AUC` con desempate por `F1`.
- Resultado: PyTorch dejo de colapsar y alcanzo F1 0.6042 en test.

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
- Estado actual: el baseline sigue mejor que PyTorch en test con F1 0.6769 vs 0.6042.

### Etapa 5 completada en nivel MVP
- Se persisten threshold, metricas, tamanos de split, seed y configuracion.
- La evidencia tecnica puede regenerarse desde los artefactos del repo.

### Etapa 6 completada: crecimiento con features historicas
- Se agregaron features historicas agregadas con `pandas`.
- Se sintetizo historial de `PlayerStat` en la base de entrenamiento.
- Se integraron features de `PlayerAttributeHistory` al entrenamiento e inferencia.
- La base de entrenamiento ahora representa trayectoria tecnica del jugador, no solo foto fija.
- Resultado actual: PyTorch queda en F1 0.6042 y PR-AUC 0.6508.
- Aun asi, el baseline lineal balanceado sigue siendo superior.

### Etapa 7 completada: contexto de partido y ScoutReport
- Se agregaron `Match` y `PlayerMatchParticipation` al esquema.
- El generador sintetico ahora crea partidos con contexto y participacion puntual del jugador.
- `PlayerStat` pasa a derivarse de esas participaciones.
- Se agregaron `ScoutReport` sinteticos al esquema y al pipeline.
- Resultado actual: PyTorch queda en F1 0.6042 y PR-AUC 0.6508.
- El baseline lineal balanceado sigue siendo superior.

## Siguiente iteracion recomendada
### Etapa 8 recomendada: target temporal y validacion metodologica
- Redefinir el target hacia una meta temporal de progresion, no solo `potential_label`.
- Evaluar si conviene sumar `Availability` o `PhysicalAssessment`.
- Revisar si PyTorch puede justificar su complejidad frente al baseline lineal.

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
