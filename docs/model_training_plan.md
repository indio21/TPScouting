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
- Resultado: PyTorch dejo de colapsar y alcanzo F1 0.6905 en test.

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
- Estado actual: el baseline sigue mejor que PyTorch en test con F1 0.8536 vs 0.6905.

### Etapa 5 completada en nivel MVP
- Se persisten threshold, metricas, tamanos de split, seed y configuracion.
- La evidencia tecnica puede regenerarse desde los artefactos del repo.

### Etapa 6 completada: crecimiento con features historicas
- Se agregaron features historicas agregadas con `pandas`.
- Se sintetizo historial de `PlayerStat` en la base de entrenamiento.
- Resultado: PyTorch mejoro hasta F1 0.6905 y PR-AUC 0.7887.
- Aun asi, el baseline lineal balanceado sigue siendo superior.

## Siguiente iteracion recomendada
### Etapa 7 opcional: calibracion y cierre metodologico
- Evaluar calibracion de probabilidades si se necesita que la probabilidad base sea mas interpretable.
- Revisar si conviene simplificar arquitectura de la MLP o ajustar regularizacion.
- Volver a comparar PyTorch contra el baseline lineal bajo el mismo split.

## Criterios de aceptacion para la siguiente iteracion
- La siguiente iteracion debe mejorar o estabilizar al menos una de estas metricas de PyTorch en test sin degradar claramente las demas:
- PR-AUC
- F1 positiva
- Recall positiva
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
- Smoke del pipeline completo con artefactos reales.

## Decisiones fijas
- No se toca todavia el documento final de tesis.
- Todo este trabajo se mantiene en la rama `training` hasta que se decida mergearlo.
- El producto sigue acotado a scouting juvenil 12-18 para clubes formativos sin grandes presupuestos.
