# Mejora de prediccion - avance tecnico

## Estado actual
- Rama de trabajo: `training`
- Ultimo commit publicado antes de esta etapa: `a7e6f7f`
- HEAD local al generar este documento: `a7e6f7f`
- Objetivo de esta iteracion: cerrar la brecha entre PyTorch y el baseline lineal despues de sumar `PhysicalAssessment` y `Availability`, sin perder la riqueza metodologica ya ganada.
- Alcance del producto mantenido: scouting juvenil de 12 a 18 anos para clubes formativos.

## Que se implemento en esta etapa
- Se mantuvo la base enriquecida con dos fuentes nuevas de senal longitudinal:
- `PhysicalAssessment`
- `PlayerAvailability`
- El pipeline del modelo sigue agregando features fisicas recientes y de evolucion corporal.
- El pipeline del modelo sigue agregando disponibilidad, fatiga, carga y lesion/inactividad.
- La generacion sintetica sigue uniendo:
- trayectoria tecnica
- maduracion fisica
- disponibilidad mensual
- rendimiento competitivo derivado de esas tres capas
- En entrenamiento se reemplazo el doble rebalanceo previo por:
- `pos_weight` completo por defecto
- `shuffle` como estrategia default de batches
- `WeightedRandomSampler` solo como opcion
- `PlayerNet` paso a una arquitectura residual inicializada desde la solucion de `LogisticRegression(class_weight="balanced")`.
- La rama lineal queda dentro del modelo PyTorch y una rama residual aprende correcciones no lineales sobre esa base.
- Se mantuvo la calibracion de probabilidades y la inferencia compartida en la app.
- La app sigue sin cambiar su interfaz; el cambio permanece en el pipeline, el target y los artefactos del modelo.

## Conteos reales de la base de entrenamiento actual
- Jugadores: 20000
- Snapshots de `PlayerAttributeHistory`: 179964
- Registros agregados de `PlayerStat`: 120696
- Partidos sinteticos (`Match`): 241300
- Participaciones por partido: 241300
- Reportes de scout: 80104
- Evaluaciones fisicas: 179964
- Registros de disponibilidad: 179964

## Comparacion con la etapa anterior
### Etapa anterior publicada: target temporal recalibrado
- PyTorch: ROC-AUC 0.9247, PR-AUC 0.3279, F1 0.4231
- Logistic balanceado: ROC-AUC 0.9431, PR-AUC 0.3923, F1 0.4409

### Etapa local inmediatamente anterior: disponibilidad + fisico + trayectorias mas ricas
- PyTorch: ROC-AUC 0.9044, PR-AUC 0.2029, F1 0.2542
- Logistic balanceado: ROC-AUC 0.9425, PR-AUC 0.2775, F1 0.3659

### Etapa actual: reentrenamiento residual apoyado en baseline lineal
- PyTorch: accuracy 0.9713, ROC-AUC 0.9306, PR-AUC 0.2371, F1 0.3768, precision 0.2989, recall 0.5098
- Logistic balanceado: accuracy 0.9653, ROC-AUC 0.9425, PR-AUC 0.2775, F1 0.3659

## Hallazgos verificados
- El target temporal sigue siendo exigente: la tasa positiva actual es 1.69%.
- Respecto de la etapa publicada anterior (`a7e6f7f`), PyTorch sigue por debajo en rendimiento global:
- cambio en F1: -0.0463
- cambio en PR-AUC: -0.0908
- Respecto de la etapa local inmediatamente anterior con fisico/disponibilidad, PyTorch mejora de forma clara:
- cambio en F1: +0.1226
- cambio en PR-AUC: +0.0342
- La calibracion de probabilidades quedo implementada y en la corrida actual el metodo seleccionado fue `isotonic`.
- El target actual deja 310 positivos por consolidacion y 28 por breakout.
- El sistema ahora tiene una senal longitudinal mucho mas rica en la base:
- disponibilidad mensual
- fatiga
- carga de trabajo
- lesion/inactividad
- maduracion fisica y crecimiento corporal
- PyTorch vuelve a acercarse al baseline lineal gracias al bootstrap residual.
- En esta corrida, PyTorch supera al baseline lineal balanceado en `F1` y en precision al threshold operativo seleccionado.
- El baseline `LogisticRegression(class_weight="balanced")` sigue siendo mejor que PyTorch en `ROC-AUC` y `PR-AUC`.
- La prediccion es metodologicamente mas defendible porque ahora la progresion futura no depende solo de tecnica y contexto de partido, sino tambien de disponibilidad y maduracion fisica.
- El sistema mantiene el pipeline compartido entre entrenamiento e inferencia, pero el entrenamiento ya no mira la trayectoria completa como si fuera toda observable en el momento de decidir.

## Limites honestos de esta etapa
- Los partidos sinteticos siguen siendo por jugador; todavia no representan encuentros compartidos entre varios jugadores del mismo plantel.
- `ScoutReport` sigue siendo sintetico, no manual ni proveniente de observacion real de usuarios.
- El target temporal actual sigue siendo sintetico y deja 338 positivos sobre 20000 jugadores.
- PyTorch ya no queda por debajo en todas las metricas, pero todavia no gana en ranking global de probabilidades.
- No seria honesto decir que PyTorch ya reemplaza al baseline: sigue perdiendo en `ROC-AUC` y `PR-AUC`.
- La base de entrenamiento SQLite ya es pesada y el repo recibio advertencia de GitHub por tamano de `players_training.db`.

## Validacion ejecutada
- `pytest -q`: 35 tests aprobados.
- Smoke de app con artefactos nuevos:
- `/` -> 200
- `/health` -> 200
- `/login` -> 200
- Reentrenamiento completo ejecutado sobre la nueva base sintetica.

## Proximo paso recomendado
- Recalibrar de nuevo el target temporal para que la nueva senal de disponibilidad/fisico no deje una clase positiva demasiado rara.
- Seguir enriqueciendo la generacion sintetica con senales longitudinales no triviales, sin aumentar volumen por aumentar.
- Intentar que PyTorch tambien supere al baseline en `PR-AUC`, no solo en `F1`, antes de dar por cerrada la ventaja del modelo no lineal.
