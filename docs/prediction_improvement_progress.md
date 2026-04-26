# Mejora de prediccion - avance tecnico

## Estado actual
- Rama de trabajo: `reformas-finales`
- Rama estable cerrada del MVP corregido: `training`
- Rama activa para nuevas reformas: `reformas-finales`
- Ultimo commit publicado antes de esta etapa: `a7e6f7f`
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
- Snapshots de `PlayerAttributeHistory`: 180212
- Registros agregados de `PlayerStat`: 116448
- Partidos sinteticos (`Match`): 232960
- Participaciones por partido: 232960
- Reportes de scout: 80062
- Evaluaciones fisicas: 180212
- Registros de disponibilidad: 180212

## Comparacion con la etapa anterior
### Etapa anterior publicada: target temporal recalibrado
- PyTorch: ROC-AUC 0.9247, PR-AUC 0.3279, F1 0.4231
- Logistic balanceado: ROC-AUC 0.9431, PR-AUC 0.3923, F1 0.4409

### Etapa local inmediatamente anterior: disponibilidad + fisico + trayectorias mas ricas
- PyTorch: ROC-AUC 0.9044, PR-AUC 0.2029, F1 0.2542
- Logistic balanceado: ROC-AUC 0.9425, PR-AUC 0.2775, F1 0.3659

### Etapa actual: reentrenamiento residual apoyado en baseline lineal
- PyTorch crudo: accuracy 0.9073, ROC-AUC 0.9102, PR-AUC 0.4826, F1 0.5088, precision 0.4417, recall 0.6000
- PyTorch calibrado: accuracy 0.9057, ROC-AUC 0.9084, PR-AUC 0.4617, F1 0.5162, precision 0.4377, recall 0.6292
- Logistic balanceado: accuracy 0.9110, ROC-AUC 0.9086, PR-AUC 0.4728, F1 0.4875

## Hallazgos verificados
- El target temporal sigue siendo exigente: la tasa positiva actual es 7.99%.
- Respecto de la etapa publicada anterior (`a7e6f7f`), PyTorch mejora en las metricas comparadas:
- cambio en F1: +0.0857
- cambio en PR-AUC: +0.1547
- Respecto de la etapa local inmediatamente anterior con fisico/disponibilidad, PyTorch mejora de forma clara:
- cambio en F1: +0.2546
- cambio en PR-AUC: +0.2797
- La calibracion de probabilidades quedo implementada y en la corrida actual el metodo seleccionado fue `isotonic`.
- La salida cruda queda como score principal porque supera al baseline en `PR-AUC` y `F1`; la calibrada queda como referencia secundaria porque mejora `F1` pero baja `PR-AUC`.
- El target actual deja 1333 positivos por consolidacion y 264 por breakout.
- El sistema ahora tiene una senal longitudinal mucho mas rica en la base:
- disponibilidad mensual
- fatiga
- carga de trabajo
- lesion/inactividad
- maduracion fisica y crecimiento corporal
- PyTorch vuelve a acercarse al baseline lineal gracias al bootstrap residual.
- En esta corrida, PyTorch crudo supera al baseline lineal balanceado en `PR-AUC` y `F1`.
- El baseline `LogisticRegression(class_weight="balanced")` se conserva como comparador formal obligatorio para futuras corridas.
- La prediccion es metodologicamente mas defendible porque ahora la progresion futura no depende solo de tecnica y contexto de partido, sino tambien de disponibilidad y maduracion fisica.
- El sistema mantiene el pipeline compartido entre entrenamiento e inferencia, pero el entrenamiento ya no mira la trayectoria completa como si fuera toda observable en el momento de decidir.

## Limites honestos de esta etapa
- Los partidos sinteticos siguen siendo por jugador; todavia no representan encuentros compartidos entre varios jugadores del mismo plantel.
- `ScoutReport` sigue siendo sintetico, no manual ni proveniente de observacion real de usuarios.
- El target temporal actual sigue siendo sintetico y deja 1597 positivos sobre 20000 jugadores.
- PyTorch crudo gana en la corrida oficial actual, pero esa ventaja sigue basada en datos sinteticos y debe validarse si cambia el target, la semilla o aparecen datos reales.
- No seria honesto eliminar el baseline: sigue siendo necesario para demostrar que PyTorch aporta valor en cada reentrenamiento.
- La base de entrenamiento SQLite ya es pesada y el repo recibio advertencia de GitHub por tamano de `players_training.db`.

## Validacion ejecutada
- `pytest -q`: 40 tests aprobados.
- Smoke de app con artefactos nuevos:
- `/` -> 200
- `/health` -> 200
- `/login` -> 200
- Reentrenamiento completo ejecutado sobre la nueva base sintetica.

## Proximo paso recomendado
- Mantener la salida cruda de PyTorch como score principal del MVP mientras conserve mejor ranking que el baseline.
- Dejar la probabilidad calibrada como evidencia secundaria y no como score principal.
- Seguir enriqueciendo la generacion sintetica con senales longitudinales no triviales, sin aumentar volumen por aumentar.
- Si se reabre el modelo, validar de nuevo PyTorch vs baseline bajo el mismo split y documentar la decision.
