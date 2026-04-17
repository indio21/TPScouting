# Mejora de prediccion - avance tecnico

## Estado actual
- Rama de trabajo: `training`
- Ultimo commit publicado antes de esta etapa: `449fbe9`
- HEAD local al generar este documento: `449fbe9`
- Objetivo de esta iteracion: sumar contexto de partido (`Match` + `PlayerMatchParticipation`) y observacion cualitativa (`ScoutReport`) al sistema de prediccion.
- Alcance del producto mantenido: scouting juvenil de 12 a 18 anos para clubes formativos.

## Que se implemento en esta etapa
- Nueva tabla `Match` para contexto minimo del partido.
- Nueva tabla `PlayerMatchParticipation` para registrar la participacion puntual del jugador por partido.
- Nueva tabla `ScoutReport` para observaciones cualitativas del scout.
- El generador sintetico ahora crea:
- trayectoria tecnica mensual
- partidos con contexto
- participacion del jugador en cada partido
- `PlayerStat` agregado a partir de esas participaciones
- reportes de scout derivados de trayectoria y rendimiento reciente
- El pipeline de entrenamiento e inferencia ahora agrega features de:
- contexto de partido
- tasa de titularidad
- minutos medios
- nivel medio del rival
- alineacion entre posicion natural y posicion jugada
- reportes cualitativos del scout

## Conteos reales de la base de entrenamiento actual
- Jugadores: 20000
- Snapshots de `PlayerAttributeHistory`: 179377
- Registros agregados de `PlayerStat`: 162180
- Partidos sinteticos (`Match`): 324165
- Participaciones por partido: 324165
- Reportes de scout: 79574

## Comparacion con la etapa anterior
### Etapa anterior: trayectoria tecnica sin contexto de partido explicito
- PyTorch: ROC-AUC 0.8468, PR-AUC 0.6153, F1 0.5751
- Logistic balanceado: ROC-AUC 0.9052, PR-AUC 0.7360, F1 0.6878

### Etapa actual: contexto de partido + ScoutReport
- PyTorch: accuracy 0.8380, ROC-AUC 0.8628, PR-AUC 0.6508, F1 0.6042, precision 0.6042, recall 0.6042
- Logistic balanceado: accuracy 0.8600, ROC-AUC 0.8996, PR-AUC 0.7373, F1 0.6769

## Hallazgos verificados
- PyTorch mejora respecto de la etapa longitudinal anterior:
- cambio en F1: +0.0291
- cambio en PR-AUC: +0.0355
- El baseline `LogisticRegression(class_weight="balanced")` sigue siendo mejor que PyTorch en esta corrida.
- La prediccion ahora es mas defendible metodologicamente porque ya no se apoya solo en atributos actuales y agregados simples.
- El sistema ya puede incorporar senal de contexto competitivo y senal cualitativa del scout sin romper el pipeline compartido entre entrenamiento e inferencia.

## Limites honestos de esta etapa
- Los partidos sinteticos siguen siendo por jugador; todavia no representan encuentros compartidos entre varios jugadores del mismo plantel.
- `ScoutReport` sigue siendo sintetico, no manual ni proveniente de observacion real de usuarios.
- El target del entrenamiento sigue siendo `potential_label` binario; todavia no pasamos a una meta temporal de progresion.
- Aunque PyTorch mejoro, todavia no supera al baseline lineal balanceado.
- La base de entrenamiento SQLite ya es pesada y el repo recibio advertencia de GitHub por tamano de `players_training.db`.

## Validacion ejecutada
- `pytest -q`: 34 tests aprobados.
- Smoke de app con artefactos nuevos:
- `/` -> 200
- `/health` -> 200
- `/login` -> 200
- Reentrenamiento completo ejecutado sobre la nueva base sintetica.

## Proximo paso recomendado
- Cambiar el target a una meta temporal de progresion.
- Evaluar si conviene introducir `Availability` o `PhysicalAssessment`.
- Replantear si el baseline lineal debe quedar como referencia principal hasta que PyTorch demuestre ventaja clara.
