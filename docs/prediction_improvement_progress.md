# Mejora de prediccion - avance tecnico

## Estado actual
- Rama de trabajo: `training`
- Ultimo commit publicado antes de esta etapa: `7df19cb`
- HEAD local al generar este documento: `7df19cb`
- Objetivo de esta iteracion: pasar de un target binario estatico a un target temporal de progresion futura.
- Alcance del producto mantenido: scouting juvenil de 12 a 18 anos para clubes formativos.

## Que se implemento en esta etapa
- El entrenamiento deja de usar `potential_label` como target principal.
- Se construye un dataset temporal con corte observado/futuro por jugador.
- Las features se calculan solo con la parte observada de la trayectoria.
- Los atributos base de entrenamiento se anclan en el punto de corte temporal para evitar fuga de informacion desde el estado final.
- El target `temporal_target_label` se marca positivo cuando el tramo futuro combina:
- crecimiento tecnico ponderado por posicion
- mejora o consolidacion del rendimiento futuro
- La app no cambia su interfaz en esta etapa; el cambio queda en el pipeline de entrenamiento y en los artefactos del modelo.

## Conteos reales de la base de entrenamiento actual
- Jugadores: 20000
- Snapshots de `PlayerAttributeHistory`: 179377
- Registros agregados de `PlayerStat`: 162180
- Partidos sinteticos (`Match`): 324165
- Participaciones por partido: 324165
- Reportes de scout: 79574

## Comparacion con la etapa anterior
### Etapa anterior: contexto de partido + ScoutReport con target estatico
- PyTorch: ROC-AUC 0.8628, PR-AUC 0.6508, F1 0.6042
- Logistic balanceado: ROC-AUC 0.8996, PR-AUC 0.7373, F1 0.6769

### Etapa actual: target temporal de progresion
- PyTorch: accuracy 0.8680, ROC-AUC 0.8374, PR-AUC 0.2598, F1 0.2528, precision 0.1754, recall 0.4527
- Logistic balanceado: accuracy 0.9380, ROC-AUC 0.9279, PR-AUC 0.3645, F1 0.3922

## Hallazgos verificados
- El target temporal deja el problema mucho mas exigente: la tasa positiva actual es 4.94%.
- PyTorch empeora respecto de la etapa anterior:
- cambio en F1: -0.3514
- cambio en PR-AUC: -0.3910
- El baseline `LogisticRegression(class_weight="balanced")` sigue siendo mejor que PyTorch en esta corrida.
- La prediccion es metodologicamente mas defendible porque el modelo ahora intenta anticipar progresion futura en lugar de reproducir una etiqueta estatica.
- El sistema mantiene el pipeline compartido entre entrenamiento e inferencia, pero el entrenamiento ya no mira la trayectoria completa como si fuera toda observable en el momento de decidir.

## Limites honestos de esta etapa
- Los partidos sinteticos siguen siendo por jugador; todavia no representan encuentros compartidos entre varios jugadores del mismo plantel.
- `ScoutReport` sigue siendo sintetico, no manual ni proveniente de observacion real de usuarios.
- El target temporal actual es sintetico y todavia puede estar demasiado restringido: deja solo 988 positivos sobre 20000 jugadores.
- Aunque el target mejora la validez metodologica, hoy empeora el rendimiento de PyTorch frente a la etapa anterior.
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
- Recalibrar los umbrales del target temporal para que la clase positiva no quede tan rara.
- Evaluar si conviene introducir `Availability` o `PhysicalAssessment`.
- Replantear si el baseline lineal debe quedar como referencia principal hasta que PyTorch demuestre ventaja clara.
