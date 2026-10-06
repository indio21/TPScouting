# Afirmaciones y datos no verificables

Fecha: 2026-07-13

`NO VERIFICABLE` no significa necesariamente falso. Significa que el repositorio, los artefactos y las evidencias disponibles no permiten demostrarlo sin una fuente adicional.

| ID | Afirmacion o dato | Motivo | Evidencia necesaria para verificar |
|---|---|---|---|
| NV-01 | El entrenamiento oficial se ejecuto sobre un commit Git exacto. | `training_metadata.json` no guarda SHA. El commit que incorporo artefactos es temporalmente compatible, pero no prueba el estado usado al entrenar. | Log original con `git rev-parse HEAD` o metadata firmada que incluya SHA. |
| NV-02 | Duracion exacta de la corrida oficial de entrenamiento. | No existe campo de duracion ni transcript original persistido. | Log original con timestamps de inicio/fin. |
| NV-03 | Validation loss de la corrida oficial. | El historial registra training loss, PR-AUC, F1, threshold y LR, pero no validation loss; `train_model.py:630-639`. | Nueva corrida instrumentada; no debe inventarse para la corrida historica. |
| NV-04 | Rama actual conectada al servicio de Render. | La evidencia historica indica `render-free-deploy`; no existe export de configuracion actual que pruebe `main`. | Captura/export vigente de Settings de Render con fecha y servicio. |
| NV-05 | Disponibilidad actual continua de la URL Render. | Smoke exitoso historico; los controles del 10/07 y 13/07/2026 terminaron por timeout. Un timeout aislado tampoco prueba caida permanente. | Monitoreo externo con ventana temporal definida o smoke exitoso vigente. |
| NV-06 | Estado y contenido actuales de la base PostgreSQL desplegada. | La auditoria no accedio con credenciales ni consulto datos productivos. | Export sanitario sin datos personales, conteos fechados o acceso read-only autorizado. |
| NV-07 | Que el deploy actual contenga el HEAD local `7d7680e`. | La CI publica certifica `fa8a50d...`; no hay evidencia de deploy del HEAD local. | Identificador de deploy y commit SHA de Render/GitHub. |
| NV-08 | Que la CI remota haya aprobado el HEAD local `7d7680e`. | La ultima run publica consultada corresponde a otro SHA. | Nueva run de Actions sobre `7d7680e` o commit posterior que lo contenga. |
| NV-09 | Que los datos de entrenamiento incluyan jugadores reales o semisinteticos. | El generador y los artefactos disponibles prueban un conjunto sintetico; no hay fuente real documentada. | Dataset de origen, consentimiento/licencia, diccionario y trazabilidad de anonimizacion. |
| NV-10 | Mejora real en decisiones de scouts, reduccion de subjetividad o deteccion de talento subrepresentado. | No hay estudio de usuarios, grupo de comparacion ni validacion de campo. | Protocolo, muestra, resultados y analisis con profesionales/clubes. |
| NV-11 | Ahorro de tiempo, costos o recursos respecto del proceso tradicional. | No se midieron tiempos ni costos antes/despues. | Estudio comparativo con definiciones y mediciones reproducibles. |
| NV-12 | Validez externa de las metricas para futbol juvenil real. | Test y train provienen del mismo proceso sintetico y el target se construye globalmente antes del split. | Evaluacion prospectiva con datos reales independientes y target definido sin usar test. |
| NV-13 | Metricas de test del `combined_prob` mostrado al usuario. | La metadata evalua sigmoid crudo y salida calibrada, no el score combinado. | Protocolo y evaluacion separada del score combinado sobre un conjunto independiente. |
| NV-14 | Tasa de disponibilidad, seguridad productiva o resistencia a ataques. | No hay SLO, monitoreo continuo, pentest ni auditoria externa. | Reportes de monitoreo y pruebas de seguridad con alcance y fecha. |
| NV-15 | Origen exacto de la base legacy `players.db`. | Su contenido puede contarse, pero no existe una cadena de procedencia suficiente. | Documento de generacion/importacion y hash de origen. |

## Inferencias que no deben convertirse en hechos

- El commit `f58fc6b...` introdujo artefactos el 19/05/2026 y es compatible temporalmente con la corrida, pero no demuestra que ese codigo haya ejecutado el entrenamiento.
- Los timeouts de Render sugieren indisponibilidad o arranque lento durante esos intentos, pero no permiten concluir una caida permanente.
- La coherencia de los datos con `generate_data.py` apoya su origen sintetico, pero no autoriza a atribuir origen a bases legacy sin trazabilidad.

