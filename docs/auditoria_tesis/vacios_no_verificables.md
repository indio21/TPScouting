# Vacios y afirmaciones no verificables

Fecha de corte: 2026-07-10. Este archivo enumera informacion que no existe en el repositorio, no queda demostrada por los artefactos o no pudo confirmarse con una inspeccion segura.

| Clasificacion | Informacion faltante | Evidencia de ausencia o limite | Redaccion academica segura |
|---|---|---|---|
| **NO VERIFICABLE** | Duracion del entrenamiento | `scouting_app/training_metadata.json` no contiene duracion; no existe log de consola persistido | "La duracion de la corrida no fue registrada." |
| **NO VERIFICABLE** | Validation loss | El historial de `training_metadata.json` guarda `loss` de train, PR-AUC/F1, threshold, calibracion y LR; `scouting_app/train_model.py:630-639` no calcula/persiste validation loss | "No se dispone de validation loss para esta corrida." |
| **NO VERIFICABLE** | Commit exacto de entrenamiento | El metadata y checkpoint no guardan SHA Git. El commit `f58fc6b...` es temporalmente compatible, pero eso solo permite una inferencia | "La corrida esta fechada, pero no fue vinculada criptograficamente a un commit." |
| **NO VERIFICABLE** | Duracion/command line exacta ejecutada | El RUNBOOK contiene el comando oficial, pero no hay transcript original del proceso | "Se documenta el comando reproducible; no se conserva el transcript de la ejecucion original." |
| **NO VERIFICABLE** | Ejecucion remota actual de GitHub Actions | `.github/workflows/ci.yml` prueba configuracion, no estado de una run; no hay resultado remoto persistido localmente | "La suite paso localmente; el workflow esta configurado. No se afirma el estado de una run remota actual." |
| **NO VERIFICABLE** | Base operativa local actual | `scouting_app/players_updated_v2.db` no esta presente; solo hay backups ignorados y `players.db` legacy | "No se informa un conteo operativo local vigente." |
| **NO VERIFICABLE** | Uso de datos reales de jugadores | Generadores y documentos identifican datos sinteticos; no existe fuente, convenio, dataset o trazabilidad de datos reales | "La validacion se realizo con datos sinteticos; faltan datos reales longitudinales." |
| **NO VERIFICABLE** | Caracter semisintetico de la corrida oficial | No se encontro mezcla demostrable entre observaciones reales y sinteticas en los 20,000 casos | Usar "sintetico", no "semisintetico", para la corrida oficial |
| **NO VERIFICABLE** | Origen exacto de `scouting_app/players.db` | La base versionada tiene 1,000 registros legacy y edades 16-22, pero no incluye metadata de procedencia | "Base legacy de origen no documentado"; no usarla como evidencia del dataset oficial |
| **NO VERIFICABLE** | Disponibilidad publica vigente de Render | El smoke del 2026-07-10 termino por read timeout. El repo conserva evidencia historica del 20/05/2026 | Fechar la evidencia: "deploy validado el 20/05/2026"; no afirmar disponibilidad continua |
| **NO VERIFICABLE** | Resultado autenticado actual de Render | No se usaron credenciales durante esta auditoria | Citar solo el smoke historico versionado, con fecha |
| **NO VERIFICABLE** | Metricas del score combinado mostrado | `combine_probability` se usa en app, pero `train_model.py` evalua sigmoid crudo y salida calibrada, no score combinado | "Las metricas corresponden al modelo/calibracion; el score operativo combinado no tiene evaluacion independiente." |
| **NO VERIFICABLE** | Validez predictiva en futbol real | Test, target y datos son sinteticos; no hay cohorte externa ni seguimiento real | "Los resultados validan el pipeline tecnico, no eficacia deportiva externa." |
| **NO VERIFICABLE** | Causalidad de las features | El pipeline mide asociacion predictiva sobre simulacion; no hay diseño causal | No usar "causa" o "impacta"; usar "se asocia" o "contribuye al score sintetico" |
| **NO VERIFICABLE** | Superioridad general de PlayerNet | LogisticRegression es competitivo; la ventaja cambia segun metrica y si se usa salida cruda/calibrada | Presentar ambas variantes y evitar superioridad general |

## Contradicciones que deben corregirse o aclararse

1. **HECHO VERIFICADO:** `potential_label` no es el target de la corrida temporal. Fuente: `scouting_app/train_model.py:194-211`; `scouting_app/preprocessing.py:106,1264-1799`.
2. **HECHO VERIFICADO:** los valores `accuracy=0.9303`, `PR-AUC=0.5241` y matriz `[[2674,86],[123,117]]` son la salida calibrada isotonic (`pytorch.test`). Fuente: `training_metadata.json:38-68`; `train_model.py:674-682`.
3. **HECHO VERIFICADO:** PlayerNet crudo obtiene PR-AUC 0.5461, superior al baseline logistico 0.5378. Por eso la frase "LogisticRegression supera a PlayerNet en PR-AUC" solo es cierta si "PlayerNet" significa la variante calibrada. Fuente: `training_metadata.json:103-120,275-327`.
4. **HECHO VERIFICADO:** la cobertura actual es 79%, no 80%, en la ejecucion del 2026-07-10.
5. **HECHO VERIFICADO:** no hay evidencia de datos reales ni de mezcla semisintetica en la corrida oficial; `generate_data.py` genera el dataset.
6. **HECHO VERIFICADO - RIESGO:** el target usa cuantiles/cuotas del dataset completo antes del split. Fuente: `preprocessing.py:1700-1799`; `train_model.py:194-211,525-540`.

## Evidencia que convendria persistir en una futura corrida

Estas son recomendaciones documentales; no se ejecuto ni modifico nada:

- SHA Git y estado dirty/clean.
- Comando exacto y versiones de dependencias.
- Hora de inicio, fin y duracion.
- Training loss y validation loss por epoca.
- Hashes de dataset/cache/split y artefactos.
- Construccion del target ajustada solo sobre train, con reglas aplicadas luego a validation/test.
- Evaluacion separada de sigmoid crudo, probabilidad calibrada y score combinado.
- Evidencia de ejecucion CI identificada por URL/run ID.
- Smoke de deploy con fecha, commit y resultado por endpoint.
