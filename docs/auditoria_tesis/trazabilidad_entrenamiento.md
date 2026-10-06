# Trazabilidad de la corrida de entrenamiento

## Evidencia disponible

| Evidencia | Estado | Uso academico |
|---|---|---|
| `training_metadata_snapshot_2026-07-10.json` | Copia exacta del metadata versionado | Probar configuracion, split, metricas, historial, calibracion y baselines |
| `training_history.csv` | Derivado mecanicamente de `pytorch.history` | Tabla de 15 epocas para anexos o graficos |
| `training_curve_snapshot_2026-07-10.png` | Copia exacta de la curva ya versionada | Figura de evolucion de loss/metricas, sujeta a lo contenido en metadata |
| `metricas_test.csv` | Derivado mecanicamente del metadata | Comparar PlayerNet calibrado, PlayerNet crudo y baselines |
| `target_distribution.csv` | Conteos verificados contra target y splits | Documentar desbalance total/train/validation/test |
| `artefactos_sha256.csv` | Hashes calculados el 2026-07-10 | Identificar checkpoint y artefactos sin ambiguedad |

## Lo que no existe

**NO VERIFICABLE.** No se encontro un log crudo de consola de la ejecucion original del 19/05/2026. No se genero uno retroactivamente porque hacerlo exigiria reentrenar o simular una salida historica.

`training_history.csv` no debe llamarse "log original". Es una exportacion fiel de la historia persistida en `scouting_app/training_metadata.json`, seccion `pytorch.history`, con:

- epoca;
- training loss;
- validation PR-AUC;
- validation F1;
- threshold;
- metodo de calibracion;
- learning rate.

Tampoco existen validation loss, duracion total ni SHA Git de la corrida. Estas ausencias se mantienen declaradas en `vacios_no_verificables.md`.

## Texto sugerido para el Trabajo Final

> La corrida conserva metadata estructurada con semilla, hiperparametros, particiones, metricas e historial por epoca. No se dispone del log de consola original ni de validation loss. La identificacion de los artefactos se refuerza mediante hashes SHA-256 calculados durante la auditoria tecnica.

## Identificacion del checkpoint

| Elemento | Valor |
|---|---|
| Archivo | `scouting_app/model.pt` |
| SHA-256 | `D9FAC23BDFC294C91CB8E7A10AAAA3E4BF90F4E6A3CDA14C022880B2AB12BB19` |
| Formato | checkpoint version 1 |
| Clase | `PlayerNet` |
| Input | 68 |
| Seed | 42 |
| Parametros entrenables | 17,994 |

Fuentes: `scouting_app/model.pt`, `scouting_app/train_model.py:75-126` y `training_metadata_snapshot_2026-07-10.json`.
