# Evidencia ML — Bloque 5

Fecha de auditoría: 2026-10-05. Esta evidencia analiza la corrida histórica
persistida del 2026-05-19. No reentrena, no modifica bases y no sobrescribe
artefactos.

## Parámetros y estado

- Parámetros entrenables (`model.parameters()`): **17.608**.
- Total de parámetros: **17.608**.
- Buffers: **386**.
- Elementos del `state_dict`: **17.994**.

Por lo tanto, 17.994 no es el número de parámetros entrenables. Es la suma de
parámetros y buffers persistidos.

## Features, etiquetas y splits

- Target de entrenamiento: `temporal_target_label`.
- Columnas de entrada antes del encoding: 64; dimensión transformada: 68.
- `potential_label` y `temporal_target_label` existen en el dataframe temporal,
  pero ninguna integra `MODEL_FEATURE_COLUMNS`.
- Tampoco entran como features `combined_prob`, probabilidades crudas o
  calibradas ni predicciones persistidas.
- Splits persistidos: train 14.000, validación 3.000 y test 3.000, seed 42.
- Intersecciones entre los tres conjuntos: cero.

Esto descarta una fuga directa por inclusión de etiquetas/predicciones como
features y solapamiento de IDs. No elimina la circularidad conceptual del
generador sintético: `potential_label` se deriva de atributos actuales, mientras
`temporal_target_label` se construye con evolución futura sintética. Además, los
cuantiles y cupos del target temporal se calculan sobre el conjunto completo
antes de aplicar el split. La tasa positiva cercana al 8% es una decisión de
diseño, no una prevalencia observada en futbolistas reales.

## Probabilidades y umbrales

- Probabilidad cruda: salida sigmoid de PlayerNet.
- Probabilidad calibrada: transformación isotónica de la salida cruda.
- Score combinado: mezcla operativa de la probabilidad cruda, historial y ajuste
  posicional. No hay predicciones de test persistidas para evaluarlo como modelo.
- Umbral crudo seleccionado en validación: 0,825.
- Umbral calibrado seleccionado en validación: 0,25.
- Bandas visuales de la aplicación: 0,60 y 0,80 sobre el score combinado.

Los umbrales 0,25/0,825 son puntos de operación para métricas binarias y no son
las bandas de presentación 0,60/0,80. Las métricas publicadas no validan las
bandas ni el score combinado.

## Métricas reproducidas y bootstrap pareado

Sobre el test persistido se reprodujeron:

| Salida | ROC-AUC | PR-AUC |
|---|---:|---:|
| PlayerNet crudo | 0,920341 | 0,546127 |
| PlayerNet calibrado | 0,917431 | 0,524116 |
| Regresión logística balanceada | 0,920506 | 0,537776 |

Bootstrap pareado no paramétrico, 2.000 remuestreos del mismo test, seed de
auditoría 20261005; diferencia PlayerNet crudo menos regresión logística:

- ROC-AUC: diferencia media -0,000151; IC percentil 95% [-0,002693; 0,002471].
- PR-AUC: diferencia media 0,008194; IC percentil 95% [-0,002508; 0,019543].

Ambos intervalos incluyen cero. Esta corrida no aporta evidencia suficiente para
afirmar superioridad estadística de uno de los dos modelos. Tampoco prueba
equivalencia: un intervalo que incluye cero no es una prueba de equivalencia y el
análisis usa un solo dataset/test sintético.

## Trazabilidad y límites históricos

El JSON reproducible contiene hashes SHA-256 de modelo, preprocesador,
calibrador, metadata, splits y dataframe temporal:
`evidencia_ml_bloque5.json`.

El commit actualmente checkout es
`49aa51c0167fdb24c5f2f3a6ab6e3f397830b462`. No se afirma que sea el commit de
entrenamiento: el metadata histórico no lo registra. Tampoco registra duración,
versiones de bibliotecas ni validation loss. Esos valores no pueden reconstruirse
como hechos históricos. La lista de loss disponible corresponde a train loss por
época.
