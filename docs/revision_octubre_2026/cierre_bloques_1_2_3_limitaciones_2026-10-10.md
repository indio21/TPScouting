# Cierre técnico de limitaciones 1, 2 y 3

Fecha: 2026-10-10, America/Buenos_Aires.

Repositorio trabajado: `C:\Tesis\TPScouting`.

El repositorio `C:\Tesis\TPScouting-entrega` no fue leído para copiar cambios,
modificado, sincronizado ni publicado durante estos bloques.

## Veredicto

- Bloque 1, evaluación independiente del score combinado: **RESUELTO**.
- Bloque 2, target ajustado después del split: **RESUELTO PARA LA CORRIDA NUEVA**.
- Bloque 3, corrida nueva reproducible: **RESUELTO COMO CANDIDATA EXPERIMENTAL**.
- Artefactos utilizados por la aplicación: **NO REEMPLAZADOS**.
- Documento Word: **NO MODIFICADO**; los cambios necesarios quedan identificados
  para la actualización documental integral.

## Bloque 1 — score combinado

Se evaluaron sobre las mismas 3.000 filas de test histórico:

| Variante | ROC-AUC | PR-AUC | Brier | Log loss | F1 |
|---|---:|---:|---:|---:|---:|
| Probabilidad cruda | 0,920341 | 0,546127 | 0,104880 | 0,329400 | 0,529148 |
| Probabilidad calibrada | 0,917431 | 0,524116 | 0,049552 | 0,168487 | 0,528217 |
| Combinado desde cruda | 0,910888 | 0,506254 | 0,184748 | 0,557933 | 0,548276 |
| Combinado desde calibrada | 0,898656 | 0,518373 | 0,143940 | 0,470627 | 0,527881 |

Los umbrales de clasificación se seleccionaron en validation. Las bandas `0,60`
y `0,80` se evaluaron por separado como bandas visuales; no se presentaron como
umbrales optimizados.

Bootstrap pareado de 2.000 remuestreos, combinado desde cruda menos probabilidad
cruda:

- ROC-AUC: diferencia media `-0,009494`; IC 95 %
  `[-0,015975; -0,003779]`.
- PR-AUC: diferencia media `-0,040005`; IC 95 %
  `[-0,067787; -0,013795]`.
- Brier: diferencia media `+0,079889`; IC 95 %
  `[+0,076678; +0,083045]`. En Brier, menor es mejor.

Conclusión limitada a estos datos sintéticos: no existe evidencia para presentar
el score combinado histórico como mejora de la probabilidad cruda. Aumenta recall
con el umbral elegido, pero reduce discriminación y empeora marcadamente la calidad
probabilística. La probabilidad calibrada mejora Brier y log loss, aunque pierde
PR-AUC frente a la cruda. Por tanto, cada salida responde a un objetivo distinto.

## Bloque 2 — target sin ajuste sobre validation/test

La corrida nueva usa esta secuencia:

1. split determinista por `player_id`, estratificado por posición y grupo etario;
2. ajuste de todos los cuantiles y umbrales solo con las 14.000 filas de train;
3. congelamiento de la política;
4. aplicación sin recalcular cuantiles ni cuotas sobre las 3.000 filas de
   validation y las 3.000 de test.

Prevalencias resultantes:

- train: `0,080000` (1.120/14.000);
- validation: `0,074333` (223/3.000);
- test: `0,080333` (241/3.000).

El 8 % continúa siendo una decisión explícita del generador sintético. Se ajusta
en train mediante un umbral de `progression_score` dentro del filtro de calidad;
validation y test conservan la prevalencia que resulta al aplicar ese umbral.

Las pruebas verifican splits disjuntos, determinismo, cobertura total de IDs,
ausencia de features prohibidas y que una modificación de test no altera la
política aprendida con train.

## Bloque 3 — corrida candidata reproducible

Directorio local autoritativo:

`artifacts/runs/20261010T_local_seed42_leakage_safe_v2/`

Configuración observada:

- seed: 42;
- Python: 3.11.9;
- Windows/AMD64;
- PyTorch: 2.14.1+cpu;
- épocas solicitadas: 30;
- épocas ejecutadas: 15 por early stopping;
- mejor época: 7;
- duración interna de entrenamiento: 15,0532 segundos;
- duración completa, incluida evaluación y bootstrap: 72,7944 segundos;
- 17.608 parámetros entrenables, 386 elementos de buffers y 17.994 elementos
  del `state_dict`;
- input transformado: 68 columnas.

Métricas de test de la corrida nueva:

| Variante | ROC-AUC | PR-AUC | Brier | Log loss | F1 |
|---|---:|---:|---:|---:|---:|
| Probabilidad cruda | 0,905581 | 0,488843 | 0,117485 | 0,367323 | 0,493103 |
| Probabilidad calibrada | 0,903533 | 0,462045 | 0,053906 | 0,184553 | 0,502415 |
| Combinado desde cruda | 0,903743 | 0,471040 | 0,188587 | 0,566371 | 0,511327 |
| Combinado desde calibrada | 0,894159 | 0,488111 | 0,144294 | 0,471052 | 0,472103 |

La repetición completa del entrenamiento comprobó igualdad exacta de:

- política de target;
- IDs y orden de los splits;
- estado del modelo;
- transformación del preprocesador;
- predicciones del calibrador;
- historial de entrenamiento;
- métricas de test.

Los timestamps y las duraciones de reloj no se comparan porque necesariamente
varían. El primer intento (`v1`) detectó que la semilla solo se inicializaba al
importar el módulo; la evaluación previa consumía el generador y cambiaba el
modelo. Se corrigió reinicializando Python, NumPy y PyTorch al comienzo de cada
entrenamiento. La evidencia del intento fallido se conserva en
`reproducibility_failure_v1_2026-10-10.json`; sus artefactos candidatos se
eliminaron para evitar confundirlos con la corrida autoritativa `v2`.

## Evidencia y artefactos

La corrida `v2` guarda:

- modelo, preprocesador y calibrador candidatos;
- dataframe etiquetado y splits;
- política de target;
- metadata con train loss y validation loss por época;
- duración, configuración y conteo de parámetros;
- predicciones de test;
- evaluación histórica y nueva;
- bootstrap pareado;
- `pip freeze` completo;
- hashes SHA-256 y tamaños de archivos;
- hashes del código fuente usado;
- comprobación de reproducibilidad.

`manifest.json` contiene 14 archivos controlados y la verificación no encontró
discrepancias de hash.

## Validación de código

- pruebas específicas nuevas: 6 aprobadas;
- suite completa: 122 aprobadas, 1 omitida y 4 advertencias conocidas;
- cobertura total: 83,22 %, por encima del umbral de 80 %;
- Ruff crítico: aprobado;
- `compileall`: aprobado;
- `pip check`: aprobado;
- checkpoint candidato: carga correcta, `input_dim=68`, 20 entradas de estado;
- modelo, preprocesador y calibrador operativos: sin cambios.

## Cambios documentales identificados y diferidos

Cuando se prepare la siguiente copia del Word se deberá:

1. retirar cualquier interpretación del score combinado como mejora demostrada;
2. explicar el compromiso entre ranking crudo y calibración;
3. presentar las bandas `0,60/0,80` únicamente como categorías visuales;
4. describir la nueva política train-only y sus prevalencias resultantes;
5. separar completamente la corrida histórica de la candidata `v2`;
6. incorporar duración, validation loss, hashes y prueba de reproducibilidad de
   la candidata;
7. aclarar que el modelo de runtime no fue reemplazado y que los resultados siguen
   limitados a datos sintéticos.

No se modifica el Word todavía para evitar versiones parciales antes del cierre
integral de las limitaciones restantes.

## Decisión adoptada

El usuario confirmó que la probabilidad cruda continuará como score principal del
modelo. El score combinado, si permanece visible, se presentará como heurística
secundaria de priorización y no como probabilidad ni como mejora demostrada.
