# Auditoría reproducible de redacción — Bloque 8

Documento: `docs\revision_octubre_2026\final\TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-20.docx`
Párrafos y celdas examinados: 940.

Este control detecta candidatos para revisión humana; una coincidencia no equivale por sí sola a un error.

## Resumen

| Control | Coincidencias |
|---|---:|
| doble espacio | 0 |
| espacio antes de puntuacion | 0 |
| primera persona plural | 0 |
| segunda persona | 0 |
| lenguaje promocional | 4 |
| pendiente editorial | 0 |
| palabra consecutiva repetida | 0 |
| oraciones de 45 palabras o más | 4 |

## doble espacio

Sin coincidencias.

## espacio antes de puntuacion

Sin coincidencias.

## primera persona plural

Sin coincidencias.

## segunda persona

Sin coincidencias.

## lenguaje promocional

- **párrafo 170:** La incorporación de inteligencia artificial al análisis deportivo ofrece herramientas para organizar y comparar información. Plataformas como Wyscout centralizan video y datos para apoyar la observación, la comparación y el reclutamiento (Hudl, s. f.). Sin embargo, disponer de más datos no garantiza decisiones más precisas: los resultados dependen de la calidad del registro, del objetivo definido y de la revisión humana.
- **párrafo 329:** La regresión logística cumple el papel de modelo de referencia. En la corrida documentada, su desempeño fue muy similar al de PlayerNet: obtuvo valores levemente superiores en ROC-AUC y F1, mientras que PlayerNet crudo alcanzó el mayor PR-AUC. En consecuencia, el trabajo no sostiene una superioridad global de la red neuronal; la conclusión es que una arquitectura más compleja no garantiza mejores resultados y debe seleccionarse según evidencia.
- **párrafo 491:** El proceso mostró que una arquitectura sencilla, trazable y verificable fue más adecuada para el alcance que una solución distribuida. La comparación experimental confirmó que una red neuronal más compleja no garantiza una mejora frente a un modelo lineal bien configurado.
- **tabla 19, fila 25, celda 2:** Reajuste no paramétrico de una probabilidad para mejorar su interpretación; no garantiza mejores métricas de ranking.

## pendiente editorial

Sin coincidencias.

## Palabras consecutivas repetidas

Sin coincidencias.

## Oraciones extensas

- **párrafo 544, 88 palabras:** Windows:
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

Linux CPU:
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -r requirements-torch-cpu.txt
.venv/bin/python -m pip install -r requirements-dev.txt

macOS:
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
- **párrafo 554, 49 palabras:** Windows PowerShell:
..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
Linux/macOS:
../.venv/bin/python generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
- **párrafo 556, 79 palabras:** Windows PowerShell:
..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///players_training.db --model-out model_candidate.pt --preprocessor-out preprocessor_candidate.joblib --calibrator-out probability_calibrator_candidate.joblib --metadata-out training_metadata_candidate.json --splits-out training_splits_candidate.json --epochs 45 --lr 5e-4 --patience 10
Linux/macOS:
../.venv/bin/python train_model.py --db-url sqlite:///players_training.db --model-out model_candidate.pt --preprocessor-out preprocessor_candidate.joblib --calibrator-out probability_calibrator_candidate.joblib --metadata-out training_metadata_candidate.json --splits-out training_splits_candidate.json --epochs 45 --lr 5e-4 --patience 10
- **párrafo 560, 53 palabras:** Windows PowerShell:
..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18 --replace
Linux/macOS:
../.venv/bin/python sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18 --replace
