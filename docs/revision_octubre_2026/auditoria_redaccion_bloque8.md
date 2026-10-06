# Auditoría reproducible de redacción — Bloque 8

Documento: `docs\revision_octubre_2026\word\TRABAJO_FINAL_TPScouting_CANDIDATO_FINAL_BLOQUE8_2026-10-06.docx`
Párrafos y celdas examinados: 929.

Este control detecta candidatos para revisión humana; una coincidencia no equivale por sí sola a un error.

## Resumen

| Control | Coincidencias |
|---|---:|
| doble espacio | 1 |
| espacio antes de puntuacion | 0 |
| puntuacion repetida | 30 |
| primera persona plural | 0 |
| segunda persona | 0 |
| lenguaje promocional | 4 |
| pendiente editorial | 2 |
| palabra consecutiva repetida | 0 |
| oraciones de 45 palabras o más | 5 |

## doble espacio

- **párrafo 520:** •   README.md: instalación, pruebas, deploy y limitaciones.

## espacio antes de puntuacion

Sin coincidencias.

## puntuacion repetida

- **párrafo 143:** El trabajo se sitúa en el uso de datos y aprendizaje automático como apoyo al análisis deportivo. La literatura muestra aplicaciones para valorar acciones, comparar perfiles y estudiar la evolución de jugadores, aunque la utilidad de cada enfoque depende del origen de los datos y de su contexto de validación (Pappalardo et al., 2019; Decroos et al., 2019; Lacan, 2024).
- **párrafo 145:** El análisis de eventos, estadísticas e historiales permite construir evaluaciones multidimensionales y comparar perfiles de jugadores con criterios consistentes (Pappalardo et al., 2019; Decroos et al., 2019).
- **párrafo 147:** La literatura reciente también explora modelos de aprendizaje automático para detectar perfiles de alto potencial y pronosticar la evolución futura de los jugadores, aunque sus resultados dependen de la calidad de los datos, la definición del objetivo y el contexto de aplicación (Lacan, 2024; van Arem et al., 2025).
- **párrafo 183:** Machine Learning (ML) es una rama de la inteligencia artificial que permite a las máquinas aprender y mejorar a partir de la experiencia, sin ser explícitamente programadas para cada tarea. Esencialmente, ML se centra en el desarrollo de algoritmos que pueden procesar datos y realizar predicciones o tomar decisiones basadas en esos datos (Hastie et al., 2009; Goodfellow et al., 2016).
- **párrafo 184:** Aplicación en el deporte: el aprendizaje automático puede apoyar tareas de evaluación de rendimiento, valoración de acciones y detección de perfiles, siempre que el objetivo y los datos estén definidos de manera verificable (Pappalardo et al., 2019; Decroos et al., 2019; Lacan, 2024). En este trabajo su uso se limita a una clasificación experimental sobre datos sintéticos.
- **párrafo 186:** El análisis de datos deportivos permite resumir eventos, comparar perfiles y conservar evidencia de las evaluaciones. Los enfoques de PlayeRank y VAEP muestran que la utilidad de una métrica depende de la función asignada al jugador y del contexto de cada acción (Pappalardo et al., 2019; Decroos et al., 2019).
- **párrafo 189:** En este trabajo, la inteligencia artificial se utiliza en una tarea acotada de clasificación sobre datos sintéticos. No se evaluaron recomendaciones tácticas, programas de entrenamiento ni efectos sobre el rendimiento real. Los antecedentes de scouting predictivo se consideran comparaciones metodológicas y no pruebas de validez para TPScouting (Lacan, 2024; van Arem et al., 2025).
- **párrafo 190:** El uso de información deportiva de menores plantea riesgos de privacidad, seguridad y sesgo. La edad relativa, la maduración y el contexto pueden afectar tanto los registros como su interpretación; por ello se requieren transparencia y revisión humana (Cobley et al., 2009). Las obligaciones para un uso real se desarrollan en la Sección 6.5.
- **párrafo 192:** La ciencia de datos combina procedimientos de obtención, preparación, análisis y comunicación de datos. El aprendizaje automático aporta modelos para tareas predictivas, cuya validez depende de la separación entre entrenamiento y evaluación (Hastie et al., 2009; Goodfellow et al., 2016).
- **párrafo 204:** La discriminación y la calibración responden a preguntas diferentes: ROC-AUC evalúa el orden de los casos, mientras que una probabilidad calibrada busca coherencia entre frecuencias observadas y probabilidades estimadas. La calibración isotónica debe ajustarse fuera del conjunto de test para conservar una evaluación final independiente (Niculescu-Mizil & Caruana, 2005). Con una clase positiva minoritaria, PR-AUC complementa ROC-AUC porque expone la relación entre precisión y recall para la clase de interés (Saito & Rehmsmeier, 2015). Cualquier estadístico calculado con todo el conjunto antes del split puede transferir información de validación o test y debe declararse como riesgo de fuga (Kaufman et al., 2012).
- **párrafo 206:** Los datos sintéticos permiten probar el flujo técnico sin exponer datos personales, pero su distribución y su etiqueta reflejan reglas de generación y no evidencia observada en futbolistas. Además, agrupar juveniles por edad cronológica puede favorecer sistemáticamente a quienes nacieron antes dentro del año de selección o maduraron antes. La literatura identifica este efecto de edad relativa como un sesgo relevante en el desarrollo deportivo (Cobley et al., 2009). Por ello, edad, maduración y contexto deben analizarse antes de cualquier uso real del score.
- **párrafo 208:** Los trabajos relacionados pueden agruparse en cuatro líneas. PlayeRank propone una evaluación multidimensional y sensible al rol a partir de registros masivos de eventos de partido, y contrasta sus resultados con valoraciones de scouts profesionales (Pappalardo et al., 2019). El enfoque VAEP valora las acciones individuales según su efecto sobre las probabilidades de marcar y recibir goles, incorporando el contexto de cada acción (Decroos et al., 2019).
- **párrafo 280:** PyTorch se emplea para implementar PlayerNet (Paszke et al., 2019; PyTorch Foundation, s. f.). Scikit-learn aporta preprocesamiento, partición, métricas, regresión logística y calibración isotónica (Pedregosa et al., 2011). La inferencia web utiliza PlayerNet como modelo operativo; la regresión logística se conserva como referencia experimental.
- **párrafo 456:** Cobley, S., Baker, J., Wattie, N., & McKenna, J. (2009). Annual age-grouping and athlete development: A meta-analytical review of relative age effects in sport. Sports Medicine, 39(3), 235–256. https://doi.org/10.2165/00007256-200939030-00005
- **párrafo 457:** Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in soccer. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 1851–1861. https://doi.org/10.1145/3292500.3330758
- **párrafo 458:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.
- **párrafo 459:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7
- **párrafo 462:** Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4), Article 15. https://doi.org/10.1145/2382577.2382579
- **párrafo 464:** Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. Proceedings of the 22nd International Conference on Machine Learning, 625–632. https://doi.org/10.1145/1102351.1102430
- **párrafo 467:** Pappalardo, L., Cintia, P., Ferragina, P., Massucco, E., Pedreschi, D., & Giannotti, F. (2019). PlayeRank: Data-driven performance evaluation and player ranking in soccer via a machine learning approach. ACM Transactions on Intelligent Systems and Technology, 10(5), Article 59. https://doi.org/10.1145/3343172
- **párrafo 468:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
- **párrafo 469:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html
- **párrafo 472:** Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
- **párrafo 473:** Schwaber, K., & Sutherland, J. (2020). La Guía Scrum: La guía definitiva de Scrum: Las reglas del juego. https://scrumguides.org/docs/scrumguide/v2020/2020-Scrum-Guide-Spanish-Latin-South-American.pdf
- **párrafo 476:** van Arem, K. W., Goes-Smit, F., & Söhl, J. (2025). Forecasting the future development in quality and value of professional football players. Applied Sciences, 15(16), 8916. https://doi.org/10.3390/app15168916
- **párrafo 501:** ..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
- **párrafo 503:** ..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///players_training.db --model-out model.pt --preprocessor-out preprocessor.joblib --calibrator-out probability_calibrator.joblib --metadata-out training_metadata.json --splits-out training_splits.json --epochs 45 --lr 5e-4 --patience 10
- **párrafo 505:** ..\.venv\Scripts\python.exe evaluate_saved_model.py --db-url sqlite:///players_training.db --metadata-path training_metadata.json
- **párrafo 507:** ..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18 --replace
- **párrafo 509:** Set-Location ..

## primera persona plural

Sin coincidencias.

## segunda persona

Sin coincidencias.

## lenguaje promocional

- **párrafo 127:** La incorporación de inteligencia artificial al análisis deportivo ofrece herramientas para organizar y comparar información. Plataformas como Wyscout centralizan video y datos para apoyar la observación, la comparación y el reclutamiento (Hudl, s. f.). Sin embargo, disponer de más datos no garantiza decisiones más precisas: los resultados dependen de la calidad del registro, del objetivo definido y de la revisión humana.
- **párrafo 285:** La regresión logística cumple el papel de modelo de referencia. En la corrida documentada, su desempeño fue muy similar al de PlayerNet: obtuvo valores levemente superiores en ROC-AUC y F1, mientras que PlayerNet crudo alcanzó el mayor PR-AUC. En consecuencia, el trabajo no sostiene una superioridad global de la red neuronal; la conclusión es que una arquitectura más compleja no garantiza mejores resultados y debe seleccionarse según evidencia.
- **párrafo 444:** El proceso mostró que una arquitectura sencilla, trazable y verificable fue más adecuada para el alcance que una solución distribuida. La comparación experimental confirmó que una red neuronal más compleja no garantiza una mejora frente a un modelo lineal bien configurado.
- **tabla 19, fila 25, celda 2:** Reajuste no paramétrico de una probabilidad para mejorar su interpretación; no garantiza mejores métricas de ranking.

## pendiente editorial

- **párrafo 13:** Alumno: Solari, Pablo
Legajo: [PENDIENTE DE INFORMAR]
- **párrafo 204:** La discriminación y la calibración responden a preguntas diferentes: ROC-AUC evalúa el orden de los casos, mientras que una probabilidad calibrada busca coherencia entre frecuencias observadas y probabilidades estimadas. La calibración isotónica debe ajustarse fuera del conjunto de test para conservar una evaluación final independiente (Niculescu-Mizil & Caruana, 2005). Con una clase positiva minoritaria, PR-AUC complementa ROC-AUC porque expone la relación entre precisión y recall para la clase de interés (Saito & Rehmsmeier, 2015). Cualquier estadístico calculado con todo el conjunto antes del split puede transferir información de validación o test y debe declararse como riesgo de fuga (Kaufman et al., 2012).

## Palabras consecutivas repetidas

Sin coincidencias.

## Oraciones extensas

- **párrafo 123, 54 palabras:** Esta situación plantea una paradoja: mientras el talento juvenil puede convertirse en una fuente de valor deportivo y económico en el mediano plazo, la falta de inversión y de procesos de registro y análisis de información reduce la capacidad de los clubes para identificar oportunidades, planificar la formación y sostener decisiones basadas en evidencia.
- **párrafo 147, 49 palabras:** La literatura reciente también explora modelos de aprendizaje automático para detectar perfiles de alto potencial y pronosticar la evolución futura de los jugadores, aunque sus resultados dependen de la calidad de los datos, la definición del objetivo y el contexto de aplicación (Lacan, 2024; van Arem et al., 2025).
- **párrafo 213, 45 palabras:** Se utilizan datos sintéticos para validar el flujo completo de captura, persistencia, entrenamiento e inferencia, ya que esta decisión favorece la reproducibilidad y evita depender de bases privadas de clubes o proveedores externos durante la evaluación académica, pero limita la validez externa de los resultados.
- **párrafo 222, 48 palabras:** El score previo a los controles pondera crecimiento (0,15), nivel futuro (0,12), rendimiento (0,13), presión (0,13), consistencia (0,10), rol (0,09), disponibilidad (0,09), recuperación (0,09), evaluación scout (0,10) y breakout (0,14), con una penalización por inestabilidad de 0,08.
- **párrafo 428, 46 palabras:** Un uso real debería definir una base jurídica y una finalidad específica, informar a los titulares y a sus representantes, obtener el consentimiento que corresponda, recolectar sólo datos necesarios, aplicar plazos de retención y mecanismos de acceso, rectificación y supresión, y limitar el acceso por rol.
