# Diagramas PlantUML TPScouting

Diagramas generados para complementar el documento final del trabajo.

## Auditoria de correlacion

Los diagramas se armaron contra evidencia actual del repositorio:

- Rutas reales:
  - `/dashboard` en `scouting_app/routes/dashboard.py`.
  - `/player/<int:player_id>/predict` en `scouting_app/routes/players.py`.
- Templates reales:
  - `dashboard.html`.
  - `prediction.html`.
- Artefactos reales de Machine Learning:
  - `scouting_app/model.pt`.
  - `scouting_app/preprocessor.joblib`.
  - `scouting_app/probability_calibrator.joblib`.
- Runtime ML real:
  - `scouting_app/ml/runtime.py`.
  - `scouting_app/preprocessing.py`.
  - `scouting_app/train_model.py`.
- Persistencia real:
  - SQLite para desarrollo local.
  - PostgreSQL administrado en Render.
- Despliegue real:
  - Render + Gunicorn.
  - `render.yaml` usa `gunicorn app:app --chdir scouting_app --workers 1 --threads 2`.
- Modelo de datos real:
  - Entidades tomadas de `scouting_app/models.py`.

No se incluye SQL Server como componente implementado porque no aparece como tecnologia real del repositorio actual. Si se menciona, debe figurar solo como alternativa futura.

## Archivos fuente

- `plantuml/01_secuencia_prediccion.puml`
- `plantuml/02_secuencia_dashboard.puml`
- `plantuml/03_componentes.puml`
- `plantuml/04_despliegue.puml`
- `plantuml/05_clases.puml`
- `plantuml/06_casos_uso_general.puml`
- `plantuml/07_casos_uso_gestion_jugadores.puml`
- `plantuml/08_casos_uso_analisis_decision.puml`

## Imagenes exportadas

Cada diagrama se exporto en PNG y SVG:

- `export/01_secuencia_prediccion.png`
- `export/01_secuencia_prediccion.svg`
- `export/02_secuencia_dashboard.png`
- `export/02_secuencia_dashboard.svg`
- `export/03_componentes.png`
- `export/03_componentes.svg`
- `export/04_despliegue.png`
- `export/04_despliegue.svg`
- `export/05_clases.png`
- `export/05_clases.svg`
- `export/06_casos_uso_general.png`
- `export/06_casos_uso_general.svg`
- `export/07_casos_uso_gestion_jugadores.png`
- `export/07_casos_uso_gestion_jugadores.svg`
- `export/08_casos_uso_analisis_decision.png`
- `export/08_casos_uso_analisis_decision.svg`

## Ubicacion sugerida en el Word

- Diagrama de componentes: capitulo 4, arquitectura del sistema.
- Diagrama de clases: capitulo 4.4, modelo de datos.
- Secuencia de prediccion: capitulo 5, integracion funcional de ML.
- Secuencia de dashboard: capitulo 5, flujo del panel general.
- Diagrama de despliegue: capitulo 5.6 o capitulo 6, evidencia operativa del despliegue.
- Casos de uso general: capitulo 4.7, como resumen funcional.
- Casos de uso de gestion de jugadores: capitulo 4.7, como detalle del modulo jugadores.
- Casos de uso de analisis y decision: capitulo 4.7 o capitulo 6, como apoyo a la validacion de uso.

## Insercion en Word

Los diagramas fueron insertados en:

- `C:\Users\Usuario\Desktop\TRABAJO_FINAL - Version corregida TPScouting.docx`
- `docs/TRABAJO_FINAL_corregido_TPScouting.docx`

Figuras agregadas:

- `Figura 4-1. Diagrama de componentes de TPScouting.`
- `Figura 4-2. Diagrama de clases del modelo de dominio.`
- `Figura 5-1. Secuencia de prediccion de potencial de jugador.`
- `Figura 5-2. Secuencia de carga del panel general.`
- `Figura 5-3. Diagrama de despliegue en Render.`

Tambien se prepararon tres diagramas de casos de uso:

- `06_casos_uso_general`: vista general reducida para evitar exceso de lineas.
- `07_casos_uso_gestion_jugadores`: gestion de jugadores, carga CSV e historiales.
- `08_casos_uso_analisis_decision`: dashboard, comparadores, ficha y potencial.

Estos diagramas fueron insertados en el Word:

- `Figura 4-3. Diagrama de casos de uso general.`
- `Figura 4-4. Diagrama de casos de uso de gestión de jugadores.`
- `Figura 4-5. Diagrama de casos de uso de análisis y decisión scout.`

Tambien fueron agregados en tamaño ampliado en anexo:

- `Figura 10-6. Diagrama de casos de uso general ampliado.`
- `Figura 10-7. Diagrama de casos de uso de gestión de jugadores ampliado.`
- `Figura 10-8. Diagrama de casos de uso de análisis y decisión scout ampliado.`

Backup previo:

- `C:\Users\Usuario\Desktop\TRABAJO_FINAL - Version corregida TPScouting.backup_diagramas_20260522.docx`

## Generacion

El entorno local no tenia Java ni PlantUML instalados. Por eso se uso `scripts/render_plantuml_diagrams.py`, que codifica los `.puml` y renderiza contra el servidor publico de PlantUML sin agregar dependencias al proyecto.

Comando:

```powershell
.\.venv\Scripts\python.exe scripts\render_plantuml_diagrams.py
```
