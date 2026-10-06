# Correcciones propuestas para el Word

Fecha: 2026-07-13

Estas propuestas no fueron aplicadas al `.docx`. El texto puede trasladarse despues de revisar estilo y numeracion definitiva.

## C-01. Persistencia SQLite

**Ubicacion:** P390, seccion 4.1.

**Texto actual**

> La base de datos se diseño de forma relacional. En el MVP se implementa en SQLite para manejar grandes cantidades de datos historicos relacionados con jugadores juveniles.

**Texto propuesto**

> La base de datos se diseño de forma relacional mediante SQLAlchemy. En desarrollo local y pruebas, el MVP utiliza SQLite por su simplicidad y portabilidad; el despliegue publico en Render utiliza PostgreSQL administrado para conservar los datos fuera del sistema de archivos efimero del servicio. El MVP no implementa infraestructura Big Data ni demuestra escalabilidad para grandes volumenes.

## C-02. Migraciones

**Ubicacion:** P371 y Tabla 4-2, fila "Sistema de migraciones".

**Texto actual**

> Migraciones con Flask-Migrate: A menudo se usa junto con Flask-Migrate, que permite gestionar los cambios en el esquema de la base de datos (...) utilizando el sistema de migraciones de Alembic.

**Texto propuesto**

> Migraciones del MVP: SQLAlchemy puede integrarse con Flask-Migrate y Alembic, pero TPScouting no incorpora esas herramientas. La version auditada crea el esquema con `Base.metadata.create_all` y aplica ampliaciones compatibles mediante funciones de migracion manual en `db_utils.py`. La adopcion de migraciones versionadas con Alembic queda como mejora futura.

**Celda propuesta para Tabla 4-2**

> El MVP usa creacion de esquema y migraciones manuales; Flask-Migrate/Alembic constituye una alternativa futura para versionar cambios.

## C-03. Variables reales del modelo de datos

**Ubicacion:** Tabla 4-5.

| Dimension | Variables propuestas | Fuente real |
|---|---|---|
| Tecnica | `pace`, `shooting`, `passing`, `dribbling`, `technique`, `vision` | Player e historial de atributos |
| Fisica | `physical`, `estimated_speed`, `endurance`, `height_cm`, `weight_kg` | Player y PhysicalAssessment |
| Defensiva | `defending`, `tackling` (mostrado como Marcaje en la interfaz) | Player e historial de atributos |
| Mental/scout | `determination`, `decision_making`, `tactical_reading`, `mental_profile`, `adaptability` | Player y ScoutReport |
| Rendimiento | `minutes`, `goals`, `assists`, `pass_accuracy`, `shot_accuracy`, `duels_won_pct`, `final_score` | PlayerStat y participaciones |

Eliminar `stamina`, `strength`, `agility`, `marking`, `work_rate` y `composure`, salvo que se los identifique expresamente como conceptos no persistidos.

## C-04. Generacion de informes

**Ubicacion:** P192, objetivo/alcance.

**Texto actual**

> Visualizacion y comunicacion de datos: Mediante la presentacion grafica y la generacion de informes basados en los analisis realizados (...)

**Texto propuesto**

> Visualizacion y comunicacion de datos: Mediante paneles, graficos, comparadores, vistas imprimibles y el registro de reportes scout, el sistema facilita la interpretacion y comunicacion de la informacion entre entrenadores, directivos y otras partes interesadas. La exportacion formal de informes y PDF se mantiene como trabajo futuro.

## C-05. Pesos del score combinado

**Ubicacion:** P423.

**Texto actual**

> (...) con pesos 0,35, 0,35 y 0,30.

**Texto propuesto**

> (...) con pesos por defecto de 0,35 para el modelo, 0,35 para el promedio historico y 0,30 para el ajuste posicional. Estos pesos son configurables mediante variables de entorno; cuando falta un componente, los pesos disponibles se renormalizan.

## C-06. Figura 5-1, secuencia de prediccion

**Reemplazo conceptual del flujo**

1. Cargar jugador e historiales.
2. Construir y transformar las 68 features.
3. Ejecutar PlayerNet y aplicar sigmoid para obtener `base_prob` cruda.
4. Aplicar el calibrador isotónico solo como referencia secundaria, si esta disponible.
5. Calcular promedio historico y ajuste posicional.
6. Ejecutar `combine_probability(base_prob, stats_summary, fit_score)`.
7. Clasificar `combined_prob` con umbrales por defecto 0,60/0,80 y mostrarlo como resultado principal.
8. Si el modelo no esta disponible, responder HTTP 500; si faltan datos suficientes con el modelo cargado, renderizar la vista controlada sin proyeccion.

**Pie propuesto**

> Figura 5-1. Secuencia real de inferencia. La salida principal de la interfaz es el score combinado; la probabilidad calibrada se presenta como referencia secundaria y no sustituye al score operativo.

## C-07. Figura 5-3, despliegue

**Textos a reemplazar**

- Actual: `Deploy desde rama main`.
- Propuesto: `Deploy automatico desde la rama configurada en Render`.

- Actual: `Los locks/cache son in-memory`.
- Propuesto: `Cache y rate limit en memoria; lock del pipeline mediante thread y archivo atomico`.

**Nota propuesta**

> La evidencia historica del 20/05/2026 corresponde a la rama `render-free-deploy`. La auditoria local no pudo demostrar la rama conectada actualmente al servicio.

## C-08. Figura 6-6, captura de prediccion

Reemplazar la captura por una tomada con la version final y un jugador cuya edad se muestre correctamente. Cambiar el rotulo visible o el pie:

- Actual: `Ajuste historial`.
- Propuesto: `Ajuste combinado`.
- Aclaracion: `Diferencia respecto de PlayerNet crudo producida por el historial y el ajuste posicional.`

## C-09. Anexo B, comandos reproducibles

**Texto introductorio propuesto**

> Los comandos generales se ejecutan desde la raiz. Para el pipeline de datos y entrenamiento se cambia al directorio `scouting_app`, de modo que las rutas SQLite y los artefactos coincidan con el RUNBOOK. Los comandos siguientes se reproducen completos; no se ejecuto entrenamiento durante la auditoria final.

**Pipeline exacto propuesto**

```powershell
Set-Location .\scouting_app
..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///players_training.db --model-out model.pt --preprocessor-out preprocessor.joblib --calibrator-out probability_calibrator.joblib --metadata-out training_metadata.json --splits-out training_splits.json --epochs 45 --lr 5e-4 --patience 10
..\.venv\Scripts\python.exe evaluate_saved_model.py --db-url sqlite:///players_training.db --metadata-path training_metadata.json
..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18 --replace
Set-Location ..
```

## C-10. Indice, listas y captions

No reemplazar numeros a mano. Aplicar en este orden:

1. Intercambiar/renumerar Tabla 4-3 y Tabla 4-4 segun orden de aparicion.
2. Mover la actual Figura 10-9 despues de Figura 10-8 o renumerar todas las figuras del Anexo en orden.
3. Seleccionar todo el documento y actualizar campos.
4. Actualizar por separado indice general, lista de figuras y lista de tablas.
5. Exportar un PDF nuevo y comprobar que el capitulo 5, captions y paginas coincidan.

La diferencia entre paginas preliminares y `Introduccion = 1` no es un error: existe un reinicio de numeracion por secciones. El problema son los campos desactualizados dentro de cada regimen.

## C-11. Diagrama de clases

En Figura 4-3, marcar:

```text
current_age {derived}
category_year {derived}
```

Ambos valores se calculan desde `birth_date`; no son columnas persistidas.

## C-12. Smoke Render mas reciente

**Texto opcional para actualizar P514**

> Como evidencia operativa adicional, la Tabla 6-5 presenta el smoke HTTP ejecutado contra Render el 20/05/2026. Las rutas principales respondieron correctamente en esa verificacion historica. Controles posteriores del 10/07/2026 y 13/07/2026 finalizaron por timeout; por ello no se afirma disponibilidad continua ni estado vigente del servicio.

