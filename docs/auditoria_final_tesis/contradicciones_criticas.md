# Contradicciones que deben corregirse antes del envio

Fecha: 2026-07-13

Este archivo contiene solo hallazgos que pueden inducir a una lectura tecnica incorrecta o deteriorar la presentacion formal. Los detalles completos estan en `matriz_trazabilidad_final.md`.

## 1. Indice, listas y captions desactualizados

**Estado:** AFIRMACIÓN DESACTUALIZADA  
**Ubicacion:** indice general, lista de figuras, lista de tablas y captions de capitulos 4 y 10.

- El indice muestra `5. DESARROLLO 24`, pero el capitulo comienza en pagina logica 43.
- Tabla 4-4 aparece antes que Tabla 4-3.
- Figura 10-9 aparece antes que Figuras 10-1 a 10-8.
- Varias paginas de la lista de figuras estan desplazadas; Tabla 6-5 tambien tiene pagina incorrecta.

**Riesgo:** alto editorial. Un tribunal puede interpretar que el documento no fue actualizado despues de las correcciones.

**Accion obligatoria:** ordenar captions, actualizar todos los campos de Word, regenerar listas y verificar el PDF pagina por pagina.

## 2. Migraciones atribuidas a Flask-Migrate/Alembic

**Estado:** NO COINCIDE  
**Ubicacion:** Tabla 4-2, fila "Sistema de migraciones", y parrafo P371.

El repositorio no usa Flask-Migrate ni Alembic. El esquema se crea con `Base.metadata.create_all` y las ampliaciones se resuelven con migraciones manuales en `scouting_app/db_utils.py:76-138`.

**Riesgo:** alto tecnico. Declara una tecnologia que no forma parte de la aplicacion.

**Accion obligatoria:** presentar Flask-Migrate/Alembic solo como alternativa general y describir el mecanismo manual real del MVP.

## 3. Campos inexistentes en la Tabla 4-5

**Estado:** NO COINCIDE  
**Ubicacion:** Tabla 4-5, dimensiones fisica, defensiva y mental.

No existen `stamina`, `strength`, `agility`, `marking`, `work_rate` ni `composure` como campos persistidos. El codigo usa:

- fisico: `physical`, `estimated_speed`, `endurance`, `height_cm`, `weight_kg`;
- defensivo: `defending`, `tackling`;
- mental/scout: `determination`, `decision_making`, `tactical_reading`, `mental_profile`, `adaptability`.

Evidencia: `scouting_app/player_logic.py:13-24`; `scouting_app/models.py:40-59,371-429`.

**Riesgo:** alto tecnico. El modelo de datos documentado no coincide con SQLAlchemy.

**Accion obligatoria:** reemplazar las variables por nombres reales o rotularlas expresamente como dimensiones conceptuales no persistidas.

## 4. Diagrama de inferencia incompleto e incorrecto

**Estado:** NO COINCIDE  
**Ubicacion:** Figura 5-1, diagrama de secuencia de prediccion.

El diagrama muestra la calibracion como origen del resultado final y omite `combine_probability`. En la aplicacion, el score principal proviene del sigmoid crudo de PlayerNet combinado con promedio historico y ajuste posicional; la calibrada es una referencia secundaria. Evidencia: `scouting_app/app.py:1050-1120,1225-1257`; `scouting_app/routes/players.py:1475-1501`.

Ademas, el diagrama agrupa "modelo no disponible" y "datos insuficientes" bajo HTTP 200. El primer caso devuelve HTTP 500 (`routes/players.py:1442-1443`); el segundo puede renderizar la vista 200 (`1469-1501`).

**Riesgo:** alto metodologico. Confunde la salida evaluada con la salida visible para el usuario.

**Accion obligatoria:** redibujar la secuencia con salida cruda, calibracion secundaria, `combine_probability` y ramas de error separadas.

## 5. Comandos de entrenamiento no reproducibles

**Estado:** NO COINCIDE  
**Ubicacion:** Anexo B, P616-P620.

`generate_data.py` y `evaluate_saved_model.py` no incluyen interprete/ruta para ejecutarse desde la raiz. `train_model.py ...` contiene una elipsis y omite argumentos obligatorios para reproducir la corrida documentada.

**Riesgo:** alto academico. El anexo promete comandos reales, pero tres no son ejecutables literalmente.

**Accion obligatoria:** indicar `Set-Location .\scouting_app` y copiar sin abreviar los comandos exactos de `RUNBOOK.md:109-117`.

## 6. Diagrama de despliegue mezcla hechos no demostrados

**Estado:** NO COINCIDE / NO VERIFICABLE  
**Ubicacion:** Figura 5-3.

- La evidencia historica registra deploy desde `render-free-deploy`, no desde `main`; no se pudo probar una reconfiguracion posterior (`docs/cierre_pre_entrega_word_render_2026-05-18.md:133-147`).
- "Los locks/cache son in-memory" es inexacto: el cache y rate limit son de memoria, pero el lock de pipeline usa thread y archivo atomico (`scouting_app/services/locks.py:12-58`).

**Riesgo:** medio tecnico.

**Accion obligatoria:** eliminar la rama o fecharla como historica, y corregir la nota sobre locks.

## 7. SQLite presentado como solucion para grandes volumenes

**Estado:** COINCIDE PARCIALMENTE  
**Ubicacion:** P390.

SQLite es la base local del MVP, pero el despliegue usa PostgreSQL y el codigo bloquea SQLite en produccion salvo excepcion (`app.py:281-297`; `render.yaml:20-27`).

**Riesgo:** medio conceptual, especialmente por el marco de Big Data.

**Accion obligatoria:** describir SQLite como opcion liviana de desarrollo/pruebas, no como infraestructura para grandes cantidades de datos.

## 8. "Generacion de informes" excede lo implementado

**Estado:** COINCIDE PARCIALMENTE  
**Ubicacion:** objetivo especifico, P192.

Existen reportes scout persistidos y vistas imprimibles, pero no un generador formal de informes ni exportacion PDF. El propio P574 ubica esa capacidad en trabajo futuro.

**Riesgo:** medio de alcance.

**Accion obligatoria:** hablar de visualizacion, comunicacion y registro de reportes scout; reservar informes/exportaciones formales para trabajo futuro.

## 9. Captura de prediccion con dato incompleto

**Estado:** COINCIDE PARCIALMENTE  
**Ubicacion:** Figura 6-6.

El chip de edad muestra solo "anos" sin numero, aunque otra captura del mismo jugador indica 17. El template actual imprime `player.current_age` (`templates/prediction.html:17`). Ademas, "Ajuste historial" incluye tambien ajuste posicional.

**Riesgo:** medio de calidad de evidencia.

**Accion obligatoria:** reemplazar la captura por una actual y rotular el delta como ajuste combinado de historial y posicion.

