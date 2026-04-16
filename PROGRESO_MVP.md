# Progreso Del MVP

Fecha de actualizacion: 2026-04-16

Este archivo resume, sin inventar nada, las etapas ya trabajadas sobre el MVP real del proyecto `TPScouting`.

## Estado Git

- Rama local actual: `main`
- Remoto configurado: `origin -> https://github.com/indio21/TPScouting.git`
- Estado: cambios locales pendientes de commit y push
- Importante: hasta esta fecha los cambios no fueron subidos a GitHub

## Etapas Ya Trabajadas

### 1. Preparacion para PostgreSQL

- Se agrego una capa comun de conexion y normalizacion de URLs en `scouting_app/db_utils.py`.
- La app y scripts principales quedaron preparados para usar SQLite o PostgreSQL.
- `render.yaml` quedo orientado a despliegue con PostgreSQL administrado.

### 2. Instalacion de entorno y validacion funcional PostgreSQL

- Se creo `.venv` en la raiz del repo.
- Se instalaron dependencias del proyecto y de testing.
- Se valido el soporte PostgreSQL con pruebas reales de:
- `generate_data.py`
- `train_model.py`
- `sync_shortlist.py`
- app Flask usando `APP_DB_URL` y `TRAINING_DB_URL`

### 3. Correcciones criticas del MVP

- Se unifico la logica de `potential_label`, categoria textual y filtro de alto potencial.
- El dashboard paso a clasificar potencial con la misma logica base de proyeccion.
- Se corrigio la recarga del modelo en memoria despues del pipeline.
- Se agrego invalidacion de cache del dashboard tras cambios de datos.

### 4. Seguridad minima y reproducibilidad

- `APP_SECRET_KEY` insegura por defecto fue reemplazada por clave efimera en desarrollo y obligatoria en produccion.
- `create_admin.py` y registro de usuarios ya no aceptan contraseñas debiles.
- `generate_data.py` ahora acepta semilla reproducible.
- `train_model.py` usa `SEED` tambien en el split y deja de fallar en silencio al calcular metricas.

### 5. Multirol real del MVP

- Se normalizaron roles para:
- `administrador`
- `scout` / `ojeador`
- `director`
- Se implementaron permisos reales en backend.
- Se ocultaron acciones no permitidas en la interfaz.
- `director` quedo en modo lectura para jugadores, historial, atributos y comparadores.

### 6. Seguridad operativa y auditoria

- Se agrego rate limiting basico al login.
- Se agregaron timestamps `created_at` y `updated_at` en entidades clave.
- `db_utils.py` asegura esas columnas tambien en bases existentes.

### 7. Rendimiento inicial

- Se creo `batch_project_players(...)` para calcular proyecciones por lote.
- El listado de jugadores dejo de recalcular proyeccion jugador por jugador con consultas repetidas al historial.
- El dashboard reutiliza logica batch para potencial.
- Los comparadores quedaron mas livianos en calculo de proyeccion.
- Se documento el comportamiento y limite del cache in-memory del dashboard en `RUNBOOK.md`.

## Tests Ejecutados

- La suite automatizada actual termina pasando en esta maquina.
- Ultimo estado validado: `16 passed`

## Puntos Mejorados De Forma Clara

- Coherencia del potencial entre modelo, categoria, filtro y dashboard
- Soporte real para PostgreSQL
- Seguridad minima de secretos y credenciales
- Multirol y control de acceso
- Trazabilidad temporal basica
- Reproducibilidad del pipeline
- Mejora inicial de rendimiento en listado, dashboard y comparadores

## Puntos Que Siguen Parciales O Pendientes

- Persistencia/despliegue final en Render
- Revisión completa de CSRF y endurecimiento adicional
- Limpieza de datos demo para defensa
- Optimizaciones adicionales de rendimiento
- Tests CRUD mas amplios, invalidos y cobertura mas fuerte
- Revision final integral del MVP
- Correccion del documento Word, que todavia no se empezo en esta fase

## Bloques Restantes

Tomando como base la checklist operativa del MVP:

- Quedan 3 bloques principales del MVP real por cerrar o fortalecer:
- consistencia final y limpieza de datos/demo
- fortalecimiento de tests y cobertura funcional
- revision final integral del MVP con evidencia real

- Luego queda 1 bloque grande aparte:
- correccion del documento Word para alinearlo con el MVP ya corregido

## Regla De Trabajo

- No inventar nada
- No vender humo
- Si algo no se puede verificar, dejarlo explicitado
- No avanzar al documento hasta terminar antes el MVP real
