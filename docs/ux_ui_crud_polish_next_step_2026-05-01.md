# UX/UI y CRUD Polish - Proximo Paso 2026-05-01

Este documento deja preparado el siguiente bloque chico y verificable para continuar el pulido visual del MVP sin mezclarlo con la etapa ya cerrada.

## Estado Actual

- Rama activa: `ux-crud-polish`.
- Commit base del contexto para retomar: `06fb998 docs: prepare next ux polish step`.
- Commit UX principal ya publicado: `8414d06 ux: polish navigation and crud screens`.
- La rama esta sincronizada con `origin/ux-crud-polish`.
- El servidor local quedo probado en `http://127.0.0.1:5000/`.

## Ya Cerrado En UX/UI Etapa 1

- Menu agrupado con links activos.
- `Dashboard` renombrado visualmente a `Panel general`.
- `Jugadores` separado en `Listado` y `Gestion`.
- `Comparadores` agrupado en un unico desplegable.
- `Staff` agrupa cuerpo tecnico y dirigentes.
- `Administracion` agrupa configuracion y usuarios/permisos.
- Mensajes flash con estilos claros, icono y cierre.
- Acciones CRUD compactas y consistentes en listados principales.
- Formularios principales ordenados por secciones.
- Correccion de arranque local con `ADMIN_PASSWORD` configurado.

## Proximo Bloque Recomendado

UX/UI etapa 2: llevar la misma consistencia visual a pantallas internas que todavia quedaron mas heterogeneas.

## Regla Visual A Mantener

- Mantener la misma linea visual ya aprobada: cabecera tipo ficha profesional, fondo verde/azul oscuro, metricas clave en tarjetas blancas, acciones compactas con iconos y formularios por secciones.
- Usar la pantalla de ficha/proyeccion como referencia para nuevas secciones: estructura densa, sobria y de scouting, sin estilo de landing ni tarjetas decorativas innecesarias.
- Priorizar consistencia entre pantallas antes que sumar efectos visuales nuevos.

## Avance 2026-05-04

Primer bloque de UX/UI etapa 2 aplicado:

- `scouting_app/templates/player_detail.html` paso a una ficha mas visual, con cabecera de jugador, metricas clave, acciones compactas, ranking de puestos y atributos en formato denso.
- `scouting_app/templates/prediction.html` paso a una pantalla de decision con resultado principal, desglose de ficha/historial, puesto recomendado, sugerencias y graficos existentes mejor integrados.
- `scouting_app/static/styles.css` incorpora clases reutilizables para cabecera de perfil, metricas, atributos, sugerencias y paneles de proyeccion.
- La referencia visual de ficha de jugador se uso solo como inspiracion de estructura. No se copiaron logos ni datos externos.
- El grafico tipo ejes/scatter con imagenes de jugadores queda reservado para el bloque de comparadores, porque requiere diseno y validacion especificos de esa pantalla.

Validacion del bloque:

- `tests/test_pages.py`: `7 passed`.
- Suite completa: `52 passed`, cobertura total `77%`, con los `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN.
- Smoke especifico con test client sobre jugador real demo: `/player/30101` y `/player/30101/predict` respondieron `200`.
- Smoke HTTP local: `/health`, `/` y `/login` respondieron `200` en `http://127.0.0.1:5000/`.

Segundo bloque de UX/UI etapa 2 aplicado:

- `scouting_app/templates/player_stats.html` adopta la misma cabecera visual, metricas clave, formulario por secciones y tabla compacta de historial.
- `scouting_app/templates/player_attributes.html` adopta cabecera visual, metricas clave, formulario de atributos por secciones, grafico integrado y tabla compacta.
- `scouting_app/static/styles.css` suma utilidades reutilizables para icono de perfil, layout de carga/historial y tablas densas.

Validacion del segundo bloque:

- `tests/test_pages.py`: `7 passed`.
- Suite completa: `52 passed`, cobertura total `77%`, con los `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN.
- Smoke especifico con test client sobre jugador real demo: `/player/30101/stats` y `/player/30101/attributes` respondieron `200`.
- Smoke HTTP local: `/health`, `/` y `/login` respondieron `200` en `http://127.0.0.1:5000/`.

Orden restante recomendado, de menor a mayor riesgo:

1. Comparadores:
   - `scouting_app/templates/compare.html`
   - `scouting_app/templates/compare_multi.html`
2. Administracion/configuracion:
   - `scouting_app/templates/settings.html`
   - `scouting_app/templates/register.html`

## Criterio De Cambio

- Priorizar templates y CSS.
- Evitar cambios de logica backend salvo que aparezca un bug verificable.
- Mantener endpoints y nombres historicos.
- Reusar clases ya creadas en `styles.css`: `page-heading`, `page-actions`, `crud-actions`, `form-section`, `table-action-cell`.
- Mantener botones compactos y consistentes: `Nuevo`, `Editar`, `Eliminar`, `Ver detalle`, `Guardar`, `Cancelar`.
- No tocar modelo, datos ni entrenamiento en esta etapa.

## Validacion Al Cerrar El Bloque

Ejecutar:

```powershell
cd C:\Tesis\TPScouting
.\.venv\Scripts\python.exe -m pytest tests\test_pages.py -q
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Smoke local sugerido:

- `GET /health`
- `GET /`
- `GET /players`
- `GET /dashboard`
- `GET /compare`
- `GET /compare/multi`
- `GET /settings`
- detalle de al menos un jugador real
- proyeccion de al menos un jugador real
- historial y atributos de al menos un jugador real

## Arranque Sugerido Para El Proximo Chat

Antes de editar:

```powershell
cd C:\Tesis\TPScouting
git status -sb
git log --oneline --decorate -5
```

Luego revisar visualmente `compare.html` y `compare_multi.html`. En ese bloque se puede evaluar el grafico tipo ejes/scatter con imagenes de jugadores, manteniendo la misma linea visual.
