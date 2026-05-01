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

Orden recomendado, de menor a mayor riesgo:

1. Detalle y proyeccion de jugador:
   - `scouting_app/templates/player_detail.html`
   - `scouting_app/templates/prediction.html`
2. Historial y atributos:
   - `scouting_app/templates/player_stats.html`
   - `scouting_app/templates/player_attributes.html`
3. Comparadores:
   - `scouting_app/templates/compare.html`
   - `scouting_app/templates/compare_multi.html`
4. Administracion/configuracion:
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

Luego revisar visualmente primero las plantillas de detalle/proyeccion y aplicar un cambio acotado. Si ese bloque pasa tests y smoke, seguir con historial/atributos.
