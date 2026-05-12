# UX/UI y CRUD Polish - Proximo Paso 2026-05-01

Este documento deja preparado el siguiente bloque chico y verificable para continuar el pulido visual del MVP sin mezclarlo con la etapa ya cerrada.

## Estado Actual

- Rama activa: `ux-crud-polish`.
- Commit base original del contexto UX: `06fb998 docs: prepare next ux polish step`.
- Ultimo commit publicado antes de este bloque: `f97c684 ux: polish player comparators`.
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

Tercer bloque de UX/UI etapa 2 aplicado:

- `scouting_app/templates/compare.html` adopta cabecera visual, selector compacto, tarjetas 1v1, radar de atributos y tabla densa.
- `scouting_app/templates/compare_multi.html` adopta cabecera visual, ranking por puesto, selector compacto y mapa visual de talento.
- `scouting_app/routes/compare.py` agrega datos visuales ya existentes (`photo_url`, edad, club y fit score) para renderizar el mapa sin cambiar la logica de negocio.
- El mapa de talento usa hasta 40 jugadores: top 10 por cada familia de puesto.

Validacion del tercer bloque:

- `tests/test_pages.py`: `7 passed`.
- Suite completa: `52 passed`, cobertura total `77%`, con los `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN.
- Smoke especifico con test client: `GET /compare`, `GET /compare/multi`, `POST /compare` y `POST /compare/multi` respondieron `200`.
- Smoke HTTP local: `/health`, `/` y `/login` respondieron `200` en `http://127.0.0.1:5000/`.

Cuarto bloque de UX/UI etapa 2 aplicado y ajustado:

- `scouting_app/templates/player_stats.html` ahora separa la visualizacion del historial y la carga de nuevos registros: el formulario de rendimiento se abre en modal centrado.
- `scouting_app/templates/player_attributes.html` aplica el mismo criterio: grafico/historial quedan como foco de lectura y la carga de atributos se abre en modal centrado.
- Ambos formularios conservan los endpoints existentes (`POST` sobre la misma pantalla), los mismos nombres de campos y el mismo CSRF.
- `scouting_app/templates/manage_players.html` separa visualmente la carga individual de la carga masiva. La importacion masiva queda como bloque propio, con guia de formato, ejemplo y textarea mas legible.
- `scouting_app/static/styles.css` suma utilidades para modales de registros, estados vacios y carga masiva.

Validacion del cuarto bloque:

- `tests/test_pages.py`: `7 passed`.
- Suite completa: `52 passed`, cobertura total `77%`, con los `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN.
- Smoke especifico con test client sobre jugador real demo: `/players/manage`, `/player/30101/stats` y `/player/30101/attributes` respondieron `200` y renderizaron los nuevos elementos.
- Smoke HTTP local: `/health`, `/` y `/login` respondieron `200` en `http://127.0.0.1:5000/`.

Quinto bloque de UX/UI etapa 2 aplicado:

- Se completo la segunda etapa CRUD de historiales de jugadores.
- `scouting_app/routes/players.py` agrega endpoints POST especificos para editar/eliminar registros de rendimiento y editar/eliminar registros de atributos.
- `scouting_app/app.py` conserva aliases historicos/planos para los nuevos endpoints.
- `scouting_app/templates/player_stats.html` muestra acciones por fila y abre editar/eliminar en modales centrados.
- `scouting_app/templates/player_attributes.html` muestra acciones por fila y abre editar/eliminar en modales centrados.
- Al editar/eliminar atributos, la ficha tecnica del jugador se resincroniza con el historial restante.
- Se agregaron tests para aliases, CSRF, permisos de director, edicion/eliminacion de rendimiento y edicion/eliminacion de atributos.

Validacion del quinto bloque:

- Pruebas focales: `9 passed`.
- `tests/test_pages.py`: `7 passed`.
- Suite completa: `57 passed`, cobertura total `77%`, con los `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN.
- Smoke especifico con test client sobre jugador real demo: `/player/30101/stats` y `/player/30101/attributes` respondieron `200` y renderizaron modales de nuevo/editar/eliminar.
- Smoke HTTP local: `/health`, `/` y `/login` respondieron `200` en `http://127.0.0.1:5000/`.

Sexto bloque de UX/UI etapa 2 aplicado:

- Se llevo el mismo patron de CRUD modal a los historiales complementarios visibles desde la ficha del jugador:
  - partidos/participaciones (`Match` + `PlayerMatchParticipation`)
  - evaluaciones fisicas (`PhysicalAssessment`)
  - disponibilidad (`PlayerAvailability`)
  - reportes scout (`ScoutReport`)
- `scouting_app/routes/players.py` agrega endpoints POST de alta, edicion y eliminacion para esos cuatro historiales, con CSRF, permisos de administrador/scout y refresco de proyeccion/cache.
- `scouting_app/app.py` conserva aliases historicos/planos para los endpoints nuevos.
- `scouting_app/templates/player_detail.html` muestra una seccion de historiales complementarios con pestanas y acciones por fila; las altas, ediciones y eliminaciones se abren en modales centrados.
- `scouting_app/templates/settings.html` adopta la misma linea visual: cabecera, metricas de calidad de base, acciones operativas y logs.
- `scouting_app/templates/register.html` adopta la misma linea visual: cabecera, formulario por secciones y resumen de roles.
- Se agregaron tests para aliases, CSRF, permisos, CRUD de los cuatro historiales y renderizado de los modales en ficha.

Validacion del sexto bloque:

- Pruebas focales de modales complementarios: `5 passed`.
- `tests/test_pages.py`: `7 passed`.
- Suite completa: `62 passed`, cobertura total `77%`, con los `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN.
- Smoke HTTP local con login admin: `/health`, `/`, `/players`, `/dashboard`, `/compare`, `/compare/multi`, `/settings`, `/register`, `/player/30101`, `/player/30101/stats`, `/player/30101/attributes` y `/player/30101/predict` respondieron `200` en `http://127.0.0.1:5000/`.

Septimo bloque de UX/UI etapa 2 aplicado:

- `scouting_app/templates/login.html` fue alineado con la linea visual aprobada: cabecera verde/azul oscuro, icono de acceso, formulario en tarjeta, roles de acceso y acciones compactas.
- `scouting_app/static/styles.css` suma ajustes responsive para celular/tablet: shell mas compacto, cabecera apilada, chips en columna, botones a ancho completo, formularios mas comodos y modales con margen reducido.
- Se agrego una prueba de render de `/login` para verificar que el layout nuevo conserve formulario y CSRF.
- Ajuste posterior: el hero de login se hizo mas sobrio y compacto; el titulo principal paso a `TPScouting` y se removio el impacto visual excesivo de `Iniciar sesion`.
- Corroboracion local con `admin/admin`: `POST /login` respondio `302` a `/players` en aproximadamente `136 ms`, y la carga posterior de `/players` respondio `200` en aproximadamente `352 ms`.

Validacion del septimo bloque:

- `tests/test_pages.py`: `8 passed`.
- Suite completa: `63 passed`, cobertura total `77%`, con los `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN.
- Smoke HTTP local: `/login` respondio `200` y renderizo `auth-hero`; con login admin, `/health`, `/`, `/login`, `/players`, `/dashboard`, `/settings`, `/register` y `/player/30101` respondieron `200`.
- Nota de verificacion: Playwright no esta instalado en la `.venv`, por lo que no se generaron capturas automatizadas responsive.

Octavo bloque de UX/UI etapa 2 aplicado:

- `scouting_app/templates/dashboard.html` se redisenio como panel de control global y no como pagina de graficos grandes.
- `scouting_app/routes/dashboard.py` ahora calcula metricas accionables con datos reales existentes:
  - jugadores evaluados
  - alto potencial
  - actividad/seguimiento reciente
  - jugadores a revisar por falta de actividad
  - reportes scout recientes
  - alertas fisicas por lesion, disponibilidad baja o fatiga alta
  - mejor forma reciente y evolucion en periodo
  - oportunidades por puesto
- El panel es dinamico por rol:
  - `administrador` y `scout` ven una `Mesa de scouting`.
  - `director` ve `Estado del plantel`.
- La cache HTML del dashboard ahora incluye el modo de rol en la clave para no mezclar vistas entre scout/admin y director.
- Los graficos se mantienen, pero pasan a tarjetas compactas al final del panel.
- Se agrego test para verificar que el dashboard cambia su texto por rol y que admin no recibe la vista de director.

Validacion del octavo bloque:

- `tests/test_pages.py`: `9 passed`.
- Suite completa: `64 passed`, cobertura total `77%`, con los `4 warnings` conocidos de scikit-learn por fixtures con columnas all-NaN.
- Smoke HTTP local con login admin: `/health`, `/dashboard`, `/players` y `/player/30101` respondieron `200`; `/dashboard` renderizo `Mesa de scouting`.

## Punto De Retome 2026-05-06

- Rama: `ux-crud-polish`.
- Ultimo commit funcional publicado: `a8ee6ea ux: move player history forms to offcanvas`.
- Estado al cerrar: rama limpia y sincronizada con `origin/ux-crud-polish`.
- Ese punto de retome fue completado el 2026-05-07.
- Lo que ya esta hecho: carga, edicion y eliminacion de rendimiento y atributos en modales centrados; carga masiva de jugadores separada visualmente.
- Tambien quedo hecho el CRUD modal de partidos/participaciones, fisico, disponibilidad y reportes scout, mas el pulido visual de `settings.html` y `register.html`.

Arranque recomendado:

1. Hacer una pasada visual manual en navegador sobre ficha de jugador, settings y registro.
2. Si el usuario aprueba, cerrar este bloque con commit/push.
3. Despues elegir entre revisar pantallas puntuales en navegador, ajustar detalles del dashboard o pasar al documento Word.

Orden restante recomendado, de menor a mayor riesgo:

1. Pasada visual en navegador y ajuste responsive si aparece algo concreto.
2. Documento Word de tesis: alinear narrativa, capturas y afirmaciones con el MVP real ya corregido.
3. Rendimiento dashboard a escala, si se quiere seguir por codigo.

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

Luego analizar el flujo CRUD de jugadores antes de tocar codigo. La idea a evaluar es separar visualizacion de carga/edicion/eliminacion, posiblemente con modales o pantallas dedicadas, sin ocultar los datos principales.
