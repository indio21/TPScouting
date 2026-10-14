# Cierre del bloque 5: pruebas por dominio

Fecha: 2026-10-14.

## Objetivo

Dividir `tests/test_mvp_regressions.py`, de 3328 líneas, sin modificar los casos,
fixtures, comportamiento ni cobertura.

## Cambios

Las 76 regresiones del archivo original se distribuyeron en:

- `test_mvp_runtime.py`: 18 pruebas de demo, configuración, runtime y settings;
- `test_mvp_auth_security.py`: 13 pruebas de autenticación, permisos, CSRF,
  rate limiting, health y datos operativos;
- `test_mvp_players.py`: 26 pruebas de jugadores, importación, historiales y staff;
- `test_mvp_ml_data.py`: 19 pruebas de inferencia, entrenamiento, migraciones,
  sincronización y datos sintéticos;
- `mvp_regression_helpers.py`: 10 constructores y acciones comunes usados por
  esos módulos.

El archivo monolítico fue retirado. Ruff eliminó solamente imports que quedaron
sin uso después de la separación.

## Evidencia

Línea base anterior:

- 76 pruebas recolectadas;
- 76 aprobadas;
- 4 advertencias conocidas de scikit-learn;
- 37,29 segundos para el archivo monolítico.

Después de la división:

- la colección conserva los mismos 76 nombres, sin duplicados y en el mismo orden;
- la representación AST de los 76 cuerpos de prueba coincide exactamente con la
  versión anterior;
- runtime: 18 aprobadas;
- autenticación/seguridad: 13 aprobadas;
- jugadores: 26 aprobadas;
- ML/datos: 19 aprobadas, con las mismas 4 advertencias conocidas;
- suite integral: 122 aprobadas, 1 omitida, 4 advertencias y 83,23 % de cobertura;
- `compileall` y Ruff crítico: aprobados.

## Límites

No se cambió código de aplicación, no se agregaron pruebas nuevas y no se alteró
el repositorio de entrega. La ampliación gradual de Ruff y logging corresponde al
bloque 6.
