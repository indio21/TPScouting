# Cierre del bloque 8: datos reales de menores

Fecha: 2026-10-14.

## Veredicto

**DISEÑO COMPLETADO; USO REAL NO HABILITADO.** El MVP continúa limitado a datos
sintéticos. No se incorporaron pantallas ni tablas que aparenten resolver
consentimiento, derechos, retención o auditoría sin una institución responsable y
decisiones jurídicas y operativas concretas.

## Estado verificado de la aplicación

Controles existentes relevantes:

- autenticación y revalidación de usuario/rol;
- roles `administrador`, `scout` y `director`;
- CSRF en operaciones mutantes;
- validaciones de entradas y fotografías;
- eliminación en cascada de historiales asociados a un jugador;
- datos sintéticos declarados en demo y documentación.

Brechas que impiden un uso real:

- no hay responsable institucional, finalidad o base habilitante configurados;
- no hay registro versionado de información, consentimiento, representación o
  participación del menor;
- no hay aislamiento multiinstitución ni permisos por finalidad/tipo de dato;
- no hay auditoría protegida de lectura, exportación, cambio o predicción;
- no hay flujo de acceso, rectificación, bloqueo, exportación y supresión;
- no hay política ni automatización de retención para base, backups, exportaciones
  y artefactos ML;
- no hay evidencia de cifrado en reposo, restauración o respuesta a incidentes;
- el modelo solo fue validado con datos sintéticos y no puede fundamentar
  decisiones reales sobre jugadores.

## Contraste normativo oficial

Se verificaron el 2026-10-14:

1. [Ley 25.326](https://www.argentina.gob.ar/normativa/nacional/ley-25326-2000-64790/texto):
   artículos 4 a 7, 9 a 11 y 14 a 16 sobre calidad/finalidad, información,
   consentimiento, datos sensibles, seguridad, confidencialidad, cesión y
   derechos.
2. [Derechos sobre datos personales, AAIP](https://www.argentina.gob.ar/aaip/datospersonales/derechos):
   acceso, rectificación, actualización o supresión y acreditación de
   representación para personas menores.
3. [Ley 26.061](https://www.argentina.gob.ar/normativa/nacional/ley-26061-110778/texto):
   artículos 3, 10 y 24 sobre interés superior, privacidad y derecho a ser oído.
4. [Resolución AAIP 47/2018](https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-47-2018-312662/texto):
   medidas recomendadas de seguridad para datos informatizados.

La fuente normativa no decide por sí sola cómo debe operar un club, academia o
institución concreta. Esa definición requiere responsable identificado y revisión
competente. El plan no se presenta como certificación de cumplimiento.

## Resultado

Se creó `docs/REAL_DATA_READINESS.md` con:

- prohibición actual de cargar datos reales;
- diez puertas de habilitación y evidencia de cierre;
- diez familias de cambios técnicos pendientes;
- pruebas mínimas antes de un piloto;
- responsables institucionales requeridos.

El archivo puede incluirse en la entrega porque describe una limitación material
del producto y evita que el evaluador interprete la demo sintética como sistema
habilitado para operar con menores reales.

## Decisiones que siguen pendientes

- institución responsable y jurisdicción del eventual piloto;
- finalidad exacta y campos estrictamente necesarios;
- base jurídica y mecanismo de representación/participación;
- plazos de retención y excepciones;
- alojamiento, proveedores y transferencias;
- responsables de derechos, seguridad e incidentes;
- criterio institucional sobre fotografías, DNI, salud y scores de potencial.

No corresponde implementar estas decisiones por inferencia.

## Verificación de cierre

- Enlaces a las cuatro fuentes oficiales: accesibles con estado HTTP 200.
- Enlaces locales y secciones obligatorias del documento público: aprobados.
- Ruff crítico y gradual: aprobados.
- `compileall`: aprobado.
- Suite integral: 123 aprobadas, 1 omitida y 4 advertencias conocidas.
- Cobertura: 83,29 %, por encima del umbral de 80 %.

Las pruebas verifican que la documentación no introdujo una regresión. No
constituyen una prueba de cumplimiento para datos reales, porque esos controles
todavía no están implementados.
