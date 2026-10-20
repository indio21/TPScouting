# Cierre de los ajustes posteriores — 20/10/2026

## Correcciones y evidencia

| ID | Estado | Cambio y comprobación |
|---|---|---|
| SEG-09 | RESUELTO | Login rechaza usuario/contraseña ausentes o vacíos con HTTP 400. Cuatro casos parametrizados y control de sesión sin autenticar. Cierre de base en `finally`, probado también ante fallo del hash. |
| SEG-10 | RESUELTO | CSRF no ASCII se rechaza con HTTP 400 antes de `compare_digest`. Tres tokens de regresión y reproducción equivalente a la que antes provocaba HTTP 500. |
| DOC-TRAZA-01 | RESUELTO | Word cita el commit funcional `2a27d53ae4f8a4849bee7d1bea362fd9146886b5`, CI `39509728373` y Container `39509728476`, todos comprobados. Publicación documental `fdf2805975efbf1de78dfe7d350f0ebb30f1c1bd` registrada externamente. |
| DOC-TRAZA-02 | RESUELTO | Fecha local 20/10/2026, 131 passed, 1 skipped, 4 warnings, 83,31 %; valores leídos de pytest.txt y coverage.json. |
| DOC-SEG-01 | RESUELTO | Descripción HSTS alineada con la condición real de producción. No se cambió el código para justificar una afirmación documental. |
| FORMA-01/02 | RESUELTO | Títulos de todos los niveles a 14 pt, preservando numeración y jerarquía de estilos. |
| FORMA-03 | RESUELTO | Caption de Tabla 6-4 a 14 pt, consistente con los demás. |
| FORMA-04 | RESUELTO | Por instrucción del usuario, comandos Arial 10 pt; todo el texto editable en Arial. El PDF solo utiliza ArialMT, Arial-BoldMT y Arial-ItalicMT. |

## Validación

- 37 pruebas del subconjunto auth/seguridad aprobadas.
- Suite completa: 131 passed, 1 skipped, 4 warnings; cobertura 83,31 %.
- Ruff crítico/gradual y pip check aprobados.
- Demo aislada: 60 jugadores sintéticos, semilla 42, todas las rutas comprobadas
  devuelven 200; login, edad/categoría, historiales, silueta y predicción correctos.
- CI, Container y macOS del commit funcional aprobados.
- Word/PDF: 97 páginas; 28 imágenes, 19 tablas y 9 secciones.
- 88 entradas del índice y 45 de figuras/tablas coinciden con las páginas reales.
- Sin errores de campo, numeración, texto fuera de página ni tamaño Word
  explícito menor de 10 pt. Portada, índice, resumen, abstract y bibliografía
  comienzan de forma independiente.
- Imágenes y tablas de configuración/métricas ML conservadas exactamente.
- Control visual dirigido de portada, títulos, tabla de evidencia y anexo de
  comandos. No se afirma una inspección exhaustiva de píxeles o una validación
  deportiva externa.

## Publicación y límites

Entrega publicada en `fdf2805975efbf1de78dfe7d350f0ebb30f1c1bd`.
La CI de esa publicación documental se registra en `publication_evidence.json`.
Los candidatos, reportes internos y scripts de edición documental no se copiaron
al repositorio del profesor.

Se mantienen las limitaciones ya declaradas del MVP y las cuatro advertencias
conocidas. Aprobar las pruebas cubiertas no garantiza ausencia absoluta de errores.

**LISTO PARA REVISIÓN FINAL HUMANA.**
