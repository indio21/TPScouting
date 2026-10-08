# Método de auditoría inversa

Este procedimiento reconstruye el tipo de revisión que produjo
`correccion-octubre.md`. No afirma conocer las herramientas exactas de su autor;
reproduce sus resultados observables mediante controles equivalentes.

## 1. Identidad y cadena de custodia

1. Localizar el documento rector y cualquier PDF exportado.
2. Registrar ruta, nombre, tamaño, fecha, hash SHA-256 y páginas reales.
3. Registrar repositorio, remoto, rama, HEAD, estado y diferencias locales.
4. Identificar el commit que representa la entrega. No confundir CI del repositorio
   de desarrollo con CI de la entrega.
5. Inventariar correcciones anteriores, artefactos ML, datasets, metadata y splits.
6. Declarar qué material falta o no pudo abrirse.

## 2. Extracción del documento

Extraer texto, tablas, títulos, captions, hipervínculos, campos e índices desde el
DOCX cuando exista. Usar el PDF para páginas, cortes y apariencia final. Buscar:

- objetivos, pregunta de investigación y conclusiones;
- arquitectura, stack, entidades, roles y funcionalidades declaradas;
- origen de datos, etiqueta, features, split, seed y métricas;
- comandos, variables, repositorio, commit, CI y despliegue;
- citas, bibliografía, metodología, ética y limitaciones;
- cifras repetidas que puedan contradecirse.

La página del PDF y la sección académica son ubicaciones distintas; registrar
ambas cuando sea útil. No asumir que el índice refleja la paginación final.

## 3. Inventario del código

Revisar primero archivos de entrada y configuración: README, RUNBOOK, manifests,
workflow CI, despliegue, aplicación, modelos, rutas, servicios, preprocessing,
entrenamiento, evaluación, scripts de demo y tests.

Construir un mapa verificable:

`afirmación del documento → archivo/símbolo → prueba o ejecución → estado`

Buscar referencias con `rg`; usar las líneas solo como ayuda y volver a calcularlas
en cada revisión. Contar archivos o parámetros mediante código, no manualmente.

## 4. Reproducción funcional

Antes de ejecutar, registrar Python, sistema operativo y entorno. Ejecutar solo
comandos compatibles con el proyecto actual:

- instalación o resolución de dependencias sin alterar el entorno, cuando sea
  posible;
- suite completa y cobertura;
- lint configurado y compilación;
- auditoría de dependencias;
- demo en base temporal vacía;
- smoke de rutas, autenticación, permisos y predicción;
- CI del commit entregado, si ya está publicado.

Anotar tests aprobados, omitidos, warnings, cobertura total y por módulos relevantes.
No extrapolar Windows a Linux/macOS; usar CI u otra ejecución real para confirmarlo.

## 5. Seguridad

Revisar con inspección y pruebas de regresión:

- redirecciones posteriores al login y variantes codificadas;
- rate limiting y confianza en headers de proxy;
- revalidación de usuario, rol y disponibilidad;
- limpieza/rotación de sesión, cookies y secreto de aplicación;
- CSRF y comparación constante;
- autorización de cada operación mutante y ausencia de cambios tras un 403;
- información de `/health`, errores y logs;
- CSP, HSTS y recursos externos;
- validación de URLs, uploads e imports;
- deserialización de joblib/PyTorch;
- acciones costosas o entrenamiento dentro de requests.

Un hallazgo de seguridad debe incluir un caso reproducible o quedar marcado como
resultado de inspección. No publicar instrucciones de explotación innecesarias.

## 6. Dependencias, CI y operaciones

Contrastar dependencias directas, lock, herramientas de desarrollo/documentación,
versión de Python, PyTorch por plataforma, serialización y archivos de despliegue.
Comprobar qué manifest usa realmente cada entorno.

Fechar `pip-audit` y registrar paquetes omitidos. No recomendar una actualización
mayor sin comprobar compatibilidad con modelos y tests. Verificar políticas de
servicios en documentación oficial vigente.

## 7. Datos y ML

Trazar el flujo completo:

`fuente → generación/limpieza → etiqueta → features → split → ajuste → calibración → evaluación → persistencia → inferencia → interfaz`

Comprobar:

- naturaleza real, sintética o mixta de los datos;
- definición y prevalencia de la etiqueta;
- circularidad, features derivadas de etiquetas y leakage antes del split;
- seed, train/validation/test y selección de umbral;
- baseline y comparación pareada;
- probabilidad cruda, calibrada, score combinado y bandas visuales;
- parámetros entrenables frente a buffers/state_dict;
- hashes, versiones, duración y metadata disponibles;
- límites de generalización y ausencia de validación con jugadores reales.

No llamar “observada” a una prevalencia fijada por diseño. No afirmar superioridad
o equivalencia estadística sin análisis suficiente. Si faltan múltiples seeds,
considerar bootstrap pareado sobre predicciones existentes y explicar sus límites.

## 8. Revisión académica

Recorrer la cadena:

`problema → pregunta → objetivos → metodología → requisitos → implementación → pruebas → resultados → discusión → conclusiones`

Marcar objetivos aspiracionales no medidos, metodología descrita pero no aplicada,
impacto social no validado y conclusiones más fuertes que la evidencia. Distinguir
funcionalidades implementadas de medidas propuestas.

Para menores, revisar consentimiento, minimización, retención/baja, acceso y riesgo
de etiquetado. Verificar legislación en fuentes oficiales y evitar asesoramiento
legal categórico.

## 9. Fuentes y forma

Auditar correspondencia cita–bibliografía, autor, título, año, DOI/URL y pertinencia.
No inventar fechas de consulta. Revisar APA 7, voz, terminología, puntuación,
enumeraciones, numeración, tablas, figuras, discusión y duplicaciones.

Usar herramientas lingüísticas como señal auxiliar. Cada propuesta debe revisarse
en contexto académico; un detector gramatical no decide verdad ni coherencia.

## 10. Consolidación

Deduplicar vinculando referencias, sin borrar IDs originales. Separar:

- prioridad previa a defensa;
- seguridad;
- dependencias/datos/calidad/pruebas/CI;
- ML y trazabilidad;
- contenido documental;
- forma, redacción y fuentes.

Cada hallazgo debe contener evidencia suficiente, impacto, corrección y estado. La
severidad evalúa riesgo para entrega/defensa, no el esfuerzo de redacción.
