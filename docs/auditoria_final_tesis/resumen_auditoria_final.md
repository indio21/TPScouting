# Resumen de auditoria tecnica final

Fecha: 2026-07-13  
Documento: `docs/tesis_final/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_AUDITADA.docx`  
Commit auditado: `7d7680e04dd15b4c4f01d9e0aee8711aeea06f0d`

## Dictamen

**APTO CON CORRECCIONES MENORES antes del envio.**

El nucleo tecnico y metodologico del Trabajo Final coincide con la aplicacion: arquitectura, funcionalidades principales, entidades, PlayerNet, 68 features, target temporal, datos sinteticos, configuracion de entrenamiento, metricas, pruebas, seguridad y limitaciones estan documentados con un nivel de cautela adecuado. No se detecto una contradiccion que invalide los resultados centrales.

La version no deberia enviarse sin corregir nueve puntos concretos. Los mas importantes son el diagrama de inferencia, los comandos incompletos del anexo, campos inexistentes en la Tabla 4-5 y los campos de indice/listas desactualizados. Son corregibles sin modificar el codigo ni reentrenar.

## Conteo de la matriz

| Estado | Cantidad |
|---|---:|
| Afirmaciones auditadas | 104 |
| COINCIDE | 82 |
| COINCIDE PARCIALMENTE | 6 |
| NO COINCIDE | 10 |
| NO VERIFICABLE | 1 |
| AFIRMACIÓN DESACTUALIZADA | 5 |

Adicionalmente, `afirmaciones_no_verificables.md` documenta 15 vacios de evidencia o afirmaciones que no deben sostenerse sin una fuente externa. Ese inventario es mas amplio que la unica fila `NO VERIFICABLE` de la matriz porque incluye informacion ausente que el Word evita afirmar o presenta correctamente como limitacion.

## Hechos confirmados

- El Word de escritorio y el del repositorio son copias identicas por SHA-256.
- PlayerNet utiliza `input_dim=68`: 11 features base, 52 historicas y 5 de posicion.
- El target oficial es `temporal_target_label`; 1.597 positivos y 18.403 negativos.
- La corrida registrada usa seed 42, split 14.000/3.000/3.000, 45 epocas solicitadas, 15 ejecutadas, mejor epoca 5 y early stopping con patience 10.
- El documento separa correctamente PlayerNet crudo, PlayerNet calibrado y score combinado.
- La suite actual mantiene `83 passed, 1 skipped, 4 warnings`; cobertura `79%` sobre 5.082 sentencias.
- CI #71 fue exitosa para `fa8a50d...`, y el Word aclara que no certifica el HEAD local.
- Las conclusiones no afirman validacion con datos reales, infraestructura Big Data ni evaluacion de test del score combinado.

## Contradicciones prioritarias

1. Flask-Migrate/Alembic se presenta como si estuviera implementado; el proyecto usa migraciones manuales.
2. La Tabla 4-5 contiene campos que no existen en SQLAlchemy.
3. Figura 5-1 omite `combine_probability` y representa mal la rama HTTP 500.
4. Tres comandos del Anexo B no son reproducibles literalmente.
5. Figura 5-3 afirma una rama no demostrada y describe mal los locks.
6. El TOC, listas y algunos captions estan desactualizados o fuera de orden.

## Riesgos academicos

- **Alto:** un evaluador podria confundir la salida calibrada con el resultado principal de la interfaz si se conserva Figura 5-1.
- **Alto:** la reproducibilidad prometida queda debilitada por comandos abreviados/no ejecutables.
- **Alto editorial:** indice y captions fuera de orden hacen visible una actualizacion incompleta del documento.
- **Medio:** campos inexistentes, uso de SQLite y "generacion de informes" sobredimensionan parcialmente el alcance.
- **Residual reconocido:** el target se construye antes del split y limita la independencia metodologica de validation/test.

## Informacion faltante

- SHA exacto del codigo usado en la corrida de entrenamiento.
- Duracion y validation loss de esa corrida.
- Rama y commit actualmente desplegados en Render.
- Disponibilidad vigente del servicio y estado actual de PostgreSQL.
- Evaluacion del score combinado y validacion externa con jugadores reales.
- Medicion real de impacto, reduccion de subjetividad, tiempo o costos.

## Verificaciones ejecutadas

- Inspeccion estatica de codigo, configuracion, tests, metadata y artefactos existentes.
- Extraccion estructural del DOCX y render temporal read-only a PDF de 90 paginas.
- Comparacion visual de figuras/capturas con rutas y templates actuales.
- Suite pytest y pytest-cov en el repositorio de entrega sincronizado.
- Consulta publica de GitHub Actions y smoke HTTP sin credenciales de Render.
- No se entreno el modelo ni se modificaron codigo, bases, checkpoints, datasets, configuracion o Word.

## Archivos de la auditoria

- `matriz_trazabilidad_final.md`
- `contradicciones_criticas.md`
- `afirmaciones_no_verificables.md`
- `correcciones_word_propuestas.md`
- `resumen_auditoria_final.md`
- `comandos_auditoria_final.txt`
