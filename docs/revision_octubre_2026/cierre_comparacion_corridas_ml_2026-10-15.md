# Comparación de las corridas ML histórica y candidata

Fecha: 2026-10-15.

## Decisión

La corrida histórica continúa como resultado documentado y como origen de los
artefactos utilizados por la aplicación. La corrida
`20261010T_local_seed42_leakage_safe_v2` se incorpora al Trabajo Final como una
comparación experimental. No reemplaza el modelo de runtime ni se presenta como
una mejora predictiva.

Esta opción reduce el riesgo: permite mostrar la corrección metodológica y la
trazabilidad alcanzada sin modificar el funcionamiento ya verificado del MVP.

## Comparación incorporada

Se comparó la salida cruda de PlayerNet sobre el test correspondiente a cada
corrida:

| Métrica | Histórica | Candidata |
|---|---:|---:|
| ROC-AUC | 0,9203 | 0,9056 |
| PR-AUC | 0,5461 | 0,4888 |
| Brier | 0,1049 | 0,1175 |
| Log loss | 0,3294 | 0,3673 |
| F1 | 0,5291 | 0,4931 |

Los cinco valores descriptivos favorecen a la corrida histórica. Esta diferencia
no se atribuye automáticamente al modelo porque las etiquetas no se definieron
de la misma manera: la corrida histórica calculó cuantiles y cuotas antes del
split, mientras que la candidata ajustó la política solo con train y la aplicó
sin recalcularla sobre validation y test.

La candidata resuelve ese riesgo metodológico y registra configuración, splits,
historial, tiempos, dependencias y hashes. Aun así, conserva datos sintéticos y
una sola seed. Por esas razones sirve como evidencia metodológica reproducible,
pero no justifica sustituir la corrida histórica ni afirmar una mejora.

## Documento resultante

- Word: `word/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_COMPARACION_ML_2026-10-15.docx`.
- PDF: `pdf/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_COMPARACION_ML_2026-10-15.pdf`.
- Evidencia: `verificacion_documento_comparacion_ml_2026-10-15.json` y
  `verificacion_estructura_comparacion_ml_2026-10-15.json`.

La comparación se agregó en la sección 6.4, sin crear tablas o figuras numeradas
ni alterar la estructura del índice. El documento quedó en 96 páginas y conservó
28 imágenes, 19 tablas, 9 secciones y 45 entradas en sus listas.

## Verificación y alcance

- métricas contrastadas directamente con los JSON de evaluación histórica y
  candidata;
- listas de figuras y tablas sin diferencias de página;
- sin campos rotos, pendientes editoriales ni páginas anormalmente vacías;
- portada, índice, resumen, abstract y bibliografía en páginas independientes;
- texto agregado en Arial 14 y sin caracteres dañados;
- sin cambios en código, artefactos de runtime o repositorio de entrega;
- imágenes conservadas sin volver a analizarlas.

No se ejecutó la suite de la aplicación porque la modificación es exclusivamente
documental y no cambia el comportamiento del software.
