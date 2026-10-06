# Plan previo para una nueva corrida ML — no ejecutado

Este plan sólo se ejecutará con autorización posterior. Su objetivo sería cerrar
D-16 y aportar evidencia multiseed adicional sin alterar la corrida histórica.

1. Crear un directorio nuevo `artifacts/runs/<fecha-hora>/`; no reemplazar
   `model.pt`, joblib, metadata, splits ni cache existentes.
2. Congelar commit, requirements, hashes del dataset/cache y configuración.
3. Ejecutar al menos cinco seeds (42–46) con el mismo protocolo y splits
   generados por seed, registrando inicio/fin, duración y plataforma.
4. Guardar train loss y validation loss por época, además de PR-AUC/F1 de
   validación. La corrida histórica sólo posee train loss.
5. Evaluar cada seed una única vez en su test reservado y agregar resultados sin
   sustituir los de mayo de 2026.
6. Generar hashes SHA-256 de cada artefacto y un manifiesto de versiones.
7. Comparar PlayerNet y baseline con intervalos entre seeds y bootstrap pareado,
   exponiendo incertidumbre y sin fijar una conclusión de antemano.

Costos todavía no medidos: duración total, uso de CPU y espacio adicional. La
primera corrida deberá medirlos; no se proyecta un tiempo ficticio a partir del
metadata, porque la duración histórica no fue guardada.
