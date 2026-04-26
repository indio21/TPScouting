# Guia De Indicadores De La App

Fecha: 2026-04-26

Este documento explica los indicadores principales que aparecen en la ficha, prediccion y comparadores del MVP.

## Rendimiento En Posicion

Ejemplo en pantalla: `Rendimiento en posicion 12.1`.

Es un puntaje de ajuste del jugador a una posicion determinada.

Escala:

- Va aproximadamente de `0` a `20`.
- No es una probabilidad.
- Sale de los atributos del jugador ponderados segun la posicion.

Interpretacion:

- Un delantero pondera mas atributos como ritmo, disparo, regate y tecnica.
- Un defensor pondera mas defensa, fisico, marcaje y determinacion.
- Un mediocampista pondera mas pase, vision, tecnica y determinacion.

Un valor como `12.1` indica un encaje medio del jugador en esa posicion segun sus atributos actuales.

## Rendimiento En Puesto Natural

Ejemplo en pantalla: `Rendimiento en puesto natural 12.1`.

Es el mismo concepto que rendimiento en posicion, pero aplicado a la posicion registrada actualmente para el jugador.

Sirve para ver si el jugador esta bien perfilado para su puesto actual o si podria convenir evaluar una posicion alternativa.

## Puntaje De Ficha

Ejemplo en pantalla: `Puntaje de ficha: 48.7%`.

Es la salida principal del modelo PyTorch para ese jugador.

En el MVP se usa la salida cruda de PyTorch (`raw_pytorch_sigmoid`) como score principal porque en la evidencia actual ordena mejor a los jugadores por `PR-AUC`.

Interpretacion:

- Resume lo que el modelo estima a partir de la ficha y las variables preprocesadas del jugador.
- Es un score de priorizacion, no una sentencia definitiva.
- Ayuda a ordenar candidatos para revision del scout.

## Probabilidad Combinada

Ejemplo en pantalla: `Probabilidad combinada 57.5%`.

Es el score final que muestra la app.

Combina:

- el puntaje de ficha del modelo;
- senales del historial de rendimiento, como `final_score`;
- el ajuste del jugador a su posicion.

Por eso puede ser mayor o menor que el puntaje de ficha.

Ejemplo:

- Puntaje de ficha: `48.7%`.
- Rendimiento en posicion: `12.1/20`.
- Historial de rendimiento favorable.
- Probabilidad combinada: `57.5%`.

En ese caso, el historial y el encaje posicional suben el score final.

## Referencia Calibrada

La app puede mostrar una referencia calibrada cuando existe calibrador.

No es el score principal del MVP.

La calibracion se conserva como evidencia secundaria porque en la corrida oficial actual mejora levemente `F1` y `recall`, pero baja `PR-AUC` frente a la salida cruda.

## Frase Recomendada Para Presentacion

El puntaje de ficha es la estimacion del modelo a partir de los datos del jugador. La probabilidad combinada ajusta esa estimacion con senales adicionales del MVP, como rendimiento historico y adecuacion a la posicion. El rendimiento en posicion no es una probabilidad, sino un puntaje de 0 a 20 que mide que tan bien encajan sus atributos con el puesto en el que juega.

