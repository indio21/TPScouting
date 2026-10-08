# Plantilla del informe consolidado

Usar esta forma cuando el usuario pida un informe comparable con
`correccion-octubre.md`. Adaptar las secciones al material disponible sin crear
hallazgos para llenar la plantilla.

```markdown
# Corrección <mes y año> — TPScouting (<autor>)
**Trabajo Final de Grado — <carrera e institución verificadas>**
**Directores:** <datos verificados o NO VERIFICADO>
**Documento revisado:** `<archivo>` (<páginas>, SHA-256 `<hash>`)
**Código revisado:** `<repo/remoto>`, rama `<rama>`, HEAD `<sha>`
**Revisión:** <fecha y zona horaria>
**Material no disponible:** <lista o “ninguno”>

## Veredicto
- **Código:** <estado y evidencia principal>.
- **Documento:** <estado y magnitud de las correcciones>.
- **Veredicto general:** <uno de los cuatro estados permitidos>.

## A. Prioridad 1 — corregir antes de la defensa
| ID | Dónde | Hallazgo | Evidencia | Corrección | Estado |
|---|---|---|---|---|---|

## B. Código — seguridad
| ID | Sev. | Ubicación | Hallazgo | Evidencia | Corrección | Estado |
|---|---|---|---|---|---|---|

## C. Código — dependencias, datos, calidad, pruebas, CI
| ID | Sev. | Hallazgo | Evidencia | Corrección | Estado |
|---|---|---|---|---|---|

## D. Código — validez de ML y trazabilidad
| ID | Sev. | Hallazgo | Evidencia | Corrección | Estado |
|---|---|---|---|---|---|

## E. Documento — contenido y coherencia
| ID | Ubicación | Hallazgo | Evidencia | Corrección | Estado |
|---|---|---|---|---|---|

## F. Documento — forma, redacción y fuentes
| ID | Ubicación | Hallazgo | Corrección | Estado |
|---|---|---|---|---|

## Evidencia reproducida
| Comando/control | Entorno | Resultado | Limitación |
|---|---|---|---|

## Decisiones y datos pendientes
- <pregunta concreta, responsable y efecto sobre el cierre>

## Orden recomendado de corrección
1. <bloque pequeño, criterio de cierre>

## Límites de esta revisión
- <qué no se ejecutó, inspeccionó o pudo demostrar>
```

## Reglas de identificación

- Conservar IDs docentes existentes.
- Para nuevos hallazgos usar prefijos: `SEG`, `DEP`, `DATOS`, `CAL`, `TEST`,
  `CI`, `INFRA`, `ML`, `DOC`, `FTE` y `FORMA`.
- Numerar de forma estable dentro del informe. Si un problema duplica otro, incluir
  “Relacionado con …” en vez de eliminarlo.
- Estado: `ABIERTO`, `PARCIAL`, `RESUELTO`, `NO APLICA` o `NO VERIFICADO`.
- Severidad: `Crítico`, `Alto`, `Medio`, `Bajo`. Explicar los críticos y altos.

## Criterios de redacción

- Escribir hechos en presente y pruebas ejecutadas en pasado, con fecha.
- Evitar “todo funciona”, “seguro” o “correcto” sin delimitar el alcance.
- Citar rutas y símbolos actuales; las líneas son orientativas y deben corresponder
  al commit indicado.
- Expresar números con su origen: comando, tabla, artefacto o cálculo.
- Si una recomendación es opcional, decirlo explícitamente.
- No asignar una nota numérica salvo que exista una rúbrica o el usuario la pida.
