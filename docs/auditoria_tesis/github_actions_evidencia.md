# Evidencia publica de GitHub Actions

Fecha de consulta: 2026-07-10

Repositorio: `https://github.com/indio21/TPScouting`

Workflow: `CI` (`.github/workflows/ci.yml`)

## Hechos verificados

| Campo | Valor | Fuente |
|---|---|---|
| Cantidad de runs publicas | 71 | API publica, `github_actions_runs_2026-07-10.json` |
| Ultima run publicada | `CI #71` | pagina publica y API GitHub |
| Run ID | `26317958817` | API publica |
| Commit | `fa8a50d1173f2760329a45f11fbf93a709721235` | API publica |
| Branch | `main` | API publica |
| Evento | `push` | API publica |
| Inicio | `2026-05-23T00:09:59Z` | API publica |
| Fin registrado | `2026-05-23T00:12:10Z` | API publica |
| Estado/conclusion | `completed / success` | API publica |
| Job Python 3.11 | `completed / success` | `github_actions_run71_jobs_2026-07-10.json` |
| Job Python 3.12 | `completed / success` | `github_actions_run71_jobs_2026-07-10.json` |
| Paso tests | `success` en ambos jobs | JSON de jobs |
| Upload coverage | `success` en ambos jobs | JSON de jobs |
| Artefactos visibles | `coverage-python-3.11` y `coverage-python-3.12` | captura `github_actions_run71_2026-07-10.png` |

URL directa: `https://github.com/indio21/TPScouting/actions/runs/26317958817`

## Advertencias visibles

**HECHO VERIFICADO.** La pagina de la run muestra dos annotations de advertencia, una por job, por uso de acciones basadas en Node.js 20 deprecado. No son fallos de pytest: la run concluyo `success`. Evidencia: `github_actions_run71_2026-07-10.png`.

## Limite de la evidencia

**HECHO VERIFICADO.** La ultima run publica corresponde a `fa8a50d...`, que coincide con `origin/main` durante la auditoria. El HEAD local es `7d7680e...` y esta un commit por delante. Por lo tanto, Actions prueba el commit remoto `fa8a50d...`, no el commit local aun no publicado ni los nuevos anexos de auditoria.

**NO VERIFICABLE.** Los logs internos completos requieren autenticacion en la interfaz publica consultada. Se conservaron el resumen publico, el estado de cada job y cada paso disponible mediante la API, sin inventar contenido de logs.

## Capturas listas para el Word

| Archivo | Pie de figura sugerido |
|---|---|
| `github_actions_2026-07-10.png` | "Historial publico del workflow CI de TPScouting, consultado el 10/07/2026." |
| `github_actions_run71_2026-07-10.png` | "Ejecucion CI #71 completada correctamente para Python 3.11 y 3.12, con artefactos de cobertura." |

## Fuentes conservadas

- `github_actions_runs_2026-07-10.json`: respuesta sin transformar de la API publica de runs, limitada a las diez mas recientes.
- `github_actions_run71_jobs_2026-07-10.json`: respuesta sin transformar de la API publica de jobs de la run 71.
- `ci_workflow_snapshot_2026-07-10.yml`: copia exacta del workflow local auditado.
- Capturas PNG obtenidas directamente de las paginas publicas con Playwright 1.59.0.
