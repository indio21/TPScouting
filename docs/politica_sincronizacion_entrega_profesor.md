# Politica interna de sincronizacion con TPScouting-entrega

Fecha: 2026-07-02

Este documento queda como regla de trabajo para continuar el proyecto sin volver a
mezclar el repositorio completo con el repositorio publico de entrega.

## Repositorios

- Repositorio completo de trabajo: `C:\Tesis\TPScouting`
- Repositorio publico para el profesor: `C:\Tesis\TPScouting-entrega`
- GitHub publico de entrega: `https://github.com/indio21/TPScouting-entrega`
- Rama publica de entrega: `main`
- Commit inicial de entrega: `a6f8b74 Initial clean delivery repository`

## Regla principal

Todo cambio se trabaja primero en `TPScouting`.

Despues, solo se replica a `TPScouting-entrega` lo que pueda ver el profesor:

- codigo de la app;
- tests;
- configuracion necesaria;
- dependencias;
- diagramas tecnicos presentables;
- documentacion tecnica limpia;
- artefactos chicos de runtime necesarios para inferencia demo.

Nunca se debe copiar el repositorio completo al repo de entrega.

## Permitido en TPScouting-entrega

- `.github/workflows/ci.yml`
- `.gitignore` saneado de entrega
- `README.md`, `README_TESTS.md`, `RUNBOOK.md` saneados de entrega
- `render.yaml` saneado de entrega
- `requirements.txt`
- `requirements-dev.txt` saneado de entrega
- `requirements-lock.txt`
- `scouting_app/*.py`
- `scouting_app/ml/`
- `scouting_app/routes/`
- `scouting_app/services/`
- `scouting_app/static/`
- `scouting_app/templates/`
- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`
- `tests/`
- `scripts/smoke_render.py`
- `scripts/render_plantuml_diagrams.py`
- `docs/diagramas/plantuml/`
- `docs/diagramas/export/`
- `docs/diagramas/README.md` saneado de entrega
- `docs/guia_indicadores_app.md` saneado de entrega
- `docs/modelo_ml.md`
- `docs/puesta_en_marcha.md`

## Prohibido en TPScouting-entrega

- `docs/contexto_para_nuevo_chat.md`
- documentos Word (`*.docx`);
- documentos de correccion del profesor;
- notas internas de auditoria, cierre o retome;
- capturas/evidencia interna del Word;
- scripts que editan o generan Word;
- bases locales (`*.db`, `*.sqlite3`);
- backups (`*.bak`, `*.before_*`, `*.db.*`);
- metadata generada de entrenamiento si contiene rutas locales (`training_metadata.json`, `training_splits.json`, `experiments.csv`);
- `.env`;
- `.vscode`;
- `scouting_app/registro_mejoras.md`;
- `scouting_app/templates/dashboard.html.bak`;
- cualquier archivo con rutas tipo `C:\Users\Usuario\Desktop`;
- cualquier archivo con contexto de chat o decisiones privadas.

## Procedimiento recomendado para futuros cambios

1. Trabajar y probar cambios en `C:\Tesis\TPScouting`.
2. Ejecutar la suite en el repo completo si corresponde.
3. Sincronizar hacia entrega con:

```powershell
cd C:\Tesis\TPScouting
.\scripts\sync_entrega_repo.ps1
```

4. Revisar el diff en `C:\Tesis\TPScouting-entrega`.
5. Revisar manualmente si tambien corresponde actualizar `README.md`,
   `RUNBOOK.md`, `README_TESTS.md`, `render.yaml`, `requirements-dev.txt`,
   `docs/diagramas/` o documentacion tecnica limpia.
6. Ejecutar tests en `TPScouting-entrega`.
7. Ejecutar auditoria de archivos sensibles.
8. Si esta OK, commitear y pushear `TPScouting-entrega`.

El script copia codigo, tests y archivos tecnicos claramente seguros. No copia
automaticamente docs ni diagramas porque en el repo completo pueden contener
contexto interno, nombres de ramas viejas, rutas locales o texto especifico de
otro cierre.

## Auditoria minima antes de publicar entrega

Desde `C:\Tesis\TPScouting-entrega`:

```powershell
git status --short --branch
rg --files -g '*.docx' -g '*.db' -g '*.zip' -g '*.bak' -g '*.csv' -g 'experiments.csv' -g 'training_metadata.json' -g 'training_splits.json' -g '.env'
rg -n --hidden --glob '!.git/**' --glob '!*.png' --glob '!*.jpg' --glob '!*.jpeg' --glob '!*.pt' --glob '!*.joblib' --glob '!*.svg' "(contexto_para_nuevo_chat|ChatGPT|profesor|Correccion TP|TRABAJO_FINAL|Desktop|C:\\Users|observaciones del profesor|auditoria|cierre_|Version corregida|backup_|AdminDemo123|tpscouting-mvp\.onrender\.com|render-free-deploy|ux-crud-polish|reformas-finales|reformas-complejas|auditoria-correcciones-mvp|CodeGPT|Gemini CLI|registro_mejoras|dashboard\.html\.bak)"
```

Los dos `rg` deben no devolver resultados peligrosos.

## Estado actual al guardar esta politica

- `TPScouting` estaba limpio y sincronizado con `origin/main` antes de agregar esta politica.
- `TPScouting-entrega` estaba limpio y sincronizado con `origin/main`.
- La suite de `TPScouting-entrega` paso con `83 passed, 1 skipped, 4 warnings`.
- El repo publico de entrega quedo creado como publico en GitHub.
