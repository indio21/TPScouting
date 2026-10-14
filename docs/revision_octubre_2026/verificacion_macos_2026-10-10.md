# Verificación de compatibilidad con macOS

Fecha de preparación: 2026-10-10.

## Alcance

Se incorporó un workflow aislado, `.github/workflows/macos.yml`, para verificar el
proyecto en una máquina virtual macOS limpia provista por GitHub Actions. La
plataforma soportada es Apple Silicon (`macos-15`, arm64) con Python 3.11.

Cada trabajo debe comprobar:

1. instalación desde `requirements.txt` y `requirements-dev.txt`;
2. consistencia del entorno mediante `pip check`;
3. carga de `model.pt`, `preprocessor.joblib` y `probability_calibrator.joblib`;
4. controles críticos de Ruff;
5. suite completa con cobertura mínima del 80 %;
6. creación y verificación de la demo sobre una base SQLite temporal.

No se entrena un modelo ni se modifica una base del usuario.

## Resultado comprobado

Commit evaluado: `39bda92da4324acc2317c25e13c38aa33cdcf0f0`.

Ejecución: https://github.com/indio21/TPScouting/actions/runs/38643910674

- `macos-15 / Python 3.11` (Apple Silicon, arm64): **aprobado**. Completó
  instalación, `pip check`, carga de artefactos, Ruff crítico, suite con cobertura
  y preparación de la demo temporal.
- `macos-15-intel / Python 3.11` (x64): **no compatible con el entorno fijado**.
  La instalación se detuvo porque PyPI no publica un wheel macOS x86_64 de
  `torch==2.14.1`. La consulta del índice mostró seis wheels macOS de esa versión,
  todos arm64.

El fallo Intel delimita el soporte de plataforma; no demuestra un defecto del
código ejecutado, ya que la dependencia no llega a instalarse. No se rebajó
PyTorch ni se creó un conjunto alternativo de dependencias sin validación. Tras
este diagnóstico, el workflow permanente se limitó a Apple Silicon.

## Fundamento técnico

- GitHub documenta que cada trabajo de un runner hospedado se ejecuta en una máquina
  virtual nueva y ofrece runners macOS estándar para arm64 e Intel.
- PyTorch documenta instalación mediante `pip` en macOS. Por eso macOS usa
  `requirements.txt`, que fija el wheel estándar de PyPI, y no
  `requirements-torch-cpu.txt`, cuyo índice y sufijo `+cpu` corresponden al cierre
  reproducible usado por Windows/Linux.

Fuentes oficiales consultadas el 2026-10-10:

- https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job
- https://pytorch.org/get-started/locally/
- https://pypi.org/pypi/torch/2.14.1/json
