# Verificación de compatibilidad con macOS

Fecha de preparación: 2026-10-10.

## Alcance

Se incorporó un workflow aislado, `.github/workflows/macos.yml`, para verificar el
proyecto en máquinas virtuales macOS limpias provistas por GitHub Actions. La matriz
incluye Apple Silicon (`macos-15`, arm64) e Intel (`macos-15-intel`, x64), ambos con
Python 3.11.

Cada trabajo debe comprobar:

1. instalación desde `requirements.txt` y `requirements-dev.txt`;
2. consistencia del entorno mediante `pip check`;
3. carga de `model.pt`, `preprocessor.joblib` y `probability_calibrator.joblib`;
4. controles críticos de Ruff;
5. suite completa con cobertura mínima del 80 %;
6. creación y verificación de la demo sobre una base SQLite temporal.

No se entrena un modelo ni se modifica una base del usuario.

## Estado inicial

Validación remota pendiente. La existencia del workflow y la documentación de los
comandos no constituyen por sí mismas evidencia de compatibilidad. Este documento se
actualizará con el commit, la ejecución y el resultado de cada arquitectura después
de recibir evidencia de GitHub Actions.

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
