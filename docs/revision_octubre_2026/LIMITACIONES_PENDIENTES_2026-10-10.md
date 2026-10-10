# Limitaciones pendientes para tratamiento paso a paso

Actualizado: 2026-10-10, America/Buenos_Aires.

Estas limitaciones no impiden evaluar el MVP entregado. Tampoco deben presentarse
como funcionalidades o evidencias ya disponibles.

## Orden recomendado

1. **Evaluación independiente del score combinado.** El score que mezcla salida
   del modelo, historial y adecuación posicional no conserva una evaluación
   independiente en test. Preparar un evaluador offline, fijar datos y split,
   comparar contra la probabilidad cruda/calibrada y guardar resultados e hashes.

2. **ML-01: target definido antes del split.** Los cuantiles y cuotas de la
   etiqueta sintética se calcularon antes de separar train/validation/test. Una
   nueva corrida debería ajustar toda transformación dependiente de distribución
   solamente con train y aplicar sus parámetros sin recalcular en validation/test.
   Debe conservarse separada de la corrida histórica y no sobrescribir artefactos.

3. **D-16: trazabilidad incompleta de la corrida histórica.** No existen
   validation loss, duración ni SHA exacto de aquella corrida. Esos valores no se
   pueden recuperar honestamente. La solución posible es ejecutar una corrida
   nueva y versionada que registre configuración, dependencias, seed, splits,
   métricas, tiempos y hashes, manteniéndola diferenciada de la histórica.

4. **Validación en macOS.** Existen instrucciones, pero no una ejecución
   comprobada. Verificar en un equipo o runner macOS y registrar versión de Python,
   arquitectura, instalación, pruebas y resultado. Hasta entonces debe seguir
   declarado como no probado.

5. **CAL-03: organización de pruebas.** `test_mvp_regressions.py` todavía puede
   dividirse por dominio. Hacerlo sin cambiar fixtures ni cobertura y comprobar la
   suite antes y después.

6. **CAL-02: lint y logging graduales.** El CI aplica solo reglas críticas de
   Ruff. Medir primero el conjunto completo, corregir por módulos y reemplazar
   `print` operativos cuando corresponda, sin reformatear masivamente.

7. **CAL-01: application factory y módulos grandes.** Es un refactor amplio.
   Preparar diseño, dependencias y pruebas de caracterización antes de autorizarlo;
   no mezclarlo con correcciones funcionales menores.

8. **Uso futuro de datos reales de menores.** El MVP usa datos sintéticos. Antes
   de cualquier uso real se necesitan consentimiento de tutores, información al
   titular, reglas de retención y baja, control de acceso, trazabilidad y revisión
   normativa/institucional. No corresponde simular que estos controles existen.

9. **Artefactos opcionales de distribución.** Dockerfile, `.env.example` y LICENSE
   continúan fuera del alcance obligatorio. Docker y el ejemplo de entorno pueden
   evaluarse por utilidad operativa; la licencia requiere una decisión explícita
   del titular del proyecto.

## Primera tarea recomendada

Comenzar por la evaluación independiente del score combinado. Es acotada, produce
evidencia útil y permite decidir si ese indicador debe conservarse, ajustarse o
presentarse solamente como heurística del MVP.
