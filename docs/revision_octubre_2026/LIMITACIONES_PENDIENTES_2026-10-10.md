# Limitaciones pendientes para tratamiento paso a paso

Actualizado: 2026-10-10, America/Buenos_Aires.

Estas limitaciones no impiden evaluar el MVP entregado. Tampoco deben presentarse
como funcionalidades o evidencias ya disponibles.

## Orden recomendado

1. **RESUELTA — Evaluación independiente del score combinado.** El score que mezcla salida
   del modelo, historial y adecuación posicional no conserva una evaluación
   independiente en test. Preparar un evaluador offline, fijar datos y split,
   comparar contra la probabilidad cruda/calibrada y guardar resultados e hashes.

2. **RESUELTA EN CORRIDA CANDIDATA — ML-01: target definido antes del split.** Los cuantiles y cuotas de la
   etiqueta sintética se calcularon antes de separar train/validation/test. Una
   nueva corrida debería ajustar toda transformación dependiente de distribución
   solamente con train y aplicar sus parámetros sin recalcular en validation/test.
   Debe conservarse separada de la corrida histórica y no sobrescribir artefactos.

3. **RESUELTA PARA LA CORRIDA NUEVA; IRRECUPERABLE EN LA HISTÓRICA — D-16.** No existen
   validation loss, duración ni SHA exacto de aquella corrida. Esos valores no se
   pueden recuperar honestamente. La solución posible es ejecutar una corrida
   nueva y versionada que registre configuración, dependencias, seed, splits,
   métricas, tiempos y hashes, manteniéndola diferenciada de la histórica.

4. **RESUELTA PARA APPLE SILICON — Validación en macOS.** El runner real
   `macos-15` arm64 con Python 3.11 completó instalación, carga de artefactos,
   suite, cobertura y demo el 2026-10-14. macOS Intel queda explícitamente fuera
   del soporte: PyPI no publica un wheel x86_64 de `torch==2.14.1`. No se cambió
   la versión fijada para ocultar esa incompatibilidad.

5. **RESUELTA — CAL-03: organización de pruebas.** Las 76 regresiones del archivo
   monolítico se dividieron en runtime, autenticación/seguridad, jugadores y
   ML/datos, con helpers compartidos. Se conservaron exactamente los cuerpos y la
   colección; la suite integral mantuvo 83,23 % de cobertura.

6. **RESUELTA EN ALCANCE GRADUAL — CAL-02: lint y logging.** Además de las reglas
   críticas, CI controla imports/variables sin uso y excepciones silenciadas en
   aplicación y pruebas. Se mejoró logging puntual y se preservaron las salidas
   CLI. La medición completa mantiene 732 avisos declarados, principalmente
   modernización masiva de tipos; su ampliación futura debe hacerse por regla.

7. **EVALUADA; NO IMPLEMENTAR EN EL MVP — CAL-01: application factory y
   módulos grandes.** La evaluación verificó estado global, efectos durante el
   import, 85 reglas Flask y 218 referencias directas desde pruebas. El riesgo de
   una migración transversal supera su beneficio inmediato para la entrega ya
   validada. Quedó definido un plan por etapas y criterios previos; la
   implementación requiere una autorización futura específica.

8. **DISEÑO COMPLETADO; USO REAL NO HABILITADO — datos reales de menores.** Se
   documentaron puertas de habilitación, cambios técnicos, pruebas y responsables
   en `docs/REAL_DATA_READINESS.md`, contrastados con fuentes oficiales. El MVP
   sigue usando datos sintéticos y no implementa aún consentimiento/representación,
   derechos, retención, auditoría, gestión de incidentes ni validación externa.
   Esos controles dependen de decisiones institucionales que no deben inventarse.

9. **RESUELTA PARA DISTRIBUCIÓN Y DERECHOS.** Se incorporaron
   `Dockerfile`, `.dockerignore` y `.env.example`, sin secretos. La imagen se
   construyó y superó smoke tests independientes en los repositorios propio y de
   entrega. Por decisión del titular no se agregó `LICENSE`: todos los derechos
   se encuentran reservados y no se concede una autorización general de
   reutilización.

## Próxima tarea recomendada

La lista técnica de nueve limitaciones quedó tratada. La preparación para datos
reales quedó diseñada, pero su uso continúa inhabilitado. La siguiente tarea es
la revisión final integral del documento actualizado.
