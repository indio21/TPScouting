# Auditoría final de entrega — 15/10/2026

## Resultado

La entrega académica quedó publicada y verificable en el commit exacto
`4c5ef2c88494e257a3bd8232bccba7f33f79b66a` de la rama `main` del repositorio
`indio21/TPScouting-entrega`. El HEAD local y `origin/main` coinciden y el árbol
de trabajo del repositorio de entrega está limpio.

## Contenido documental publicado

Se sustituyeron selectivamente los documentos anteriores por una única pareja
final:

- `documento/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-15.docx`
  - SHA-256: `8385153268405E1069496E3E39D774C97DB4C47B35F295C4B064324171CC97AE`
- `documento/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_2026-10-15.pdf`
  - SHA-256: `2D1393893F6E7F84E9BEAA706FE30871DFCB573D94C04D993873C6C8C7ACAEE4`

Los hashes coinciden con las copias verificadas del repositorio de trabajo. La
verificación estructural del documento aprobó: 96 páginas PDF, 28 imágenes, 19
tablas, 9 secciones, 45 entradas de índices/listas, cero diferencias, cero
pendientes de campos y ninguna página de contenido anormalmente vacía. No se
volvieron a analizar las imágenes en esta auditoría.

## Verificación local del código publicado

La ejecución se hizo desde `C:\Tesis\TPScouting-entrega`, usando Python 3.11.9
de `C:\Tesis\TPScouting\.venv`:

- Suite: 123 pruebas aprobadas, 1 omitida y 4 advertencias conocidas.
- Cobertura: 83,29 %, por encima del umbral configurado de 80 %.
- Ruff crítico (`E9,F63,F7,F82`): aprobado.
- Ruff gradual (`F401,F841,RUF059,S110,S112`): aprobado.
- `compileall`: aprobado.
- `pip check`: sin dependencias rotas.
- `pip-audit --local`: sin vulnerabilidades conocidas; el build local
  `torch 2.14.1+cpu` no existe en PyPI y fue informado como omitido por la
  herramienta.
- `pip-audit -r requirements.txt --no-deps`: sin vulnerabilidades conocidas
  en el manifiesto canónico, incluido `torch==2.14.1`.

Versiones principales observadas: Flask 3.1.3, NumPy 2.4.4, pandas 3.0.2,
scikit-learn 1.8.0 y PyTorch 2.14.1+cpu.

## Demo desde base temporal

`scripts/iniciar_demo.py --recrear --solo-preparar` aprobó con una base SQLite
temporal: generó 60 jugadores sintéticos, validó fechas, edades y categorías,
creó el administrador y dejó disponibles los historiales. Una segunda prueba
con el cliente Flask comprobó login con CSRF y respuestas HTTP 200 en listado,
detalle, estadísticas, atributos, predicción, silueta y `/health`. La base
temporal se eliminó al terminar.

## Evidencia remota sobre el commit publicado

- CI: ejecución `38815440730`, finalizada correctamente.
  - Python 3.11: instalación, Ruff, suite y cobertura aprobados.
  - Python 3.12: instalación, Ruff, suite y cobertura aprobados.
  - Auditoría de dependencias instalada y del manifiesto directo: aprobada.
  - URL: <https://github.com/indio21/TPScouting-entrega/actions/runs/38815440730>
- Container: ejecución `38815440745`, finalizada correctamente.
  - Construcción de imagen, arranque con datos sintéticos, smoke test y cierre:
    aprobados.
  - URL: <https://github.com/indio21/TPScouting-entrega/actions/runs/38815440745>

Docker no está instalado en el equipo local. Por esa razón, la prueba real de
construcción y ejecución del contenedor se realizó en el runner Linux de GitHub
Actions y no se afirma una ejecución local inexistente.

## Control de publicación

- 101 archivos rastreados en la entrega.
- Cero coincidencias rastreadas con patrones de documentos internos, contexto,
  evidencias, auditorías, candidatos, corridas experimentales, bases `.db` o
  archivos `.env` reales.
- `.env.example` contiene exclusivamente marcadores y valores de ejemplo; no
  contiene credenciales reales.
- No se publicó ningún archivo `LICENSE`. README, guía del profesor y
  Dockerfile declaran que todos los derechos están reservados.
- Los artefactos locales `.coverage`, `coverage.xml`, `.pytest_cache` y
  `.ruff_cache` están ignorados y no forman parte del commit.

## Commits de esta publicación

- `30a15e6cfbc4c08a68841e410e59f341b2b1dc61`: actualización selectiva de las
  instrucciones y sustitución del Word/PDF final.
- `4c5ef2c88494e257a3bd8232bccba7f33f79b66a`: comentario declarativo de
  derechos en Dockerfile y commit exacto auditado/publicado.

## Conclusión

El repositorio de entrega queda listo para la revisión del profesor dentro del
alcance comprobado del MVP: instalación reproducible, suite y controles de
calidad aprobados, demo sintética funcional, contenedor validado y documento
final único. Las limitaciones futuras ya declaradas en el trabajo no bloquean
esta entrega y no se presentan como funcionalidades implementadas.
