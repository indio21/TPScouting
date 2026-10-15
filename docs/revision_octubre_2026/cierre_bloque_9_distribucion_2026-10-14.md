# Cierre del bloque 9: distribución y licencia

Fecha: 2026-10-14.

## Decisiones

| Artefacto | Decisión | Motivo |
|---|---|---|
| `.env.example` | Incorporar | Hace visible la configuración sin publicar secretos y alinea desarrollo, Docker y despliegue |
| `Dockerfile` y `.dockerignore` | Incorporar | Ofrecen un entorno Linux reproducible y limitan el contenido enviado al build |
| `LICENSE` | No incorporar todavía | Falta elegir qué derechos se conceden; una licencia abierta no puede revocarse retroactivamente sobre copias ya entregadas |

## Implementación

El contenedor:

- usa Python 3.11 slim;
- instala `requirements-lock.txt` y `requirements-torch-cpu.txt`;
- ejecuta `pip check` durante el build;
- copia solo la aplicación y sus artefactos runtime;
- utiliza un usuario sin privilegios;
- expone el puerto 8000 y verifica `/health`;
- ejecuta seed demo idempotente y Gunicorn con un worker;
- no incluye repositorio Git, `.venv`, bases, logs, pruebas, documentos internos
  ni datasets/artefactos de entrenamiento.

`.env.example` enumera configuración de base, seguridad, demo sintética, cache,
rate limiting, límites y score. Contiene marcadores visibles en lugar de secretos.
No se agregó `python-dotenv`: el archivo sirve para Docker y como referencia, no
modifica implícitamente la carga de configuración de la aplicación.

## Límite de verificación

Docker no está instalado en la máquina de trabajo. Por lo tanto, el bloque debe
validar estáticamente el Dockerfile, comprobar manifests/comando/artefactos y
ejecutar la aplicación y la suite en el entorno existente. No se afirmará que la
imagen fue construida hasta que CI o una máquina con Docker complete el build y
el smoke test.

Se incorporó `.github/workflows/docker.yml` para cerrar esa verificación en un
runner Linux real. El workflow construye la imagen, inicia una demo sintética en
SQLite, comprueba `/health` y `/login`, muestra logs ante fallos y elimina el
contenedor. No publica la imagen.

## Decisión de licencia pendiente

“Gratuita al principio y cambiarla después” admite modelos distintos:

1. **Sin licencia por ahora:** conserva todos los derechos; permite evaluación y
   lectura del repositorio, pero no otorga reutilización general.
2. **Licencia abierta ahora:** MIT, Apache-2.0 o GPL/AGPL conceden permisos que
   permanecen para las versiones ya publicadas. El titular puede cambiar la
   licencia de versiones futuras si conserva los derechos necesarios.
3. **Licencia dual o source-available:** puede ofrecer uso gratuito bajo
   condiciones y una licencia comercial separada, pero requiere definir con
   precisión usuarios, finalidad, plazo y derechos; conviene revisión jurídica.

Hasta que el titular elija uno de esos modelos, no corresponde generar una
licencia en su nombre. La ausencia de `LICENSE` debe declararse y no confundirse
con dominio público o código abierto.

## Verificación local

- estructura y campos obligatorios de Dockerfile, `.dockerignore`,
  `.env.example` y workflow: aprobados;
- parseo YAML del workflow y job `build-and-smoke`: aprobado;
- presencia de los tres artefactos ML y manifests usados por el build: aprobada;
- ausencia de credenciales reales en `.env.example`: comprobada;
- Ruff crítico y gradual: aprobados;
- `compileall`: aprobado;
- suite integral: 123 aprobadas, 1 omitida, 4 advertencias conocidas;
- cobertura: 83,29 %.

La construcción y el smoke del contenedor quedan sujetos al resultado del nuevo
workflow remoto. La suite local no sustituye esa evidencia.

## Verificación remota y entrega

- Repositorio propio: commit `754ccb3601a489ed1249a078ad700d1eb546a494`.
- Container propio: ejecución `38655727631`, aprobada. Construyó la imagen,
  inició la demo sintética y verificó `/health` y `/login`.
- CI propia: ejecución `38655727560`, aprobada en Linux/Python 3.11 y 3.12 y
  auditoría de dependencias.
- Entrega: commit `8093fe616c9c50b72078ff4f241d9ab3427e763d`.
- Container de entrega: ejecución `38656129892`, aprobada con el mismo build y
  smoke.
- CI de entrega: ejecución `38656130065`, aprobada.

La limitación de construcción local queda suplida por dos ejecuciones reales e
independientes en runners Linux. No se publicó ninguna imagen en un registry.

## Estado final

- Dockerfile, `.dockerignore` y `.env.example`: incorporados y verificados.
- Construcción y smoke del contenedor: aprobados en repositorio propio y entrega.
- Licencia: decisión diferida; por ahora no se conceden permisos generales de
  reutilización. Esto preserva la posibilidad de elegir luego un modelo abierto,
  comercial o dual sin intentar revocar derechos ya otorgados.

## Decisión posterior del titular — 2026-10-15

La decisión pendiente quedó cerrada: el proyecto no incorporará un archivo
`LICENSE` en esta entrega y todos los derechos se encuentran reservados. La
publicación permite la revisión académica del proyecto, pero no concede una
autorización general para copiarlo, modificarlo, redistribuirlo, sublicenciarlo
o explotarlo comercialmente. README y guía de demo deben reflejar esta decisión.
