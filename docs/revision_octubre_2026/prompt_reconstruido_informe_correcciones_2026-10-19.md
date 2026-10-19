# Prompt reconstruido para un informe de correcciones como el de octubre

**Fecha de reconstrucción:** 19/10/2026.  
**Naturaleza:** propuesta basada en la estructura y las afirmaciones observables de
`C:\Users\Usuario\Desktop\correccion-octubre.md`. No es una transcripción del
pedido original ni una nueva auditoría del proyecto.
**SHA-256 del informe adjunto:**
`DDC8168AF86432511477226C780BC9CA96BCEBC44E64AD4D9A78B6CB1400C6A5`.

## Qué permite afirmar la evidencia

- El informe adjunto se presenta como revisión del PDF `TRABAJO_FINAL v2.pdf`
  de 98 páginas y del repositorio `TPScouting-entrega` en `ffdefdf`, fechada el
  04/10/2026. Esa es **la identificación que declara el informe**, no una nueva
  verificación de esos artefactos.
- Su estructura es A–F: prioridades previas a defensa; seguridad; dependencias,
  datos, calidad, pruebas y CI; ML; contenido documental; forma y fuentes.
- Distingue algunos casos reproducidos de observaciones por inspección. Declara
  resultados de pytest/cobertura, resolución de dependencias y revisión de
  referencias; el informe consolidado por sí solo no contiene todos los
  comandos, registros ni artefactos con los que confirmar cada ejecución.
- El encabezado menciona `correccion_documento.md` y `correccion_codigo.md` como
  detalle completo. Esos archivos no se encontraron en las ubicaciones locales
  revisadas para esta reconstrucción.
- En `C:\Tesis` existen `Prompt_Correccion_MVP.md` y
  `Prompt_Etapas_Correccion_MVP_Profesor.md`, ambos de abril de 2026. Su
  propósito es corregir el MVP por etapas. Su fecha y contenido no demuestran
  que hayan sido el pedido que generó el informe de octubre.

**No recuperable con la evidencia disponible:** texto literal del prompt
original; identidad, herramientas y criterios privados del revisor; comandos
exactos que ejecutó; rúbrica que produjo la nota orientativa 6,5; contenido de
los dos informes de detalle mencionados. La similitud del siguiente prompt es
metodológica y estructural, no una prueba de autoría.

## Prompt reutilizable — copiar desde aquí

> Actuá como revisor independiente de un Trabajo Final de Grado de Ingeniería
> en Informática sobre TPScouting. Prepará **solo un informe de correcciones**
> técnico, académico y reproducible. No modifiques archivos, bases, modelos ni
> repositorios; no hagas commits, despliegues ni reentrenamientos. No inventes
> resultados, citas, métricas, páginas, líneas, verificaciones ni fallos.
>
> **Material de entrada**
>
> - Documento rector: `[ruta del DOCX]` y/o PDF exportado `[ruta del PDF]`.
> - Repositorio de código que se entregará: `[ruta y URL]`.
> - Correcciones anteriores, si existen: `[rutas]`.
> - Detalles separados de documento/código, si existen: `[rutas]`.
> - Fecha, sistema operativo y entorno de revisión: registralos al comenzar.
>
> Primero identificá la versión concreta de cada artefacto: ruta, tamaño,
> fecha, SHA-256 y páginas reales del PDF; repositorio, remoto, rama, commit
> completo y diferencias locales. Si Word y PDF no corresponden a la misma
> versión, indicá cuál es el rector. No mezcles el CI del repositorio de
> desarrollo con el de entrega. Si faltan archivos, consignalo y continuá con
> las partes que sí puedan verificarse.
>
> Contrastá **cada afirmación central del documento** con implementación,
> configuración, pruebas y artefactos: alcance real del MVP, requisitos,
> arquitectura, entidades, roles, seguridad, datos sintéticos, origen de
> etiquetas, features, splits, entrenamiento, inferencia, métricas, interfaz,
> demo, CI y despliegue. No cambies código para sostener una afirmación del
> documento. Diferenciá hecho reproducido, inspección, afirmación documental,
> inferencia y asunto no verificado.
>
> Ejecutá las verificaciones pertinentes en un entorno aislado: suite y
> cobertura, lint o análisis estático disponible, dependencias por plataforma,
> auditoría de vulnerabilidades, demo con base temporal y pruebas de flujos
> críticos. Indicá comando, fecha, versión, resultado y límite de cada prueba.
> No alteres bases existentes. Una prueba aprobada solo demuestra lo que cubre.
>
> Revisá seguridad con casos concretos cuando sea posible: redirección tras
> login, intentos de acceso con usuarios eliminados o roles cambiados,
> rate limiting y cabeceras de proxy confiables, CSRF, cookies y secreto,
> autorizaciones sobre operaciones que modifican datos, `/health`, CSP/HSTS,
> recursos externos, `photo_url`, importaciones, deserialización de modelos
> y entrenamiento o generación costosos dentro de requests. Describí como
> inspección lo que no hayas ejecutado. No propongas `ProxyFix` sin verificar
> la topología de proxies.
>
> Auditá dependencias y operación: `requirements`, lock realmente utilizado,
> compatibilidad de PyTorch y Python por plataforma, versiones de serialización,
> CI, cache, cobertura, auditoría, Render y contenedor si existen. Consultá
> fuentes oficiales actuales para vulnerabilidades, políticas de servicios,
> normas y leyes; fechá la consulta. No trates una vulnerabilidad histórica
> como vigente sin reproducirla.
>
> Trazá el recorrido ML completo: generación/fuente de datos → etiqueta →
> columnas de entrada → split → preprocesamiento → entrenamiento → calibración
> → evaluación → persistencia → inferencia → bandas de interfaz. Comprobá
> circularidad y fuga, cuantiles calculados antes del split, tasa positiva
> fijada por diseño, seeds, selección de umbral, baseline, incertidumbre y
> relación entre probabilidad cruda, calibrada y score combinado. Contá
> parámetros entrenables con `model.parameters()` y separalos de buffers y
> elementos de `state_dict`. No afirmes superioridad, equivalencia ni
> indistinguibilidad estadística sin evidencia adecuada. No reconstruyas
> metadata histórica ausente como si se hubiera registrado en esa corrida.
>
> Revisá la cadena académica problema → pregunta → objetivos → metodología →
> implementación → pruebas → resultados → discusión → conclusiones. Detectá
> objetivos aspiracionales no validados, comparaciones de tecnologías sin
> justificación, cifras que contradicen el código, comandos incompletos,
> ausencia de trazabilidad Git/CI, alcances futuros presentados como hechos y
> riesgos éticos de tratar datos de menores. Separá medidas implementadas de
> propuestas.
>
> Revisá también forma y fuentes: portada institucional, resumen y abstract,
> índices y listas frente al PDF final, estructura y numeración, tablas y
> figuras, voz académica, puntuación, terminología, bibliografía y citas APA 7.
> Verificá autores, títulos, años, DOI/URL y pertinencia antes de recomendar
> nuevas fuentes. No inventes datos institucionales ni fechas de consulta.
>
> **Formato de salida:** informe Markdown comparable con
> `correccion-octubre.md`, con encabezado que identifique documento y commit,
> veredicto separado de código y documento, y secciones:
>
> A. Prioridad 1 antes de defensa.  
> B. Código: seguridad.  
> C. Código: dependencias, datos, calidad, pruebas, CI e infraestructura.  
> D. Código: ML y trazabilidad.  
> E. Documento: errores de contenido y coherencia con código.  
> F. Documento: forma, redacción y fuentes.
>
> Para **cada observación**, asigná un ID estable, ubicación actual (archivo y
> símbolo o sección y página PDF), severidad justificada, descripción precisa,
> evidencia o método de comprobación, corrección propuesta, criterio de cierre
> y estado (`ABIERTO`, `PARCIAL`, `RESUELTO`, `NO APLICA` o `NO VERIFICADO`).
> Conservá los IDs anteriores y vinculá duplicados sin borrarlos. Separá
> explícitamente lo observado de lo recomendado. No repitas hallazgos antiguos
> como hechos actuales sin volver a contrastarlos.
>
> Cerrá con evidencias reproducidas, material faltante, limitaciones, decisiones
> que requieren datos del autor y orden sugerido de corrección en bloques
> pequeños. Podés dar un veredicto cualitativo, pero **no asignes nota numérica**
> sin una rúbrica verificable. Si un punto no se puede comprobar, decí qué
> faltaría para hacerlo. Entregá el informe y detenete: la auditoría no autoriza
> a aplicar correcciones.

## Uso correcto en la próxima revisión

Para auditar nuevas observaciones del profesor, sustituir las rutas y revisar el
Word/PDF y el commit **vigentes entonces**. El informe de octubre funcionará como
antecedente histórico y lista de riesgos, no como fuente de estado actual. La
skill local `tpscouting-correction-report` contiene un método y una plantilla
compatibles, pero fue creada para reproducir este tipo de diagnóstico; tampoco
prueba cuál fue el prompt original.
