# Correcciones menores del profesor — 12/11/2026

## Alcance autorizado

El usuario pidió tomar el último Word, conservarlo como respaldo y generar uno
nuevo con solo cuatro correcciones. Eligió retirar la leyenda huérfana, sin
agregar una tabla. Solicitó aplicar primero las correcciones y luego renumerar,
conservando tipografía, color, formato e información restante. La nueva copia
fue revisada y aprobada por el usuario, que luego autorizó su publicación.

Durante la corrección no se modificó código. Tras la aprobación se publicó
la entrega documental en `84b6e0a30a269c6f6cb2ac6c1c1ceef148347793`.
Leer primero `../PUNTO_GUARDADO_NOVIEMBRE_2026-11-12.md` para el estado de cierre.

## Archivos

Base anterior aprobada, preservada como respaldo:
`Trabajo Final_Solari Pablo_21-10-2026.docx`, SHA-256
`81A743DF38FB32820EA7DC3BEC247F57CCE6F3F826510DFEA2ED9CF3D63A131C`.

Respaldo adicional byte a byte en esta carpeta:
`BACKUP_Trabajo Final_Solari Pablo_21-10-2026.docx`.

Nueva versión aprobada y publicada:
`Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_12-11-2026.docx`.
SHA-256: `C6BEB90582E818D371AC9CF96EBAB0C1A1DA025C6DE689DC2383EB606DBFDC79`.

PDF correspondiente publicado con el mismo nombre:
SHA-256: `E43883FC89601B5E14B7144BA885BD6454C84C2FBCF0CC139B9DBC5621EAE5F0`.

## Matriz de las cuatro observaciones

| ID | Observación original | Corrección realizada | Evidencia y estado |
|---|---|---|---|
| NOV-01 | Tabla 4-2 huérfana | Retirada la leyenda «Comparación entre motores relacionales» y su entrada de lista. Después se renumeraron 13 apariciones de tablas/referencias del capítulo 4. | 19 tablas reales y 19 leyendas, cada tabla con una leyenda adyacente. Secuencia 4-1 a 4-7. RESUELTO. |
| NOV-02 | La fila /health de Tabla 4-6 atribuye control de calidad de datos | Propósito corregido: «Comprobación mínima de disponibilidad del servicio y conectividad con la base de datos». Tras renumerar está en Tabla 4-5. | Coherente con app.py: health ejecuta SELECT 1 y devuelve status mínimo. Es la única celda modificada del documento. RESUELTO. |
| NOV-03 | URL repetida en §9.4 | Retirado el hyperlink duplicado que seguía a la URL ya escrita en el comando Linux/macOS. Conservado el texto negro del comando y una sola URL. | Comando con --base-url admitido por scripts/smoke_render.py; comprobación textual y visual. No se ejecutó un smoke remoto nuevo. RESUELTO. |
| NOV-04 | Omisión del bloqueo por cuota de intentos | Ampliado únicamente el párrafo de §4.2.4: cinco fallos en ventana móvil de 15 minutos, siguientes intentos con HTTP 429 aunque las credenciales sean correctas; posible bloqueo temporal provocado por terceros y registro en memoria por proceso. | Contrastado con defaults 5/900 de app.py, orden de validación en routes/auth.py y agrupación por usuario de services/security.py. RESUELTO como corrección documental. |

La descripción no introduce un bloqueo persistente de usuario ni implica
revocación de sesiones existentes. Describe la restricción de nuevos logins.
Los títulos de capítulos y subtítulos no cambiaron. La antigua Tabla 4-6 de
endpoints es ahora Tabla 4-5; la nueva Tabla 4-2 es la de dimensiones deportivas.

## Verificaciones nuevas

- Aplicación de cambios mediante edición de nodos de texto OOXML; antes de la
  actualización de Word solo cambió word/document.xml.
- Comparación íntegra de los párrafos desde Resumen: únicamente las cuatro
  correcciones y renumeraciones autorizadas.
- Comparación de todas las celdas: solo cambió el propósito de /health.
- Imágenes byte a byte idénticas, sin regeneración ni análisis exhaustivo.
- styles.xml, theme1.xml, numbering.xml y todos los encabezados/pies idénticos
  byte a byte al original, incluso después de guardar con Microsoft Word.
- Estilos de párrafo, títulos de capítulos y geometría de secciones conservados.
- Fuentes Word y PDF: Arial y sus variantes originales; ningún tamaño explícito
  menor de 10 pt. Se conservó el color original sin marcado rojo.
- 97 páginas, 28 imágenes, 19 tablas y 9 secciones.
- Índice general: 88 entradas con páginas reales correctas.
- Listas: 44 entradas (25 figuras y 19 tablas), con páginas correctas y números
  alineados en la misma coordenada derecha.
- Sin campos rotos, numeración discontinua, desbordamientos ni leyendas de
  figuras sin imagen. Preliminares y bibliografía mantienen inicio separado.
- Control visual dirigido de páginas físicas 8, 31, 32, 36, 39 y 84.
- Ruff crítico aprobado para el script de corrección.

Evidencias: verificacion_integridad.json, control_formato.json,
control_formato_conservado.json y muestras de las páginas modificadas.
No se repitió la suite de aplicación: la intervención fue solo documental.

## Para continuar

Se amplió luego la comprobación a numeración de capítulos/subtítulos,
secuencias de leyendas, todas las tablas/figuras y formato de URL. Resultado:
84 títulos numerados sin saltos ni duplicaciones; 19 tablas y 25 figuras con
leyendas asociadas; 88 destinos internos del índice existentes; 23 apariciones
de URL sin concatenaciones ni duplicaciones accidentales. La Figura 6-5 tiene
tres imágenes con rótulos a), b), c), no una leyenda huérfana. Véase
auditoria_numeracion_leyendas_enlaces.json. No se modificó el Word/PDF ni se
comprobó la disponibilidad actual de los sitios externos.

El usuario pidió después una comparación independiente anterior/corregido.
Se completó sin modificar los Word/PDF: ver COMPARACION_ANTES_DESPUES.md y
comparacion_independiente_word_pdf.json. Se comprobaron 107 referencias, una sola
celda distinta, gráficos y formato preservados, y coherencia del agregado con
el limitador existente. No se detectaron cambios ajenos a lo autorizado.

El usuario aprobó ambos archivos y autorizó su publicación y el commit de cierre.
Los originales y las evidencias quedan en el respaldo; la entrega contiene solo
la nueva versión y su identificación pública. Esperar nuevas indicaciones.
La CI de publicación informó diez avisos en dos paquetes y falló su auditoría;
ver `alertas_dependencias_ci.json` y el punto de guardado de noviembre. No
confundir el cierre documental con una aprobación de dependencias actual.
El contexto histórico completo permanece en docs/revision_octubre_2026/.

LISTO PARA REVISIÓN FINAL HUMANA
