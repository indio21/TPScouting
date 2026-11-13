# Comparación de la versión anterior y la corregida — 12/11/2026

Se compararon nuevamente ambos Word y ambos PDF, sin guardar modificaciones
en ellos. También se contrastó el párrafo de autenticación con el código actual.

## Diferencias comprobadas

1. Eliminación de la leyenda huérfana y su entrada en la lista de tablas.
2. Renumeración de las tablas 4-3 a 4-8 como 4-2 a 4-7 y de su referencia en
   el párrafo sobre entidades. El resto del texto de ese párrafo no cambió.
3. Ampliación de un único párrafo de §4.2.4, manteniendo sus frases originales:
   cinco fallos/ventana móvil de 15 minutos, HTTP429 antes de comprobar nuevas
   credenciales, posible restricción provocada por terceros y memoria por proceso.
4. Corrección de una única celda: propósito de /health en la actual Tabla4-5.
5. Retirada de una repetición de la URL en el comando Linux/macOS de §9.4.

Las diferencias de numeración derivan de la primera observación del profesor;
no constituyen un cambio adicional de contenido.

## Conservación y coherencia

- La comparación íntegra de párrafos confirma que no hay cambios de contenido
  ajenos a los autorizados y a la renumeración.
- Solo una celda difiere entre las 19 tablas: /health. Las tablas de métricas,
  configuración ML, resultados y evidencia conservan todos sus valores.
- Las 28 imágenes conservan sus bytes, sin sustituciones ni regeneración.
- Los estilos, tema, numeración de listas, encabezados y pies son idénticos
  byte a byte. Se conservan Arial, tamaños, geometría y nueve secciones.
- Se comprobaron 107 apariciones de referencias/leyendas de tablas y figuras:
  todas tienen una leyenda existente. Hay 19 tablas y 19 leyendas asociadas.
- Índice: 88 entradas con páginas correctas. Listas: 44 entradas con páginas
  correctas y números alineados.
- Ambos PDF tienen 97 páginas. En 86 páginas el texto extraído, excluido el pie,
  es idéntico; las restantes son la lista de tablas, páginas del capítulo4
  afectadas por renumeración/reflujo y la página del comando corregido en §9.4.
- No hay cambios en portada, resumen, abstract, resultados, conclusiones,
  bibliografía ni anexos gráficos.

El párrafo agregado describe el limitador que ya existe: clave por usuario
normalizado y comprobación del límite antes de validar contraseña. No introduce
un bloqueo persistente en base ni una función nueva, y es consistente con la
advertencia previa sobre memoria local y ausencia de limitador distribuido.
La fila /health describe la consulta SELECT1 real, sin atribuir auditoría de datos.

Los cambios internos adicionales del paquete DOCX corresponden a relaciones del
hyperlink eliminado, ajustes de Word y metadatos de guardado; estilos y recursos
gráficos permanecen intactos.

Evidencia: comparacion_independiente_word_pdf.json, verificacion_integridad.json,
control_formato.json y control_formato_conservado.json.
El original y el repositorio de entrega permanecen intactos. No se ejecutó una
nueva suite de aplicación ni un despliegue: esta fue una comparación documental.

Resultado: no se detectaron diferencias no autorizadas ni contradicciones nuevas
en las cuatro correcciones revisadas.

LISTO PARA REVISIÓN FINAL HUMANA
