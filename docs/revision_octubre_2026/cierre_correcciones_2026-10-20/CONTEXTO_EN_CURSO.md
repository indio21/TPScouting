# Cierre de correcciones de autenticación y Arial — 20/10/2026

Este registro describe el trabajo intermedio. El cierre posterior vigente está
en `../CONTEXTO_FINAL_ARIAL_2026-10-20.md` y en `CIERRE.md`.

El usuario autorizó aplicar las correcciones encontradas, usar Arial en todo el
texto editable y actualizar pruebas, commits y documentos.

## Completado

- SEG-09: login rechaza credenciales ausentes o vacías con HTTP 400; la sesión
  de base de datos se cierra también si falla la comprobación de contraseña.
- SEG-10: CSRF rechaza tokens no ASCII sin provocar TypeError/HTTP 500.
- Ocho casos nuevos de regresión; autenticación/seguridad: 37 aprobados.
- Suite completa local: 131 passed, 1 skipped, 4 warnings; cobertura 83,31 %.
- Ruff crítico/gradual y pip check aprobados.
- Demo nueva con 60 jugadores sintéticos en SQLite temporal: login, nacimiento,
  edad/categoría, historiales, silueta y predicción aprobados.
- Los modelos y metadatos históricos permanecen intactos.
- Commit funcional de respaldo publicado:
  `8809760cd20d804927d3604c160aee2459506b3d`.
- Commit funcional de entrega publicado:
  `2a27d53ae4f8a4849bee7d1bea362fd9146886b5`.
- Container `39509728476`: success.

## Punto de continuación

1. Consultar CI `39509728373` y macOS `39509728473` del commit funcional de
   entrega. No declarar éxito antes de comprobarlo.
2. Actualizar `functional_evidence.json` con los resultados reales.
3. Ejecutar `scripts/update_thesis_auth_arial_2026_10_20.py` cuando CI y contenedor
   hayan aprobado. Lee los resultados locales de pytest/cobertura, cita el commit
   funcional y conserva como históricas las capturas CI anteriores.
4. Word nuevo previsto:
   `TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20.docx` y PDF equivalente
   en esta carpeta. Fuente inmutable: la entrega del 20/10 previa, hash
   `2EA4A6EA4C99617CB3EDA36F8C9218E28D4D9B479D812F1240F81347569AAD16`.
5. Verificar Arial, jerarquía de títulos, mínimo 10 pt, imágenes intactas,
   índices/listas, numeración y revisión visual dirigida. Resolver únicamente
   problemas concretos de maquetación antes de publicar.
6. Sustituir selectivamente Word/PDF en entrega, actualizar enlaces README y
   agregar VERIFICACION_ENTREGA.md con hashes y referencia funcional.
7. Registrar fuera del Word el commit exacto que publique ese paquete; el Word
   no puede incorporar anticipadamente su propio hash futuro de Git.
8. Guardar y publicar el respaldo con contexto final actualizado.

No agregar archivos internos ni auditorías al repositorio de entrega. Preservar
los archivos antiguos no seguidos por Git. No usar `git add .` ni borrar en masa.
