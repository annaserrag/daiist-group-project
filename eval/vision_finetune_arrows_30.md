# Opciones con flecha en los 30 PDFs

Se volvieron a leer los 30 PDFs con el lector de Anna en modo página (`8086331`). Se verificó el SHA-256 de cada PDF contra el paquete y se evaluaron los adaptadores `bridge-v1` y `bridge-v2` sobre la misma salida de lectura. Para desarrollo, las dos pasadas compartieron las respuestas de Gemini y cualquier fallo del servicio. Las notas de referencia se consultaron para puntuar los resultados. [Referencias del paquete](case_references.json) · [Primera evaluación](vision_finetune_30.md)

## Cambios atribuibles al adaptador

El adaptador acepta el símbolo → y una única letra permitida en esa pregunta de test, con espacios opcionales. Conserva las confianzas originales del lector. Las 23 preguntas seleccionadas salieron de revisión y las 23 notas coincidieron con la referencia de test. El resto de las 1.050 decisiones del corrector (30 salidas de 35 preguntas) se cotejó por nota, motivo y error de ejecución. Solo cambiaron esas 23. Los porcentajes de revisión usan las 105 preguntas presentes en el paquete.

| Métrica | Adaptador anterior | Adaptador nuevo |
|---|---:|---:|
| Casos que cumplen todo | 10/30 | 11/30 |
| Transcripciones exactas | 51/105 | 74/105 |
| Notas propuestas con referencia | 28/100 | 51/100 |
| Notas exactas con referencia | 28/100 | 51/100 |
| Notas propuestas discordantes | 0 | 0 |
| Preguntas a revisión | 77/105 | 54/105 |

## Notas de las 23 preguntas que salieron de revisión

| PDF | Preguntas | Nota propuesta y de referencia, en orden |
|---|---|---|
| c04 | Q17–Q20 | 0, 1, 1, 0 |
| c16 | Q1–Q19 | 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1 |

En c16 Q20, el texto de la opción incluye el encabezado de la página siguiente y la confianza de lectura es baja. La pregunta sigue en revisión.

## Comparación con la primera evaluación

La primera pasada produjo 28 notas exactas sobre 100 referencias, 77 revisiones sobre 105 preguntas y 10 PDFs que cumplían todos los criterios. Las tablas de esta prueba comparan también ambas versiones en las condiciones de la nueva ejecución.
Fuera de las 23 flechas, se reprodujeron todas las decisiones de proponer nota o revisar de la primera evaluación.

## Resultados por PDF

| PDF | Pasaba en el primer informe | Pasa con v1 ahora | Pasa con v2 ahora | Revisión v1 → v2 | Notas exactas v1 → v2 |
|---|---|---|---|---:|---:|
| c01 | Sí | Sí | Sí | 0 → 0 | 5 → 5 |
| c02 | Sí | Sí | Sí | 0 → 0 | 5 → 5 |
| c03 | Sí | Sí | Sí | 0 → 0 | 6 → 6 |
| c04 | No | No | Sí | 4 → 0 | 0 → 4 |
| c05 | No | No | No | 1 → 1 | 0 → 0 |
| c06 | No | No | No | 1 → 1 | 3 → 3 |
| c07 | No | No | No | 1 → 1 | 0 → 0 |
| c08 | No | No | No | 1 → 1 | 0 → 0 |
| c09 | No | No | No | 1 → 1 | 0 → 0 |
| c10 | No | No | No | 1 → 1 | 0 → 0 |
| c11 | No | No | No | 1 → 1 | 0 → 0 |
| c12 | No | No | No | 1 → 1 | 1 → 1 |
| c13 | No | No | No | 20 → 20 | 0 → 0 |
| c14 | Sí | Sí | Sí | 0 → 0 | 5 → 5 |
| c15 | No | No | No | 4 → 4 | 0 → 0 |
| c16 | No | No | No | 31 → 12 | 0 → 19 |
| c17 | No | No | No | 1 → 1 | 0 → 0 |
| c18 | No | No | No | 1 → 1 | 0 → 0 |
| c19 | No | No | No | 1 → 1 | 0 → 0 |
| c20 | No | No | No | 1 → 1 | 0 → 0 |
| c21 | Sí | Sí | Sí | 1 → 1 | 0 → 0 |
| c22 | Sí | Sí | Sí | 1 → 1 | 1 → 1 |
| c23 | Sí | Sí | Sí | 1 → 1 | 1 → 1 |
| c24 | Sí | Sí | Sí | 1 → 1 | 1 → 1 |
| c25 | No | No | No | 1 → 1 | 0 → 0 |
| c26 | Sí | Sí | Sí | 1 → 1 | 0 → 0 |
| c27 | Sí | Sí | Sí | 0 → 0 | 0 → 0 |
| c28 | No | No | No | 0 → 0 | 0 → 0 |
| c29 | No | No | No | 0 → 0 | 0 → 0 |
| c30 | No | No | No | 0 → 0 | 0 → 0 |
