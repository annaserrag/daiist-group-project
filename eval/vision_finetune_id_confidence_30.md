# Confianza del número de pregunta en 30 PDFs

Las revisiones bajaron de 50 a 47. Se recuperaron tres respuestas. Dos tienen nota de referencia y ambas coinciden. En las otras cinco decisiones que cambiaron, se confirmó el número pero la confianza del texto siguió siendo insuficiente. Ningún PDF perdió su resultado de aceptación ni ninguna respuesta perdió una nota correcta.

## Cómo se midió

Se comprobaron las huellas SHA-256 de los 30 PDFs y se usó el lector de Anna en modo página, commit `8086331`. El control usa el adaptador anterior (`bridge-v3`); la nueva versión (`bridge-v4`) conserva las puntuaciones de los tokens durante la misma lectura. Los dos adaptadores recibieron las mismas observaciones y el mismo resultado de Gemini para cada petición idéntica. Se usaron los criterios del evaluador del proyecto y se compararon 1.050 decisiones por nota, motivo y fallo de ejecución. Solo las 105 preguntas presentes se incluyen en las tasas por respuesta. [Referencias por caso](case_references.json) · [Informe anterior](vision_finetune_option_text_30.md)

Anna aplicaba el mínimo de la línea inicial tanto al número como al texto. La nueva versión mide los dígitos y la respuesta por separado. Para elevar un ID que estaba por debajo de 0,85 se pide una segunda lectura. Con una sola etiqueta se exige un recorte localizado por el detector de preguntas. Con varias etiquetas se comprueban su cantidad, su posición y sus dígitos. Se conserva el límite de 0,80 para el contenido. La segunda lectura reutiliza el modelo y puede repetir errores; las puntuaciones son señales de decisión sin calibración probabilística.

## Respuestas recuperadas

| PDF | Pregunta | ID anterior → confirmado | Confianza usada para la respuesta | Propuesta | Referencia |
|---|---:|---:|---:|---:|---:|
| c15 | Q4 | 0,588 → 1,000 | 1,000 en la letra B, cuyo texto «regression» concuerda con la opción | 0/1 | 0/1 |
| c08 | Q25 | 0,804 → 0,996 | 0,804 en el texto | 2/2 | 2/2 |
| c16 | Q28 | 0,830 → 1,000 | 0,830 en el texto | 2/2 | Sin nota publicada |

En c06 Q21, c11 Q31 y c16 Q26, Q29 y Q31 se confirmó el número. Los cinco textos siguieron por debajo de 0,80. En c17, Q31 se transcribió como Q34; incluso una segunda lectura de la página completa repitió Q34 con confianza alta. Al exigir un recorte para un número aislado no se obtuvo confirmación. En c25, Q22 siguió apareciendo como Q26 con confianza insuficiente. Ambas detecciones erróneas permanecieron en revisión tanto por su número como en la decisión final.

## Comparación sobre la misma ejecución

| Métrica | Anterior | Nueva |
|---|---:|---:|
| Preguntas a revisión, de 105 | 50 | 47 |
| Notas propuestas con referencia, de 100 | 55 | 57 |
| Notas coincidentes con referencia | 55 | 57 |
| Transcripciones exactas, de 105 | 77 | 77 |
| PDFs que cumplen todos los criterios, de 30 | 12 | 13 |
| Decisiones con fallo de proveedor | 0 | 0 |

De las 24 respuestas manuscritas de desarrollo, las detenidas por el número bajaron de 14 a 7. El total de revisiones de desarrollo pasó de 22 a 20.

El informe anterior registró 52 revisiones y dos fallos del proveedor. Al repetir la versión anterior, c07 Q21 y c16 Q21 recibieron 2/2, de acuerdo con el profesor. En estas pasadas no hubo fallos del proveedor. Se atribuye al adaptador únicamente el cambio de 50 a 47 revisiones; c08 fue el único PDF que pasó a cumplir todos sus criterios.

## Resultado por PDF

| PDF | Revisiones antes → después | Notas exactas comparables antes → después | Cumple los criterios antes → después |
|---|---:|---:|---|
| c01 | 0 → 0 | 5 → 5 | Sí → Sí |
| c02 | 0 → 0 | 5 → 5 | Sí → Sí |
| c03 | 0 → 0 | 6 → 6 | Sí → Sí |
| c04 | 0 → 0 | 4 → 4 | Sí → Sí |
| c05 | 1 → 1 | 0 → 0 | No → No |
| c06 | 1 → 1 | 3 → 3 | No → No |
| c07 | 0 → 0 | 1 → 1 | Sí → Sí |
| c08 | 1 → 0 | 0 → 1 | No → Sí |
| c09 | 1 → 1 | 0 → 0 | No → No |
| c10 | 1 → 1 | 0 → 0 | No → No |
| c11 | 1 → 1 | 0 → 0 | No → No |
| c12 | 1 → 1 | 1 → 1 | No → No |
| c13 | 20 → 20 | 0 → 0 | No → No |
| c14 | 0 → 0 | 5 → 5 | Sí → Sí |
| c15 | 2 → 1 | 2 → 3 | No → No |
| c16 | 11 → 10 | 20 → 20 | No → No |
| c17 | 1 → 1 | 0 → 0 | No → No |
| c18 | 1 → 1 | 0 → 0 | No → No |
| c19 | 1 → 1 | 0 → 0 | No → No |
| c20 | 1 → 1 | 0 → 0 | No → No |
| c21 | 1 → 1 | 0 → 0 | Sí → Sí |
| c22 | 1 → 1 | 1 → 1 | Sí → Sí |
| c23 | 1 → 1 | 1 → 1 | Sí → Sí |
| c24 | 1 → 1 | 1 → 1 | Sí → Sí |
| c25 | 1 → 1 | 0 → 0 | No → No |
| c26 | 1 → 1 | 0 → 0 | Sí → Sí |
| c27 | 0 → 0 | 0 → 0 | Sí → Sí |
| c28 | 0 → 0 | 0 → 0 | No → No |
| c29 | 0 → 0 | 0 → 0 | No → No |
| c30 | 0 → 0 | 0 → 0 | No → No |
