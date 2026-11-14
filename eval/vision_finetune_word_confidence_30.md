# Relectura de palabras dudosas en los 30 PDFs

Dos de las cinco respuestas de desarrollo salieron de revisión y recibieron 2/2, de acuerdo con el profesor. Una tercera superó el control de lectura y fue a revisión por discrepancia en la corrección. Las otras dos conservaron el motivo de revisión original. Ningún PDF perdió su resultado de aceptación y ninguna nota correcta se perdió.

## Qué generaba la duda

El número indica la puntuación mínima observada en la primera lectura. Los fragmentos entre comillas se transcriben tal como los produjo el lector. Los cinco casos tienen una nota de referencia de 2/2.

| PDF y pregunta | Fragmento dudoso | Qué ocurrió al releer | Resultado desde PDF |
|---|---|---|---|
| c06 Q21 | Punto final de «labeled.» (0,591) | Se repitió la frase completa; la confianza del texto subió a 0,963. | 2/2, exacta |
| c11 Q31 | «k» inicial de «k-means» (0,663) | Se repitió la respuesta completa. La confianza pasó a 0,940. | 2/2, exacta |
| c16 Q26 | «Struct» de «Structured» (0,738) | Ambas lecturas coincidieron en las dos líneas; puntuación final de 0,907. | Revisión por discrepancia entre las comprobaciones de nota. |
| c16 Q29 | Guion en «k-NN» (0,542) | La segunda lectura produjo «K-NN» y esa palabra obtuvo 0,676. Se conservó la puntuación original. | Revisión de lectura |
| c16 Q31 | «k» (0,571) y «number» (0,617) | La primera lectura decía «number number» e incorporó un encabezado de la página siguiente. La inspección de la página de inicio leyó «number» una sola vez. El adaptador excluye del reintento automático las respuestas que abarcan más de una página. | Revisión de lectura |

## Regla usada

Para una respuesta escrita de una sola página, con número único y confianza de identificación suficiente, el lector vuelve a mirar la imagen a otra escala. Recibe el número de pregunta y una instrucción de transcripción literal; no recibe la primera lectura, la solución ni la nota. Se exige coincidencia de todas las palabras en orden, incluidas mayúsculas y puntuación. Para cada palabra se toma su mejor puntuación completa entre las dos lecturas y, para la respuesta, la menor de esas puntuaciones. Se mantiene el umbral de 0,80 y el texto original. Son dos lecturas del mismo modelo; la puntuación sirve para decidir cuándo pedir revisión y no es una probabilidad calibrada.

## Comparación de los 30 PDFs

Se comprobaron las huellas SHA-256 de los 30 PDFs y se repitió la lectura con el modelo de Anna en modo página, commit `8086331`. Se pasó la salida de cada PDF por los adaptadores `bridge-v4` y `bridge-v5` y por el corrector habitual. Las peticiones idénticas a Gemini compartieron respuestas o fallos; las preguntas que solo alcanzaron el corrector en la versión nueva se evaluaron normalmente. Se contrastaron 1.050 decisiones por nota, motivo y fallo del proveedor. Las métricas de respuesta usan las 105 preguntas presentes. Las referencias escritas son notas del profesor; las de test se deducen de la opción registrada y la clave. El paquete contiene recortes repetidos y controles construidos. [Referencias por caso](case_references.json) · [Evaluación anterior](vision_finetune_id_confidence_30.md)

| Métrica | Anterior | Nueva |
|---|---:|---:|
| Respuestas a revisión, de 105 | 47 | 45 |
| Notas propuestas con referencia, de 100 | 57 | 59 |
| Notas coincidentes con referencia | 57 | 59 |
| Transcripciones exactas, de 105 | 77 | 77 |
| PDFs que cumplen todos los criterios, de 30 | 13 | 15 |
| Decisiones con fallo de proveedor | 0 | 0 |

En las 24 respuestas manuscritas de desarrollo, las revisiones bajaron de 20 a 18. Las seis propuestas finales incluyen cinco notas de 2/2 que coinciden con el profesor y una propuesta previa de 2/2 en c16 Q28 sin nota publicada.

| PDF | Revisiones antes → después | Notas exactas comparables antes → después | Cumple los criterios antes → después |
|---|---:|---:|---|
| c01 | 0 → 0 | 5 → 5 | Sí → Sí |
| c02 | 0 → 0 | 5 → 5 | Sí → Sí |
| c03 | 0 → 0 | 6 → 6 | Sí → Sí |
| c04 | 0 → 0 | 4 → 4 | Sí → Sí |
| c05 | 1 → 1 | 0 → 0 | No → No |
| c06 | 1 → 0 | 3 → 4 | No → Sí |
| c07 | 0 → 0 | 1 → 1 | Sí → Sí |
| c08 | 0 → 0 | 1 → 1 | Sí → Sí |
| c09 | 1 → 1 | 0 → 0 | No → No |
| c10 | 1 → 1 | 0 → 0 | No → No |
| c11 | 1 → 0 | 0 → 1 | No → Sí |
| c12 | 1 → 1 | 1 → 1 | No → No |
| c13 | 20 → 20 | 0 → 0 | No → No |
| c14 | 0 → 0 | 5 → 5 | Sí → Sí |
| c15 | 1 → 1 | 3 → 3 | No → No |
| c16 | 10 → 10 | 20 → 20 | No → No |
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
