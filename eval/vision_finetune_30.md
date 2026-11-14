# Evaluación conjunta de los 30 PDFs

Lector de Anna en modo página, commit `8086331`, adaptador local `bridge-v1` y corrector en `f4329ef`. Prueba sobre 30 archivos verificados por SHA-256. Las cuatro pruebas anteriores se repitieron. El modelo de lectura, los umbrales y la rúbrica se mantuvieron fijos durante la medición.

Las 21 hojas manuscritas proceden de ocho exámenes del mismo conjunto. Hay recortes repetidos, un PDF usado para desarrollar el adaptador y nueve controles construidos por el equipo. Cada unidad de la tabla es una pregunta en un PDF, incluso cuando la hoja contiene dos lecturas del mismo número.

Las notas de desarrollo proceden del CSV del profesor. Las de test se calculan con la opción registrada y la clave publicada. Los controles se comparan con el comportamiento que fijó el equipo. Se verificaron las referencias reales contra el archivo publicado con SHA-256 `51e194a98970dddf738dd81bdbc51e2d7dea5637666ee1ff7402e7165154b017`. [Referencias del paquete](case_references.json) · [Fuente de notas y clave](https://doi.org/10.17632/sf3kvjwknt.1)

## Resumen de respuestas

| Grupo | Preguntas objetivo | Con nota de referencia | Notas comparables propuestas | Notas exactas | A revisión |
|---|---:|---:|---:|---:|---:|
| Test manuscrito | 72 | 72 | 24 | 24 | 48 |
| Desarrollo manuscrito | 24 | 23 | 0 | 0 | 24 |
| Controles construidos | 9 | 5 | 4 | 4 | 5 |
| Total | 105 | 100 | 28 | 28 | 77 |

El control repite el corrector con la transcripción, el estado y la identificación verificados, confianza 1 y la misma política. Solo recibe los textos y estados de referencia. Las notas objetivo se consultan después. Las respuestas del proveedor se reutilizan por huella exacta cuando están en caché. Un fallo del servicio deja indeterminado qué nota habría propuesto. La coincidencia de texto ignora mayúsculas y espacios. Puede fallar por puntuación aun cuando la nota sea correcta.

## Atribución de los fallos

Pasaron 10 de 30 PDFs con el lector y 20 de 30 con la entrada verificada. La lectura coincidió en texto, número de observaciones y estado para 51 de 105 preguntas objetivo. El sistema propuso 28 de las 100 notas comparables y las 28 coincidieron. Las 77 revisiones incluyen 72 preguntas con referencia numérica y cinco sin ella.

- **Adaptador de opciones.** En c04, Q17–Q20, y c16, Q1–Q19, el lector entregó una flecha seguida de la letra correcta en las 23 ocasiones. El adaptador conservó la flecha y el corrector rechazó el formato. En c15, Q1–Q3 llevaban la letra seguida del texto de la opción. En c16, Q20 llevaba además el encabezado de la página siguiente y cayó por confianza baja.
- **Asignación de números.** En c13 quedaron ocho observaciones sueltas. El lector interpretó varios números como 41, 51 o 61, y la política bloqueó las 20 preguntas. En c17 leyó Q31 como Q34, y en c25, Q22 como Q26. Las seis detecciones inesperadas del paquete quedaron sin nota.
- **Desarrollo manuscrito.** De sus 24 revisiones, 14 se debieron al umbral de confianza en el número, dos a falta de detección, una a confianza baja en el texto, tres a fallos del proveedor, tres a la política de revisión de notas bajas y una a la figura ausente. El modo página reutiliza la confianza de toda la línea para el número; esta señal puede ser baja con el número correcto.
- **Corrector con la entrada verificada.** Propuso cinco notas de desarrollo comparables. Acertó cuatro y dio 2/2 en c19 Q28 frente al 1,5/2 del profesor. Propuso además 2/2 en c16 Q28, que carece de referencia numérica. Quedaron 18 revisiones en desarrollo, 13 por fallo del proveedor, dos por falta de evidencia, dos por depender de una figura y una por la política de notas bajas. En la pasada desde PDF hubo fallos del proveedor en c07 Q21 y c16 Q21 y Q25.
- **Tipo de documento.** c27 fue identificado como ilegible. c28 y c29, documentos administrativos, también recibieron ese estado. c30, una hoja de geografía, se clasificó como examen válido. Los tres archivos ajenos quedaron sin nota. El adaptador infiere el estado a partir de las detecciones y carece de un verificador de identidad del examen.

En un reintento diagnóstico aislado se usaron las mismas observaciones para las tres respuestas que fallaron por el proveedor. c16 Q21 obtuvo 2/2 y coincidió con el profesor. c07 Q21 y c16 Q25 volvieron a quedar sin nota por indisponibilidad. Las tablas conservan los resultados de la pasada de los PDFs completos.

## Resultados por PDF

El caso pasa cuando se cumplen los objetivos del manifiesto y la estructura de lectura es correcta, sin detecciones inesperadas ni líneas sin asignar. Una nota histórica puede coexistir con una revisión esperada. Las casillas de revisión cuentan solo preguntas presentes en la referencia.

| PDF | Caso pasa | Pasaría con lectura verificada | Lecturas exactas / objetivo | Notas exactas / propuestas / referencias numéricas | A revisión | Estado observado / esperado | Líneas sueltas |
|---|---|---|---|---|---:|---|---:|
| c01 | Sí | Sí | 5/5 | 5/5/5 | 0 | in_scope / in_scope | 0 |
| c02 | Sí | Sí | 5/5 | 5/5/5 | 0 | in_scope / in_scope | 0 |
| c03 | Sí | Sí | 6/6 | 6/6/6 | 0 | in_scope / in_scope | 0 |
| c04 | No | Sí | 0/4 | 0/0/4 | 4 | in_scope / in_scope | 0 |
| c05 | No | Sí | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c06 | No | No | 4/4 | 3/3/4 | 1 | in_scope / in_scope | 0 |
| c07 | No | No | 1/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c08 | No | No | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c09 | No | Sí | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c10 | No | No | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c11 | No | No | 1/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c12 | No | Sí | 1/2 | 1/1/2 | 1 | in_scope / in_scope | 0 |
| c13 | No | Sí | 10/20 | 0/0/20 | 20 | in_scope / in_scope | 8 |
| c14 | Sí | Sí | 5/5 | 5/5/5 | 0 | in_scope / in_scope | 0 |
| c15 | No | Sí | 0/4 | 0/0/4 | 4 | in_scope / in_scope | 0 |
| c16 | No | No | 5/31 | 0/0/30 | 31 | in_scope / in_scope | 0 |
| c17 | No | Sí | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c18 | No | No | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c19 | No | No | 1/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c20 | No | No | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c21 | Sí | Sí | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c22 | Sí | Sí | 2/2 | 1/1/1 | 1 | in_scope / in_scope | 0 |
| c23 | Sí | Sí | 2/2 | 1/1/1 | 1 | in_scope / in_scope | 0 |
| c24 | Sí | Sí | 2/2 | 1/1/1 | 1 | in_scope / in_scope | 0 |
| c25 | No | No | 0/1 | 0/0/1 | 1 | in_scope / in_scope | 0 |
| c26 | Sí | Sí | 1/1 | 0/0/0 | 1 | in_scope / in_scope | 0 |
| c27 | Sí | Sí | 0/0 | 0/0/0 | 0 | unreadable / unreadable | 0 |
| c28 | No | Sí | 0/0 | 0/0/0 | 0 | unreadable / out_of_scope | 4 |
| c29 | No | Sí | 0/0 | 0/0/0 | 0 | unreadable / out_of_scope | 4 |
| c30 | No | Sí | 0/0 | 0/0/0 | 0 | in_scope / out_of_scope | 1 |

## Detalle por respuesta

Las notas de test son sobre 1 y las de desarrollo sobre 2. La columna de control muestra qué decidió el mismo corrector con la entrada verificada. Las detecciones inesperadas se incluyen para auditar posibles notas espurias y quedan fuera de las tasas por pregunta objetivo.

| PDF | Pregunta | Referencia | Propuesta desde PDF | Con lectura verificada | Lectura del PDF | Interpretación |
|---|---:|---|---|---|---|---|
| c01 | 1 | 0 | 0 | 0 | exacta | Nota exacta |
| c01 | 2 | 1 | 1 | 1 | exacta | Nota exacta |
| c01 | 3 | 0 | 0 | 0 | exacta | Nota exacta |
| c01 | 4 | 1 | 1 | 1 | exacta | Nota exacta |
| c01 | 5 | 1 | 1 | 1 | exacta | Nota exacta |
| c02 | 6 | 0 | 0 | 0 | exacta | Nota exacta |
| c02 | 7 | 0 | 0 | 0 | exacta | Nota exacta |
| c02 | 8 | 0 | 0 | 0 | exacta | Nota exacta |
| c02 | 9 | 0 | 0 | 0 | exacta | Nota exacta |
| c02 | 10 | 1 | 1 | 1 | exacta | Nota exacta |
| c03 | 11 | 1 | 1 | 1 | exacta | Nota exacta |
| c03 | 12 | 1 | 1 | 1 | exacta | Nota exacta |
| c03 | 13 | 1 | 1 | 1 | exacta | Nota exacta |
| c03 | 14 | 1 | 1 | 1 | exacta | Nota exacta |
| c03 | 15 | 1 | 1 | 1 | exacta | Nota exacta |
| c03 | 16 | 1 | 1 | 1 | exacta | Nota exacta |
| c04 | 17 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c04 | 18 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c04 | 19 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c04 | 20 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c05 | 31 | 2 | Rev. · confianza baja en el número | 2 | texto distinto | Lectura/asignación o puerta de confianza |
| c06 | 18 | 1 | 1 | 1 | exacta | Nota exacta |
| c06 | 19 | 0 | 0 | 0 | exacta | Nota exacta |
| c06 | 20 | 1 | 1 | 1 | exacta | Nota exacta |
| c06 | 21 | 2 | Rev. · confianza baja en el número | Rev. · proveedor no disponible | exacta | Lectura/asignación o puerta de confianza |
| c07 | 21 | 2 | Rev. · fallo del proveedor | Rev. · fallo del proveedor | exacta | Ejecución del corrector |
| c08 | 25 | 2 | Rev. · confianza baja en el número | Rev. · proveedor no disponible | texto distinto | Lectura/asignación o puerta de confianza |
| c09 | 23 | 2 | Rev. · confianza baja en el número | 2 | texto distinto | Lectura/asignación o puerta de confianza |
| c10 | 25 | 2 | Rev. · confianza baja en el número | Rev. · evidencia insuficiente | texto distinto | Lectura/asignación o puerta de confianza |
| c11 | 31 | 2 | Rev. · confianza baja en el número | Rev. · proveedor no disponible | exacta | Lectura/asignación o puerta de confianza |
| c12 | 1 | 1 | 1 | 1 | exacta | Nota exacta |
| c12 | 2 | 0 | Rev. · opción o formato no admitido | 0 | estado o nº de lecturas distinto | Lectura/asignación o puerta de confianza |
| c13 | 1 | 0 | Rev. · línea sin asignar en la hoja | 0 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 2 | 0 | Rev. · línea sin asignar en la hoja | 0 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 3 | 0 | Rev. · línea sin asignar en la hoja | 0 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 4 | 1 | Rev. · línea sin asignar en la hoja | 1 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 5 | 0 | Rev. · línea sin asignar en la hoja | 0 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 6 | 0 | Rev. · línea sin asignar en la hoja | 0 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 7 | 0 | Rev. · línea sin asignar en la hoja | 0 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 8 | 1 | Rev. · línea sin asignar en la hoja | 1 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 9 | 1 | Rev. · línea sin asignar en la hoja | 1 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 10 | 1 | Rev. · línea sin asignar en la hoja | 1 | sin detectar | Lectura/asignación o puerta de confianza |
| c13 | 11 | 1 | Rev. · línea sin asignar en la hoja | 1 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 12 | 1 | Rev. · línea sin asignar en la hoja | 1 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 13 | 1 | Rev. · línea sin asignar en la hoja | 1 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 14 | 1 | Rev. · línea sin asignar en la hoja | 1 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 15 | 0 | Rev. · línea sin asignar en la hoja | 0 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 16 | 1 | Rev. · línea sin asignar en la hoja | 1 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 17 | 0 | Rev. · línea sin asignar en la hoja | 0 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 18 | 1 | Rev. · línea sin asignar en la hoja | 1 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 19 | 1 | Rev. · línea sin asignar en la hoja | 1 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 20 | 0 | Rev. · línea sin asignar en la hoja | 0 | exacta | Lectura/asignación o puerta de confianza |
| c13 | 21 | sin referencia | Rev. · línea sin asignar en la hoja | Rev. · respuesta sin detectar | pregunta inesperada | Detección fuera de la referencia |
| c13 | 31 | sin referencia | Rev. · línea sin asignar en la hoja | Rev. · respuesta sin detectar | pregunta inesperada | Detección fuera de la referencia |
| c14 | 16 | 1 | 1 | 1 | exacta | Nota exacta |
| c14 | 17 | 1 | 1 | 1 | exacta | Nota exacta |
| c14 | 18 | 1 | 1 | 1 | exacta | Nota exacta |
| c14 | 19 | 1 | 1 | 1 | exacta | Nota exacta |
| c14 | 20 | 0 | 0 | 0 | exacta | Nota exacta |
| c15 | 1 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c15 | 2 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c15 | 3 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c15 | 4 | 0 | Rev. · confianza baja en el número | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 1 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 2 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 3 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 4 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 5 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 6 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 7 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 8 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 9 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 10 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 11 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 12 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 13 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 14 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 15 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 16 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 17 | 0 | Rev. · opción o formato no admitido | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 18 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 19 | 1 | Rev. · opción o formato no admitido | 1 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 20 | 0 | Rev. · confianza baja en el texto | 0 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 21 | 2 | Rev. · fallo del proveedor | Rev. · proveedor no disponible | exacta | Ejecución del corrector |
| c16 | 22 | 1 | Rev. · confianza baja en el número | Rev. · proveedor no disponible | exacta | Lectura/asignación o puerta de confianza |
| c16 | 23 | 1 | Rev. · política de revisión de notas bajas | Rev. · proveedor no disponible | exacta | Origen sin aislar · falló el control |
| c16 | 24 | 2 | Rev. · política de revisión de notas bajas | Rev. · proveedor no disponible | texto distinto | Origen sin aislar · falló el control |
| c16 | 25 | 2 | Rev. · proveedor no disponible | Rev. · evidencia insuficiente | texto distinto | Ejecución del corrector |
| c16 | 26 | 2 | Rev. · confianza baja en el número | Rev. · proveedor no disponible | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 27 | 2 | Rev. · política de revisión de notas bajas | Rev. · política de revisión de notas bajas | exacta | Persiste con lectura verificada |
| c16 | 28 | sin referencia | Rev. · confianza baja en el número | 2 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 29 | 2 | Rev. · confianza baja en el número | Rev. · proveedor no disponible | exacta | Lectura/asignación o puerta de confianza |
| c16 | 31 | 2 | Rev. · confianza baja en el número | 2 | texto distinto | Lectura/asignación o puerta de confianza |
| c16 | 34 | 2 / revisión (requiere la figura) | Rev. · confianza baja en el número | Rev. · requiere la figura | texto distinto | Lectura/asignación o puerta de confianza |
| c17 | 31 | 2 | Rev. · respuesta sin detectar | 2 | sin detectar | Lectura/asignación o puerta de confianza |
| c17 | 34 | sin referencia | Rev. · confianza baja en el número | Rev. · respuesta sin detectar | pregunta inesperada | Detección fuera de la referencia |
| c18 | 23 | 2 | Rev. · confianza baja en el número | Rev. · proveedor no disponible | texto distinto | Lectura/asignación o puerta de confianza |
| c19 | 28 | 1.5 | Rev. · confianza baja en el número | 2 | exacta | Lectura/asignación o puerta de confianza |
| c20 | 24 | 2 | Rev. · confianza baja en el texto | Rev. · proveedor no disponible | texto distinto | Lectura/asignación o puerta de confianza |
| c21 | 34 | 2 / revisión (requiere la figura) | Rev. · requiere la figura | Rev. · requiere la figura | texto distinto | Revisión prevista |
| c22 | 1 | 1 | 1 | 1 | exacta | Nota exacta |
| c22 | 17 | revisión (opción o formato no admitido) | Rev. · opción o formato no admitido | Rev. · opción o formato no admitido | exacta | Revisión prevista |
| c23 | 2 | 1 | 1 | 1 | exacta | Nota exacta |
| c23 | 3 | revisión (número duplicado) | Rev. · número duplicado | Rev. · número duplicado | exacta | Revisión prevista |
| c24 | 1 | 1 | 1 | 1 | exacta | Nota exacta |
| c24 | 2 | revisión (ilegible) | Rev. · ilegible | Rev. · ilegible | exacta | Revisión prevista |
| c25 | 22 | 1 / revisión (diagnóstico parcial) | Rev. · respuesta sin detectar | Rev. · proveedor no disponible | sin detectar | Lectura/asignación o puerta de confianza |
| c25 | 26 | sin referencia | Rev. · confianza baja en el número | Rev. · respuesta sin detectar | pregunta inesperada | Detección fuera de la referencia |
| c26 | 32 | revisión (un solo hiperparámetro) | Rev. · un solo hiperparámetro | Rev. · un solo hiperparámetro | exacta | Revisión prevista |
| c30 | 1 | sin referencia | Rev. · línea sin asignar en la hoja | Rev. · fuera de alcance | pregunta inesperada | Detección fuera de la referencia |
| c30 | 2 | sin referencia | Rev. · línea sin asignar en la hoja | Rev. · fuera de alcance | pregunta inesperada | Detección fuera de la referencia |

Las cuatro hojas sin respuestas objetivo, c27–c30, se evalúan por estado documental y por la ausencia de notas. Las filas de preguntas ausentes que la interfaz crea para completar las 35 posiciones del examen quedan fuera del denominador.
