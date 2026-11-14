# Letra y texto de la opción en los 30 PDFs

Se repitió la lectura de los 30 PDFs con la versión `8086331` del lector de Anna después de comprobar el SHA-256 de cada archivo. La salida de cada lectura se pasó por los adaptadores `bridge-v2` y `bridge-v3` y por la ruta PDF del corrector. Las dos versiones recibieron los mismos resultados o fallos de Gemini. Se contrastaron las 1.050 decisiones por pregunta (nota, motivo y error de ejecución), los criterios de aceptación por PDF y las referencias del paquete. Los denominadores de cobertura excluyen las preguntas ausentes del recorte. [Referencias](case_references.json) · [Evaluación anterior](vision_finetune_arrows_30.md)

## Las cuatro respuestas de c15

El adaptador consulta el texto completo de la opción marcada en esa pregunta del examen. Admite variaciones de mayúsculas, espacios y punto final. La referencia de test usa la opción registrada y la clave publicada.

| Pregunta | Lectura | Resultado | Referencia |
|---|---|---|---|
| Q1 | D, All of the mentioned | 0/1. Coincide con el texto de D. | 0/1 |
| Q2 | B, Minimum. | 0/1. Coincide con el texto de B. | 0/1 |
| Q3 | D, All of the mentioned. | Revisión. La opción D de Q3 dice «Minkowski Distance». | 0/1 |
| Q4 | B, regression | Revisión. Concuerda con B, pero la confianza del número es 0,588 y el mínimo es 0,85. | 0/1 |

## Resultado conjunto

| Métrica | Antes | Después |
|---|---:|---:|
| Notas exactas | 51 | 53 |
| Notas propuestas con referencia (100 disponibles) | 51 | 53 |
| Revisiones entre 105 preguntas presentes | 54 | 52 |
| Lecturas exactas entre 105 preguntas presentes | 74 | 77 |
| PDFs que cumplen todos los criterios (30 evaluados) | 11 | 11 |

Las dos respuestas que salieron de revisión coincidieron con la referencia. Las otras 1.048 decisiones conservaron su nota y su motivo. Las 23 respuestas con flecha recuperadas en la evaluación anterior siguen puntuadas en la comparación. Ambas versiones registraron 2 decisiones de revisión por fallo del proveedor en esta pasada.

Frente al informe anterior, las cifras por PDF del adaptador anterior se reprodujeron.

| PDF | Revisiones antes → después | Notas exactas antes → después | Cumple todos los criterios antes → después |
|---|---:|---:|---|
| c01 | 0 → 0 | 5 → 5 | Sí → Sí |
| c02 | 0 → 0 | 5 → 5 | Sí → Sí |
| c03 | 0 → 0 | 6 → 6 | Sí → Sí |
| c04 | 0 → 0 | 4 → 4 | Sí → Sí |
| c05 | 1 → 1 | 0 → 0 | No → No |
| c06 | 1 → 1 | 3 → 3 | No → No |
| c07 | 1 → 1 | 0 → 0 | No → No |
| c08 | 1 → 1 | 0 → 0 | No → No |
| c09 | 1 → 1 | 0 → 0 | No → No |
| c10 | 1 → 1 | 0 → 0 | No → No |
| c11 | 1 → 1 | 0 → 0 | No → No |
| c12 | 1 → 1 | 1 → 1 | No → No |
| c13 | 20 → 20 | 0 → 0 | No → No |
| c14 | 0 → 0 | 5 → 5 | Sí → Sí |
| c15 | 4 → 2 | 0 → 2 | No → No |
| c16 | 12 → 12 | 19 → 19 | No → No |
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
