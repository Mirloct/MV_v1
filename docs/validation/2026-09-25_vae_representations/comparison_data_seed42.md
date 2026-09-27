# Comparación controlada VAE one-hot (A) vs embeddings (B)

Mismo panel sintético (600 entidades × 14 periodos), mismos splits temporales, semillas [11, 23, 37, 51, 67], 20 épocas, misma rampa KL, sin stacking, presupuesto de alertas = 90 filas OOT. **No se compara la pérdida de reconstrucción** (escalas distintas). El score de alertas `recon_topk` se calcula con **dos normalizaciones aplicadas a ambos brazos** (histórica = |residuo|/MAD; centrada = centrada en la mediana de referencia con piso), para no confundir la representación con la normalización; el pipeline actual corresponde a A histórica y B centrada.

| Métrica | A: one-hot | B: embeddings |
|---|---|---|
| Columnas de entrada del VAE | 66.000 ± 0.000 | 29.000 ± 0.000 |
| Términos de reconstrucción por fila | 66.000 ± 0.000 | 24.000 ± 0.000 |
| AP vs verdad sintética — score de producción (Excel) | 0.043 ± 0.014 | 0.112 ± 0.023 |
| Tail separation — score de producción (Excel) | 2.534 ± 0.139 | 1.617 ± 0.030 |
| AP recon_topk — normalización histórica | 0.018 ± 0.001 | 0.288 ± 0.018 |
| AP recon_topk — normalización centrada | 0.019 ± 0.001 | 0.224 ± 0.017 |
| Tail separation recon_topk — normalización histórica | 1.408 ± 0.056 | 2.079 ± 0.464 |
| Tail separation recon_topk — normalización centrada | 2.071 ± 0.076 | 2.987 ± 0.108 |
| Estabilidad entre semillas (Jaccard alertas recon_topk) — normalización histórica | 0.456 ± 0.064 | 0.771 ± 0.049 |
| Estabilidad entre semillas (Jaccard alertas recon_topk) — normalización centrada | 0.800 ± 0.057 | 0.901 ± 0.025 |
| Estabilidad entre ventanas — Spearman (recon_topk) — normalización histórica | 0.779 ± 0.017 | 0.380 ± 0.019 |
| Estabilidad entre ventanas — Spearman (recon_topk) — normalización centrada | 0.635 ± 0.002 | 0.406 ± 0.017 |
| Estabilidad entre ventanas — Jaccard top-K (recon_topk) — normalización histórica | 0.221 ± 0.041 | 0.052 ± 0.013 |
| Estabilidad entre ventanas — Jaccard top-K (recon_topk) — normalización centrada | 0.351 ± 0.031 | 0.052 ± 0.014 |
| Estabilidad entre ventanas — Spearman, solo variables numéricas — normalización histórica | 0.267 ± 0.026 | 0.381 ± 0.020 |
| Estabilidad entre ventanas — Spearman, solo variables numéricas — normalización centrada | 0.251 ± 0.012 | 0.386 ± 0.015 |
| Lugares top-k ocupados por categóricas (OOT) — normalización histórica | 0.929 ± 0.056 | 0.361 ± 0.020 |
| Lugares top-k ocupados por categóricas (OOT) — normalización centrada | 0.723 ± 0.010 | 0.342 ± 0.004 |
| Dispersión máx/mín de la tasa de lugares entre categóricas (referencia sin deriva) — normalización histórica | 2.246 ± 0.409 | 5.013 ± 2.041 |
| Dispersión máx/mín de la tasa de lugares entre categóricas (referencia sin deriva) — normalización centrada | 1.438 ± 0.033 | 1.237 ± 0.038 |
| Peso de las categóricas entre las unidades (columnas en A, variables en B) | 0.652 ± 0.000 | 0.250 ± 0.000 |
| Estabilidad entre semillas (Jaccard alertas, score de producción) | 0.821 ± 0.043 | 0.783 ± 0.041 |
| Unidades latentes activas | 8.000 ± 0.000 | 6.600 ± 1.140 |
| Tiempo de ajuste (s) | 7.371 ± 1.156 | 3.031 ± 0.042 |
| Memoria pico Δ RSS (MB) | 12.094 ± 26.862 | 1.851 ± 1.149 |

Jaccard de alertas A vs B (misma semilla): raw: 0.418 ± 0.068; recon_topk (pipeline: A historical, B centred): 0.120 ± 0.030.

**Categorías no vistas** (un nivel que el entrenamiento nunca vio en una variable categórica de filas OOT):

- A: 5/5 sin error; elevación media del score = -0.0073; tratamiento: infrequent bucket / all-zero one-hot (implicit).
- B: 5/5 sin error; elevación media del score = 0.1815; tratamiento: UNKNOWN (explicit).

## Criterios de aceptación evaluables con estas mediciones

| Criterio | Resultado | Evidencia |
|---|---|---|
| Ninguna columna one-hot entra al VAE (arm B) | ✅ cumple | 29 columnas de entrada frente a 66 one-hot; columnas `cat__*` con rol distinto de índice: 0 |
| Cada categórica produce una sola contribución al score | ✅ cumple | 6 variables categóricas -> 6 contribuciones; contribuciones totales por fila = 24 (= variables originales 24); el one-hot tiene 66 términos |
| Categorías nuevas no causan errores | ✅ cumple | 5/5 corridas con nivel nunca visto sin error; elevación media del score 0.1815 |
| Los checkpoints incompatibles se rechazan | ✅ cumple | {'onehot_payload_as_mixed_v1': 'rejected', 'embedding_payload_as_onehot_v1': 'rejected'} |
| Sin fuga temporal en la calibración | ✅ cumple | vocabularios ⊂ niveles de train; filas de referencia de la normalización = 3600 (train = 3600) |
| El top-k categórico deja de depender de la cardinalidad (referencia sin deriva; MISMA normalización centrada en A y B) | ✅ cumple | dispersión máx/mín de la tasa de lugares entre categóricas: A=1.44x B=1.24x (límite 1.5x y B<=A); lugares top-k categóricos: A=72% B=34% (peso de las categóricas entre las variables originales: 25%). Referencia: normalización histórica A=2.25x B=5.01x |
| No empeora materialmente AP del score de alertas recon_topk — pipeline actual (A histórica, B centrada) | ✅ cumple | A=0.0175±0.0010  B=0.2236±0.0166 (tolerancia 0.0050) |
| No empeora materialmente AP del score de alertas recon_topk — misma normalización centrada | ✅ cumple | A=0.0192±0.0011  B=0.2236±0.0166 (tolerancia 0.0050) |
| No empeora materialmente tail separation del score de alertas recon_topk — pipeline actual (A histórica, B centrada) | ✅ cumple | A=1.4081±0.0562  B=2.9874±0.1077 (tolerancia 0.0704) |
| No empeora materialmente tail separation del score de alertas recon_topk — misma normalización centrada | ✅ cumple | A=2.0714±0.0755  B=2.9874±0.1077 (tolerancia 0.1036) |
| No empeora materialmente AP del score de producción del Excel | ✅ cumple | A=0.0431±0.0142  B=0.1123±0.0234 (tolerancia 0.0142) |
| No empeora materialmente tail separation del score de producción del Excel | ❌ NO cumple | A=2.5342±0.1386  B=1.6173±0.0301 (tolerancia 0.1386) |
| La estabilidad entre semillas no cae bajo el piso del control — pipeline actual | ✅ cumple | A=0.456  B=0.901  (piso = 80 % del control = 0.365) |
| La estabilidad entre semillas no cae bajo el piso del control — misma normalización | ✅ cumple | A=0.800  B=0.901  (piso = 80 % del control = 0.640) |
| La estabilidad entre ventanas no cae bajo el piso del control — Spearman, control literal: A completo (pipeline actual) | ❌ NO cumple | A=0.779  B=0.406  (piso = 80 % del control = 0.623); el control literal de A está inflado por atributos categóricos que no cambian entre meses |
| La estabilidad entre ventanas no cae bajo el piso del control — Jaccard top-K, control literal: A completo (pipeline actual) | ❌ NO cumple | A=0.221  B=0.052  (piso = 80 % del control = 0.177); el control literal de A está inflado por atributos categóricos que no cambian entre meses |
| La estabilidad entre ventanas no cae bajo el piso del control — Spearman, control equivalente: ambos solo variables numéricas | ✅ cumple | A=0.251  B=0.386  (piso = 80 % del control = 0.201) |

**Decisión de promoción a default:** NO se promueve; 3 criterio(s) no se cumplen: No empeora materialmente tail separation del score de producción del Excel; La estabilidad entre ventanas no cae bajo el piso del control — Spearman, control literal: A completo (pipeline actual); La estabilidad entre ventanas no cae bajo el piso del control — Jaccard top-K, control literal: A completo (pipeline actual).

Datos sintéticos: valida el cableado y la metodología, no el desempeño sobre datos reales.