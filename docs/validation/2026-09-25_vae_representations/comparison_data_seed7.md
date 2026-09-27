# Comparación controlada VAE one-hot (A) vs embeddings (B)

Mismo panel sintético (600 entidades × 14 periodos), mismos splits temporales, semillas [11, 23, 37, 51, 67], 20 épocas, misma rampa KL, sin stacking, presupuesto de alertas = 90 filas OOT. **No se compara la pérdida de reconstrucción** (escalas distintas). El score de alertas `recon_topk` se calcula con **dos normalizaciones aplicadas a ambos brazos** (histórica = |residuo|/MAD; centrada = centrada en la mediana de referencia con piso), para no confundir la representación con la normalización; el pipeline actual corresponde a A histórica y B centrada.

| Métrica | A: one-hot | B: embeddings |
|---|---|---|
| Columnas de entrada del VAE | 66.000 ± 0.000 | 29.000 ± 0.000 |
| Términos de reconstrucción por fila | 66.000 ± 0.000 | 24.000 ± 0.000 |
| AP vs verdad sintética — score de producción (Excel) | 0.063 ± 0.039 | 0.133 ± 0.006 |
| Tail separation — score de producción (Excel) | 2.198 ± 0.080 | 1.536 ± 0.019 |
| AP recon_topk — normalización histórica | 0.018 ± 0.002 | 0.311 ± 0.001 |
| AP recon_topk — normalización centrada | 0.020 ± 0.001 | 0.220 ± 0.045 |
| Tail separation recon_topk — normalización histórica | 1.389 ± 0.108 | 2.261 ± 0.263 |
| Tail separation recon_topk — normalización centrada | 2.013 ± 0.053 | 2.293 ± 0.028 |
| Estabilidad entre semillas (Jaccard alertas recon_topk) — normalización histórica | 0.365 ± 0.068 | 0.782 ± 0.057 |
| Estabilidad entre semillas (Jaccard alertas recon_topk) — normalización centrada | 0.837 ± 0.031 | 0.839 ± 0.034 |
| Estabilidad entre ventanas — Spearman (recon_topk) — normalización histórica | 0.797 ± 0.026 | 0.390 ± 0.046 |
| Estabilidad entre ventanas — Spearman (recon_topk) — normalización centrada | 0.629 ± 0.005 | 0.382 ± 0.016 |
| Estabilidad entre ventanas — Jaccard top-K (recon_topk) — normalización histórica | 0.292 ± 0.111 | 0.083 ± 0.022 |
| Estabilidad entre ventanas — Jaccard top-K (recon_topk) — normalización centrada | 0.259 ± 0.035 | 0.040 ± 0.006 |
| Estabilidad entre ventanas — Spearman, solo variables numéricas — normalización histórica | 0.280 ± 0.017 | 0.343 ± 0.035 |
| Estabilidad entre ventanas — Spearman, solo variables numéricas — normalización centrada | 0.246 ± 0.009 | 0.348 ± 0.013 |
| Lugares top-k ocupados por categóricas (OOT) — normalización histórica | 0.961 ± 0.017 | 0.414 ± 0.010 |
| Lugares top-k ocupados por categóricas (OOT) — normalización centrada | 0.728 ± 0.002 | 0.347 ± 0.003 |
| Dispersión máx/mín de la tasa de lugares entre categóricas (referencia sin deriva) — normalización histórica | 2.055 ± 0.330 | 9.313 ± 3.813 |
| Dispersión máx/mín de la tasa de lugares entre categóricas (referencia sin deriva) — normalización centrada | 1.494 ± 0.074 | 1.138 ± 0.040 |
| Peso de las categóricas entre las unidades (columnas en A, variables en B) | 0.652 ± 0.000 | 0.250 ± 0.000 |
| Estabilidad entre semillas (Jaccard alertas, score de producción) | 0.729 ± 0.057 | 0.822 ± 0.030 |
| Unidades latentes activas | 8.000 ± 0.000 | 6.400 ± 0.894 |
| Tiempo de ajuste (s) | 6.616 ± 0.780 | 3.030 ± 0.031 |
| Memoria pico Δ RSS (MB) | 12.473 ± 27.340 | 1.452 ± 1.280 |

Jaccard de alertas A vs B (misma semilla): raw: 0.385 ± 0.011; recon_topk (pipeline: A historical, B centred): 0.066 ± 0.033.

**Categorías no vistas** (un nivel que el entrenamiento nunca vio en una variable categórica de filas OOT):

- A: 5/5 sin error; elevación media del score = -0.0069; tratamiento: infrequent bucket / all-zero one-hot (implicit).
- B: 5/5 sin error; elevación media del score = 0.1769; tratamiento: UNKNOWN (explicit).

## Criterios de aceptación evaluables con estas mediciones

| Criterio | Resultado | Evidencia |
|---|---|---|
| Ninguna columna one-hot entra al VAE (arm B) | ✅ cumple | 29 columnas de entrada frente a 66 one-hot; columnas `cat__*` con rol distinto de índice: 0 |
| Cada categórica produce una sola contribución al score | ✅ cumple | 6 variables categóricas -> 6 contribuciones; contribuciones totales por fila = 24 (= variables originales 24); el one-hot tiene 66 términos |
| Categorías nuevas no causan errores | ✅ cumple | 5/5 corridas con nivel nunca visto sin error; elevación media del score 0.1769 |
| Los checkpoints incompatibles se rechazan | ✅ cumple | {'onehot_payload_as_mixed_v1': 'rejected', 'embedding_payload_as_onehot_v1': 'rejected'} |
| Sin fuga temporal en la calibración | ✅ cumple | vocabularios ⊂ niveles de train; filas de referencia de la normalización = 3600 (train = 3600) |
| El top-k categórico deja de depender de la cardinalidad (referencia sin deriva; MISMA normalización centrada en A y B) | ✅ cumple | dispersión máx/mín de la tasa de lugares entre categóricas: A=1.49x B=1.14x (límite 1.5x y B<=A); lugares top-k categóricos: A=73% B=35% (peso de las categóricas entre las variables originales: 25%). Referencia: normalización histórica A=2.05x B=9.31x |
| No empeora materialmente AP del score de alertas recon_topk — pipeline actual (A histórica, B centrada) | ✅ cumple | A=0.0183±0.0015  B=0.2199±0.0449 (tolerancia 0.0050) |
| No empeora materialmente AP del score de alertas recon_topk — misma normalización centrada | ✅ cumple | A=0.0202±0.0014  B=0.2199±0.0449 (tolerancia 0.0050) |
| No empeora materialmente tail separation del score de alertas recon_topk — pipeline actual (A histórica, B centrada) | ✅ cumple | A=1.3885±0.1083  B=2.2928±0.0282 (tolerancia 0.1083) |
| No empeora materialmente tail separation del score de alertas recon_topk — misma normalización centrada | ✅ cumple | A=2.0127±0.0532  B=2.2928±0.0282 (tolerancia 0.1006) |
| No empeora materialmente AP del score de producción del Excel | ✅ cumple | A=0.0633±0.0391  B=0.1327±0.0064 (tolerancia 0.0391) |
| No empeora materialmente tail separation del score de producción del Excel | ❌ NO cumple | A=2.1982±0.0797  B=1.5357±0.0188 (tolerancia 0.1099) |
| La estabilidad entre semillas no cae bajo el piso del control — pipeline actual | ✅ cumple | A=0.365  B=0.839  (piso = 80 % del control = 0.292) |
| La estabilidad entre semillas no cae bajo el piso del control — misma normalización | ✅ cumple | A=0.837  B=0.839  (piso = 80 % del control = 0.670) |
| La estabilidad entre ventanas no cae bajo el piso del control — Spearman, control literal: A completo (pipeline actual) | ❌ NO cumple | A=0.797  B=0.382  (piso = 80 % del control = 0.638); el control literal de A está inflado por atributos categóricos que no cambian entre meses |
| La estabilidad entre ventanas no cae bajo el piso del control — Jaccard top-K, control literal: A completo (pipeline actual) | ❌ NO cumple | A=0.292  B=0.040  (piso = 80 % del control = 0.233); el control literal de A está inflado por atributos categóricos que no cambian entre meses |
| La estabilidad entre ventanas no cae bajo el piso del control — Spearman, control equivalente: ambos solo variables numéricas | ✅ cumple | A=0.246  B=0.348  (piso = 80 % del control = 0.197) |

**Decisión de promoción a default:** NO se promueve; 3 criterio(s) no se cumplen: No empeora materialmente tail separation del score de producción del Excel; La estabilidad entre ventanas no cae bajo el piso del control — Spearman, control literal: A completo (pipeline actual); La estabilidad entre ventanas no cae bajo el piso del control — Jaccard top-K, control literal: A completo (pipeline actual).

Datos sintéticos: valida el cableado y la metodología, no el desempeño sobre datos reales.