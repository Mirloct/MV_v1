# Comparación controlada: VAE one-hot (A) vs embeddings (B) — 2026-09-25

Generada con `py tools/compare_vae_representations.py --individuals 600 --periods 14 --epochs 20 --seeds 11 23 37 51 67 --data-seed <42|7>`
(temporal working directory; el `artifacts/` del proyecto no se toca). Datos **sintéticos**.

| Archivo | Contenido |
|---|---|
| `comparison_data_seed42.md` / `.json` | Panel sintético con semilla de datos 42: tabla A vs B, criterios de aceptación, decisión |
| `comparison_data_seed7.md` / `.json` | Segundo panel independiente (semilla de datos 7) |

## Qué se mantiene idéntico entre A y B

Mismo panel, mismos splits temporales (2 meses de validación, 3 de test, 3 OOT), mismas semillas de modelo, mismas filas de
entrenamiento y meses de validación, mismo presupuesto de épocas (20) y misma rampa KL (`min(10, épocas//2)`), misma arquitectura base
(latente 8, oculta 64×2, lote 256), **sin** stacking del IF (para que solo cambie la representación) y el mismo presupuesto de alertas
(top-5 % de las filas OOT). **No se compara la pérdida de reconstrucción** (escalas distintas: MSE sobre columnas vs Huber/BCE/NLL sobre variables).

## Qué cambió tras la revisión independiente

Una primera versión aplicaba el centrado + piso solo al brazo B y atribuía a la representación un efecto que en parte era de la normalización. Ahora
`recon_topk` se calcula con **las dos normalizaciones en ambos brazos**, y los criterios que antes se afirmaban (una contribución por variable, sin one-hot,
calibración solo con train) se **miden** (contribuciones por fila = variables originales; filas de referencia = filas de train). La estabilidad entre ventanas
se juzga con el control **literal** (A completo) y con el control **equivalente** (variables numéricas).

## Lectura

* Con la **misma** normalización, los lugares top-k categóricos son 72–73 % con one-hot y 34–35 % con embeddings (25 % esperado por número de variables
  originales). El AP de `recon_topk` es 0.019–0.020 con one-hot (azar) y 0.22 con embeddings: ese salto es de la representación.
* La normalización centrada es **necesaria** para B: con la histórica, la dispersión de la tasa de lugares entre categóricas es 5.0×–9.3×; con la centrada, 1.1×–1.2×
  (A centrado: 1.4×–1.5×).
* La estabilidad entre semillas de las alertas sube de 0.46 a 0.90 (pipeline tal cual) y de 0.80–0.84 a 0.84–0.90 (misma normalización).
* La dependencia de la cardinalidad se mide sobre el **bloque de referencia sin deriva**. Sobre OOT la variable `transaction_channel` domina con ambos modelos
  porque su distribución **deriva de verdad** (46 % `branch` → 39 % `mobile_app`): es señal, no sesgo.
* **Tres criterios no se cumplen** y por eso **el default sigue en `onehot`**: (1) `tail_separation` del score de producción del Excel (1.62 vs 2.53; su AP sí
  mejora, 0.112 vs 0.043); (2) y (3) estabilidad entre ventanas medida contra el one-hot **completo** (Spearman 0.41 vs 0.78; Jaccard top-K 0.05 vs 0.22). El valor
  del one-hot completo está inflado por atributos categóricos que no cambian entre meses; con el control equivalente (solo variables numéricas) B es mayor
  (0.39 vs 0.25) y ese criterio se cumple, pero el literal no.
