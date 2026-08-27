# Diagnóstico del proyecto — hallazgos, riesgos y puntos débiles

Revisión de inicio a fin del pipeline (2026-08-22). Busca bugs, condiciones de
bloqueo, puntos débiles y consideraciones metodológicas.

**Separado en dos bloques que se corrigen de forma distinta:**
[§A Código](#a-nivel-de-código) son defectos con una corrección concreta;
[§B Metodología](#b-nivel-metodológico) son decisiones de diseño que requieren
criterio, no un parche.

## Escala de criticidad

| Nivel | Significado |
| --- | --- |
| 🔴 **CRÍTICO** | Invalida resultados. Lo que el sistema reporta hoy no es lo que cree reportar. |
| 🟠 **ALTO** | Sesga la selección de modelo o esconde un fallo. No invalida todo, pero altera conclusiones. |
| 🟡 **MEDIO** | Riesgo real bajo condiciones alcanzables (datos reales, panel corto, alta cardinalidad). |
| 🔵 **BAJO** | Deuda técnica, robustez o claridad. No afecta correctitud hoy. |

---

# Resumen ejecutivo

**El hallazgo dominante de esta revisión: el VAE del proyecto estaba
colapsado, en toda corrida real hecha hasta el momento de esta revisión.** Se
verificó empíricamente: **0 de 8 dimensiones latentes activas**, KL media
0.0001. El decodificador ignoraba por completo el código latente. Como el
puntaje de anomalía **es** el error de reconstrucción, el VAE había estado
emitiendo puntajes finitos, ordenando filas y exportando su cola de Excel OOT
a partir de un modelo degenerado — sin que nada fallara. **Corregido y
verificado el mismo día (0/8 → 8/8 dimensiones activas) — ver A-1.** Todo
resultado de VAE producido antes de esa corrección describe el modelo
degenerado; ver `CONTEXT.md` "Known open problems".

La causa raíz es un desajuste de escala entre los dos términos de la pérdida
(§A-1), que multiplica el `beta` efectivo por el número de features. Con 22
features, el `beta=1.0` por defecto entrena como si fuera `beta=22`.

Tres hallazgos 🔴/🟠 se refuerzan entre sí y explican por qué nadie lo notó:

1. El escalado colapsa el modelo (§A-1).
2. El objetivo de Optuna no es comparable entre trials, así que el tuneo no
   podía corregirlo por la vía correcta (§A-2).
3. No existía ninguna verificación de colapso — **corregido en esta ronda**
   (§A-3).

| # | Hallazgo | Nivel | Estado |
| --- | --- | --- | --- |
| A-1 | `beta` efectivo = `beta × n_features` → VAE colapsado | 🔴 | **CORREGIDO** y verificado (0/8 → 8/8) |
| A-2 | Objetivo Optuna no supervisado depende de `beta`, que es dimensión de búsqueda | 🟠 | **CORREGIDO** (ELBO a beta=1) |
| A-3 | Ninguna detección de posterior collapse | 🔴 | **CORREGIDO** |
| A-4 | Ruta sin tuneo del VAE usaba split de validación aleatorio (fuga temporal) | 🔴 | **CORREGIDO** |
| A-9 | Rampa de KL más larga que el presupuesto de épocas del trial | 🟠 | **CORREGIDO** |
| A-10 | `--quick` no alcanza para entrenar el VAE; el trial ganador colapsa | 🟠 | **PENDIENTE** (decisión abierta) |
| A-11 | Artefactos de corridas sintéticas sobrescribían los oficiales | 🟠 | **CORREGIDO** |
| A-12 | Marcador de procedencia por carpeta: marcaba datos reales como sintéticos | 🔴 | **CORREGIDO** (regresión de A-11) |
| A-13 | Ground truth sintético residual capturado por corrida real | 🟡 | **CORREGIDO** (mensaje explicativo) |
| A-5 | 53 bloques `except Exception` que degradan en silencio | 🟡 | Abierto |
| A-6 | Suite de tests eliminada | 🟠 | Decisión del usuario, con consecuencia registrada |
| A-7 | Reporte HTML de 6 MB por el bundle de Plotly | 🔵 | Abierto |
| A-8 | Sin persistencia de scores/rankings por corrida | 🔵 | Abierto |

## Verificación de las correcciones (2026-08-23)

Medido sobre el pipeline real, no en aislamiento:

| Escenario | Dimensiones activas | Antes |
| --- | --- | --- |
| `--no-tune` (15 épocas, `beta=1.0`) | **8/8** | 0/8 |
| `--quick --vae-epochs 15` (tuneado, eligió `beta=1.4`, 6 épocas) | **21/21** | 0/9 |
| `--quick` tal cual (5 épocas máx.) | 0/9 | 0/9 |

Las dos primeras filas confirman A-1: con el escalado corregido, todo el rango
de `beta` que explora Optuna produce espacios latentes sanos. La tercera es
A-10, abajo.

---

# A. Nivel de código

## 🔴 A-1. El `beta` efectivo es `beta × n_features` — el VAE colapsa — CORREGIDO

**Dónde:** `src/models/vae.py::vae_loss`, `reduction="mean"`.

```python
recon = F.mse_loss(x_recon, x, reduction="mean")   # divide entre (B × D)
kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
kl = kl_per_sample.mean()                          # divide entre B solamente
total = recon + beta * kl
```

El término de reconstrucción se promedia sobre **batch y features**; la KL se
suma sobre dimensiones latentes y se promedia **solo sobre el batch**. Respecto
de la convención ELBO por muestra, la KL pesa `D` veces más, con `D` = número
de features.

**Verificación numérica** (`B=64, D=12, L=8`):

```
recon_sum/B ÷ recon_mean = 12.00   ← igual a D
kl_sum/B    ÷ kl_mean    =  1.00   ← igual a 1
```

**Impacto según el ancho de la matriz:**

| Features | `beta=0.1` (piso Optuna) | `beta=1.0` (default) | `beta=2.0` (tope Optuna) |
| --- | --- | --- | --- |
| 22 (panel sintético) | 2.2 | **22** | **44** |
| 50 (feature mart real) | 5.0 | **50** | **100** |
| 120 (one-hot alta cardinalidad) | 12 | **120** | **240** |

**Evidencia experimental** (mismos datos, arquitectura y semilla; 2000×22, 4
factores latentes reales):

| Configuración | Activas | KL media | Pérdida val | Veredicto |
| --- | --- | --- | --- | --- |
| `beta=1.0` (default del proyecto) | **0/8** | 0.0005 | 0.928 | 🔴 colapso |
| `beta=2.0` (tope Optuna) | **0/8** | 0.0002 | 0.929 | 🔴 colapso |
| `beta=0.1` (piso Optuna) | 8/8 | 2.84 | 0.404 | ok |
| `beta=1/D` (efectivo 1.0) | 8/8 | 4.79 | 0.254 | ok |
| `beta=0.1/D` (efectivo 0.1) | 8/8 | 13.9 | 0.076 | ok |

**Confirmado en el pipeline real** (200 individuos × 6 períodos, `--no-tune`):

```
VAE latent health: 0/8 active units (delta=0.01), mean KL=0.0001
```

**Consecuencia.** Todo resultado del VAE producido hasta ahora describe un
modelo degenerado. El Excel OOT del VAE, sus métricas no supervisadas, su
espacio latente y su error por feature son artefactos de un decodificador que
reconstruye aproximadamente la media.

**Por qué era invisible.** El puntaje sigue siendo finito y con dispersión
razonable (rango [0.009, 6.45], σ=0.83 en el caso de prueba). Ninguna
aserción falla, ningún NaN aparece, el `best_params.yaml` se escribe poblado.

### Corrección aplicada (2026-08-22)

Se aplicó la **Opción A**: `vae_loss` cambió a MSE **sumado sobre features**,
promediado sobre batch (`reduction="mean"` ahora escala ambos términos igual),
en vez de dividir el término de reconstrucción entre `batch × features` como
antes. `beta=1.0` vuelve a significar lo que dice la literatura (Higgins et
al. 2017), y el rango `[0.1, 2.0]` de Optuna deja de ser catastrófico.

De las tres opciones consideradas (A: cambiar la reducción de la pérdida — la
aplicada; B: dividir la KL entre `D` dentro de `vae_loss`, mismo efecto pero
más alejado de la forma publicada; C: mover el rango de `beta` sin tocar la
pérdida, que solo enmascara el problema), A era la única que hace que `beta`
signifique lo mismo para cualquier ancho de matriz `D`.

**Costo, tal como se anticipó:** todo `best_params_vae.yaml` generado antes
de este cambio codifica un `beta` en la escala vieja y no es reutilizable —
hay que re-tunear. Ver `CONTEXT.md` "Known open problems".

---

## 🟠 A-2. El objetivo de Optuna no es comparable entre trials — CORREGIDO

**Dónde:** `src/models/vae.py::tune_vae`, modo no supervisado (el **default**
del proyecto).

```python
# Default unsupervised: validation loss at the best epoch.
return float(detector.best_val_loss_)
```

`best_val_loss_` es la pérdida **total** — `recon + beta·KL` — y `beta` es una
dimensión del espacio de búsqueda. Optuna compara trials que optimizan
**funciones objetivo distintas**: un trial con `beta=0.1` obtiene un valor
mecánicamente menor que uno con `beta=2.0` aunque el modelo sea idéntico.

Se observa directamente en la tabla de A-1: la pérdida de validación cae
monótonamente con `beta` (0.928 → 0.404 → 0.254 → 0.076), casi por completo
por el peso del término, no por calidad del modelo.

**Efecto neto.** El sesgo empuja hacia `beta` bajo, que resulta ser la
dirección *sana* — así que compensa parcialmente A-1 por accidente. Pero está
seleccionando sobre un artefacto de la parametrización, no sobre calidad.

**Corrección aplicada (2026-08-22).** El objetivo no supervisado por defecto
ya no es `best_val_loss_` (la pérdida ponderada por `beta` de ese trial); es
la **ELBO negativa a `beta=1`**, calculada aparte
(`VAEDetector.best_val_elbo_`) precisamente porque no depende de la dimensión
de búsqueda. Los trials vuelven a ser comparables entre sí sin importar qué
`beta` sorteó cada uno. `objective_metric="recon_p50"` (error de
reconstrucción de validación puro, sin KL) sigue disponible como alternativa
explícita.

---

## 🔴 A-3. No existía detección de posterior collapse — CORREGIDO

**Estado: corregido en esta ronda.**

* `src/models/vae.py::VAEDetector.latent_diagnostics` — calcula
  `A_j = Var_x(E_q[z_j|x])` y la KL por dimensión.
  Umbral `δ = 0.01` (Burda, Grosse & Salakhutdinov, IWAE, ICLR 2016).
* `src/models/vae.py::collapse_verdict` — emite el juicio. Separado de la
  medición a propósito: el umbral de "cuántas activas son suficientes"
  (`1/3`) es una convención **de este proyecto**, no de la literatura, y está
  documentado como tal.
* `main.py` Fase 7 — corre el gate justo después del ajuste, registra un
  `observability.check` de severidad crítica, y publica el conteo al dashboard.
* `src/reporting/report.py::_latent_health_html` — callout de estado en la
  tarjeta del modelo, con chip de texto visible (nunca solo color).

**Diseño deliberado:** un KL bajo por sí solo **no** declara colapso. Se exige
evidencia conjunta (fracción activa baja **y** KL cercana a cero), o el caso
degenerado de ≤1 dimensión activa. Una dimensión silenciosa es normal en
cualquier VAE entrenado; alertar por eso convertiría el check en ruido.

---

## 🔴 A-4. Fuga temporal en la ruta sin tuneo del VAE — CORREGIDO

**Dónde:** `main.py`, ajuste del VAE cuando `--no-tune`.

```python
vae_detector.fit(X_vae_in)          # ANTES: sin valid_mask
```

La ruta con tuneo pasaba `valid_mask=valid_local` (los meses de validación);
la ruta sin tuneo no. Sin ese argumento, `fit` cae a `val_fraction=0.1`
**aleatorio**, que en un panel toma filas de todos los períodos — incluidos
posteriores a los de entrenamiento. El early stopping y la selección de la
mejor época quedaban juzgados sobre el futuro.

Es exactamente la fuga que `docs/leakage_free_pipeline.md` prohíbe, en un modo
documentado y de uso habitual (`--quick` y `--no-tune` lo usan).

**Corregido:** ambas rutas pasan ahora `valid_mask=valid_local`.

**Consideración de fondo:** que dos rutas del mismo modelo difirieran en algo
load-bearing es el patrón de riesgo, no el argumento faltante. Conviene que la
construcción del detector sea una sola función usada por ambas ramas.

---

## 🟠 A-9. La rampa de KL era más larga que el presupuesto del trial — CORREGIDO

**Dónde:** `src/models/vae.py::tune_vae`, construcción del detector por trial.

El objetivo no pasaba `kl_anneal_epochs`, así que cada trial usaba el default
de **10 épocas de rampa**. Con `--quick` (`max_epochs=5`) o cualquier trial que
sorteara menos de 10 épocas, la rampa **nunca terminaba**: el modelo entrenaba
toda su vida con un peso KL parcial y jamás veía el `beta` con el que estaba
siendo evaluado. El `beta` del trial era en buena medida ficticio.

**Corregido:** `kl_anneal = min(10, max(1, epochs // 2))`, de modo que la rampa
siempre cabe en el presupuesto y al menos la mitad del entrenamiento ocurre con
el `beta` completo.

---

## 🟠 A-10. `--quick` no alcanza para entrenar el VAE — PENDIENTE

**Observado:** con el preset `--quick` tal cual (`vae_epochs=5`), el trial
ganador entrena 2 épocas y el modelo colapsa (0/9 activas). Subiendo solo esa
variable a `--vae-epochs 15`, el mismo preset produce **21/21 activas**.

**No es un bug del objetivo.** Es el objetivo funcionando: con 5 épocas ningún
trial alcanza a recuperar el costo de KL mediante mejor reconstrucción, así que
el ELBO prefiere —correctamente— al que no lo pagó. El problema es el
presupuesto, no el criterio.

Se subió el piso de `epochs` de 1 a 2 (un trial de 1 época ni siquiera
entrena), pero eso **no resuelve** el caso: con techo 5, ningún valor del rango
basta.

**Opciones, pendientes de decisión:**

| Opción | Efecto |
| --- | --- |
| Subir `vae_epochs` en `--quick` de 5 a ~15 | `--quick` deja de ser tan rápido, pero su VAE pasa a ser utilizable |
| Documentar `--quick` como humo puro | Barato y honesto: su resultado de VAE no se interpreta |
| Penalizar el colapso dentro del objetivo | Evita seleccionar modelos degenerados aun con poco presupuesto; añade un umbral arbitrario al criterio |

**Mitigación vigente:** el gate de A-3 marca la corrida con un health check
crítico fallido, así que un VAE colapsado ya no pasa inadvertido — que era el
riesgo real. Ninguna corrida `--quick` debería tomarse como resultado de VAE
mientras esto siga abierto.

---

## 🟠 A-11. Las corridas sintéticas sobrescribían artefactos oficiales — CORREGIDO

Una corrida sobre datos generados escribía `best_params_*.yaml`, los
checkpoints de modelo y los estudios de Optuna **en la misma ruta** que una
corrida real. El último en escribir ganaba, y nada en el artefacto decía de
qué tipo de datos provenía.

**Corregido en tres piezas:**

1. `generate_synthetic_panel` escribe un marcador de procedencia
   (`.synthetic.json`) junto al CSV. Sin él, solo la corrida que *genera* el
   panel sabe que es sintético; a partir de la segunda, el CSV en disco es
   indistinguible de datos reales.
2. `PanelSchema.is_synthetic` propaga ese hecho, poniéndolo donde el resto del
   pipeline ya mira.
3. `main.py` redirige modelo, parámetros y estudio bajo `_dev/` cuando la
   corrida no es oficial, y lo registra como health check `run.official`.

**Verificado:** tras una corrida `--quick` completa sobre datos sintéticos,
`artifacts/tuning/` y `artifacts/models/` quedan **vacíos** y todo lo ajustable
aparece bajo `_dev/`. El reporte y los entregables Excel sí se generan
normalmente.

---

## 🔴 A-12. El marcador de procedencia era por carpeta, no por archivo — CORREGIDO

**Regresión introducida por la propia corrección A-11**, detectada al usarla
con datos reales.

El marcador se escribía como `.synthetic.json` **en el directorio** del panel,
y la lectura solo comprobaba su existencia. Consecuencia: cualquier CSV en esa
carpeta quedaba marcado como sintético — incluidos **datos reales colocados en
`artifacts/data/`**, que es justamente la ubicación por defecto para ambos. Una
corrida oficial se clasificaba como desarrollo y sus parámetros ajustados
terminaban en `_dev/` en vez de la ubicación definitiva.

**Corregido con dos comprobaciones independientes**, porque equivocarse en
cualquiera de las dos direcciones cuesta caro:

1. El marcador se nombra por el panel: `data.csv` → `data.csv.synthetic.json`.
2. Su campo `panel` debe nombrar ese mismo archivo, lo que cubre el caso de un
   marcador copiado o renombrado junto a otro CSV.

Un marcador ilegible se resuelve como **datos reales**, que es la dirección
conservadora: protege los artefactos oficiales de ser evitados en silencio, y
es ruidosa, así que se nota.

*Verificado:* panel generado → sintético; recarga del mismo → sintético; panel
real en la misma carpeta → **real**; marcador copiado junto a otro panel →
ignorado.

---

## 🟡 A-13. Ground truth sintético residual capturado por una corrida real

**Síntoma:** `WARNING ... Ground-truth file ... missing expected columns`.

`_discover_ground_truth` busca `ground_truth.parquet` / `.csv` **por
convención de nombre** junto al panel. Un archivo dejado por una corrida
sintética anterior se encuentra igual, pero sus columnas clave llevan los
nombres del esquema generado (`entity_id`, `period`), no los del panel real
(p. ej. `id`, `codmes`).

**No es un fallo:** la comprobación de columnas es la red de seguridad
funcionando — detecta el desajuste y continúa **sin etiquetas** (no supervisado),
en vez de cruzar mal las claves y producir métricas falsas.

Lo que faltaba era que el mensaje dijera *por qué*. Ahora nombra la causa
probable y la acción concreta, en lugar de solo listar columnas.

**Acción del usuario:** borrar el `ground_truth.*` sintético residual de la
carpeta del panel, o renombrar el propio para que sus columnas clave coincidan
con las del panel.

---

## 🟡 A-5. 53 bloques `except Exception` que degradan en silencio

**Distribución:** `console_ui.py` (21), `report_content.py` (9),
`vae.py` (5), `iforest_explain.py` (3), `statistics.py` (3), y otros.

La mayoría son correctos por diseño: una figura que falla no debe tumbar el
pipeline, y el dashboard es decoración. Pero hay dos patrones de riesgo:

1. **`report_content.py`** — cada gráfico va en su propio `try/except` que
   registra un warning y sigue. Si una entrada cambia de forma, el reporte se
   genera **con menos gráficos** y nadie lo nota salvo leyendo el log.
   *Mitigación existente:* el validador cuenta los gráficos producidos.
2. **`main.py` Fase 10** — cada bloque de interpretabilidad es best-effort. Un
   fallo sistemático de SHAP daría reportes sin explicabilidad de forma
   indefinida.

**Sugerencia:** que los `except` de figuras registren un `observability.check`
fallido además del warning, para que el conteo aparezca en el resumen de salud
en vez de solo en el log.

---

## 🟠 A-6. Suite de tests eliminada

Removida por decisión explícita (2026-08-22). Se registra aquí porque cambia
el perfil de riesgo del proyecto, no para discutir la decisión.

**Consecuencia concreta:** los tres bugs de este informe (A-1, A-2, A-4) son
del tipo que **no** produce excepciones — dan números plausibles pero
incorrectos. Ese es justamente el tipo que una aserción atrapa y una corrida
manual no.

**Recomendación acotada:** si se implementan las métricas de
`docs/validacion_no_supervisada.md` (EM, MV, Jaccard, active units), conviene
tests **al menos para ellas**: son fórmulas con convenciones de signo y
normalización fáciles de invertir en silencio.

---

## 🔵 A-7 / A-8. Deuda menor

* **A-7 — Reporte de 6 MB.** El 79% es el bundle de plotly.js embebido, costo
  fijo por el requisito de funcionar offline. Palanca disponible:
  `include_plotlyjs="cdn"` (~40 KB) a cambio de perder el offline.
* **A-8 — Sin `scores.parquet` / `rankings.parquet`.** Los puntajes solo
  persisten dentro del Excel OOT, así que no se puede recomputar una métrica
  sobre una corrida pasada sin reejecutarla.

---

## Condiciones de bloqueo — revisadas, sin hallazgos

Se buscaron específicamente loops y bloqueos:

| Riesgo | Veredicto |
| --- | --- |
| `while n_test + n_val >= n_periods` (`splits.py`) | **Seguro.** `n_periods ≥ 3` está garantizado y ambos contadores tienen piso 1, así que la suma converge a 2 < 3. |
| Bucle de repintado del dashboard (`console_ui.py`) | **Seguro.** Hilo daemon con `_stop.wait(0.1)`; termina con el proceso y en el `finally` de `main`. |
| Servidor HTTP de la vista en vivo | **Seguro.** `serve_forever()` en hilo daemon; muere con el proceso. Solo escucha en 127.0.0.1. |
| Optuna sin `timeout` | **Aceptable.** Sin límite de reloj por defecto, pero el early stopping por trials y `n_trials` acotan la corrida. |
| Reintentos / backoff | No hay bucles de reintento en el código. |

---

# B. Nivel metodológico

## 🔴 B-1. El VAE nunca fue validado como VAE

Consecuencia directa de A-1/A-3, pero es un punto metodológico propio: el
proyecto trató al VAE como una caja que produce puntajes, y validó **los
puntajes** (distribución, umbral, estabilidad de ranking) sin validar nunca
**el modelo generativo** que los produce.

Un VAE tiene condiciones de salud propias — dimensiones activas, balance
reconstrucción/KL, ausencia de colapso — que ninguna métrica de ranking
detecta. Es la razón de que un modelo completamente degenerado atravesara todo
el pipeline y llegara a un entregable.

**Corregido parcialmente** con el gate de A-3. Faltan: gap de reconstrucción
(§15 del marco), estabilidad estocástica (§19), capacidad latente (§18).

---

## 🟠 B-2. Selección de modelo sobre una sola métrica

Optuna optimiza un escalar. En modo supervisado es PR-AUC; en no supervisado,
la pérdida de validación (con el problema de A-2).

Ninguno captura lo que el proyecto declara buscar: estabilidad, robustez
temporal, calidad del ranking sin etiquetas. Un modelo puede ganar en PR-AUC y
ser inestable entre semillas, y hoy nada lo detectaría.

**Referencia:** `docs/validacion_no_supervisada.md` §24-25 propone el perfil
multidimensional que reemplazaría esta selección escalar.

---

## 🟠 B-3. Sin estabilidad entre semillas

`PipelineConfig.seed = 42`, una sola corrida. `_rank_stability` en
`metrics.py` calcula un proxy por *bootstrap jitter* — mide sensibilidad del
ranking al ruido en los datos, **no** al azar del ajuste.

Para el Isolation Forest la distinción importa: el muestreo de árboles es
estocástico, y dos semillas pueden producir rankings distintos sobre los mismos
datos. Hoy no hay forma de saber cuánto.

**Costo de cerrarlo:** bajo. Repetir el ajuste con 8 semillas y calcular
Spearman/Jaccard. Es la prueba con mejor relación valor/esfuerzo del marco.

---

## 🟡 B-4. Umbral calibrado, curva sin explorar

`thresholds.py` calibra **un** punto de operación (POT con Pareto
Generalizada, o percentil). No se examina cómo cambia el conjunto de alertas al
mover el umbral, ni cuánto se concentran las alertas en pocas unidades.

Para un panel bancario esto es operativamente relevante: un umbral que marca
siempre a las mismas 20 entidades tiene un problema de utilidad que ninguna
métrica actual expresa.

---

## 🟡 B-5. Solapamiento de entidades del 100% no se cuestiona

`measure_person_overlap` mide y documenta correctamente que en un panel
balanceado cerrado el 100% es *esperado por diseño*. Correcto para datos
sintéticos.

**El riesgo aparece con datos reales:** ahí hay altas, bajas y churn, y un
solapamiento del 100% dejaría de ser esperado — pasaría a ser señal de que el
panel se filtró a entidades presentes en todos los períodos, lo que sesga la
población evaluada. Hoy la métrica se registra pero no se contrasta contra
ninguna expectativa.

---

## 🟡 B-6. Sin calibración entre detectores

IF y VAE producen puntajes en escalas distintas y no comparables. El reporte
los muestra lado a lado con la convención "mayor = más anómalo", pero un
puntaje IF de 0.6 y uno VAE de 0.6 no significan lo mismo.

El gráfico de concordancia usa percentiles (correcto), pero los umbrales se
calibran por separado, así que "top 10% del IF" y "top 10% del VAE" pueden
tener tasas de alerta reales distintas.

---

## 🔵 B-7. El `robust` que gana está prohibido

Registrado en `docs/leakage_free_pipeline.md`: `numeric_transform="robust"` es
la mejor opción medida para el Isolation Forest (PR-AUC OOT 0.272 vs 0.117 de
`yeo-johnson`), pero produce 100% de puntajes NaN en el VAE.

El default es el compromiso que mantiene ambos vivos. Es una decisión
consciente y documentada, pero significa que **el bosque corre
permanentemente con una transformación subóptima** por una restricción del
otro modelo.

**Nota:** con A-1 corregido convendría re-medir. El desbordamiento del VAE bajo
`robust` puede estar agravado por el mismo desbalance de escala.

---

# Prioridades

| Orden | Acción | Nivel | Esfuerzo |
| --- | --- | --- | --- |
| 1 | **Re-tunear el VAE con datos reales.** Los `best_params_vae.yaml` previos al cambio de escala codifican un `beta` de la escala vieja y no son reutilizables | 🔴 | Medio |
| 2 | Decidir A-10: presupuesto de épocas de `--quick` | 🟠 | Bajo |
| 3 | Estabilidad entre semillas del IF (B-3) | 🟠 | Bajo |
| 4 | Re-medir `robust` para el IF con el VAE ya sano (B-7) | 🔵 | Bajo |
| 5 | Curva de umbral completa (B-4) | 🟡 | Bajo |
| 6 | EM/MV (`validacion_no_supervisada.md` §9) | 🟡 | Alto |

**Lo que haría primero:** re-tunear sobre datos reales. Los cuatro defectos
🔴/🟠 del VAE ya están corregidos y verificados, pero **ningún ajuste anterior
al 2026-08-23 sirve**: se hizo sobre una pérdida con otra escala. Cualquier
reporte de VAE generado antes de esa fecha describe un modelo colapsado.

---

## Nota sobre el alcance de esta revisión

Se revisó: `main.py`, `src/models/`, `src/preprocessing/`, `src/evaluation/`,
`src/interpretability/`, `src/reporting/`, `src/utils/`, y la documentación.

**No se revisó con la misma profundidad:** `src/data/synthetic.py` (generador
sintético — su corrección afecta al desarrollo, no a producción con datos
reales) ni el detalle de `flow_visualization.py` más allá del ciclo de vida
del servidor.

Los hallazgos A-1, A-2 y A-4 se verificaron **empíricamente** con experimentos
reproducibles, no por lectura de código. A-5 a A-8 y todo el bloque B son
observaciones de revisión, no fallos demostrados.
