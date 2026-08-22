# Validación de detección de anomalías no supervisada (IF + VAE)

Marco metodológico para validar los dos detectores del proyecto **sin depender
de ground truth**. Adapta una especificación general de 31 secciones a la
arquitectura real de este repositorio.

> **Cómo leer este documento.** Cada sección lleva una etiqueta de estado:
>
> | Etiqueta | Significado |
> | --- | --- |
> | **[IMPLEMENTADO]** | Existe hoy en el código, con la ruta del módulo |
> | **[PARCIAL]** | Existe una versión reducida; se indica qué falta |
> | **[PROPUESTO]** | No existe; es diseño pendiente de decisión |
>
> Esa distinción es el punto del documento: sin ella, una especificación
> ambiciosa se lee como si ya estuviera construida.

---

## 0. Fuentes bibliográficas

Toda referencia metodológica citada aquí fue contrastada con la fuente
primaria.

| Método | Referencia primaria | Nota |
| --- | --- | --- |
| Isolation Forest | Liu, F.T., Ting, K.M., Zhou, Z.H. *Isolation Forest*. ICDM 2008 (ext. TKDD 2012). | Defaults: `n_estimators=100`, `max_samples='auto'` (= min(256, n)), `max_features=1.0`, `bootstrap=False`. |
| Excess-Mass (EM) | Goix, N., Sabourin, A., Clémençon, S. *On Anomaly Ranking and Excess-Mass Curves*. AISTATS 2015 (PMLR v38). | EM*(t) = sup_Ω {P(X∈Ω) − t·Leb(Ω)}, no creciente en t, acotada en [0,1]. |
| Mass-Volume (MV) | Clémençon, S., Jakubowicz, J. *Scoring anomalies: a M-estimation formulation*. AISTATS 2013. Formalizado en Clémençon, S., Thomas, A. *Mass volume curves and anomaly ranking*. EJS 12(2), 2018. | **Corrección respecto de borradores previos**: la referencia no es "Clémençon & Robbiano" (esa es *bipartite ranking*, ICML 2014). |
| EM/MV en alta dimensión | Goix, N. *How to Evaluate the Quality of Unsupervised Anomaly Detection Algorithms?* arXiv:1607.01152, 2016. | Resuelve la estimación de Leb(Ω) vía muestreo uniforme + sub-muestreo de features. Sin esto, EM/MV no son implementables. |
| Local Outlier Factor | Breunig, M., Kriegel, H.P., Ng, R., Sander, J. *LOF*. ACM SIGMOD 2000. | Método de consenso, no ground truth. |
| Mahalanobis robusto (MCD) | Rousseeuw, P.J., Van Driessen, K. *A Fast Algorithm for the MCD Estimator*. Technometrics, 1999. | Base del "Robust Mahalanobis" del consenso. |
| VAE / ELBO | Kingma, D.P., Welling, M. *Auto-Encoding Variational Bayes*. ICLR 2014 (arXiv:1312.6114). | ELBO = E_q[log p(x\|z)] − KL(q(z\|x)‖p(z)). |
| β-VAE | Higgins, I. et al. *β-VAE*. ICLR 2017. | Justifica `Loss = Recon + β·KL`. |
| Posterior collapse / active units | Burda, Y., Grosse, R., Salakhutdinov, R. *Importance Weighted Autoencoders*. ICLR 2016 (arXiv:1509.00519). | Aⱼ = Var_x(E_q[z_j\|x]); activa si Aⱼ > δ, con δ = 0.01 como umbral de referencia. |

### Limitación declarada

**No existe una fuente única que defina "estabilidad de ranking para datos
panel" como marco unificado.** Las secciones 11–13 combinan prácticas de
validación panel (leave-one-period-out, análogo a leave-one-group-out) con las
métricas de estabilidad de Goix et al. Esa combinación es un **diseño de
validación propio**, metodológicamente razonable pero construido a partir de
piezas validadas por separado. No debe presentarse como un método publicado con
ese nombre. Si se implementa el dashboard de la sección 26, esta limitación
debe aparecer en él.

---

## 1. Objetivo

Validar los dos detectores del proyecto:

* **Isolation Forest** — `src/models/iforest.py` (Liu, Ting & Zhou, 2008/2012).
* **VAE** — `src/models/vae.py` (Kingma & Welling, 2014).

El objetivo **no** es medir accuracy, recall, precision ni F1: el escenario de
producción de este proyecto es **no supervisado por defecto**
(`PipelineConfig.supervised = False`), y las etiquetas del panel sintético
existen solo como instrumento de desarrollo.

Se evalúa: estabilidad del ranking, sensibilidad a semillas e hiperparámetros,
estabilidad temporal, comportamiento de los scores, concentración de alertas,
rareza estructural, generalización, calidad estadística del ranking sin
etiquetas (EM/MV), estabilidad para datos panel, y calidad de la representación
latente del VAE.

### Relación con el reporte que ya existe

El proyecto **ya genera** un reporte HTML interactivo
(`src/reporting/report.py`, 100% Plotly, offline) con métricas no supervisadas,
explicabilidad por modelo y una sección de confiabilidad estadística. Este
documento **no lo reemplaza**: describe el marco de validación más profundo que
ese reporte podría incorporar. Ver `docs/interpretability_and_reporting.md`.

---

## 2. Principios metodológicos obligatorios

Estos principios **ya rigen el código actual** y deben mantenerse:

* No usar accuracy, precision, recall, F1 ni ROC-AUC como métricas principales.
  → *Cumplido*: cuando `supervised=False`, `main.py` **no adjunta etiquetas** a
  `chart_data`, y el reporte declara explícitamente que no se calculó ninguna
  métrica basada en etiquetas.
* No interpretar una alerta como anomalía verdadera por tener score alto.
* Diferenciar estabilidad, rareza, concentración, generalización, calidad del
  ranking y exactitud supervisada.
* Cuando una métrica no sea interpretable sin etiquetas, declararlo.
  → *Cumplido parcialmente*: el glosario de `report_content.py` explica cómo
  leer cada indicador, pero no lleva la etiqueta literal *"Métrica no
  supervisada"*.
* Declarar cuándo un diseño de validación es construcción propia y no un método
  publicado (ver sección 0).

---

## 3. Arquitectura: propuesta vs. estructura real

La especificación original propone un paquete `validation/`. El proyecto usa
`src/evaluation/` para lo mismo. Mapeo:

| Módulo propuesto | Estado en este repositorio |
| --- | --- |
| `validation/stability.py` | **[PARCIAL]** `src/evaluation/metrics.py::_rank_stability` (proxy por bootstrap, no multi-semilla) |
| `validation/hyperparameters.py` | **[PARCIAL]** Optuna en `src/models/{iforest,vae}.py` optimiza, pero no analiza sensibilidad |
| `validation/threshold_analysis.py` | **[PARCIAL]** `src/evaluation/thresholds.py` calibra un umbral; no barre la curva completa |
| `validation/rarety.py` | **[PROPUESTO]** |
| `validation/panel_validation.py` | **[PARCIAL]** `src/utils/assumptions.py::measure_person_overlap` |
| `validation/leave_one_period_out.py` | **[PROPUESTO]** |
| `validation/excess_mass.py` | **[PROPUESTO]** |
| `validation/mass_volume.py` | **[PROPUESTO]** |
| `validation/vae_validation.py` | **[PARCIAL]** `src/interpretability/vae_explain.py` (espacio latente, error por feature); sin active units |

**Recomendación**: no crear un paquete `validation/` paralelo. Extender
`src/evaluation/` mantiene una sola ubicación para "cómo se juzga un detector"
y respeta la convención de rutas de `src/utils/paths.py`.

### Trazabilidad de corridas — [IMPLEMENTADO]

La especificación pide registrar configuración, semillas, hiperparámetros,
ventanas, métricas, timestamp, duración y errores. Eso **ya existe**:
`src/utils/observability.py` emite un stream JSON Lines
(`artifacts/logs/run_events.jsonl`) con `run_id`, hash de configuración,
huella del dataset, eventos de fase con duración, y health checks tipados.

---

## 4. Sistema de progreso en vivo — [IMPLEMENTADO]

Estados `PENDING`/`RUNNING`/`COMPLETED`/`FAILED` con progreso incremental: **ya
existe en dos formas**.

* **Terminal** — `src/utils/console_ui.py`: checklist fijo de las 15 fases
  (cajas que se encienden en su sitio), panel de supuestos IF/VAE en vivo, KPIs
  por modelo, salud del equipo (RAM/CPU) y cola de log.
* **Navegador** — `src/reporting/flow_visualization.py::start_live_view`:
  servidor HTTP local (127.0.0.1) que sirve el flujo en vivo durante la
  corrida, alimentado por el mismo stream de eventos.

**Diferencia con la especificación**: el progreso se organiza por **fase del
pipeline**, no por *experimento de validación*. Si se implementan las pruebas
de las secciones 6–23, el hook de observadores
(`logging_config.add_phase_observer`) permite añadirlas sin tocar los ~15
call-sites existentes.

---

## 5. Identificación de las observaciones — [IMPLEMENTADO]

`PanelSchema` (`src/data/loader.py`) infiere `entity_col` y `time_col`. El
identificador de observación es el par `(entity_id, period)`, preservado
como `keys` a través de todo el preprocesamiento para poder cruzar scores con
la verdad oculta.

**`entity_id` nunca se usa como variable predictora** — `PanelFeatureEngineer`
elimina las columnas clave antes de construir la matriz. Es una garantía del
código, no una convención.

---

## 6. Isolation Forest: estabilidad ante semillas — [PROPUESTO]

Hoy el proyecto corre **una sola semilla** (`PipelineConfig.seed = 42`).

Propuesta: `seeds = [7, 21, 42, 84, 123, 256, 512, 1024]`.

Por corrida: score, ranking, top 1%/5%/10%. Métricas a calcular:

* **Spearman entre rankings** (matriz por pares; media, mediana, mín, máx, SD).
* **Pearson entre scores**. Reportar **por separado** de Spearman: Pearson mide
  relación lineal de valores, Spearman preservación de orden. Son propiedades
  distintas y promediarlas no significa nada.
* **Variabilidad individual**: σᵢ = SD(score_{i,1..S}); distribución, mediana,
  P95, P99, observaciones más inestables.
* **Jaccard** top 1%/5%/10%: J(A,B) = |A∩B| / |A∪B|.
* **Frecuencia de selección**: pᵢ = #{s : i ∈ Top_k^(s)} / S.

> **Interpretación.** pᵢ mide la estabilidad de selección bajo distintas
> semillas. **No** es la probabilidad de que la observación sea una anomalía
> real.

**Relación con lo existente**: `_rank_stability` en `metrics.py` ya calcula un
proxy — Spearman medio bajo *bootstrap jitter*, no bajo re-entrenamiento con
semillas distintas. Es más barato y más débil; mide sensibilidad del ranking al
ruido, no al azar del ajuste.

---

## 7. Estabilidad ante hiperparámetros — [PARCIAL]

Optuna **ya explora** el espacio (`tune_iforest`, `tune_vae`), pero para
*optimizar*, no para *medir sensibilidad*. La diferencia importa: un estudio de
Optuna dice cuál es el mejor punto, no cuánto cambia el ranking al moverse.

Espacio real del IF (`src/models/iforest.py`): `n_estimators` ∈ [100,600] paso
50; `max_samples_mode` ∈ {auto, int, float}; `max_features` ∈ [0.3,1.0];
`bootstrap` ∈ {True, False}.

Por configuración calcular **solo**: Spearman entre rankings, Pearson entre
scores, Jaccard top-k, cambios en percentiles del score, concentración por
unidad y temporal. **No** usar recall/precision/accuracy/F1.

Salida: heatmaps `configuración × configuración`.

> Los presupuestos de Optuna y su reparto explorar/explotar están justificados
> en `docs/decisiones_de_modelado.md` §1, incluido el hallazgo de que TPE no
> guiaba ningún trial con los presets pequeños.

---

## 8. Curva tasa de alertas–umbral — [PARCIAL]

`src/evaluation/thresholds.py` **ya calibra** un umbral sobre validación (POT
con Pareto Generalizada, o percentil), y el reporte muestra cuántas filas OOT
superan ese umbral. Lo que falta es **barrer la curva completa**.

Percentiles propuestos: 0.1%, 0.5%, 1%, 2%, 5%, 10%, 15%, 20%. Por umbral:
observaciones seleccionadas (n y %), unidades afectadas, concentración
temporal/por segmento/por unidad, solapamiento entre umbrales (Jaccard).

Curvas: `% alertas vs % unidades afectadas`, `umbral vs concentración`.

> **Interpretación.** Estas curvas describen el comportamiento **operativo**
> del umbral. No demuestran que las alertas sean correctas.

---

## 9. Excess-Mass y Mass-Volume — [PROPUESTO]

* **EM** (Goix et al., 2015): EM*(t) = sup_Ω{P(X∈Ω) − t·Leb(Ω)}, no creciente
  en t, acotada en [0,1]. Mayor área bajo la curva es preferible.
* **MV** (Clémençon & Jakubowicz 2013; Clémençon & Thomas 2018):
  MV(α) = inf{Leb(s≥u) : P_n(s≥u) ≥ α}. Menor volumen para cubrir masa α es
  preferible.

### Nota de implementación (obligatoria)

`Leb(Ω)` **no es calculable analíticamente en dimensión > 1**. Procedimiento
estándar (Goix, 2016):

1. Generar un conjunto de referencia uniforme sobre un hiper-rectángulo que
   contenga el soporte empírico (bounding box con margen).
2. Estimar `Leb(s ≥ u)` como la proporción de puntos uniformes con score ≥ u,
   multiplicada por el volumen del hiper-rectángulo.
3. En alta dimensión, aplicar **sub-muestreo de features** para evitar
   degeneración del estimador.

Sin este paso, EM/MV quedan como fórmulas teóricas, no como código ejecutable.

**Nota específica para este proyecto:** la matriz post-preprocesamiento incluye
one-hot de categóricas, así que el bounding box tiene dimensiones binarias
donde el muestreo uniforme no representa el soporte real. Al implementarlo hay
que decidir explícitamente si EM/MV se calculan solo sobre el bloque continuo
(`categorical_feature_mask` en `src/preprocessing/pipeline.py` ya separa
ambos) o con un muestreo mixto.

No interpretar EM/MV como accuracy.

---

## 10. Rareza estructural y consenso — [PROPUESTO]

Para observaciones con score alto: densidad local, distancia al vecino más
cercano, Mahalanobis robusto (MCD, Rousseeuw & Van Driessen 1999), profundidad
estadística, percentiles marginales, rareza condicional dentro de la unidad.

Comparar: Isolation Forest, LOF, Robust Mahalanobis (MCD), densidad local,
score por percentiles.

```
consenso_i = (# métodos que seleccionan i) / (# métodos)
```

Mostrar matriz de solapamiento, Jaccard, Spearman, distribución del consenso.

> **Interpretación.** El consenso mide acuerdo entre mecanismos no
> supervisados. **No** es ground truth: cinco métodos pueden coincidir y
> equivocarse juntos si comparten el mismo sesgo.

---

## 11. Validación específica para datos panel — [PARCIAL]

```
r_i = (# periodos en que i aparece en Top_k) / (# periodos observados para i)
```

Reportar: distribución de rᵢ (mediana, P95, P99), concentración de alertas en
pocas unidades, concentración por periodo, persistencia, y dependencia
`score ~ tamaño de unidad` — **sin concluir causalidad**.

**Lo que ya existe**: `measure_person_overlap` (`src/utils/assumptions.py`)
mide qué fracción de entidades OOT aparece en train/val, y documenta por qué
un solapamiento del 100% es *esperado por diseño* en un panel balanceado
cerrado, no un defecto.

---

## 12. Persistencia de alertas — [PROPUESTO]

Por unidad: periodos consecutivos alertada, racha máxima, total de periodos
alertados, % alertado, tiempo hasta primera/última alerta.

Visualizaciones: distribución de persistencia, heatmap unidad × periodo.

> **Distinción crítica.** *Persistencia* = patrón temporal de un individuo.
> *Estabilidad del modelo* = comportamiento del detector bajo perturbación.
> Son conceptos distintos y no intercambiables.

---

## 13. Leave-one-period-out — [PROPUESTO]

Análogo a leave-one-group-out aplicado a ranking no supervisado.
**Construcción metodológica propia** — ver la limitación de la sección 0.

Por periodo t: retirar t del entrenamiento, reentrenar, evaluar, comparar con
el modelo base. Calcular: correlación de scores, Spearman, Jaccard top-k,
concentración, EM/MV cuando la sección 9 esté implementada.

**Separar tres efectos distintos**, que de otro modo se confunden:

1. cambio de composición (menos datos de entrenamiento);
2. cambio del modelo (parámetros reajustados);
3. cambio del ranking (resultado observable).

**Restricción de este proyecto**: la partición cronológica
(`src/evaluation/splits.py`) exige ≥3 períodos distintos. Con paneles cortos,
retirar un período puede invalidar los horizontes de contraste — ver los dos
peligros de varianza cero en `docs/decisiones_de_modelado.md` §2.3.

---

## 14. VAE: convergencia — [IMPLEMENTADO]

`VAEDetector.fit` **ya registra** por época: pérdida de reconstrucción, KL,
pérdida total, y las mismas sobre validación, con early stopping y restauración
de los pesos de la mejor época.

Convención (Kingma & Welling, 2014):

```
ELBO = E_q[log p(x|z)] − KL(q(z|x) ‖ p(z))
Loss = ReconstructionLoss + β·KL        (= −ELBO cuando β = 1)
```

El código minimiza la forma β-VAE (Higgins et al., 2017) con `beta`
configurable y rampa lineal de KL sobre `kl_anneal_epochs`. La equivalencia de
signos está documentada en `src/models/vae.py` y en el reporte técnico
(`model_documentation.md`).

**Falta**: gráficos de convergencia por época en el reporte. Hoy se registran
en el log, pero no se grafican.

---

## 15. Gap de reconstrucción — [PROPUESTO]

```
gap_rec = median(e_validation) − median(e_training)
```
(ídem con P95 y P99).

> **Interpretación.** Un gap grande *puede* indicar overfitting, pero no es
> prueba por sí solo. Verificar también que un gap pequeño no sea consecuencia
> de **sobrecapacidad**: una red que reconstruye todo bien, incluido el ruido,
> también produce un gap pequeño — y es exactamente el fallo que destruye la
> señal de anomalía, porque el puntaje *es* el error de reconstrucción.

---

## 16. Error de reconstrucción por variable — [IMPLEMENTADO]

`reconstruction_error_by_feature` (`src/interpretability/vae_explain.py`)
calcula `e_itj = |x_itj − x̂_itj|` agregado por feature, y el reporte lo
muestra como gráfico Plotly interactivo ordenado por magnitud.

**Falta**: contribución porcentual al error total, error por unidad, error por
período, y evolución temporal.

**Control de escala**: la matriz ya llega estandarizada del preprocesamiento,
así que las contribuciones son comparables entre variables. Si se cambiara a
`numeric_transform="passthrough"`, esta comparación dejaría de ser válida.

---

## 17. Posterior collapse — [PROPUESTO]

Métrica de referencia (Burda et al., 2016):

```
A_j = Var_x( E_q[z_j | x] )
dimensión j activa  ⟺  A_j > δ,   δ = 0.01
```

Reportar: `active_units`, `inactive_units`, KL media por dimensión,
distribución de KL, `Var(mu)`.

> **No declarar posterior collapse solo porque una KL sea baja.** Requiere
> evidencia conjunta: KL≈0 en múltiples dimensiones, pocas dimensiones activas
> (Aⱼ < δ), baja variabilidad de las representaciones, y posterior ≈ prior.

**Por qué importa aquí**: el espacio de búsqueda incluye `beta` ∈ [0.1, 2.0].
Un β alto empuja hacia el colapso, y hoy **nada en el pipeline lo detecta** —
un VAE colapsado seguiría produciendo puntajes y un `best_params.yaml`
poblado. Es el hueco de validación más relevante de esta lista.

---

## 18. Capacidad latente — [PARCIAL]

Optuna ya explora `latent_dim` ∈ [4, 32], pero optimizando una sola métrica.

Propuesta: por dimensión, reportar reconstrucción de validación (+P95/P99), KL,
dimensiones activas, entropía, estabilidad del ranking y Jaccard top-k.

> **No** seleccionar la dimensión solo por mínimo error de reconstrucción:
> ese criterio premia la sobrecapacidad, que es justo lo que destruye la señal
> (§15). Buscar equilibrio entre generalización, estabilidad y ausencia de
> sobrecapacidad.

---

## 19. Estabilidad estocástica del VAE — [PROPUESTO]

Varias semillas; comparar correlación de scores, Spearman, Jaccard top-k,
variación de la reconstrucción, del KL y de las dimensiones activas.

> **Advertencia técnica.** **No comparar** `z_seed_1[:,0]` contra
> `z_seed_2[:,0]`: el espacio latente de un VAE no tiene base canónica, y las
> coordenadas pueden rotarse o permutarse entre entrenamientos. Comparar en su
> lugar: score final, reconstrucción, distribución latente agregada, distancias
> y rankings.

---

## 20. Coherencia entre componentes del VAE — [PROPUESTO]

Tres rankings: error de reconstrucción, rareza latente, score combinado.

Calcular Spearman, Jaccard, solapamiento y contribución porcentual de cada
componente. Emitir advertencia cuando la contribución de un componente supere
un umbral configurable (documentar el umbral y su justificación).

**Estado actual**: el puntaje del VAE es **solo** error de reconstrucción
(`score_kl_weight` existe pero es 0 por defecto), así que hoy no hay tres
rankings que comparar. Esta sección aplica si se activa el término KL.

---

## 21. Estabilidad temporal del VAE — [PROPUESTO]

Por ventana temporal: reconstrucción, KL, score total, P95/P99, top-k, Jaccard
entre períodos, persistencia por unidad, dimensiones activas.

Comparar **distribuciones** entre ventanas, no solo un umbral absoluto fijo.

---

## 22. Densidad latente — [PROPUESTO]

Ajustar la distribución de referencia **únicamente con el conjunto de
referencia**, nunca con datos de evaluación — esto es leakage, y el proyecto ya
lo prohíbe estructuralmente (ver `docs/leakage_free_pipeline.md` fase 3).

Calcular Mahalanobis robusto, log-densidad, percentiles, densidad local,
distancia al vecino más cercano. Comparar `ranking de rareza latente` contra
`ranking de reconstrucción` como prueba de coherencia interna.

---

## 23. EM/MV para el VAE — [PROPUESTO]

Aplicar la sección 9 a: score final, error de reconstrucción, KL, distancia
latente, score combinado.

> No seleccionar el modelo por una métrica EM/MV aislada. Buscar consistencia
> conjunta entre EM/MV, estabilidad, generalización, comportamiento latente y
> robustez temporal.

---

## 24. Comparación global IF vs VAE — [PARCIAL]

| Dimensión | Estado |
| --- | --- |
| Estabilidad semillas | [PROPUESTO] §6, §19 |
| Estabilidad top-k | [PROPUESTO] §6 |
| Estabilidad temporal | [PROPUESTO] §21 |
| Sensibilidad hiperparámetros | [PROPUESTO] §7 |
| Concentración por unidad | [PROPUESTO] §11 |
| Persistencia | [PROPUESTO] §12 |
| Rareza estructural | [PROPUESTO] §10 |
| Excess-Mass / Mass-Volume | [PROPUESTO] §9, §23 |
| Generalización | [PARCIAL] evaluación OOT |
| Estabilidad del score | [PARCIAL] `_rank_stability` |
| Interpretabilidad | [IMPLEMENTADO] SHAP, longitud de camino, latente, error por feature |

**El reporte ya compara ambos modelos lado a lado** y evita deliberadamente
nombrar un "ganador": el pipeline ejecuta y reporta *ambos* detectores. No
producir un único "accuracy score" — construir un **perfil multidimensional**.

---

## 25. Scorecard final — [PROPUESTO]

Categorías: `STABILITY`, `TEMPORAL ROBUSTNESS`, `PANEL ROBUSTNESS`,
`STRUCTURAL RARITY`, `RANKING QUALITY`, `GENERALIZATION`, `MODEL HEALTH`,
`OPERATIONAL CONCENTRATION`.

Cada una con: métricas, tendencia, nivel de alerta, explicación y evidencia.

> Un índice agregado es **opcional** y debe documentarse como *índice de
> evaluación metodológica*, nunca como accuracy.

---

## 26. Dashboard HTML — [PARCIAL]

**Ya existe** un reporte HTML offline con: header con metadatos, figura hero,
KPIs agrupados, resultados interactivos, explicabilidad por modelo, tarjetas por
modelo, guía de indicadores, sección de confiabilidad estadística, glosario de
parámetros y posicionamiento ML-vs-econometría. Todo en Plotly, con tema
claro/oscuro y tablas responsivas.

Secciones de la especificación **que faltarían**: Excess-Mass, Mass-Volume,
rareza estructural, análisis panel, leave-one-period-out, diagnósticos VAE
(active units, gap), sistema de advertencias, log de experimentos y
**referencias bibliográficas**.

---

## 27. Sistema de alertas — [PROPUESTO]

```text
WARNING: Jaccard top-5% entre semillas es bajo.
WARNING: Las alertas se concentran en muy pocas unidades.
WARNING: Gap de reconstrucción train-validation grande.
WARNING: Muy pocas dimensiones latentes activas (< δ).
WARNING: El ranking cambia sustancialmente al retirar un solo período.
INFO: El ranking del IF es muy estable entre semillas.
```

> **No usar** `TRUE ANOMALY`, `FALSE POSITIVE` ni `TRUE POSITIVE` salvo que se
> disponga de ground truth confiable.

**Base existente**: `observability.check(...)` ya implementa health checks
tipados con severidad, y el dashboard de consola los muestra en vivo. Las
alertas de esta sección encajan en ese mecanismo sin infraestructura nueva.

---

## 28. Reproducibilidad — [IMPLEMENTADO]

```text
artifacts/logs/run_events.jsonl     # config, semilla, hash, fases, health checks
artifacts/tuning/best_params_*.yaml # hiperparámetros seleccionados
artifacts/models/                   # checkpoints
artifacts/reports/                  # HTML, MD, model_documentation.md
```

Cada corrida lleva `run_id`, `config_hash` y huella del dataset. `model_documentation.md`
incluye el catálogo completo de artefactos escritos.

**Falta** respecto de la especificación: `scores.parquet` y `rankings.parquet`
por experimento (hoy los puntajes solo persisten dentro del Excel OOT).

---

## 29. Criterios de calidad del código

Funciones independientes, type hints, docstrings, validación de inputs,
logging, manejo de excepciones, configuración centralizada, reproducibilidad.

> **Nota sobre tests.** La especificación pide tests unitarios de cada métrica
> (Spearman, Jaccard, EM, MV, active units con δ=0.01 como caso de prueba).
> El proyecto **tuvo** una suite de 319 tests que se eliminó por decisión
> explícita el 2026-08-22 (ver `CONTEXT.md`). Si se implementan las métricas de
> este documento, conviene reconsiderar esa decisión **al menos para ellas**:
> son fórmulas con convenciones de signo y normalización fáciles de invertir en
> silencio, y un error ahí no se manifiesta como excepción sino como un número
> plausible pero incorrecto.

Validar con datasets sintéticos de comportamiento conocido (mezcla gaussiana
con outliers de posición conocida) — sin usarlos como ground truth de
producción. `src/data/synthetic.py` ya genera cuatro geometrías de anomalía
(`global`, `local`, `contextual`, `collective`), lo que sirve exactamente para
esto.

---

## 30. Regla crítica de interpretación

Distinguir explícitamente en todo dashboard y documentación:

* **Estabilidad** — "El modelo produce resultados similares bajo perturbaciones."
* **Rareza** — "La observación es inusual según uno o varios detectores."
* **Consenso** — "Distintos detectores no supervisados coinciden en seleccionarla."
* **Robustez** — "El resultado se mantiene ante cambios de datos, períodos o hiperparámetros."
* **Exactitud** — "El modelo identifica correctamente una anomalía real."
  **Esta afirmación no puede hacerse sin ground truth adecuado.**

---

## 31. Entregables y estado

| # | Entregable | Estado |
| --- | --- | --- |
| 1 | Código completo | [IMPLEMENTADO] pipeline de 12 fases |
| 2 | Dashboard HTML interactivo | [PARCIAL] §26 |
| 3 | Ejecución reproducible | [IMPLEMENTADO] §28 |
| 4 | Progreso en vivo | [IMPLEMENTADO] §4 |
| 5 | Resultados parciales durante ejecución | [IMPLEMENTADO] vista de flujo en vivo |
| 6 | Gráficos y tablas | [IMPLEMENTADO] 9 gráficos Plotly |
| 7 | Logs | [IMPLEMENTADO] texto + JSONL |
| 8 | Configuración experimental | [IMPLEMENTADO] `PipelineConfig` + CLI |
| 9 | Métricas exportables | [PARCIAL] JSONL sí; parquet no |
| 10 | Scorecard final | [PROPUESTO] §25 |
| 11 | Documentación metodológica | [IMPLEMENTADO] este documento + `docs/` |

### Validaciones automáticas previas a la ejecución — [PARCIAL]

`src/utils/assumptions.py` **ya valida**: esquema, calidad de datos,
integridad de la partición temporal, solapamiento de entidades, finitud de la
matriz antes de cada `.fit()`, y rango de `contamination`. Las que faltan son
específicas de este marco: convención EM/MV, y consistencia ELBO/KL con la
pérdida implementada.

### Sección "Methodological Validation" — [PROPUESTO]

Por prueba: `Test | Fórmula | Implementación | Estado | Interpretación | Limitaciones`,
con estado `VALID` / `WARNING` / `INVALID` / `NOT APPLICABLE`.

> No ejecutar en silencio una métrica que no cumple sus supuestos: documentar
> el problema y mostrarlo.

---

## Prioridades sugeridas

Si se implementa parte de este marco, el orden por relación valor/esfuerzo:

1. **§17 Posterior collapse** — el hueco más serio: hoy un VAE colapsado pasa
   inadvertido y produce puntajes plausibles. Barato de calcular.
2. **§6 Estabilidad ante semillas (IF)** — el más citado y el más simple; solo
   requiere repetir el ajuste variando la semilla.
3. **§8 Curva de umbral completa** — extiende algo que ya existe.
4. **§15 Gap de reconstrucción** — dos líneas sobre datos ya calculados.
5. **§9 EM/MV** — el más valioso metodológicamente y el más caro; requiere
   resolver antes la cuestión del muestreo con one-hot.
