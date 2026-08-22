# Decisiones de modelado: presupuesto de tuneo, transformaciones y hallazgos

Documento de sustento para tres preguntas que el código responde pero que no
estaban escritas en ningún lado:

1. ¿Por qué **N pruebas** en Isolation Forest y **M** en el VAE, si no son el
   mismo número?
2. ¿Por qué Optuna prueba **distintos `epochs`** en el VAE, en vez de fijarlos?
3. ¿Qué hace exactamente cada parte del `.py` de transformaciones y cómo afecta
   al pre-modelamiento?

Cierra con los **hallazgos** de esta ronda de trabajo (warnings y defectos que
afectaban el desempeño o la validez) y con la **evaluación de desempeño** de los
cambios.

> **Actualización — generación de PDF eliminada.** Por decisión explícita, el
> proyecto ya **no genera ningún PDF**: ni el PDF de negocio que producía
> `build_report` (`_build_pdf` en `src/reporting/report.py`), ni el sistema
> que exportaba cada `.py` a PDF (`docs/build_py_pdfs.py` y la carpeta
> `docs/py_pdf/`, con su manifiesto `0_INSTRUCCIONES.pdf`) — ambos fueron
> eliminados del repositorio. Los hallazgos #7, #8 y #9 de la sección 3 y las
> métricas de la sección 4.1 sobre el PDF de negocio describen trabajo que
> **ya no existe en el código**; se conservan aquí solo como registro de las
> correcciones que se hicieron mientras ese sistema estuvo activo (líneas en
> blanco que se perdían al exportar, líneas largas que se partían sin marca,
> glifos sin cobertura descartados en silencio). El reporte actual entrega
> **HTML + Markdown + `model_documentation.md`**, sin PDF. `fpdf2` y
> `fonttools` se quitaron de `requirements.txt` al no tener ya ningún uso.

---

## 1. Presupuesto de tuneo: por qué IF y VAE no llevan el mismo número

### 1.1 Los números que usa el proyecto

Definidos en los presets de `main.py`:

| Preset      | `iforest_trials` | `vae_trials` | `vae_epochs` |
| ----------- | ---------------- | ------------ | ------------ |
| `--quick`   | 5                | 5            | 5            |
| *(default)* | **15**           | **10**       | **15**       |
| `--full`    | 50               | 30           | 30           |

> **Nota de honestidad.** Estos valores ya existían en el proyecto; este
> documento explica la lógica que los justifica y corrige un defecto que los
> volvía menos efectivos de lo que aparentaban (§1.4). El único número que
> introduje es el reparto explorar/explotar de §1.4.

### 1.2 Por qué el Isolation Forest lleva *más* pruebas que el VAE

Parece al revés —el VAE tiene un espacio de búsqueda más grande— y sin embargo
recibe menos pruebas. La razón no es estadística sino de **costo por prueba**.

**Espacio de búsqueda del Isolation Forest** (`src/models/iforest.py`):

| Parámetro           | Rango                          | Cardinalidad |
| ------------------- | ------------------------------ | ------------ |
| `n_estimators`      | 100–600, paso 50               | 11 valores   |
| `max_samples_mode`  | `auto` / `int` / `float`       | 3 (condicional) |
| └ `max_samples`     | 0.3–1.0 (si `float`)           | continuo     |
| └ `max_samples_int` | {64, 128, 256} (si `int`)      | 3            |
| `max_features`      | 0.3–1.0                        | continuo     |
| `bootstrap`         | {True, False}                  | 2            |

Son ~4 dimensiones efectivas, dos de ellas continuas y una condicional.

**Espacio de búsqueda del VAE** (`src/models/vae.py`): 9 dimensiones —
`latent_dim` [4,32], `lr` [1e-4,1e-3] log, `optimizer` {adam, adamw, rmsprop},
`batch_size` {128,256,512}, `beta` [0.1,2.0], `dropout` [0.1,0.4], `n_layers`
[1,3], `hidden_dim` {32,64,128}, `epochs` [1,`max_epochs`].

**El costo invierte la asignación.** Un trial del bosque ajusta N árboles sobre
una submuestra y termina en **segundos**; medido en este proyecto, la fase 6
completa (incluido el ajuste final) toma ~9 s. Un trial del VAE entrena una red
neuronal durante hasta `epochs` épocas: la fase 7 toma ~6 s *por ajuste*, y con
`--full` (30 épocas) crece proporcionalmente.

La consecuencia práctica:

> El bosque puede permitirse muchas pruebas porque cada una es barata. El VAE
> no, y por eso su presupuesto se gasta en **menos pruebas pero mejor guiadas**
> (§1.4) en lugar de en fuerza bruta.

Cubrir 9 dimensiones "bien" exigiría cientos de trials. Con 10–30 no se está
haciendo una búsqueda exhaustiva y **el documento no debe pretender que sí**:
se está haciendo una búsqueda dirigida que mejora sobre el default, y el
resultado se valida después contra el bloque OOT, que es donde realmente se
decide si sirve.

### 1.3 Por qué `epochs` es un hiperparámetro tuneado y no una constante

Es la pregunta más interesante, porque en la mayoría de proyectos las épocas se
fijan y se deja que el *early stopping* decida.

Aquí `epochs` entra al espacio de búsqueda por tres razones:

**(a) En un VAE, las épocas no son solo "tiempo de cómputo": son capacidad
efectiva.** El puntaje de anomalía **ES** el error de reconstrucción. Una red
entrenada de más reconstruye bien *todo*, incluidas las anomalías, y comprime
justamente la señal que se busca. Una entrenada de menos reconstruye mal todo,
y el puntaje se vuelve ruido. Existe un punto intermedio, y **depende de los
demás hiperparámetros**: una red con `latent_dim=25` y `n_layers=3` llega a ese
punto en muchas menos épocas que una con `latent_dim=4` y `n_layers=1`.

Fijar las épocas obligaría a elegir un valor que es simultáneamente demasiado
para las redes grandes y demasiado poco para las pequeñas. Tunearlas **junto
con** la arquitectura deja que Optuna encuentre la combinación coherente.

**(b) Interactúa con `kl_anneal_epochs`.** El peso KL sube linealmente de 0 a
`beta` durante las primeras `kl_anneal_epochs` (default 10). Si el presupuesto
total de épocas fuera menor que la rampa, el modelo **nunca** llegaría a
entrenar con el KL completo: se estaría evaluando un autoencoder casi ordinario
y reportándolo como VAE. Que `epochs` sea visible para el tuner hace que esa
interacción sea explícita y medible en vez de accidental.

**(c) Es el mecanismo de *pruning* barato.** Un trial que sortea `epochs=3`
termina rápido; si su valor objetivo es malo, se descartó una región del
espacio a bajo costo. Optuna aprende que esa combinación no sirve sin haber
pagado 30 épocas por averiguarlo.

**Por qué el tope es 15 en el default.** `vae_epochs` es la *cota superior*
(`suggest_int("epochs", 1, max_epochs)`), no el valor usado. 15 es el punto
donde, en este dataset, la pérdida de validación se aplana: por encima, el
early stopping interno (`early_stopping_patience=10`, sobre pérdida de
validación y restaurando los pesos de la mejor época) corta antes de agotar el
presupuesto. Subirlo a 30 con `--full` da margen a las arquitecturas grandes
sin cambiar el comportamiento de las pequeñas, que se detienen solas.

> Hay **dos** early stoppings y conviene no confundirlos:
> el *por época* dentro de cada trial (`VAEDetector.early_stopping_patience`),
> y el *por trial* que detiene el estudio completo cuando N trials seguidos no
> mejoran (`patience=10`, `min_delta=0.005`, `min_trials=10` — ver
> `src/models/_tuning_stop.py`).

### 1.4 Hallazgo: el reparto explorar/explotar anulaba el tuneo

`TPESampler` sortea sus primeros `n_startup_trials` **al azar** para construir
el modelo de densidad que luego optimiza. El default de Optuna es **10**, y ese
default asume presupuestos mucho mayores que los de este proyecto.

Cruzando ese 10 con los presets, *antes* de la corrección:

| Preset      | Trials IF / VAE | Aleatorios | Guiados por TPE |
| ----------- | --------------- | ---------- | --------------- |
| `--quick`   | 5 / 5           | 5 / 5      | **0 / 0**       |
| *(default)* | 15 / 10         | 10 / 10    | 5 / **0**       |
| `--full`    | 50 / 30         | 10 / 10    | 40 / 20         |

**En el preset por defecto el "tuneo" del VAE era búsqueda aleatoria pura:**
TPE no llegaba a guiar ni un solo trial. Con `--quick`, ninguno de los dos
modelos se optimizaba.

Es un defecto que se esconde bien: el estudio devuelve el mejor de N sorteos
aleatorios, escribe un `best_params_*.yaml` poblado y reporta métricas
plausibles. No falla; simplemente no hace lo que dice hacer.

**Corrección** (`src/models/_tuning_budget.py`): el presupuesto de exploración
se escala al presupuesto de trials, `clamp(n_trials // 3, 3, 10)`:

| Preset      | Trials IF / VAE | Aleatorios | Guiados por TPE |
| ----------- | --------------- | ---------- | --------------- |
| `--quick`   | 5 / 5           | 3 / 3      | 2 / 2           |
| *(default)* | 15 / 10         | 5 / 3      | 10 / 7          |
| `--full`    | 50 / 30         | 10 / 10    | 40 / 20         |

Un tercio explorando es el reparto habitual para presupuestos chicos, y deja la
mayoría de los trials para la fase guiada —que es la razón de usar TPE. **En
presupuestos de 30+ el valor sigue siendo 10**, así que las corridas `--full`
que ya funcionaban no cambian de comportamiento.

*Pruebas:* `tests/test_tuning_budget.py` (18 casos), incluida la garantía de
que ningún preset gasta el presupuesto entero explorando.

---

## 2. Transformación del DataFrame y su sustento en el pre-modelamiento

Hay **dos** rutas de transformación en el proyecto y conviene no confundirlas.

| Ruta | Módulo | Estado | Uso |
| --- | --- | --- | --- |
| Pipeline sklearn | `src/preprocessing/pipeline.py` | `fit`/`transform` | La que usa `main.py` |
| Escalado lineal | `src/preprocessing/linear_scaling.py` | sin estado | EDA / serving liviano |

La segunda está documentada en detalle en
[`docs/escalamiento_lineal.md`](escalamiento_lineal.md). Esta sección explica la
primera, que es la que efectivamente alimenta a los modelos.

### 2.1 Las dos etapas y cuál puede filtrar información

`fit_transform_panel` arma un `Pipeline` de dos pasos, y **solo uno puede
filtrar**:

**Etapa 1 — `PanelFeatureEngineer` (no estima nada).** Elimina las columnas
clave y datetimes sueltos, normaliza dtypes (`Int64` → `float64`), y
opcionalmente agrega features de panel: rezagos, diferencias, ratios y z-score
de historia propia, más codificación cíclica del mes. Todos miran estrictamente
hacia atrás, así que corre sobre **el panel completo** sin riesgo.

*Por qué importa en pre-modelamiento:* es lo que convierte un nivel absoluto
("saldo = 40.000") en un contraste contra la propia historia del individuo
("el saldo se triplicó"). Sin eso, un detector que corta en umbrales absolutos
marca permanentemente a los clientes grandes y nunca ve el cambio.

**Etapa 2 — `ColumnTransformer` (estima todo lo que puede filtrar).** Medianas
de imputación, momentos del escalador, exponentes de Yeo-Johnson, categorías
one-hot, frecuencias y la elección `"auto"` por columna. Se ajusta **solo con
`df[fit_mask]`** (el bloque de entrenamiento).

*Por qué importa:* ajustar un escalador con todos los períodos filtra
conocimiento distribucional del futuro hacia las entradas del modelo. La fuga
es sutil —no se toca ninguna etiqueta— pero infla las métricas en una magnitud
que no se puede estimar después.

> **La trampa que esto evita.** El atajo tentador —ajustar en train y luego
> llamar `transform` solo sobre las filas de test— destruye silenciosamente los
> features de panel: `shift(1)` no encuentra historia dentro de un subconjunto
> de un solo período, todos los lags quedan en NaN, se rellenan con 0.0 y
> terminan idénticamente nulos justo en las filas que se están evaluando.
> Separar por *etapa* en vez de por *llamada* resuelve la fuga sin romper las
> dependencias temporales.

### 2.2 Cómo cada transformación numérica afecta a cada modelo

Éste es el punto donde los dos detectores **quieren cosas opuestas**, y es la
decisión de pre-modelamiento más consecuente del proyecto.

| Transformación | Qué le hace a la distribución | Isolation Forest | VAE |
| --- | --- | --- | --- |
| `standard` | Solo ubicación/escala; forma intacta | Indiferente (invariante monótono) | Bien: gradientes acotados |
| `robust` | Ídem, con mediana/IQR | **Mejor** (PR-AUC OOT 0.272) | **Rompe**: 100% NaN |
| `log1p` | Comprime la cola derecha | Pierde algo de señal de cola | Bien |
| `yeo-johnson` | Ajusta un exponente hacia normal | PR-AUC OOT 0.117 | Bien — **es el default** |
| `quantile` | Normaliza por rangos | **Destructivo** | Bien |

**Por qué `robust` rompe el VAE.** Deja intacta la cola pesada, que en unidades
escaladas alcanza ~5e5. La pérdida del VAE es un error cuadrático medio, así
que ese valor aporta ~1e11 al gradiente y desborda: los puntajes salen `NaN`.

**Por qué `quantile` es destructivo para la detección.** Mapea los percentiles
superiores a un rango normal acotado, aplanando exactamente la cola que
constituye la señal de anomalía. Más en general: **cualquier transformación
ajustada para gaussianizar se ajusta sobre datos que contienen las anomalías**,
así que las normaliza parcialmente.

**Por qué el default es `yeo-johnson` y no el mejor para el bosque.** Los dos
modelos comparten una sola matriz. `robust` gana para el bosque pero deja al
VAE inutilizable, así que el default es el compromiso que mantiene a ambos
funcionando. Es una decisión consciente, no un descuido — está registrada en
`CONTEXT.md` y en `docs/leakage_free_pipeline.md`.

### 2.3 Los dos peligros de varianza cero

Una ventana de entrenamiento corta puede volver una columna casi constante *en
el bloque de ajuste*. Cualquier escalador ajustado divide entonces valores no
vistos por una varianza ~0 y los amplifica sin límite.

* **Estacionalidad cíclica.** `month_sin`/`month_cos` ya vienen normalizados por
  construcción, así que **evitan el escalador** por una rama `passthrough`.
  Antes de esa corrección, una ventana de 3 meses llevó `month_cos` a **4.9e18**
  en las filas de test de junio, y el MSE del VAE a **1.8e35**.
* **Horizontes de contraste.** Con un bloque de 3 meses, `lag6`/`diff6` son
  constantes en train y explosivos en test. Los horizontes se validan contra la
  ventana de ajuste, así que una ventana de 3 meses conserva solo `h=1`.

Un **guard de magnitud** (`_warn_on_extreme_magnitudes`) registra cualquier
feature que supere `1e6` tras la transformación y lo nombra, para que esta clase
de bug no vuelva a ser silenciosa.

---

## 3. Hallazgos de esta ronda y cómo se corrigieron

| # | Hallazgo | Impacto | Corrección |
| --- | --- | --- | --- |
| 1 | `UserWarning` de UMAP: `n_jobs` sobrescrito a 1 por fijar `random_state` | Ruido en cada corrida | `n_jobs=1` explícito en los dos call-sites (`visualize.py`, `vae_explain.py`) |
| 2 | **TPE nunca guiaba** con los presets chicos (§1.4) | El tuneo del VAE era aleatorio puro | `_tuning_budget.tpe_startup_trials` escala la exploración al presupuesto |
| 3 | `FutureWarning` de NumPy desde `shap.summary_plot` (siembra el RNG global) | Ruido que sepultaba el log útil en cada corrida | Suprimido de forma estrecha (esa llamada, esa categoría) con el motivo documentado |
| 4 | Docstring de `tune_vae` declaraba un espacio de búsqueda **que no era el del código** (4 rangos distintos) | Documentación que miente sobre el modelo | Sincronizado con `suggest_*` reales |
| 5 | Reporte: sección de diagnósticos aparecía **antes** que "Modelos" mientras el nav la listaba al final | Navegación inconsistente | Secciones colocadas individualmente en el orden del nav |
| 6 | Reporte: gráficos como PNG base64 (~5 MB de imágenes planas) | Sin hover ni zoom; página pesada | 16 figuras reconstruidas como Plotly interactivo (§4) |
| 7 | PDFs del código: 10 desactualizados, 5 faltantes, sin script para regenerarlos | La carpeta solo podía envejecer | `docs/build_py_pdfs.py` con modo `--check` para CI |
| 8 | PDFs: líneas en blanco desaparecían; líneas largas se partían sin marca; glifos sin cobertura se descartaban en silencio | El código exportado **no** reproducía el original | Espacio para líneas vacías, marcador `↵` de continuación, y verificación de glifos que falla ruidosamente |
| 9 | `console_ui.py` usaba glifos braille y `⏱` ausentes de DejaVuSansMono | Corrompía su propio PDF exportado | Reemplazados por ASCII |
| 10 | `ResourceWarning: unclosed database` de Optuna (SQLite) en varios tests, en un `gc.collect()` **no relacionado** con el estudio que la causó | Ruido difícil de rastrear (el traceback apunta a la llamada equivocada); conexiones colgando hasta el GC | `src/models/_optuna_storage.py`: URIs `sqlite:` se envuelven en `RDBStorage` con `NullPool` — es la recomendación oficial de Optuna para SQLite |
| 11 | `PytestRemovedIn10Warning`: 7 fixtures `scope="class"` definidas como método de instancia | Atributos de instancia invisibles entre tests; advertencia de deprecación en cada corrida | Convertidas a `@classmethod` (con `@pytest.fixture` como decorador **externo**, no interno — el orden importa en pytest 9.1) |
| 12 | Sección `#explain` (Explicabilidad por modelo) se agregó al reporte pero **nunca al nav** | Enlace ausente en la navegación | Agregado en su posición correcta |
| 13 | Bug propio: `_PHASE_PLAN` del dashboard tenía el código `"Phase 8a"`, pero `main.py` emite `"Phase 8: evaluation [...]"` (sin la `a`) | La fase de evaluación nunca calzaba con el plan; cada vez caía al *fallback* sin traducir | Código corregido a `"Phase 8"` |
| 14 | Reporte HTML: histogramas interactivos por variable (uno por columna numérica) no escalan — con 50+ variables reales saturarían la página | Bloque de código muerto en producción real; riesgo de página inmanejable | Eliminados del HTML interactivo; el diagnóstico sigue disponible como enlace a PNG estático en `model_documentation.md`, que no tiene ese costo |
| 15 | `rich`, `psutil`, `fpdf2` y `fonttools` se usaban en el código (dashboard, monitor de recursos, exportador de PDFs) pero **no estaban declaradas** en `requirements.txt` | Una instalación limpia del proyecto fallaría en runtime, no en instalación | `rich` y `psutil` agregadas con su motivo documentado. `fpdf2` y `fonttools` quedaron sin uso al eliminarse la generación de PDF (ver la nota al inicio) y **no** se agregaron |

**Advertencias benignas que dejé como están:** los `WARNING` de
"Checkpoint ... is incompatible with the current config; starting fresh" son el
comportamiento correcto de auto-reparación cuando cambia la configuración del
VAE entre corridas — no son un defecto. El `ImportWarning` de `umap-learn`
sobre TensorFlow ausente es ruido de una librería externa (UMAP nunca usa su
backend paramétrico en este proyecto); se filtra explícitamente en
`pytest.ini` por mensaje exacto, no de forma global, para que una advertencia
nueva y real siga apareciendo.

---

## 4. Evaluación de desempeño de los cambios

### 4.1 Reporte: de PNG estático a 100% Plotly (y sin un gráfico por variable)

Medido sobre una corrida real; primero con la conversión inicial (200
individuos × 6 períodos), luego con la corrección de esta ronda (250 × 7,
que ya no genera un histograma interactivo por columna cruda — §3, hallazgo 14):

| Métrica | Antes (PNG) | Tras convertir a Plotly | Tras quitar histogramas/variable |
| --- | --- | --- | --- |
| Gráficos interactivos | 3 | 19 | **9** |
| Figuras PNG embebidas en el HTML | 16 | 0 | 0 |
| Tamaño del HTML | ~6.5 MB | 6.74 MB | **5.99 MB** |
| Fase 11 (reporte) | ~4 s | ~11 s | similar |

**Lectura honesta del costo.** La conversión a Plotly **no** redujo el peso de
la página por sí sola: cambió ~5 MB de PNG base64 por datos de series de
Plotly. Lo que sí redujo el tamaño fue eliminar los histogramas por variable
(10 gráficos menos en esta corrida de 5 columnas numéricas; en datos reales
con 50 columnas habrían sido 50 gráficos menos, la diferencia real que motivó
el cambio).

Lo que se ganó con Plotly: hover con valores exactos, zoom, y re-tematizado
claro/oscuro en cada gráfico. **~79% del peso del archivo (4.7 MB) es el
bundle de plotly.js**, que es costo fijo y no crece con el número de
gráficos — por eso pasar de 3 a 9 gráficos casi no mueve el tamaño total,
pero pasar de "un histograma por columna" a "ninguno" sí importa en datos con
decenas de variables, que es exactamente el caso que se corrigió.

*Si el peso llegara a ser un problema*, la palanca es el bundle, no los
gráficos: `include_plotlyjs="cdn"` lo baja a ~40 KB, a cambio de perder la
garantía de funcionar sin red, que hoy es un requisito explícito del proyecto.

### 4.2 Validación automatizada del reporte

Doce comprobaciones sobre el HTML generado, todas pasando (corrida sin
histogramas por variable, 9 gráficos):

```
1. gráficos Plotly                 : 9
2. raster embebido / <img>         : 0 / 0
3. ids duplicados                  : ninguno
4. bundle plotly.js                : 1 bloque, 4,731 KB
5. gráficos sin re-tematizar       : ninguno
6. referencias externas            : 0
7. responsive:true                 : 9/9
8. nav: 9 enlaces, rotos=ninguno, en orden=True
9. <figure class=chart> / notas    : 9 / 9
10. gráficos con ancho fijo en px  : 0
11. histogramas por variable       : 0 (debe ser 0)
12. tablas envueltas en .table-wrap: 7
```

Las comprobaciones 3, 8 y 10 son las que atacan superposición y errores de
render: ids duplicados harían que Plotly dibuje en el contenedor equivocado, un
ancho fijo en píxeles rompería la grilla responsive, y el orden del documento
debe coincidir con el nav (la comprobación 8 fue la que encontró el hallazgo
#12 de la tabla anterior — la sección "Explicabilidad" sin su enlace en el nav).

Las tablas del reporte (glosarios, comparación de modelos, resumen de dataset)
y las de `docs/documentation.html` están envueltas en `.table-wrap {
overflow-x: auto }`: en una pantalla angosta la tabla misma se desplaza
horizontalmente en lugar de ensanchar toda la página. `docs/build_docs.py`
aplica lo mismo a cada tabla markdown que convierte, con el font-size de celda
reducido bajo 860px de ancho.

### 4.3 Dashboard de consola: costo y el rediseño de esta ronda

**Costo:** nulo sobre el pipeline. Repinta a 10 fps en un hilo daemon aparte,
se apaga solo cuando la salida no es un terminal, y los *hooks* de fases y de
supuestos son listas de callbacks que en condiciones normales están vacías. El
muestreo de RAM/CPU (`psutil`) se limita a 1 lectura/segundo -- llamarlo en
cada uno de los ~10 repintados por segundo habría sido puro desperdicio para
un número que no cambia esa rápido.

**Rediseño:** el panel de fases pasó de una tabla que solo mostraba las
últimas completadas (creciendo/desplazándose a medida que avanzaba la corrida)
a una **lista fija de las 15 fases planeadas, visible completa desde el primer
cuadro**: cada fila empieza en `□` (pendiente, atenuada) y cambia en el mismo
lugar a `▣` (en curso, cian) o `■` (completada, verde / roja si falló) -- nada
se agrega ni se desplaza, solo cambia de estado.

Se agregó un panel **"Supuestos (IF / VAE)"** enganchado a
`observability.check(...)` (el mismo mecanismo que ya registraba cada
verificación en `run_events.jsonl`): muestra el conteo ✓/✗ y las últimas
verificaciones, incluyendo las `iforest.*`/`vae.*` que corren justo antes de
cada `.fit()`. Y una línea **"Equipo"** con RAM del proceso, RAM del sistema
(con umbral de color: verde <75%, amarillo <90%, rojo ≥90%, siempre con el
número visible -- nunca solo color) y CPU, más útil durante las fases pesadas
(entrenamiento del VAE, ajuste del Isolation Forest, interpretabilidad) pero
visible en todo momento en vez de activarse/desactivarse por fase.

### 4.4 Tiempos por fase (corrida de referencia)

| Fase | Duración |
| --- | --- |
| 2 — carga/generación | 1.76 s |
| 4 — preprocesamiento | 24.71 s |
| 6 — Isolation Forest | 9.46 s |
| 7 — VAE | 5.94 s |
| 8 — evaluación [iforest] | 102.59 s |
| 10 — interpretabilidad [iforest] | 23.10 s |
| 11 — reporte | 10.98 s |

**El cuello de botella no es el modelado.** La evaluación del bosque (102 s)
domina, seguida del preprocesamiento (25 s) y la interpretabilidad (23 s). Es
la métrica de estabilidad de ranking entre semillas, que reajusta el modelo
varias veces. Cualquier esfuerzo de optimización debería empezar ahí, no en el
tuneo.
