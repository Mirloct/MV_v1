# Modelo — Framework de Detección de Anomalías Bancarias

Framework de investigación para detección de anomalías en **datos de panel** del
sector bancario. Combina un **Isolation Forest** y un **Variational Autoencoder
(VAE)** como detectores complementarios, con tuneo de hiperparámetros vía Optuna,
recuperación ante caídas para trabajos largos, logging exhaustivo y reportería
completa. El trabajo es **no supervisado por defecto**: se genera un panel
bancario mensual sintético con estructura realista (colas pesadas, categóricas de
cola larga, deriva temporal, faltantes MCAR/MNAR) y cuatro tipos distintos de
anomalía (`global`, `local`, `contextual`, `collective`), cuyas etiquetas viven en
un archivo de ground truth **separado**, de modo que los detectores nunca las ven
al entrenar.

El pipeline es **ejecutable de punta a punta**: `python main.py` va desde la
generación de datos hasta un Excel de los 50 individuos más riesgosos y un
reporte HTML/MD. Está construido sobre un diseño estrictamente cronológico y
libre de fugas — ver [`docs/leakage_free_pipeline.md`](docs/leakage_free_pipeline.md)
para la derivación de 7 fases y el checklist anti-fugas.

## Requisitos e instalación

- **Python 3.12+** (esta máquina corre 3.13 sin problemas).
- Instalar dependencias:

  ```
  pip install -r requirements.txt
  ```

- Validar el entorno (revisa la versión de Python y que cada paquete requerido
  sea importable, intentando un `pip install` de lo que falte):

  ```
  python setup_validator.py
  ```

  Imprime un resumen por paquete y termina con código distinto de cero si algo no
  se pudo satisfacer. `pyarrow` importa aquí: sin un motor parquet, el generador
  escribe el ground truth como `.csv` en vez de `.parquet` sin avisar.

**Advertencia en Windows.** El `python` del PATH puede ser un stub de Microsoft
Store que no hace nada útil. Si los comandos parecen no ejecutarse o abren la
Store, llama al intérprete real por su ruta completa:

```
C:\Users\<usuario>\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests/ -q
```

## Cómo ejecutarlo

Para el pipeline de un solo comando, salta a
[Ejecutar `main.py`](#ejecutar-mainpy--el-pipeline-completo). Las secciones
siguientes muestran cada módulo por separado, para cuando quieras manejar uno
directamente.

### Generar datos sintéticos / cargar el panel

El módulo `src.data` está completo. Usa `load_or_generate_panel`: carga
`artifacts/data/data.csv` si existe, y lo genera (junto con el archivo de ground
truth) si no.

```python
from src.data import load_or_generate_panel

# Carga artifacts/data/data.csv si existe; si no, primero lo genera.
# Los generator_kwargs (n_individuals, n_periods, seed, ...) sólo se usan
# cuando el archivo debe generarse — un archivo existente nunca se regenera.
df, schema = load_or_generate_panel(
    data_path="artifacts/data/data.csv",
    n_individuals=1_000,   # escala chica para una corrida rápida
    n_periods=10,
    seed=42,
)
print(df.shape)                 # (n_individuals * n_periods, 22)
print(schema.time_col,          # "period"
      schema.entity_col,        # "entity_id"
      schema.target_col,        # None (los datos sintéticos no traen etiquetas)
      schema.ground_truth_path) # ruta real del archivo de GT separado
```

Para control explícito de escala, semilla y ubicaciones de salida, llama al
generador directamente:

```python
from src.data import generate_synthetic_panel

result = generate_synthetic_panel(
    n_individuals=100_000,   # default a escala completa -> 1,000,000 filas
    n_periods=10,
    out_path="artifacts/data/data.csv",
    ground_truth_path="artifacts/data/ground_truth.parquet",
    seed=42,
)
# Usa las rutas RETORNADAS — ground_truth_path puede caer a .csv
# cuando no hay motor parquet instalado.
print(result.out_path, result.ground_truth_path, result.anomaly_counts)
```

Notas:

- **`data.csv` presente vs ausente**: `load_or_generate_panel` sólo genera cuando
  falta el CSV. Si existe, se carga tal cual y los kwargs
  `n_individuals`/`n_periods`/`seed` se ignoran. Borra el archivo para forzar la
  regeneración.
- **Escala chica vs completa**: el default del generador es
  `n_individuals=100_000 * n_periods=10 = 1,000,000` filas (~22–25 s,
  `data.csv` ≈ 192 MB). Para iterar rápido usa de cientos a unos pocos miles de
  entidades.
- **Las salidas van a `artifacts/data/`**: el panel en `data.csv` y el archivo de
  ground truth (`ground_truth.parquet`, o `.csv` como respaldo) al lado.
- Las etiquetas de anomalía **nunca** son columnas de `data.csv`; viven sólo en el
  archivo de ground truth separado. Ver el contrato de datos en `CONTEXT.md`.

**Formato del campo de periodo.** El cargador detecta formatos compactos de
periodo automáticamente: `202401` (`yyyyMM`, string o entero) y `20240115`
(`yyyyMMdd`) se parsean con formato explícito. Esto importa porque
`pd.to_datetime(["202401"])` **falla** por sí solo — 6 dígitos sin separadores
son ambiguos para pandas. Si una columna de periodo no se puede parsear, se deja
intacta y la compuerta de supuestos detiene la corrida nombrando el problema, en
vez de convertirla silenciosamente a `NaT`.

### Preprocesamiento — del panel a la matriz de features

El módulo `src.preprocessing` convierte un panel crudo en una matriz de features
lista para modelar, dejando las llaves `(entity_id, period)` aparte para el
posterior cruce con el ground truth. Aliméntalo con el par `(df, schema)` de
`load_or_generate_panel`:

```python
from src.data import load_or_generate_panel
from src.preprocessing import fit_transform_panel

df, schema = load_or_generate_panel(data_path="artifacts/data/data.csv",
                                    n_individuals=1_000, n_periods=10, seed=42)

# numeric_transform y categorical_encoding se seleccionan por nombre (para que
# un estudio de Optuna pueda tunearlos como hiperparámetros categóricos).
X, keys, feature_names = fit_transform_panel(
    df, schema,
    numeric_transform="yeo-johnson",   # NUMERIC_TRANSFORMS: standard | robust |
                                       #   log1p | yeo-johnson | quantile | passthrough
    categorical_encoding="onehot",     # CATEGORICAL_ENCODINGS: onehot | ordinal | frequency
)
print(X.shape)                 # matriz de features (puede ser dispersa)
print(keys.columns.tolist())   # ['entity_id', 'period'] — para el cruce con GT, no features
print(feature_names[:5])       # nombres alineados con las columnas de X
```

`keys` está alineado fila a fila con `X`; entidad y tiempo se tratan como llaves,
no como features.

**Ruteo de features por modelo.** Los dos detectores **no** reciben las mismas
columnas. Las features derivadas de variables categóricas se excluyen del
Isolation Forest y se conservan para el VAE:

```python
from src.preprocessing import split_matrix_for_model

X_if,  names_if  = split_matrix_for_model(X, feature_names, "iforest")  # sólo numéricas
X_vae, names_vae = split_matrix_for_model(X, feature_names, "vae")      # matriz completa
```

La identificación es puramente por **tipo de dato** de la columna de origen
(`object`/`category` → rama categórica del `ColumnTransformer`), nunca por una
lista de nombres, así que una columna de texto nueva se rutea sola. El motivo de
la asimetría: el Isolation Forest parte con cortes de orden sobre una feature a
la vez, y una columna one-hot no tiene interior significativo — cada corte
degenera en "tiene este nivel / no lo tiene", y con alta cardinalidad diluye el
muestreo de `max_features` sin aportar estructura aislable. El VAE, en cambio,
reconstruye su vector completo y el contexto categórico le sirve para la
definición de anomalía `contextual`.

- **Justificación de la transformación**: `compute_transform_diagnostics` /
  `recommend_transform` puntúan cada transformación numérica por feature con
  tamaños de efecto estables a la escala, en vez de p-values de normalidad (que
  no significan nada con ~1M filas). `plot_transform_diagnostics` escribe figuras
  antes/después en `artifacts/reports/figures/`.

### Isolation Forest — tunear y puntuar

`src.models` envuelve el Isolation Forest de scikit-learn en
`IsolationForestDetector` y agrega tuneo Optuna con recuperación ante caídas. El
score sigue la convención del proyecto: **mayor = más anómalo**
(`score_samples` retorna `-sklearn.score_samples`).

```python
from src.data import load_or_generate_panel
from src.preprocessing import fit_transform_panel, split_matrix_for_model
from src.models import IsolationForestDetector, tune_iforest

df, schema = load_or_generate_panel(data_path="artifacts/data/data.csv",
                                    n_individuals=1_000, n_periods=10, seed=42)
X, keys, feature_names = fit_transform_panel(df, schema)
X_if, names_if = split_matrix_for_model(X, feature_names, "iforest")

# Estudio Optuna persistido en sqlite:///artifacts/tuning/optuna_iforest.db.
# Re-ejecutar con el mismo study_name/storage REANUDA tras una caída; los
# mejores params se guardan en artifacts/tuning/best_params_iforest.yaml tras
# cada trial, y el mejor detector reajustado va a artifacts/models/iforest.joblib.
study = tune_iforest(X_if, n_trials=25)   # pasa y=<etiquetas 0/1> para PR-AUC supervisado

detector = IsolationForestDetector.load("artifacts/models/iforest.joblib")
scores = detector.score_samples(X_if)     # mayor = más anómalo
flags = detector.predict(X_if)            # 1 = anomalía, 0 = normal
```

Sin etiquetas el objetivo de tuneo es un proxy de separación de scores; entrega
`y` para el objetivo supervisado PR-AUC / ROC-AUC. Ver
[`docs/models_isolation_forest.md`](docs/models_isolation_forest.md) para el
concepto, la API completa, el espacio de búsqueda, la referencia de parámetros y
los detalles de recuperación.

### Variational Autoencoder (VAE) — ajustar, puntuar y tunear

`src.models` también trae un VAE en PyTorch (`VAEDetector`) con las mismas
convenciones. El score es el **error de reconstrucción MSE por fila**, siguiendo
la convención **mayor = más anómalo** (un VAE entrenado sobre la masa normal
reconstruye mal las anomalías). La entrada dispersa se densifica internamente. El
entrenamiento escribe un `checkpoint.pth` por época en `artifacts/models/vae/`,
así que re-ejecutar con `resume=True` continúa desde la última época completada;
los mejores pesos se restauran al final de `fit`.

```python
from src.data import load_or_generate_panel
from src.preprocessing import fit_transform_panel
from src.models import VAEDetector, tune_vae

df, schema = load_or_generate_panel(data_path="artifacts/data/data.csv",
                                    n_individuals=1_000, n_periods=10, seed=42)
X, keys, feature_names = fit_transform_panel(df, schema)   # el VAE usa la matriz completa

# Ajustar y puntuar. Checkpoints por época en artifacts/models/vae/; resume=True
# continúa tras una caída. La pérdida es reconstrucción + beta*KL (beta=1 = VAE clásico).
detector = VAEDetector(latent_dim=8, hidden_dim=64, n_layers=2, beta=1.0, epochs=30)
detector.fit(X, checkpoint_dir="artifacts/models/vae", resume=True)
scores = detector.score_samples(X)     # mayor = más anómalo (error de reconstrucción)

# Tuneo Optuna. Estudio persistido en sqlite:///artifacts/tuning/optuna_vae.db;
# re-ejecutar con el mismo study_name/storage REANUDA. Los mejores params van a
# artifacts/tuning/best_params_vae.yaml cada trial; el mejor detector reajustado
# a artifacts/models/vae_best.pt. Objetivo no supervisado = pérdida de
# reconstrucción en validación (auto 'minimize'); pasa y=<etiquetas 0/1> para el
# objetivo supervisado PR-AUC / ROC-AUC (auto 'maximize').
study = tune_vae(X, n_trials=25)
best = VAEDetector.load("artifacts/models/vae_best.pt")
```

El VAE tiene **early stopping por época** dentro de cada ajuste (monitorea
pérdida de *validación*, no de entrenamiento, y restaura los pesos de la mejor
época), y ambos tuneos tienen además **early stopping entre trials**: el estudio
se detiene tras `patience` trials consecutivos sin mejora relativa mínima,
saltando los restantes. Son dos mecanismos distintos que conviven. Ver
[`docs/models_vae.md`](docs/models_vae.md) para el concepto, la API, el espacio
de búsqueda, la referencia de parámetros y los modos de objetivo.

### Interpretabilidad y reportería

`src.interpretability` explica *por qué* un punto puntuó alto, y `src.reporting`
arma la corrida en reportes compartibles. Toda función de interpretabilidad
guarda su figura en `artifacts/reports/figures/` y retorna un `dict`/ruta simple.

- **Isolation Forest**: `shap_summary_iforest` (atribución SHAP — intenta
  `shap.TreeExplainer` nativo, luego un `shap.Explainer` agnóstico sobre
  `score_samples`, luego importancia por permutación, logueando cuál se usó) y
  `path_length_analysis` (relaciona el score con el largo de camino promedio
  normalizado).
- **VAE**: `latent_space_plot` (codifica a medias latentes y proyecta a 2D vía
  UMAP, si no PCA) y `reconstruction_error_by_feature` (error de reconstrucción
  medio por feature — qué columnas reproduce peor el VAE).

`build_report(context, ...)` renderiza la corrida en dos entregables
autocontenidos bajo `artifacts/reports/`: un **Markdown** y un **HTML offline**
(CSS embebido, todos los gráficos interactivos vía Plotly, sin red ni
imágenes rasterizadas). No se genera PDF. El HTML incluye, además de las
métricas y figuras: cómo leer cada indicador, por qué los resultados son
estadísticamente confiables, qué significa cada parámetro del modelo
(marcando si el valor vino del tuneo o es el default), y el posicionamiento
del enfoque entre machine learning y econometría.

Junto a esos dos se genera **`model_documentation.md`**: el companion técnico
con los hiperparámetros exactos de ambos modelos, el registro completo de
calibración del umbral, la matemática del VAE, el preprocesamiento y el
catálogo de artefactos. Todo lo que el reporte de negocio deja fuera a
propósito para no abrumar al lector.

Ver [`docs/interpretability_and_reporting.md`](docs/interpretability_and_reporting.md)
para los conceptos, la API completa y el esquema del `context`.

> Todo el texto del reporte y su documentación está **en español**. Los
> comentarios y docstrings del código siguen en inglés.

### Escalamiento lineal sin estado

`src/preprocessing/linear_scaling.py` implementa reescalado afín
(`y = a·x + b`) del bloque continuo con numpy/pandas puro — sin sklearn, sin
clases y sin `.fit()`. Robust scaling `(x − mediana) / IQR` por defecto, que es
el que preserva la magnitud del outlier (lo que necesita el Isolation Forest) a
la vez que acota la escala (lo que necesita el VAE para no desbordar
gradientes). Ignora strings, booleanos y columnas clave (`id`, `codmes`, …).

```bash
python -m src.preprocessing.linear_scaling   # ejemplo ejecutable
```

Ver [`docs/escalamiento_lineal.md`](docs/escalamiento_lineal.md) — incluye la
advertencia de fuga de información y la forma de dos pasos que la evita.

### Decisiones de modelado y hallazgos

[`docs/decisiones_de_modelado.md`](docs/decisiones_de_modelado.md) sustenta las
decisiones que el código toma pero que no estaban escritas: por qué el
Isolation Forest lleva más pruebas de Optuna que el VAE (costo por trial, no
tamaño del espacio), por qué `epochs` es un hiperparámetro tuneado y no una
constante, qué hace cada etapa de la transformación del DataFrame y cómo afecta
a cada detector, y los 9 hallazgos de la última ronda con su corrección.

Incluye la evaluación de desempeño medida: tiempos por fase, costo real de
pasar el reporte a 100% Plotly, y dónde está el cuello de botella (la
evaluación del bosque, no el tuneo).

### Dashboard de consola

Durante la corrida, `main.py` muestra un panel en vivo (barra de progreso
ponderada, fase actual con cronómetro, tabla de fases completadas, KPIs de la
corrida y cola del log) en lugar de líneas de log desplazándose. Teclas: `v`
detalle, `d` todas las fases, `o` abrir la vista web, `p` pausa.

Se desactiva solo si la salida no es un terminal (redirigida, CI) o si falta
`rich`, para no corromper un log capturado. `--no-console-ui` lo fuerza off.

### Ejecutar la suite de tests

292 tests cubren `src/data`, `src/preprocessing`, ambos modelos, evaluación,
interpretabilidad y reportería. Desde la raíz del proyecto:

```
python -m pytest tests/ -q
```

`tests/conftest.py` corre toda la sesión dentro de un sandbox descartable (hace
`chdir` antes de que arranque cualquier logger o generador), así que los tests
nunca tocan el `artifacts/` real.

### Ejecutar `main.py` — el pipeline completo

```bash
python main.py                   # 2000 individuos x 15 periodos, con tuneo
python main.py --quick           # 500 x 12, 5 trials cada uno — corrida de humo
python main.py --full            # 100_000 x 15, 50/30 trials
python main.py --top-n 100       # exporta los 100 individuos más riesgosos
python main.py --panel-features  # reactiva lag/diff/ratio/own-z + estacionalidad
python main.py --supervised      # usa ground truth para tuneo/métricas (default: no supervisado)
python main.py --contamination 0.05   # punto de operación del Isolation Forest
python main.py --no-live-view    # no abre la vista de progreso local en el navegador
python main.py --help
```

**La estrategia por defecto es NO SUPERVISADA.** Las etiquetas de ground truth se
cargan siempre que exista el archivo (los diagnósticos las usan igual), pero sólo
alimentan el objetivo de tuneo y las métricas supervisadas (PR-AUC/ROC-AUC contra
etiquetas reales) cuando se pasa `--supervised` explícitamente. Que las etiquetas
simplemente existan ya no cambia la estrategia de la corrida en silencio, y el
reporte tampoco muestra métricas supervisadas cuando la corrida no las calculó.

**Vista de progreso en vivo.** Por defecto, al iniciar el pipeline se abre una
página local (`http://127.0.0.1:<puerto>/`, nunca accesible fuera de esta
máquina) que se actualiza cada segundo: porcentaje de avance, barra, fase actual
con spinner y el diagrama de flujo creciendo conforme las fases completan. Si el
proceso se cancela o muere, la página lo detecta y marca la fase interrumpida en
vez de quedarse congelada en "running". `--no-live-view` la desactiva (por
ejemplo en CI). El mismo flujo, reproducible tras la corrida, queda en
`artifacts/reports/flow_visualization.html`.

El orquestador corre datos → validación de supuestos → preprocesamiento →
Isolation Forest + VAE (tuneo/ajuste) → evaluación → **calibración de umbral** →
Excel top-N → interpretabilidad → reporte HTML/MD, sobre el diseño
cronológico libre de fugas documentado en
[`docs/leakage_free_pipeline.md`](docs/leakage_free_pipeline.md).

**Orden de la interpretabilidad.** La Fase 10 (SHAP, UMAP) corre **después** de
que todos los Excel están en disco: es la etapa más lenta y no produce entregable
propio, así que ejecutarla antes dejaría la cola de revisión esperando.

**Las panel features vienen APAGADAS aquí.** `--panel-features` /
`--no-panel-features` controla si el preprocesamiento genera features
lag/diff/ratio/own-z + estacionalidad dentro de cada entidad — apagadas por
defecto en `main.py` porque el uso con datos reales de este pipeline ya las
calcula en un flujo aguas arriba. Enciéndelas para el flujo con datos sintéticos.
Ver `CONTEXT.md` para el sustento completo y su consecuencia sobre el recall de
anomalías `local`/`contextual`.

**Partición.** Los periodos se dividen entrenamiento / validación / prueba en
orden temporal (10 / 2 / 3 por defecto, `--n-val-periods` / `--n-test-periods`).
Entrenamiento ajusta el preprocesamiento y los modelos, validación selecciona
hiperparámetros *y* calibra el umbral de alerta, prueba se lee exactamente una
vez al final.

**Entregable principal — la cola priorizada por riesgo.** Cada detector escribe

```
artifacts/reports/oot_top50_iforest.xlsx
artifacts/reports/oot_top50_vae.xlsx
```

los **50 individuos de mayor score** en los meses de prueba, con formato
**ID – SCORE – VARIABLES** y una columna `alert` que marca las filas sobre el
umbral calibrado. El tamaño es parametrizable con `--top-n` (usa `--top-n 0` para
caer a la fracción de `--top-fraction`). Una fila por individuo: cuando el bloque
de prueba abarca varios meses, cada entidad se representa por su mes de mayor
score.

**Compuerta P95 entre capas.** Al terminar el Isolation Forest y **antes** de que
arranque el VAE, se exporta `artifacts/reports/p95_checkpoint_iforest.xlsx` con
todos los registros sobre el percentil 95 (calculado sólo sobre filas in-time) y
todas las columnas originales. El artefacto se valida (existe, no vacío, relectura
reproduce el shape exacto, checksum) y si falla, se lanza `ArtifactGenerationError`
y el VAE **no** arranca.

**Umbral.** `--threshold-method pot` (default) ajusta una Pareto Generalizada a la
cola de scores de validación y la invierte para una tasa de falsa alarma objetivo
(`--threshold-target-far`); `--threshold-method percentile` usa
`--threshold-percentile`. En ambos casos se calibra sobre validación y sólo se
aplica a prueba.

## Estructura del proyecto

**Código fuente en la raíz, todo lo generado bajo `artifacts/`.** Ese es el
principio organizador: la raíz sólo contiene lo que escribe una persona, y todo
el estado generado puede inspeccionarse — o borrarse — en un solo lugar.

```
Modelo-v0.1/
├── main.py                 # orquestador (fases, CLI argparse)
├── setup_validator.py      # verificación de entorno/dependencias
├── requirements.txt
├── README.md               # esta guía
├── CONTEXT.md              # memoria persistente del proyecto
├── src/
│   ├── data/               # cargador de panel + generador sintético
│   ├── preprocessing/      # pipeline (transformaciones, features de panel) + diagnósticos
│   ├── models/             # Isolation Forest + VAE + stacking + early stopping de trials
│   ├── evaluation/         # partición OOT, cruce con GT, métricas, umbrales, exports Excel
│   ├── interpretability/   # SHAP, largo de camino, espacio latente, reconstrucción por feature
│   ├── reporting/          # constructor de reportes + visualización de flujo
│   └── utils/              # rutas, logging, observabilidad, supuestos, escritura atómica
├── tests/                  # 292 tests + sandbox de sesión
├── docs/                   # fuentes *.md + documentation.html generado
└── artifacts/              # TODO lo que el pipeline escribe (fuera de control de versiones)
```

## Documentación

- [`docs/documentation.html`](docs/documentation.html) — toda la documentación
  consolidada en una página offline navegable. Regenerar con
  `python docs/build_docs.py` tras editar cualquier `.md`; **nunca editarla a
  mano**, es un archivo derivado.
- `CONTEXT.md` — memoria del proyecto: contratos de datos, decisiones tomadas con
  su justificación, resultados medidos y problemas abiertos.
