# Escalamiento lineal sin estado (`linear_scaling`)

Módulo: `src/preprocessing/linear_scaling.py`

Reescalado afín (`y = a·x + b`) del bloque continuo de un panel, con
numpy/pandas puro: sin scikit-learn, sin clases estimadoras y sin `.fit()`.
Una función recibe un DataFrame, calcula sus constantes y devuelve el
DataFrame reescalado en una sola llamada.

Es **independiente** del pipeline de `main.py` (`fit_transform_panel`), que
tiene su propio escalado dentro de un `ColumnTransformer`. Este módulo existe
para EDA, para una ruta de serving más liviana, y para cuando se necesita el
escalado sin arrastrar el objeto sklearn serializado.

---

## 1. Qué se transforma y qué no

| Tipo de columna | Acción | Motivo |
| --- | --- | --- |
| `int` / `float` | **Se escala** | Son las variables continuas |
| `str` / `object` / `category` | Intacta | No tienen escala que normalizar |
| `bool` | Intacta | Ya son 0/1 en una escala fija; dividir un flag por su IQR no lo hace más comparable |
| `datetime` | Intacta | No es numérica para pandas |
| `id`, `codmes`, … | Intacta | Son claves, no mediciones |

La detección de claves (`is_key_column`) es **por token, no por subcadena**:
normaliza el nombre y compara el nombre completo y su primer/último token
contra `KEY_COLUMN_TOKENS`. Por eso `id`, `client_id`, `id_cliente`, `codmes`,
`cod_mes`, `periodo` y `fecha` se detectan, pero `avg_txn_to_income`,
`identidad` o `balance_to_income` —que solo *contienen* las letras— no.

Si los nombres reales no siguen esta convención, hay dos escapes:
`exclude=[...]` para nombrar las claves a mano y `detect_keys=False` para
apagar la heurística.

---

## 2. Por qué una transformación *lineal* para estos dos detectores

Un mapa afín cambia solo ubicación y escala: deja intacta la **forma** de la
distribución (asimetría, curtosis) y el **orden** de cada columna. Eso es
exactamente lo que necesitan ambos modelos, por razones distintas.

### Isolation Forest — invarianza al outlier

El bosque parte con umbrales paralelos a los ejes, sorteados uniformemente
entre el mínimo y el máximo observados de cada columna. Es invariante a
cualquier mapa monótono por columna, así que **reescalar no puede cambiar su
ranking** — pero una transformación de *forma* sí, y de forma destructiva: un
normalizador por rangos o cuantiles arrastra la cola lejana de vuelta hacia el
bulto, que es precisamente la señal que se quiere detectar.

Una transformación lineal mantiene al outlier como outlier: después de
`(x − mediana) / IQR`, un punto a 6 IQR sigue a 6 IQR.

### VAE — estabilidad de gradientes

La pérdida del VAE es un error cuadrático medio de reconstrucción. Una
variable en escala `1e5` aporta ~`1e10` al gradiente y domina a todas las
demás; en este proyecto eso desbordó a puntajes `NaN` (ver
`docs/leakage_free_pipeline.md`, sección *On RobustScaler*). Centrar y dividir
por un estadístico de dispersión deja a todas las variables en un rango
comparable, lo que mantiene los gradientes finitos e impide que una sola
columna monopolice el espacio latente.

---

## 3. Por qué mediana/IQR y no media/desviación

La media y la desviación estándar se calculan sobre **todas** las filas,
anomalías incluidas. En una muestra contaminada ambas quedan arrastradas hacia
los puntos que se quieren detectar: la media se desplaza y la desviación se
infla, así que dividir por ella **encoge** la distancia de la anomalía al
centro — el estimador enmascara la señal que debía exponer.

La mediana (punto de ruptura 50%) y el IQR (que descarta los cuartiles
externos por construcción) no se ven afectados por una minoría de valores
extremos, así que la cola escalada conserva su magnitud.

Medido sobre el mismo dato: la anomalía queda **más de 100×** más lejos del
centro con robust que con standard.

### División por cero

Si `IQR == 0` (columna constante), es `NaN` (columna toda nula) o no es
finito, el denominador se reemplaza por `1.0`. La columna queda en `x − centro`
—finita y constante— en vez de producir `inf`/`NaN`. Lo mismo aplica a la
desviación estándar y al rango de min-max.

---

## 4. Fuga de información — la advertencia importante

Los atajos de una sola llamada **estiman sus constantes de las filas que
reciben**. Llamar `robust_scale(df)` sobre el panel completo deja que las filas
del período de test influyan en la mediana y el IQR que se aplican a las filas
de entrenamiento: exactamente la fuga que prohíbe la Fase 3 de
`docs/leakage_free_pipeline.md`.

Para cualquier corrida con división cronológica, usar la forma de dos pasos
—que sigue siendo *sin fit* en el sentido pedido: un diccionario plano de
números, no un objeto con estado:

```python
params = robust_scale_params(df[train_mask])   # constantes solo del train
df     = apply_linear_scaling(df, params)      # aplicadas a todas las filas
```

`params` es JSON-serializable, así que puede registrarse en logs, guardarse
junto al modelo y recargarse sin deserializar ninguna clase.

Si las dos formas (una sola llamada vs. las dos pasos) alguna vez dejan de
diferir sobre un panel con partición cronológica, es señal de que la ruta sin
fuga dejó de estar haciendo algo.

---

## 5. API

| Función | Devuelve | Uso |
| --- | --- | --- |
| `select_continuous_columns(df, ...)` | `list[str]` | Qué columnas son elegibles |
| `is_key_column(name)` | `bool` | Detección de claves por token |
| `robust_scale_params(df, ...)` | `dict` | Constantes mediana/IQR |
| `standard_scale_params(df, ...)` | `dict` | Constantes media/desviación |
| `minmax_scale_params(df, ...)` | `dict` | Constantes mín/rango |
| `scaling_params(df, method=...)` | `dict` | Despacho por nombre |
| `apply_linear_scaling(df, params)` | `DataFrame` | Aplica constantes dadas |
| `robust_scale(df, ...)` | `DataFrame` | Una llamada (estima + aplica) |
| `standard_scale(df, ...)` | `DataFrame` | Ídem, media/desviación |
| `minmax_scale(df, ...)` | `DataFrame` | Ídem, a `[0, 1]` |

Parámetros comunes: `columns` (fija las columnas), `quantile_range` (por
defecto `(25, 75)`; ampliarlo usa más de la distribución y es *menos* robusto),
`inplace`, `exclude`, `detect_keys`, `min_unique`.

### Ejemplo ejecutable

El módulo trae un bloque `if __name__ == "__main__":` con un panel que incluye
claves, strings, un booleano, una columna constante, una con nulos y una
anomalía deliberada:

```bash
python -m src.preprocessing.linear_scaling
```

Imprime la entrada, las columnas detectadas, la salida escalada, las
comprobaciones de que strings/`id`/`codmes`/booleanos quedaron intactos, y la
comparación entre la forma de una llamada y la forma sin fuga.

---

## 6. Sobre `minmax`

Es el menos robusto de los tres: ambas constantes las fija el valor más
extremo en cada dirección, así que **una sola anomalía comprime todas las filas
normales en una banda estrecha**. Está incluido porque a veces lo exige una
activación acotada aguas abajo, no como recomendación.
