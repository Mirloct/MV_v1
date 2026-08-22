"""Explanatory content and interactive Plotly charts for the modelling report.

Separated from `report.py` (which owns document assembly: HTML/Markdown
scaffolding, figure embedding, file writing) because this module owns
*meaning*: what each indicator says in plain language, which statistical
checks make the run trustworthy and why, what every model parameter does,
and where this pipeline sits between machine learning and econometrics.

All reader-facing strings below (glossary text, statistical-check text,
parameter descriptions, chart titles/labels/hover text) are in Spanish --
the report and its documentation are a Spanish-language deliverable.
Source-level comments and docstrings stay in English for the dev audience.

Two hard rules for everything below:

* **Nothing is asserted that the code does not do.** Every parameter entry
  names the module that reads it; every statistical check corresponds to a
  real gate in `src/utils/assumptions.py` or a real metric in
  `src/evaluation/`. Where a number has a known caveat (`contamination` not
  affecting the ranking, `silhouette` being a proxy rather than proof), the
  caveat is part of the text, not omitted for tidiness.
* **Charts carry a colour palette validated for colour-vision deficiency.**
  The two model series use categorical slots 1 and 2 (blue/orange) from the
  project's data-viz reference palette, verified with the palette validator
  against this report's own light and dark surfaces (worst adjacent pair
  ΔE 24.7 protan / 33.6 normal-vision in light mode, 26.8 / 31.8 in dark --
  both far above the ≥8 / ≥15 floors). Series identity is never carried by
  colour alone: every chart also direct-labels or legends its series.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "METRIC_GLOSSARY", "FIGURE_NOTES", "STATISTICAL_CHECKS", "PARAM_GLOSSARY",
    "ML_VS_ECONOMETRICS", "SERIES_COLORS", "metric_note", "figure_note",
    "param_note", "build_plotly_figures",
]

# Categorical slots 1 & 2 of the validated reference palette, in fixed order:
# iForest always blue, VAE always orange, never cycled or reassigned.
SERIES_COLORS = {
    "light": {"iforest": "#2a78d6", "vae": "#eb6834"},
    "dark": {"iforest": "#3987e5", "vae": "#d95926"},
}

# --------------------------------------------------------------------------- #
# 1. What each headline indicator actually means, in plain language.          #
# --------------------------------------------------------------------------- #
#: metric key -> ("qué es", "cómo leer este número")
METRIC_GLOSSARY: dict[str, tuple[str, str]] = {
    "roc_auc": (
        "Probabilidad de que una anomalía real elegida al azar quede "
        "clasificada por encima de una fila normal elegida al azar.",
        "0.5 = azar puro, 1.0 = clasificación perfecta. En un panel muy "
        "desbalanceado, el ROC-AUC resulta engañosamente alto: la enorme "
        "mayoría de filas normales lo domina, por lo que un valor cercano a "
        "0.8 puede coexistir con una cola de alertas débil. Léase siempre "
        "junto al PR-AUC, nunca de forma aislada.",
    ),
    "pr_auc": (
        "Área bajo la curva de precisión-recall (precisión promedio).",
        "El indicador honesto para eventos raros. Su línea base es la "
        "propia tasa de anomalías (~2%), no 0.5 -- por lo que un PR-AUC de "
        "0.14 frente a una tasa base del 2% es aproximadamente 7 veces "
        "mejor que el azar, aunque el número parezca pequeño. Compárese con "
        "la tasa base mostrada en las tarjetas del dataset, no con 1.0.",
    ),
    "best_f1": (
        "Mejor media armónica alcanzable entre precisión y recall, sobre "
        "todos los umbrales posibles.",
        "Un techo optimista: se elige con retrospectiva sobre las mismas "
        "filas que se están evaluando, por lo que el umbral realmente "
        "desplegado rendirá peor. Útil como cota superior de lo que esta "
        "clasificación podría soportar, no como punto de operación "
        "esperado.",
    ),
    "mcc": (
        "Coeficiente de correlación de Matthews entre las etiquetas "
        "predichas y las reales.",
        "De -1 a +1, donde 0 es azar. A diferencia del F1, considera las "
        "cuatro celdas de la matriz de confusión, por lo que no favorece "
        "artificialmente a un modelo que simplemente marca muy pocas "
        "filas.",
    ),
    "precision_at_10pct": (
        "Del 10% de filas con mayor puntaje, la fracción que son realmente "
        "anómalas.",
        "Es la tasa de acierto del analista: de cada 10 casos en la cola de "
        "revisión, cuántos son reales. Directamente comparable con la tasa "
        "base de anomalías -- estar por encima de ella significa que el "
        "modelo aporta valor frente a revisar al azar.",
    ),
    "recall_at_10pct": (
        "De todas las anomalías reales, la fracción capturada dentro del "
        "10% superior de puntajes.",
        "Cobertura de la cola de revisión. 0.60 significa que seis de cada "
        "diez anomalías reales serían vistas por un analista que revisa el "
        "decil superior; las otras cuatro se pierden por completo con ese "
        "presupuesto.",
    ),
    "lift_at_10pct": (
        "Cuántas veces mejor que el azar es el 10% superior de puntajes.",
        "Lift 3.0 = el decil superior contiene tres veces la densidad de "
        "anomalías del panel completo. Es la métrica que responde más "
        "directamente '¿vale la pena clasificar frente a revisar casos al "
        "azar?' -- 1.0 significa que no se agrega valor.",
    ),
    "silhouette": (
        "Qué tan limpiamente se separan las filas marcadas del resto en el "
        "espacio de features.",
        "Proxy sin etiquetas, de -1 a +1. Mide separación geométrica, NO "
        "corrección: un modelo puede aislar un clúster compacto y bien "
        "separado de las filas equivocadas. Es evidencia de estructura, "
        "nunca evidencia de exactitud.",
    ),
    "calinski_harabasz": (
        "Razón entre la dispersión entre grupos y la dispersión dentro de "
        "los grupos para la partición marcada.",
        "No acotado y dependiente de la escala -- más alto es mejor, pero "
        "solo es comparable entre corridas sobre los mismos datos y el "
        "mismo conjunto de features. No debe leerse como una puntuación de "
        "calidad en términos absolutos.",
    ),
    "rank_stability": (
        "Coincidencia entre las filas mejor clasificadas bajo distintas "
        "semillas aleatorias.",
        "Cercano a 1.0 significa que la cola de alertas es reproducible y "
        "no un artefacto de un único sorteo aleatorio. Es una verificación "
        "de confiabilidad del modelo, no una medida de si acierta.",
    ),
    "n_flagged": (
        "Cantidad de filas por encima del umbral de alerta calibrado.",
        "Volumen operativo, no calidad: indica cuánto trabajo genera la "
        "cola de revisión, y debe leerse contra la capacidad de revisión "
        "disponible.",
    ),
    "threshold_value": (
        "El punto de corte de puntaje calibrado por encima del cual una "
        "fila genera una alerta.",
        "Ajustado únicamente sobre el bloque de validación (nunca sobre "
        "test), de modo que aplicarlo a test mide lo que realmente haría el "
        "despliegue. Ver la sección de verificaciones estadísticas para el "
        "método de estimación.",
    ),
    "test_alert_rate": (
        "Fracción de filas fuera de tiempo (OOT) que el umbral calibrado "
        "marca.",
        "Debería situarse cerca de la tasa de falsas alarmas objetivo para "
        "la que se calibró el umbral. Una brecha grande indica que la "
        "distribución del puntaje cambió entre validación y test -- una "
        "señal de drift que vale la pena investigar.",
    ),
}


# --------------------------------------------------------------------------- #
# 2. What each retained figure shows, and why it earns its place.             #
# --------------------------------------------------------------------------- #
FIGURE_NOTES: dict[str, str] = {
    "score_distribution": (
        "Cómo se distribuyen los puntajes de este detector en el bloque "
        "fuera de tiempo (OOT), con el umbral de alerta calibrado dibujado "
        "encima. La forma esperada es una masa normal densa con una cola "
        "derecha delgada: el umbral debe ubicarse en esa cola, y la masa a "
        "su derecha es exactamente el volumen de trabajo que genera la cola "
        "de alertas."
    ),
    "score_distribution_supervised": (
        "Dónde ubica el detector a las filas normales conocidas frente a "
        "las anómalas conocidas. Cierto solapamiento es esperable -- si "
        "ambas se separaran limpiamente, el problema sería trivial. Lo que "
        "importa es la cola derecha, donde corta el umbral."
    ),
    "model_agreement": (
        "Cada punto es un individuo, posicionado según su percentil de "
        "rango bajo cada detector. Este es el diagnóstico central cuando no "
        "hay etiquetas disponibles: los dos modelos se basan en principios "
        "distintos (geometría de aislamiento frente a error de "
        "reconstrucción) y ven conjuntos de features distintos, por lo que "
        "un individuo con puntaje alto en ambos cuenta con evidencia "
        "corroborada de forma independiente. Los puntos en la esquina "
        "superior derecha son los candidatos más fuertes; los puntos altos "
        "en un eje y bajos en el otro marcan casos donde los dos métodos "
        "realmente discrepan, y vale la pena inspeccionarlos precisamente "
        "porque un detector ve algo que el otro no."
    ),
    "roc_pr": (
        "Ambos modelos en los mismos ejes, evaluados solo sobre el bloque "
        "fuera de tiempo (OOT). El panel PR es el que hay que confiar en "
        "estos datos: su línea base punteada es la propia tasa de "
        "anomalías del panel, así que la distancia por encima de esa línea "
        "es la señal real. El panel ROC se incluye por ser la vista "
        "convencional, pero su diagonal de 0.5 favorece artificialmente a "
        "los problemas desbalanceados."
    ),
    "metric_comparison": (
        "Métricas principales lado a lado para cada detector ejecutado. "
        "Las barras comparten un único eje por métrica, por lo que las "
        "longitudes son directamente comparables; las métricas en escalas "
        "distintas se separan en sus propias filas en lugar de forzarlas a "
        "un eje compartido."
    ),
    "recall_by_type": (
        "Tasa de detección desglosada por las cuatro geometrías de "
        "anomalía inyectadas. Es el diagnóstico que un PR-AUC agregado no "
        "puede dar: muestra a qué tipo de anomalía es ciego un detector. Un "
        "modelo puede mostrar un puntaje global respetable sin recuperar "
        "ninguna de un tipo en particular."
    ),
    "iforest_shap_summary": (
        "Qué features mueven el puntaje del Isolation Forest, y en qué "
        "dirección, por fila. Obsérvese la dispersión vertical: un feature "
        "cuyos puntos se abren lejos del centro impulsa fuertemente el "
        "puntaje; uno agrupado en cero simplemente acompaña sin influir."
    ),
    "iforest_path_length": (
        "El mecanismo detrás del puntaje. Isolation Forest aísla una "
        "anomalía en menos divisiones aleatorias, por lo que una longitud "
        "de camino promedio más corta implica un puntaje de anomalía más "
        "alto -- este gráfico muestra esa relación cumpliéndose en datos "
        "reales, confirmando que el puntaje significa lo que el algoritmo "
        "afirma."
    ),
    "vae_recon_by_feature": (
        "Qué columnas de entrada reconstruye peor el VAE. Dado que el "
        "puntaje de anomalía ES el error de reconstrucción, los features en "
        "la parte superior de este gráfico son los que efectivamente "
        "impulsan la clasificación del VAE."
    ),
    "embedding": (
        "El espacio de features comprimido a dos dimensiones, coloreado "
        "por puntaje de anomalía. Busque puntos con puntaje alto separados "
        "de la masa principal. Precaución: la proyección 2D es una "
        "representación con pérdida -- la distancia aparente es sugestiva, "
        "nunca prueba de separación en el espacio completo."
    ),
    "vae_latent_space": (
        "La representación latente aprendida por el VAE en dos "
        "dimensiones. A diferencia de la proyección de features crudos de "
        "arriba, esta muestra el espacio que el propio modelo construyó, "
        "que es contra el cual se calcula su error de reconstrucción."
    ),
}


# --------------------------------------------------------------------------- #
# 3. The checks that make the run statistically trustworthy.                  #
# --------------------------------------------------------------------------- #
#: (nombre, qué verifica, por qué importa, qué significaría una falla)
STATISTICAL_CHECKS: list[tuple[str, str, str, str]] = [
    (
        "División cronológica (fuera de tiempo)",
        "Los bloques de entrenamiento, validación y test se cortan "
        "estrictamente por período, nunca al azar: el modelo se ajusta con "
        "meses anteriores y se evalúa sobre meses posteriores.",
        "Una división aleatoria permite seleccionar el modelo usando filas "
        "que ocurren *después* de aquellas con las que aprendió. Eso mide "
        "interpolación, no pronóstico, e infla todas las métricas. Los "
        "datos de panel siempre cargan este riesgo porque la misma entidad "
        "aparece en muchos períodos.",
        "Cualquier mezcla aleatoria aquí haría que las métricas reportadas "
        "fueran inalcanzables en producción -- el modelo luciría mucho "
        "mejor en el papel que en uso real.",
    ),
    (
        "Preprocesamiento ajustado solo con filas de entrenamiento",
        "Las medianas de imputación, los momentos del escalador, los "
        "exponentes de Yeo-Johnson y las categorías one-hot se estiman "
        "únicamente con el bloque de entrenamiento, y luego se aplican sin "
        "cambios a validación y test.",
        "Ajustar un escalador con todos los períodos filtra información del "
        "futuro hacia la transformación de filas pasadas. La fuga es sutil "
        "-- no se toca ninguna etiqueta -- pero igual traspasa conocimiento "
        "distribucional del período de test hacia las entradas del modelo.",
        "Las métricas quedarían sesgadas de forma optimista en una "
        "magnitud que no puede estimarse después del hecho, porque la fuga "
        "queda incorporada en los features.",
    ),
    (
        "Features de panel causales (solo hacia atrás)",
        "Los features de rezago, diferencia y z-score de historia propia "
        "dentro de cada entidad miran estrictamente hacia atrás, y los "
        "horizontes se validan contra la profundidad de la ventana de "
        "entrenamiento antes de usarse.",
        "Un feature que mira hacia adelante es el tipo de fuga más dañino. "
        "Existe también una segunda falla, más silenciosa: un horizonte más "
        "profundo que el bloque de entrenamiento deja a cada fila de "
        "entrenamiento en un valor de relleno, por lo que el escalador ve "
        "varianza casi nula y luego amplifica sin límite los valores reales "
        "de test.",
        "Cualquiera de las dos fallas produce un modelo que no puede "
        "reproducirse con datos en vivo. La segunda ya afectó a este "
        "proyecto antes, alcanzando magnitudes de 1e18 antes de agregar la "
        "validación de horizonte.",
    ),
    (
        "Integridad de claves del panel balanceado",
        "Cada par (entidad, período) aparece exactamente una vez, y cada "
        "valor de período se interpreta como una fecha real -- incluyendo "
        "formatos compactos yyyyMM.",
        "Las claves duplicadas vuelven ambiguos los features de rezago y "
        "diferencia, y duplican silenciosamente entidades al unir con la "
        "verdad base. Los períodos no interpretables rompen el propio orden "
        "temporal, del cual depende todo lo demás.",
        "Bloqueante. El pipeline se detiene antes de entrenar cualquier "
        "modelo, nombrando las claves o valores exactos causantes del "
        "problema.",
    ),
    (
        "Matriz de features finita",
        "La matriz entregada a cada detector no contiene NaN ni valores "
        "infinitos, verificado inmediatamente antes del ajuste.",
        "Ambos detectores fallan de forma distinta y poco útil ante "
        "entradas no finitas: scikit-learn lanza un error en las "
        "profundidades del constructor de árboles, y el VAE propaga NaN "
        "silenciosamente a través de la función de pérdida hasta que todos "
        "los puntajes son NaN.",
        "Bloqueante, y se reporta con el conteo exacto y la forma de la "
        "matriz en lugar de aparecer después como una métrica NaN "
        "inexplicada.",
    ),
    (
        "Umbral calibrado en validación, aplicado a test",
        "El punto de corte de alerta se estima ajustando una distribución "
        "Generalizada de Pareto a la cola de los puntajes de validación "
        "(peaks-over-threshold) para una tasa de falsas alarmas objetivo, o "
        "mediante un percentil simple como alternativa.",
        "Elegir el umbral sobre las mismas filas con las que se evalúa es "
        "una decisión retrospectiva que no puede repetirse en producción. "
        "Se usa teoría de valores extremos porque la cantidad de interés "
        "vive en la cola lejana, donde la distribución empírica es escasa y "
        "una estimación por percentil es inestable.",
        "La tasa de alerta reportada sería un artefacto del conjunto de "
        "test y no una propiedad de la regla.",
    ),
    (
        "Medición de superposición de individuos",
        "La proporción de entidades fuera de tiempo (OOT) que también "
        "están presentes en entrenamiento se mide y registra en cada "
        "corrida.",
        "Se reporta como diagnóstico, deliberadamente no como "
        "aprobado/reprobado. En un panel sintético balanceado, un "
        "solapamiento del 100% es la propiedad diseñada, no un defecto. Se "
        "vuelve una preocupación genuina solo cuando la población real "
        "tiene rotación, deserción y entidades nuevas -- condiciones que "
        "este generador no modela.",
        "No bloqueante. Existe para que el supuesto sea visible y pueda "
        "reexaminarse frente a datos reales en lugar de heredarse "
        "silenciosamente.",
    ),
    (
        "Estabilidad de la clasificación entre semillas",
        "El solapamiento entre las filas mejor clasificadas producidas "
        "bajo distintas semillas aleatorias, calculado como una métrica de "
        "confiabilidad sin etiquetas.",
        "Ambos detectores son estocásticos. Si la cola de alertas cambia "
        "sustancialmente con la semilla, entonces la cola de una corrida "
        "particular es en parte un artefacto del azar y no una propiedad de "
        "los datos.",
        "Un valor bajo no invalida el modelo, pero sí significa que la "
        "lista específica de individuos marcados no debe tratarse como "
        "definitiva.",
    ),
]


# --------------------------------------------------------------------------- #
# 4. Every parameter, its meaning, and what its default implies.              #
# --------------------------------------------------------------------------- #
#: model -> list of (parámetro, valor por defecto, qué controla, qué implica el valor por defecto)
PARAM_GLOSSARY: dict[str, list[tuple[str, str, str, str]]] = {
    "iforest": [
        ("n_estimators", "200",
         "Número de árboles de aislamiento promediados en un único puntaje "
         "de anomalía.",
         "Por encima del valor por defecto de scikit-learn (100), elegido "
         "porque la clasificación de puntajes con 100 árboles aún varía "
         "notablemente entre semillas aleatorias. El costo crece de forma "
         "aproximadamente lineal con este número; la precisión se "
         "estabiliza mucho antes de alcanzarlo."),
        ("max_samples", "'auto' (= min(256, n))",
         "Filas tomadas para construir cada árbol individual.",
         "Deliberadamente pequeño, y más pequeño es aquí el ajuste *más "
         "fuerte*, no el más débil: el resultado central del algoritmo es "
         "que las anomalías se aíslan más rápido en submuestras pequeñas, "
         "porque una muestra grande permite que la masa normal las rodee y "
         "las enmascare."),
        ("contamination", "0.02",
         "Fracción de filas tratadas como anómalas al convertir puntajes "
         "en etiquetas duras.",
         "Afecta solo a predict() y decision_function(). NO cambia "
         "score_samples(), que es lo que este pipeline usa para clasificar "
         "y umbralizar -- por lo tanto es una elección de punto de "
         "operación, nunca una estimación de la tasa real de anomalías, y "
         "deliberadamente no se ajusta (tuning)."),
        ("max_features", "1.0",
         "Fracción de columnas consideradas al elegir cada división.",
         "Todos los features disponibles en cada división. Reducirlo "
         "agrega una segunda fuente de diversidad entre árboles, a costa de "
         "que cada árbol vea menos datos; se incluye en la búsqueda de "
         "tuning en lugar de fijarse."),
        ("bootstrap", "False",
         "Si la muestra de filas de cada árbol se toma con reemplazo.",
         "Muestreo sin reemplazo, igual que el algoritmo original. La "
         "alternativa cambia levemente las estadísticas de diversidad "
         "entre muestras y se ofrece al tuner porque no cuesta nada "
         "incluirla."),
        ("random_state", "42",
         "Semilla para las decisiones aleatorias de división.",
         "Fija para que una métrica reportada pueda reproducirse. El valor "
         "por defecto de scikit-learn es None, lo que haría que cada "
         "corrida produjera un bosque y puntajes distintos."),
    ],
    "vae": [
        ("latent_dim", "8",
         "Ancho del cuello de botella comprimido que debe atravesar la "
         "entrada.",
         "El control central de capacidad. Demasiado angosto y hasta las "
         "filas normales reconstruyen mal, comprimiendo la brecha de la "
         "que depende el puntaje; demasiado ancho y la red también puede "
         "reproducir fielmente las anomalías, borrando la señal. Se ajusta "
         "mediante tuning en lugar de asumirse."),
        ("hidden_dim / n_layers", "64 / 2",
         "Ancho y profundidad del encoder (el decoder lo refleja).",
         "Una red deliberadamente pequeña. El dataset es tabular y de "
         "tamaño moderado, donde más profundidad principalmente compra "
         "sobreajuste; no se usan capas de normalización, razón por la "
         "cual la profundidad también está limitada a 3 en la búsqueda."),
        ("beta", "1.0",
         "Peso del término KL: pérdida = reconstrucción + beta x KL.",
         "1.0 es el VAE de manual (una cota inferior de evidencia "
         "propiamente dicha). Por encima de 1 regulariza más fuerte el "
         "espacio latente a costa de la calidad de reconstrucción -- lo "
         "cual comprime directamente el puntaje de anomalía, ya que el "
         "puntaje ES el error de reconstrucción."),
        ("kl_anneal_epochs", "10",
         "Épocas durante las cuales el peso KL sube linealmente de cero a "
         "beta.",
         "Protege contra el colapso posterior: aplicar presión KL "
         "completa antes de que el decoder haya aprendido algo útil "
         "empuja cada latente hacia el prior, tras lo cual el error de "
         "reconstrucción no lleva ninguna señal de anomalía."),
        ("early_stopping_patience", "10",
         "Épocas sin mejora en validación antes de detener el "
         "entrenamiento.",
         "Monitorea la pérdida de validación, no la de entrenamiento, y "
         "restaura los pesos de la mejor época en lugar de los de la "
         "última. La pérdida de entrenamiento sigue mejorando incluso al "
         "comenzar el sobreajuste, por lo que no puede servir como señal."),
        ("lr / optimizer / batch_size", "1e-3 / adam / 256",
         "Tamaño de paso, regla de actualización y filas por paso de "
         "gradiente.",
         "Puntos de partida estándar y conservadores. Los tres están en "
         "el espacio de búsqueda de tuning; ninguno es una afirmación "
         "sobre lo óptimo para estos datos."),
        ("score_kl_weight", "0.0",
         "Cuánto del término KL por fila se mezcla en el puntaje de "
         "anomalía.",
         "Cero significa que el puntaje es error de reconstrucción puro, "
         "que es lo que asume cada métrica de este reporte. Cambiarlo "
         "cambia el significado de cada puntaje exportado."),
    ],
}


ML_VS_ECONOMETRICS = {
    "verdict": "Machine learning — detección de anomalías no supervisada, no econometría.",
    "paragraphs": [
        (
            "Lo que distingue a ambos enfoques es el objetivo, no el "
            "algoritmo. Un modelo econométrico existe para estimar un "
            "parámetro y defenderlo: el coeficiente es el resultado, lleva "
            "un error estándar y un intervalo de confianza, y el esfuerzo "
            "de modelado se dirige a que esa estimación sea insesgada e "
            "interpretable como una cantidad causal o estructural. Los "
            "supuestos allí (exogeneidad, ausencia de autocorrelación, "
            "homocedasticidad) existen para proteger la validez del "
            "coeficiente."
        ),
        (
            "Este pipeline no estima ningún parámetro de ese tipo. Ambos "
            "detectores producen una clasificación (ranking), y esa "
            "clasificación es el producto. Ni Isolation Forest ni el VAE "
            "exponen algo parecido a un coeficiente interpretable: la "
            "salida del bosque es una longitud promedio de camino de "
            "aislamiento sobre árboles aleatorizados, y la del VAE es un "
            "error de reconstrucción a través de una red no lineal. "
            "Ninguno tiene error estándar, y ninguno admite una afirmación "
            "causal. Los supuestos que esta corrida verifica son, en "
            "consecuencia, de otro tipo — protegen la honestidad de una "
            "predicción fuera de muestra (sin fuga de información, sin "
            "mirar hacia adelante, un umbral no elegido con retrospectiva), "
            "no la insesgadez de una estimación."
        ),
        (
            "Los modelos también son no supervisados: se ajustan sin ver "
            "nunca una etiqueta de anomalía. Donde existen etiquetas aquí, "
            "se usan solo después, para puntuar qué tan buena resultó la "
            "clasificación — y por defecto este pipeline ni siquiera les "
            "permite influir en el tuning."
        ),
    ],
    "classification_driver": (
        "No existe un coeficiente que decida la clasificación, y esa "
        "diferencia es la que importa en la práctica. Lo que convierte un "
        "puntaje de anomalía continuo en una marca binaria es "
        "<strong>el umbral calibrado</strong> — estimado ajustando una "
        "distribución Generalizada de Pareto a la cola de los puntajes de "
        "<em>validación</em> para una tasa de falsas alarmas objetivo, y "
        "luego aplicado sin cambios al bloque fuera de tiempo. Mover ese "
        "único número desplaza cada precisión, recall y conteo de alertas "
        "de este reporte, mientras el modelo subyacente permanece intacto. "
        "Hay dos parámetros que es fácil confundir con él y no lo son: el "
        "<code>contamination</code> del Isolation Forest, que solo "
        "desplaza su frontera interna de <code>predict()</code> y deja "
        "intacta la clasificación que este pipeline realmente usa; y el "
        "tamaño de la cola de revisión (top-N), que es una decisión de "
        "capacidad operativa y no una decisión estadística."
    ),
}


def metric_note(metric_base: str) -> Optional[tuple[str, str]]:
    """``(what it is, how to read it)`` for a metric, or ``None`` if undocumented."""
    return METRIC_GLOSSARY.get(metric_base)


def figure_note(title: str) -> Optional[str]:
    """Match a figure title to its explanatory note, or ``None``.

    Matching is by substring against the figure's title because titles are
    composed at the call site (e.g. ``"iforest UMAP embedding"``), so an exact
    key lookup would miss. Order matters: the more specific latent-space key is
    tested before the generic embedding one.
    """
    low = title.lower()
    if "latent" in low:
        return FIGURE_NOTES["vae_latent_space"]
    if "shap" in low:
        return FIGURE_NOTES["iforest_shap_summary"]
    if "path-length" in low or "path length" in low:
        return FIGURE_NOTES["iforest_path_length"]
    if "per-feature" in low or "recon_by_feature" in low:
        return FIGURE_NOTES["vae_recon_by_feature"]
    if "embedding" in low:
        return FIGURE_NOTES["embedding"]
    if "roc" in low or "pr " in low:
        return FIGURE_NOTES["roc_pr"]
    if "distribution" in low or "score" in low:
        return FIGURE_NOTES["score_distribution"]
    return None


def param_note(model: str, param: str) -> Optional[tuple[str, str, str, str]]:
    """The glossary row for one parameter of one model, if documented."""
    for row in PARAM_GLOSSARY.get(model, []):
        if row[0] == param or param in row[0].split(" / "):
            return row
    return None


# --------------------------------------------------------------------------- #
# 5. Interactive Plotly charts.                                               #
# --------------------------------------------------------------------------- #
# Plotly cannot read CSS custom properties, so a figure cannot inherit the
# report's theme tokens the way the rest of the page does. Backgrounds are
# therefore transparent (the panel behind shows through, correct in both
# themes) and every *colour* is re-applied from JS on load and on every theme
# toggle -- see `THEME_RESTYLE_JS`. A single fixed colour set is not an option
# here: the palette validator FAILS slot-2 orange (#eb6834, OKLCH L 0.671)
# against the dark surface's 0.48-0.67 band, which is exactly why the dark
# column of SERIES_COLORS exists.
_AXIS_GRID = "rgba(128,138,160,0.20)"
_TRANSPARENT = "rgba(0,0,0,0)"

# -- Mark specs, applied uniformly to every chart in the report -------------- #
# Fixed rather than per-chart so the figures read as one system. The values are
# the project's data-viz conventions: thin marks, rounded data-ends, a surface
# gap doing the separating (never a stroke around a mark, which would add ink
# that is not data).
_BAR_CORNER_RADIUS = 4      # px rounded data-end on columns
_LINE_WIDTH = 2             # px
_MARKER_SIZE = 7            # px diameter for scatter marks
# Bar *thickness* is not set in px: Plotly sizes bars from the slot width, so
# `bargap` below is the lever -- 0.28 leaves ~28% of every band as air, which
# is what keeps the columns thin at any figure width.
#: The 2px separator between touching marks is drawn in the *surface* colour so
#: it disappears into the page. The report's surface differs per theme, so the
#: gap is created by `bargap`/`bargroupgap` (real empty space) rather than by a
#: coloured stroke that would have to be re-themed.
_BAR_GAP = 0.28
_BAR_GROUP_GAP = 0.08

#: Emitted alongside the figures; `applyPlotlyTheme()` is called on load and by
#: the report's existing theme toggle.
THEME_RESTYLE_JS = """
const PLOTLY_SERIES_COLORS = __SERIES_COLORS__;
const PLOTLY_FIGURE_SERIES = __FIGURE_SERIES__;
function applyPlotlyTheme() {
  if (typeof Plotly === "undefined") return;
  const root = document.documentElement;
  const explicit = root.getAttribute("data-theme");
  const dark = explicit === "dark" ||
    (!explicit && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const mode = dark ? "dark" : "light";
  const ink = dark ? "#8b93ab" : "#5b6479";
  Object.keys(PLOTLY_FIGURE_SERIES).forEach(divId => {
    const el = document.getElementById(divId);
    if (!el || !el.data) return;
    const series = PLOTLY_FIGURE_SERIES[divId];
    series.forEach((modelName, i) => {
      /* "__neutral__" marks a non-identity trace (the "normal rows" half of a
         score histogram): it takes recessive ink in both themes so the
         model's own colour stays reserved for the series that carries
         identity. */
      const color = modelName === "__neutral__"
        ? (dark ? "rgba(139,147,171,0.55)" : "rgba(91,100,121,0.45)")
        : PLOTLY_SERIES_COLORS[mode][modelName];
      if (!color) return;
      /* Bars/histograms carry colour on marker; lines on line. Restyling both
         is harmless for whichever the trace does not use. */
      Plotly.restyle(el, {"marker.color": color, "line.color": color}, [i]);
    });
    Plotly.relayout(el, {
      "font.color": ink,
      "xaxis.gridcolor": "rgba(128,138,160,0.20)",
      "yaxis.gridcolor": "rgba(128,138,160,0.20)",
      "xaxis2.gridcolor": "rgba(128,138,160,0.20)",
      "yaxis2.gridcolor": "rgba(128,138,160,0.20)",
      "legend.font.color": ink,
    });
  });
}
"""


def _base_layout(go, title: str, height: int = 340, **kwargs):
    """Shared layout: transparent surfaces, recessive grid, room for the title.

    ``t=56`` top margin is deliberate -- Plotly's default lets a two-line title
    collide with the plotting area, which is the overlap problem this report
    had to fix.
    """
    layout = dict(
        title=dict(text=title, font=dict(size=14), x=0, xanchor="left", y=0.97),
        height=height,
        # Bottom margin carries the legend: `b=86` reserves the strip the
        # legend sits in. Placing it above the plot (Plotly's usual `y=1.0`
        # trick) puts it in the same band as the title, which is what caused
        # the legend/title collision this layout had to fix.
        margin=dict(l=60, r=24, t=52, b=86),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font=dict(size=12, color="#5b6479",
                  family='-apple-system, "Segoe UI", Roboto, sans-serif'),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
            bgcolor=_TRANSPARENT, borderwidth=0,
        ),
        hovermode="closest",
    )
    layout.update(kwargs)
    return go.Layout(**layout)


def _style_axes(fig) -> None:
    fig.update_xaxes(gridcolor=_AXIS_GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=_AXIS_GRID, zeroline=False, showline=False)


def build_plotly_figures(chart_data: dict, log=None) -> list[dict]:
    """Build the report's interactive charts from raw run data.

    Args:
        chart_data: ``{"models": {name: {"oot_scores", "oot_labels",
            "threshold", "metrics"}}, "anomaly_rate": float}`` as assembled by
            ``main.py``. Missing pieces are tolerated -- each chart is skipped
            individually rather than failing the report.
        log: Optional logger for skip diagnostics.

    Returns:
        A list of ``{"id", "title", "note", "html", "series"}`` dicts, in the
        order they should appear. ``series`` lists the model each trace belongs
        to, so the theme restyler can recolour by index.
    """
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.io import to_html
        from plotly.subplots import make_subplots
    except Exception as exc:  # plotly is a declared dependency; degrade anyway
        if log:
            log.warning("Plotly unavailable (%s); interactive charts skipped.", exc)
        return []

    models = (chart_data or {}).get("models") or {}
    if not models:
        return []
    base_rate = float((chart_data or {}).get("anomaly_rate") or 0.0)
    out: list[dict] = []
    first = True  # only the first figure carries the ~4.8MB inline plotly.js

    def emit(fig, fig_id: str, title: str, note: str, series: list[str],
             group: str = "resultados") -> None:
        """Render one figure and record it under a display ``group``.

        ``group`` drives the report's sectioning: ``"resultados"`` charts lead,
        ``"modelo"`` holds the per-detector explanations, and
        ``"diagnostico"`` is the collapsed raw-input section. Without it all
        ~20 charts would stack into one undifferentiated wall.
        """
        nonlocal first
        _style_axes(fig)
        html = to_html(
            fig, include_plotlyjs=(True if first else False), full_html=False,
            div_id=fig_id, config={"displayModeBar": False, "responsive": True},
        )
        first = False
        out.append({"id": fig_id, "title": title, "note": note,
                    "html": html, "series": series, "group": group})

    ordered = [m for m in ("iforest", "vae") if m in models]

    # -- 1. Score distribution per model, with the calibrated threshold ------ #
    # Always available: it needs no labels. One chart PER model, never
    # overlaid -- the two detectors' scores live on different scales
    # (isolation depth vs reconstruction error), so a shared axis would be the
    # classic dual-axis error.
    for m in ordered:
        s_raw = models[m].get("oot_scores")
        if not s_raw:
            continue
        try:
            s = np.asarray(s_raw, dtype=float)
            y_raw = models[m].get("oot_labels")
            labelled = bool(y_raw) and len(set(y_raw)) > 1
            fig = go.Figure()
            if labelled:
                y = np.asarray(y_raw, dtype=int)
                fig.add_trace(go.Histogram(
                    x=s[y == 0], name="normal", opacity=0.75, nbinsx=60,
                    histnorm="probability density",
                    marker=dict(cornerradius=_BAR_CORNER_RADIUS),
                    hovertemplate="puntaje %{x:.4f}<br>densidad %{y:.2f}<extra>normal</extra>",
                ))
                fig.add_trace(go.Histogram(
                    x=s[y == 1], name="anomalía conocida", opacity=0.75, nbinsx=60,
                    histnorm="probability density",
                    marker=dict(cornerradius=_BAR_CORNER_RADIUS),
                    hovertemplate="puntaje %{x:.4f}<br>densidad %{y:.2f}<extra>anomalía</extra>",
                ))
                series = ["__neutral__", m]
                note = FIGURE_NOTES["score_distribution_supervised"]
            else:
                # One series: no legend box. There is only one colour, and the
                # chart title already names what is plotted -- a single-swatch
                # legend restates the title and costs space.
                fig.add_trace(go.Histogram(
                    x=s, name=f"puntajes de {m}", opacity=0.85, nbinsx=60,
                    showlegend=False,
                    marker=dict(cornerradius=_BAR_CORNER_RADIUS),
                    hovertemplate="puntaje %{x:.4f}<br>filas %{y}<extra></extra>",
                ))
                series = [m]
                note = FIGURE_NOTES["score_distribution"]
            thr = models[m].get("threshold")
            n_alert = None
            if thr is not None and np.isfinite(thr):
                n_alert = int((s >= float(thr)).sum())
                fig.add_vline(
                    x=float(thr), line=dict(dash="dash", width=2, color="#d03b3b"),
                    annotation_text=f"umbral de alerta ({n_alert} de {len(s)} filas)",
                    annotation_position="top right", annotation_font_size=10,
                )
            fig.update_layout(_base_layout(
                go, f"{m}: distribución de puntajes (bloque fuera de tiempo)", height=330,
                barmode="overlay", bargap=0.02,
            ))
            fig.update_xaxes(title_text="puntaje de anomalía (mayor = más anómalo)")
            fig.update_yaxes(title_text="densidad" if labelled else "filas")
            emit(fig, f"fig-scores-{m}", f"{m}: distribución de puntajes", note, series)
        except Exception as exc:
            if log:
                log.warning("Score-distribution chart for %s skipped (%s).", m, exc)

    # -- 2. Model agreement: the key label-free diagnostic ------------------- #
    if len(ordered) >= 2:
        try:
            a, b = ordered[0], ordered[1]
            sa = np.asarray(models[a].get("oot_scores") or [], dtype=float)
            sb = np.asarray(models[b].get("oot_scores") or [], dtype=float)
            if sa.size and sa.size == sb.size:
                from scipy.stats import rankdata, spearmanr

                pa = 100.0 * rankdata(sa) / len(sa)
                pb = 100.0 * rankdata(sb) / len(sb)
                rho = float(spearmanr(sa, sb).statistic)
                k = max(1, int(round(0.05 * len(sa))))
                top_a = set(np.argsort(-sa)[:k].tolist())
                top_b = set(np.argsort(-sb)[:k].tolist())
                overlap = len(top_a & top_b)
                fig = go.Figure()
                # Density is the message here, so the marks stay small and
                # semi-transparent: at ~1k points a full-size opaque dot with a
                # surface ring would read as a solid block and hide exactly the
                # crowding the chart exists to show.
                fig.add_trace(go.Scattergl(
                    x=pa, y=pb, mode="markers", name="individuo (fila OOT)",
                    showlegend=False,
                    marker=dict(size=_MARKER_SIZE, opacity=0.42),
                    hovertemplate=(f"percentil {a} " + "%{x:.1f}<br>"
                                   + f"percentil {b} " + "%{y:.1f}<extra></extra>"),
                ))
                for edge in (95,):
                    fig.add_vline(x=edge, line=dict(dash="dot", width=1,
                                                    color="rgba(128,138,160,0.7)"))
                    fig.add_hline(y=edge, line=dict(dash="dot", width=1,
                                                    color="rgba(128,138,160,0.7)"))
                fig.add_annotation(
                    x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left",
                    yanchor="top", showarrow=False, align="left", font=dict(size=11),
                    text=(f"Spearman rho = {rho:.3f}<br>"
                          f"solapamiento top-5%: {overlap} de {k} individuos"),
                )
                fig.update_layout(_base_layout(
                    go, f"¿Coinciden los dos detectores? {a} vs {b} (OOT)", height=420,
                ))
                fig.update_xaxes(title_text=f"percentil de puntaje de {a}", range=[0, 100])
                fig.update_yaxes(title_text=f"percentil de puntaje de {b}", range=[0, 100],
                                 scaleanchor="x", scaleratio=1)
                emit(fig, "fig-agreement", "Concordancia entre detectores",
                     FIGURE_NOTES["model_agreement"], ["__neutral__"])
        except Exception as exc:
            if log:
                log.warning("Model-agreement chart skipped (%s).", exc)

    # -- 3. Supervised-only charts ------------------------------------------ #
    # Gated on the run actually being supervised. The default strategy is
    # unsupervised, and showing ROC/PR against ground truth in that mode would
    # report an evaluation the run did not perform.
    have_labels = [
        m for m in ordered
        if models[m].get("supervised") and models[m].get("oot_labels")
        and models[m].get("oot_scores")
        and len(set(models[m]["oot_labels"])) > 1
    ]
    if have_labels:
        try:
            from sklearn.metrics import (
                average_precision_score, precision_recall_curve, roc_auc_score,
                roc_curve,
            )

            fig = make_subplots(
                rows=1, cols=2, horizontal_spacing=0.13,
                subplot_titles=("Curva ROC", "Curva Precisión-Recall"),
            )
            series: list[str] = []
            for m in have_labels:
                y = np.asarray(models[m]["oot_labels"], dtype=int)
                s = np.asarray(models[m]["oot_scores"], dtype=float)
                fpr, tpr, _ = roc_curve(y, s)
                auc = roc_auc_score(y, s)
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m} (AUC {auc:.3f})",
                               line=dict(width=_LINE_WIDTH),
                               hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>"),
                    row=1, col=1,
                )
                series.append(m)
            for m in have_labels:
                y = np.asarray(models[m]["oot_labels"], dtype=int)
                s = np.asarray(models[m]["oot_scores"], dtype=float)
                prec, rec, _ = precision_recall_curve(y, s)
                ap = average_precision_score(y, s)
                fig.add_trace(
                    go.Scatter(x=rec, y=prec, mode="lines", name=f"{m} (AP {ap:.3f})",
                               line=dict(width=_LINE_WIDTH), showlegend=True,
                               hovertemplate="Recall %{x:.3f}<br>Precisión %{y:.3f}<extra></extra>"),
                    row=1, col=2,
                )
                series.append(m)
            # Reference lines: the honest baselines for each panel.
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, row=1, col=1,
                          line=dict(dash="dash", width=1, color="rgba(128,138,160,0.6)"))
            if base_rate > 0:
                fig.add_shape(type="line", x0=0, y0=base_rate, x1=1, y1=base_rate,
                              row=1, col=2,
                              line=dict(dash="dash", width=1, color="rgba(128,138,160,0.6)"))
                fig.add_annotation(
                    x=0.98, y=base_rate, xref="x2", yref="y2",
                    text=f"tasa base {base_rate:.1%}", showarrow=False,
                    font=dict(size=10), yshift=10, xanchor="right",
                )
            fig.update_layout(_base_layout(go, "Calidad de la clasificación en el bloque fuera de tiempo", height=380))
            fig.update_xaxes(title_text="Tasa de falsos positivos", row=1, col=1, range=[0, 1])
            fig.update_yaxes(title_text="Tasa de verdaderos positivos", row=1, col=1, range=[0, 1])
            fig.update_xaxes(title_text="Recall", row=1, col=2, range=[0, 1])
            fig.update_yaxes(title_text="Precisión", row=1, col=2, range=[0, 1])
            emit(fig, "fig-roc-pr", "Calidad de la clasificación (OOT)", FIGURE_NOTES["roc_pr"], series)
        except Exception as exc:
            if log:
                log.warning("ROC/PR chart skipped (%s).", exc)

    # -- 4. Headline metric comparison (supervised only) -------------------- #
    # Only metrics bounded in [0,1] share this chart -- lift is unbounded and
    # would distort a shared axis, so it is left to the per-model tables.
    bounded = [
        ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
        ("best_f1", "Mejor F1"), ("precision_at_10pct", "Precisión@10%"),
        ("recall_at_10pct", "Recall@10%"),
    ]

    def _metric(model_name: str, key: str):
        return models[model_name].get("metrics", {}).get(f"oot_{key}")

    try:
        present = [
            (k, lab) for k, lab in bounded
            if any(_finite(_metric(m, k)) for m in ordered)
        ]
        if present:
            fig = go.Figure()
            series = []
            for m in ordered:
                vals = [_num_or_none(_metric(m, k)) for k, _ in present]
                fig.add_trace(go.Bar(
                    x=[lab for _, lab in present], y=vals, name=m,
                    text=[f"{v:.3f}" if v is not None else "" for v in vals],
                    textposition="outside", cliponaxis=False,
                    marker=dict(cornerradius=_BAR_CORNER_RADIUS),
                    hovertemplate="%{x}: %{y:.4f}<extra>" + m + "</extra>",
                ))
                series.append(m)
            fig.update_layout(_base_layout(
                go, "Métricas principales, bloque fuera de tiempo (mayor es mejor)",
                height=360, barmode="group", bargap=_BAR_GAP,
                bargroupgap=_BAR_GROUP_GAP, uniformtext=dict(mode="hide", minsize=9),
            ))
            fig.update_yaxes(title_text="puntaje", range=[0, 1.08])
            emit(fig, "fig-metric-compare", "Comparación de métricas principales",
                 FIGURE_NOTES["metric_comparison"], series)
    except Exception as exc:
        if log:
            log.warning("Metric-comparison chart skipped (%s).", exc)

    # -- 5. Recall by injected anomaly type (supervised only) --------------- #
    def _by_type(model_name: str) -> dict:
        return models[model_name].get("metrics", {}).get("oot_by_type") or {}

    try:
        types: list[str] = []
        for m in ordered:
            for t in _by_type(m):
                if t != "__overall__" and t not in types:
                    types.append(t)
        types.sort()
        if types:
            fig = go.Figure()
            series = []
            for m in ordered:
                bt = _by_type(m)
                vals = [_num_or_none((bt.get(t) or {}).get("recall_at_10pct")) for t in types]
                fig.add_trace(go.Bar(
                    x=types, y=vals, name=m,
                    text=[f"{v:.2f}" if v is not None else "" for v in vals],
                    textposition="outside", cliponaxis=False,
                    marker=dict(cornerradius=_BAR_CORNER_RADIUS),
                    hovertemplate="%{x}: recall %{y:.3f}<extra>" + m + "</extra>",
                ))
                series.append(m)
            fig.update_layout(_base_layout(
                go, "Recall@10% por tipo de anomalía inyectada", height=340,
                barmode="group", bargap=_BAR_GAP, bargroupgap=_BAR_GROUP_GAP,
                uniformtext=dict(mode="hide", minsize=9),
            ))
            fig.update_yaxes(title_text="recall en el 10% superior", range=[0, 1.12])
            emit(fig, "fig-recall-type", "Recall por tipo de anomalía",
                 FIGURE_NOTES["recall_by_type"], series)
    except Exception as exc:
        if log:
            log.warning("Recall-by-type chart skipped (%s).", exc)

    # -- 6. Everything that used to be an embedded PNG ---------------------- #
    # The report renders 100% Plotly: the static matplotlib figures are still
    # written to `artifacts/reports/figures/` as run evidence, but the HTML no
    # longer embeds them. Each one is rebuilt here from the numbers that
    # produced it, so the reader gets hover values and zoom instead of a flat
    # image.
    # Appends to `out` through the shared `emit` closure.
    _build_static_replacements(chart_data, go, np, emit, log)

    return out


def _build_static_replacements(chart_data, go, np, emit, log) -> list:
    """Interactive versions of the former PNG gallery.

    Driven by ``chart_data["static"]``, which ``main.py`` fills with the arrays
    the matplotlib figures were drawn from. Every block is independently
    guarded: a missing or malformed payload skips that one chart rather than
    losing the whole section.
    """
    static = (chart_data or {}).get("static") or {}
    produced: list = []

    def _emit(fig, fig_id, title, note, series, group="modelo"):
        emit(fig, fig_id, title, note, series, group=group)

    # -- 6a. Per-feature importance / reconstruction error ------------------ #
    # Horizontal bars: feature names are long, and a horizontal layout lets
    # them be read without rotating the label 90 degrees.
    for key, fig_id, title, note, model in (
        ("shap_importance", "fig-shap", "Importancia de features (SHAP) — Isolation Forest",
         FIGURE_NOTES["iforest_shap_summary"], "iforest"),
        ("recon_by_feature", "fig-recon-feat",
         "Error de reconstrucción por feature — VAE",
         FIGURE_NOTES["vae_recon_by_feature"], "vae"),
    ):
        payload = static.get(key)
        if not isinstance(payload, dict) or not payload:
            continue
        try:
            items = sorted(payload.items(), key=lambda kv: float(kv[1]), reverse=True)[:25]
            items.reverse()  # highest at the top of a horizontal bar chart
            names = [str(k) for k, _ in items]
            vals = [float(v) for _, v in items]
            fig = go.Figure(go.Bar(
                x=vals, y=names, orientation="h", showlegend=False,
                marker=dict(cornerradius=_BAR_CORNER_RADIUS),
                hovertemplate="%{y}: %{x:.5f}<extra></extra>",
            ))
            fig.update_layout(_base_layout(
                go, title, height=max(320, 18 * len(names) + 120),
                bargap=0.35,
                # Long feature names need room; without this the labels are
                # truncated by the default 60px left margin.
                margin=dict(l=220, r=24, t=52, b=48),
            ))
            fig.update_xaxes(title_text="importancia media |SHAP|" if model == "iforest"
                             else "error cuadrático medio")
            _emit(fig, fig_id, title, note, [model])
            produced.append(fig_id)
        except Exception as exc:
            if log:
                log.warning("Chart %s skipped (%s).", fig_id, exc)

    # -- 6b. iForest path length vs score ----------------------------------- #
    pl = static.get("path_length")
    if isinstance(pl, dict) and pl.get("scores"):
        try:
            fig = go.Figure(go.Scattergl(
                x=pl["scores"], y=pl["path_lengths"], mode="markers",
                showlegend=False, marker=dict(size=_MARKER_SIZE, opacity=0.4),
                hovertemplate="puntaje %{x:.4f}<br>camino %{y:.4f}<extra></extra>",
            ))
            corr = pl.get("corr")
            if corr is not None:
                fig.add_annotation(
                    x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left",
                    yanchor="top", showarrow=False, font=dict(size=11),
                    text=f"corr(puntaje, camino) = {float(corr):.3f}",
                )
            fig.update_layout(_base_layout(
                go, "Longitud de camino vs puntaje — Isolation Forest", height=380))
            fig.update_xaxes(title_text="puntaje de anomalía")
            fig.update_yaxes(title_text="longitud de camino normalizada")
            _emit(fig, "fig-pathlen", "Longitud de camino (iForest)",
                  FIGURE_NOTES["iforest_path_length"], ["iforest"])
            produced.append("fig-pathlen")
        except Exception as exc:
            if log:
                log.warning("Path-length chart skipped (%s).", exc)

    # -- 6c. 2D embeddings and the VAE latent space ------------------------- #
    for key, fig_id, title, note, model in (
        ("embedding_iforest", "fig-emb-iforest",
         "Proyección 2D del espacio de features — Isolation Forest",
         FIGURE_NOTES["embedding"], "iforest"),
        ("embedding_vae", "fig-emb-vae",
         "Proyección 2D del espacio de features — VAE",
         FIGURE_NOTES["embedding"], "vae"),
        ("latent_vae", "fig-latent",
         "Espacio latente del VAE", FIGURE_NOTES["vae_latent_space"], "vae"),
    ):
        payload = static.get(key)
        if not isinstance(payload, dict) or not payload.get("x"):
            continue
        try:
            method = str(payload.get("method", "")).upper()
            fig = go.Figure(go.Scattergl(
                x=payload["x"], y=payload["y"], mode="markers", showlegend=False,
                marker=dict(
                    size=_MARKER_SIZE, opacity=0.55,
                    color=payload.get("color"), colorscale="Viridis",
                    colorbar=dict(
                        title=dict(text=payload.get("color_label", "puntaje"),
                                   side="right"),
                        thickness=12, len=0.75, outlinewidth=0,
                    ),
                ),
                hovertemplate="%{x:.3f}, %{y:.3f}<br>color %{marker.color:.4f}"
                              "<extra></extra>",
            ))
            # `colorbar` occupies the right edge; widen the right margin so it
            # cannot sit on top of the plotting area.
            fig.update_layout(_base_layout(
                go, f"{title}{f' ({method})' if method else ''}", height=430,
                margin=dict(l=60, r=90, t=52, b=48),
            ))
            fig.update_xaxes(title_text="componente 1")
            fig.update_yaxes(title_text="componente 2", scaleanchor="x", scaleratio=1)
            _emit(fig, fig_id, title, note, ["__neutral__"])
            produced.append(fig_id)
        except Exception as exc:
            if log:
                log.warning("Chart %s skipped (%s).", fig_id, exc)

    # NOTE: deliberately no per-variable raw-feature histogram here. One chart
    # per numeric column does not scale -- a real feature mart with 50+
    # columns would embed 50+ Plotly figures and saturate the HTML for a
    # section that is diagnostic, not a modelling result. That diagnostic is
    # still available without the bloat: `model_documentation.md` links the
    # static per-column PNGs (`plot_transform_diagnostics`) by relative path
    # instead of embedding them, which is the point of that file being
    # separate from the business-facing report.

    if log and produced:
        log.info("Report: %d former PNG figure(s) rebuilt as interactive charts.",
                 len(produced))
    return []


def _finite(v) -> bool:
    try:
        f = float(v)
        return f == f and abs(f) != float("inf")
    except (TypeError, ValueError):
        return False


def _num_or_none(v):
    return float(v) if _finite(v) else None
