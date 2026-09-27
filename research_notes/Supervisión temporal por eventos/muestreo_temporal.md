# Muestreo y desbalance para supervisión temporal por eventos

## ¿Qué remedios al desbalance tienen respaldo y cuáles fallan bajo dependencia temporal?

### Takeaway
Para un panel `entity_id`–`codmes`, el orden conservador es: (1) modelo sobre la distribución original + umbral operativo, (2) pesos/costos o batches balanceados, (3) submuestreo estructurado por evento/tiempo mediante ensambles, y solo después (4) sobremuestreo sintético temporal validado. La prevalencia baja por sí sola no justifica fabricar observaciones; el problema puede ser falta absoluta de eventos, solapamiento, ruido de etiquetas o deriva, ninguno de los cuales se cura automáticamente con SMOTE.

### Cited Findings

- **Ponderación/costos.** El aprendizaje sensible a costos formaliza que distintos errores pueden tener costos distintos y permite cambiar la frontera sin alterar las filas originales; la revisión de Feng, Zhou y Tong muestra que la elección de remuestreo depende del paradigma (error global, costos o restricción Neyman–Pearson), el clasificador y la métrica, por lo que no existe una receta universal. **Estado:** artículo revisado por pares en revista de la ASA; **limitación:** experimentos principalmente i.i.d., no paneles temporales; **confianza:** alta para la ausencia de una regla universal, media para transferir resultados cuantitativos al proyecto. [Feng, Zhou & Tong, 2021](https://doi.org/10.1002/sam.11538)
- **Focal loss.** Focal loss multiplica la entropía cruzada por un factor que reduce la contribución de ejemplos ya bien clasificados y concentra el gradiente en ejemplos difíciles. **Estado:** paper original revisado por pares (ICCV); **limitación:** fue probado en detección de objetos, no en panel entidad–mes ni en probabilidades de riesgo; **confianza:** alta sobre el mecanismo, baja-media sobre beneficio en este proyecto. [Lin et al., 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)
- **Random undersampling (RUS).** El submuestreo simple reduce costo, pero descarta información de la mayoritaria; EasyEnsemble conserva varios subconjuntos y agrega modelos, mientras BalanceCascade retira negativos ya bien clasificados. En sus benchmarks i.i.d. mejoraron AUC, F-measure y G-mean frente a varios comparadores. **Estado:** paper original revisado por pares (IEEE TSMC-B); **limitación:** no preserva por diseño tiempo, entidad ni episodios; **confianza:** alta sobre algoritmo, media sobre transferencia. [Liu, Wu & Zhou, 2009](https://doi.org/10.1109/TSMCB.2008.2007853)
- **Balanced Random Forest (BRF).** BRF extrae para cada árbol una muestra bootstrap de cada clase del tamaño de la minoritaria; el informe original encontró mejora de desempeño minoritario frente a RF estándar en sus datos. **Estado:** informe técnico primario de UC Berkeley, no revista revisada por pares; **limitación:** evidencia antigua, i.i.d. y el bootstrap por fila rompe clústeres/tiempo si se usa sin adaptación; **confianza:** alta sobre definición, media sobre eficacia general. [Chen, Liaw & Breiman, 2004](https://digicoll.lib.berkeley.edu/record/85556)
- **SMOTE.** El método original interpola entre vecinos minoritarios y reportó que SMOTE + undersampling mejoraba el ROC frente a undersampling en C4.5, Ripper y Naive Bayes. **Estado:** paper original revisado por pares (JAIR); **limitación:** benchmarks i.i.d., métricas ROC y clasificadores de la época; no demuestra validez de trayectorias temporales ni calibración. **Confianza:** alta sobre mecanismo, baja-media para panel mensual. [Chawla et al., 2002](https://doi.org/10.1613/JAIR.953)
- **ADASYN.** Genera más sintéticos alrededor de ejemplos minoritarios considerados difíciles, desplazando adaptativamente la frontera. **Estado:** paper original revisado por pares (IJCNN/IEEE); **limitación:** al enfatizar zonas difíciles puede también enfatizar ruido o etiquetas dudosas; no modela secuencia; **confianza:** alta sobre mecanismo, baja-media para este uso. [He et al., 2008](https://doi.org/10.1109/IJCNN.2008.4633969)
- **Fallas conocidas de SMOTE.** Una investigación revisada por pares identifica sobregeneralización por sobremuestrear ruido, ejemplos no informativos y aumento del solapamiento cerca de fronteras; otro análisis muestra que los sintéticos quedan correlacionados con sus progenitores y que SMOTE reduce la varianza minoritaria, lo que afecta métodos que suponen observaciones independientes. **Estado:** artículos revisados por pares; **limitación:** no son específicos de paneles mensuales; **confianza:** alta sobre estos modos de falla. [Jiang et al., 2021](https://doi.org/10.1016/j.ins.2020.07.014); [Blagus & Lusa, 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3648438/)
- **Datos mixtos.** El propio trabajo de SMOTE reconoce que SMOTE estándar no maneja adecuadamente nominales y propone SMOTE-NC para mezcla continua/nominal. **Estado:** paper original revisado por pares; **limitación:** SMOTE-NC resuelve tipos de dato, no coherencia longitudinal, reglas de negocio ni consistencia entre campos. **Confianza:** alta. [Chawla et al., 2002, sección SMOTE-NC](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/node15.html)
- **Sobremuestreo temporal.** T-SMOTE crea muestras cerca de la frontera usando información temporal y reportó mejores AUC/F1/AUPRC en conjuntos de clasificación univariados y multivariados, incluida predicción temprana. **Estado:** paper revisado por pares (IJCAI 2022); **limitación:** clasifica series como objetos completos; no valida paneles entidad–mes con variables categóricas, ventanas superpuestas, eventos recurrentes ni probabilidades calibradas. **Confianza:** alta sobre los experimentos publicados, baja-media para transferibilidad. [Zhao et al., 2022](https://doi.org/10.24963/ijcai.2022/334)
- **Calibración dañada por remuestreo.** Simulaciones y un caso real hallaron que oversampling/RUS pueden producir calibración muy deficiente y no mejorar AUC; el estimador de corrección recuperó calibración manteniendo discriminación. En regresión logística clínica, RUS/ROS/SMOTE sobreestimaron riesgo sin mejorar discriminación, y cambiar el umbral logró una sensibilidad/especificidad similar. **Estado:** estudios revisados por pares; **limitación:** contexto clínico y simulaciones, no este dominio; **confianza:** alta sobre riesgo de calibración, media-alta sobre recomendación conservadora. [Piccininni et al., 2024](https://doi.org/10.1016/j.jbi.2024.104666); [van den Goorbergh et al., 2022](https://doi.org/10.1093/jamia/ocac093)

### Inferences

- **Jerarquía recomendada para el repositorio:**
  1. Baseline sin remuestreo con métrica PR-AUC, recall/precision al presupuesto de revisión, lift@k, Brier/log-loss y calibración por horizonte.
  2. Ajuste de umbral sobre validación temporal con prevalencia natural; si el costo FN/FP es conocido, optimizar utilidad/costo, no una tasa 50:50 artificial.
  3. Pesos de clase o costo por observación; limitar pesos extremos y compararlos contra baseline. Para redes, probar batches balanceados o focal loss como ablation, no como default.
  4. Si el volumen de negativos hace inviable entrenar: submuestreo de negativos **por bloques/episodios y estratos de `codmes`, entidad, régimen y riesgo**, conservando todos los positivos; mejor un ensamble tipo EasyEnsemble/BRF adaptado a bloques que un único RUS.
  5. Hard-negative mining solo con predicciones out-of-fold temporales: reincorporar falsos positivos/confusos de folds pasados y mantener una fracción aleatoria de negativos fáciles para representar la población.
  6. SMOTE/ADASYN solo como experimento tardío, dentro del fold, con progenitores del mismo régimen temporal y reglas de coherencia. T-SMOTE puede ser challenger si cada observación es una ventana completa comparable; no aplicarlo fila a fila a `entity_id`–`codmes`.
- Duplicar positivos completos (ROS) es preferible a interpolar trayectorias cuando la prioridad es simplicidad, pero no agrega información y puede sobreajustar entidades/eventos; batches ponderados suelen expresar el mismo objetivo de optimización con mejor trazabilidad.
- Nunca balancear a 50:50 por reflejo. Tratar la razón de muestreo/peso como hiperparámetro restringido y elegirla por utilidad temporal fuera de muestra y calibración.
- Definir la unidad positiva como **episodio**, no como cada mes positivo: si un evento ocupa varios `codmes`, conservar la ventana preevento y el onset, y evitar que un episodio largo domine la pérdida. Considerar peso `1 / meses_del_episodio` o muestrear una/anclas por episodio.

### Gaps

- No se encontró evidencia primaria suficiente para afirmar que un método sintético temporal sea seguro en paneles mensuales mixtos con identidad, reglas de negocio y eventos recurrentes.
- La literatura no ofrece una prevalencia universal a partir de la cual deba activarse SMOTE/RUS; importan el número absoluto de eventos independientes, el solapamiento, el ruido, la deriva, los costos y la capacidad del modelo.

## ¿Por qué el remuestreo debe ocurrir dentro de cada fold temporal/grupal y qué alternativas preservan eventos?

### Takeaway
Primero se congelan test y folds walk-forward por `codmes`, con purga/embargo equivalente al horizonte y la ventana de features; después se ajustan scaler, imputación, selección, vecinos, pesos y remuestreo **solo en el train de ese fold**. En un panel, además se controla el solapamiento por `entity_id` o episodio conforme al escenario de despliegue.

### Cited Findings

- Aplicar validación cruzada estándar a series permite que datos futuros entrenen modelos evaluados en el pasado (look-ahead bias); el preprocesamiento dependiente de datos antes del split también filtra información. **Estado:** revisión metodológica revisada por pares; **limitación:** orientación general, no estudio específico de SMOTE; **confianza:** alta. [Kapoor & Narayanan, 2023/2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11573893/)
- Un estudio empírico de validación temporal recomienda CV bloqueada para usar la información disponible evitando problemas teóricos de evolución y dependencia temporal. **Estado:** artículo revisado por pares; **limitación:** forecasting y no clasificación de eventos en panel; **confianza:** alta para respetar tiempo, media para diseño exacto. [Bergmeir & Benítez, 2012](https://doi.org/10.1016/j.ins.2011.12.028)
- En un estudio aplicado de clasificación desbalanceada, el remuestreo se ejecutó solo en train para prevenir leakage y el test conservó la distribución no balanceada. **Estado:** artículo revisado por pares; **limitación:** evidencia aplicada, no demostración formal; **confianza:** media-alta. [Solomatine et al./IRCIP, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10232287/)
- El block bootstrap fue desarrollado para dependencia y el cluster bootstrap tiene justificación teórica para datos longitudinales/agrupados. **Estado:** literatura estadística revisada por pares; **limitación:** se refiere principalmente a inferencia/incertidumbre, no a balancear etiquetas para clasificación; **confianza:** alta sobre preservar dependencia, media sobre uso como sampler de entrenamiento. [Künsch, 1989](https://doi.org/10.1214/aos/1176347265); [Cheng et al., 2013](https://doi.org/10.1016/j.jmva.2012.12.007)
- El muestreo por risk set selecciona controles que seguían en riesgo en el momento del caso; una evaluación de incidence-density sampling encontró estimaciones de riesgo relativo no sesgadas con el procedimiento correcto. **Estado:** artículo metodológico revisado por pares; **limitación:** epidemiología/inferencia causal, no optimización predictiva; **confianza:** alta sobre diseño, media sobre adaptación predictiva. [Richardson, 2004](https://doi.org/10.1136/oem.2004.014472)

### Inferences

- **Orden innegociable del pipeline por fold:**
  1. Definir corte temporal de train/validación/test y horizonte objetivo.
  2. Purgar observaciones cuyas ventanas de features o labels crucen el corte; añadir embargo si las ventanas se solapan.
  3. Aplicar regla de grupos: para generalizar a entidades nuevas, `entity_id` no puede aparecer en train y evaluación; para pronosticar meses futuros de entidades conocidas, sí puede aparecer, pero siempre con observaciones estrictamente anteriores y sin que el mismo episodio/ventana cruce folds.
  4. Ajustar transformaciones en train.
  5. Construir vecinos/sintéticos/submuestras/pesos **solo con train**.
  6. Entrenar y elegir hiperparámetros usando validación intacta con prevalencia real.
  7. Calibrar en un bloque posterior separado e intacto; evaluar una sola vez en test final.
- **Por qué SMOTE antes del split contamina:** un sintético puede usar como progenitor o vecino una fila que luego cae en validación/test; incluso sin duplicado exacto, comparte información geométrica y queda correlacionado con ella. En ventanas solapadas, dos filas adyacentes del mismo episodio ya contienen gran parte de los mismos meses; interpolarlas agrava el parentesco train–test.
- **Muestreo por episodio/evento:** conservar todos los episodios positivos; seleccionar negativos en el mismo `codmes`/régimen y del conjunto realmente en riesgo, con ventanas completas. Muestrear bloques contiguos por entidad en vez de meses aislados. Registrar `sampling_probability`, `sample_weight`, `episode_id`, `block_id`, `fold_id` y semilla.
- **Ensambles de submuestreo:** cada learner recibe todos los positivos y un bloque/subconjunto distinto de negativos, estratificado por tiempo/régimen. Agregar scores y calibrar después sobre prevalencia real. Esto reduce descarte total de negativos frente a un RUS único.
- **Hard negatives:** generarlos exclusivamente de errores OOF anteriores. No elegirlos con scores in-sample ni con el fold futuro. Limitar su proporción y conservar negativos aleatorios para no borrar regímenes fáciles pero prevalentes.
- **PU learning:** procede si `target=1` revisado es confiable pero `target=0` significa realmente “no revisado/no confirmado”. PU supone que unlabeled mezcla positivos y negativos y requiere hipótesis sobre el mecanismo de etiquetado o estimación del prior; la encuesta de Bekker y Davis distingue escenarios y advierte dependencia de supuestos como SCAR. **Estado:** survey revisado por pares en *Machine Learning*; **limitación:** muchos métodos suponen muestras i.i.d. y SCAR, probablemente irreal si analistas revisan alertas de alto score; **confianza:** alta sobre marco conceptual, media sobre aplicación. [Bekker & Davis, 2020](https://doi.org/10.1007/s10994-020-05877-5)
- Si la selección para revisión depende del score, mes, entidad o severidad, registrar `review_propensity`/motivo y no asumir SCAR. Comparar PU con: negativos confirmados, delayed labels y análisis de sensibilidad del prior por cohorte temporal.

### Gaps

- Sin conocer horizonte, duración/repetición del evento y si el despliegue puntúa entidades conocidas o nuevas, no puede fijarse una única regla de agrupación.
- No hay una proporción universal de negativos por evento para paneles predictivos. Debe elegirse mediante curvas de aprendizaje por **número de episodios independientes** y utilidad fuera de tiempo, no por filas.

## ¿Cómo corregir priors, calibrar y auditar leakage, imposibilidades y pérdida de regímenes?

### Takeaway
Validación, calibración y test deben conservar la prevalencia operacional. Todo cambio de prior de entrenamiento requiere guardar probabilidades de inclusión y comparar corrección analítica con recalibración out-of-time; con SMOTE/ADASYN no debe asumirse que una simple corrección de intercepto repara la densidad alterada.

### Cited Findings

- Submuestrear una clase cambia los priors y sesga las probabilidades posteriores aunque el ranking pueda preservarse; el sesgo de calibración afecta decisiones basadas en umbral. **Estado:** paper revisado por pares (IEEE SSCI); **limitación:** corrección teórica supone un muestreo cuya alteración puede expresarse por priors; **confianza:** alta. [Dal Pozzolo et al., 2015](https://ieeexplore.ieee.org/document/7376606/)
- Saerens, Latinne y Decaestecker derivan un procedimiento para ajustar outputs cuando cambian probabilidades a priori, bajo estabilidad de distribuciones condicionales. **Estado:** paper revisado por pares en *Neural Computation*; **limitación:** label shift/priors, no covariate o concept drift ni sintéticos que deforman `P(X|Y)`; **confianza:** alta sobre la corrección bajo supuestos, baja si esos supuestos fallan. [Saerens et al., 2002](https://doi.org/10.1162/089976602753284446)
- Beta calibration fue derivada para corregir scores sesgados; su familia incluye el mapeo identidad y en los experimentos superó calibración logística para algunos clasificadores, mientras isotónica puede sobreajustar con muestras pequeñas. **Estado:** paper revisado por pares (AISTATS/PMLR); **limitación:** resultados no específicos de deriva temporal o eventos raros; **confianza:** alta sobre método, media sobre cuál ganará aquí. [Kull, Silva Filho & Flach, 2017](https://proceedings.mlr.press/v54/kull17a.html)
- La literatura PU indica que recuperar desempeño verdadero suele depender de conocer/estimar el prior positivo y el ruido de labels; evaluar como si unlabeled fuera negativo sesga las métricas. **Estado:** investigación revisada por pares; **limitación:** el prior puede variar por `codmes`; **confianza:** alta. [Jain et al., 2017/2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417800/)

### Inferences

- **Calibración recomendada:**
  - Mantener un bloque de calibración posterior al train y anterior al test, con prevalencia natural y labels suficientemente maduros.
  - Si solo hubo RUS/ROS aleatorio por clase y se conocen probabilidades de inclusión, aplicar corrección Bayes/prior como baseline; aun así contrastar Platt, beta e isotónica out-of-time.
  - Si hubo SMOTE/ADASYN, hard-negative adaptativo o focal loss, preferir calibración empírica temporal porque no se garantiza solo cambio de prior.
  - Recalibrar por horizonte o régimen solo si hay suficientes eventos; de lo contrario usar calibrador global y medir intercepto/slope por subgrupo.
  - Separar umbral de probabilidad: calibrar primero, luego elegir umbral por costo/capacidad de revisión en validación; congelarlo antes del test.
- **Tests automáticos de integridad y fuga:**
  1. Unicidad de clave (`entity_id`, `codmes`, `label_version`) y hash de fila/ventana; cero hashes compartidos entre train, calibración y test.
  2. Para cada fila de train, `feature_max_timestamp <= as_of_date` y `label_window_start > as_of_date`; ninguna fecha/progenitor sintético posterior al corte.
  3. Cero `episode_id` o ventanas solapadas entre folds; según escenario, cero `entity_id` compartidos.
  4. Trazabilidad de cada sintético a progenitores del mismo fold, clase, horizonte y régimen permitido; vecinos calculados solo en train.
  5. Prueba canaria: permutar labels dentro de bloques temporales/entidades; desempeño debe caer a baseline. Probar además una feature deliberadamente futura y verificar que el validador la rechace.
- **Tests de imposibilidad sintética:**
  1. Tipos/rangos/dominios, enteros, categorías y nulos estructurales.
  2. Restricciones cruzadas (sumas, ratios, estados mutuamente excluyentes, fechas monotónicas).
  3. Continuidad de trayectoria: orden de `codmes`, saltos máximos plausibles, transiciones de estado permitidas y no aparición de la entidad antes de alta/después de baja.
  4. Distancia a datos reales y detección de memorization/near-duplicates; revisar manualmente una muestra estratificada de sintéticos.
  5. Comparar marginales, correlaciones y secuencias por régimen; rechazar el método aunque suba PR-AUC si viola reglas o colapsa diversidad.
- **Tests de cobertura y regímenes perdidos por undersampling:**
  1. Tabla train original vs muestreado por `codmes`, cohorte, región/segmento, antigüedad, fuente, `episode_type`, missingness y deciles de score.
  2. Cobertura mínima de cada estrato y número efectivo de entidades/episodios, no solo filas.
  3. PSI/Jensen–Shannon o distancia apropiada por feature/régimen; revisión específica de colas y negativos cercanos al evento.
  4. Estabilidad sobre múltiples semillas/submuestras; si la ganancia depende de una semilla, no promover.
- **Tests de calibración y utilidad:** Brier y log-loss; intercepto ideal 0 y slope ideal 1; curva de calibración con intervalos bootstrap por entidad/episodio; observed/expected total y por `codmes`, horizonte y segmentos; PR-AUC y precision/recall@k; alertas por 1.000 entidades; decision/utility curve con costos. Comparar antes/después de corrección y exigir no degradación material en test temporal.
- **Gates de descarte:** descartar cualquier sampler si (a) usa información futura o cruza episodio, (b) crea combinaciones imposibles, (c) elimina un régimen relevante, (d) mejora ranking pero daña calibración/utilidad sin corrección estable, o (e) la ventaja no se repite en varios cortes temporales.
- **Campos que conviene guardar por fila de entrenamiento/revisión:** `entity_id`, `codmes`, `target`, `label_status` (`positive_confirmed`, `negative_confirmed`, `unlabeled`, `pending`), `label_as_of_date`, `event_date`, `episode_id`, `episode_start/end`, `horizon`, `reviewed_at`, `reviewer/source`, `review_reason`, `selection_score_at_review`, `review_propensity` si existe, `label_version`, `fold_id`, `sampling_method`, `sampling_probability`, `sample_weight`, `block_id`, `synthetic_flag`, `parent_row_ids`, `random_seed`, `feature_as_of_date` y `data_snapshot_id`.

### Gaps

- Deben definirse tolerancias numéricas de calibración y utilidad con el dueño del proceso; no hay umbral académico universal.
- La corrección de prior solo es defendible si se registra exactamente el mecanismo de inclusión y no hay cambio relevante en `P(X|Y)`; esto debe validarse, no asumirse.
