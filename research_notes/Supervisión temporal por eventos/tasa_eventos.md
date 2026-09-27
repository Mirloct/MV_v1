# Suficiencia de eventos para supervisión temporal en panel `entity_id`-`codmes`

## ¿Qué debe contarse y por qué no existe una tasa de eventos universal?

### Takeaway
La prevalencia fila-mes (`sum(target)/filas elegibles`) sirve para dimensionar volumen y presupuesto de alertas, pero no determina por sí sola si un modelo supervisado es viable. La unidad informativa principal debe ser el **episodio positivo independiente y maduro**, junto con entidades positivas distintas, meses cubiertos, regímenes temporales y cobertura de revisión; las filas correlacionadas de un mismo episodio no pueden contarse como eventos independientes.

### Cited Findings
- Para desarrollar un modelo binario, el tamaño muestral debe considerar conjuntamente participantes/observaciones, número de eventos, número de parámetros candidatos, prevalencia y desempeño esperado; Riley et al. proponen criterios de sobreajuste, shrinkage, estimación del riesgo global y error absoluto, en vez de un único umbral de EPV — [Riley et al., *Statistics in Medicine*, 2019, DOI 10.1002/sim.7992](https://doi.org/10.1002/sim.7992).
- Las simulaciones de van Smeden et al. muestran que EPV por sí solo tiene una relación débil con el desempeño predictivo y que prevalencia, número total, distribución y fuerza de predictores y estrategia de modelado también importan; la selección hacia atrás empeoró la exactitud y la regularización ayudó — [van Smeden et al., *Statistical Methods in Medical Research*, 2019, DOI 10.1177/0962280218784726](https://doi.org/10.1177/0962280218784726).
- En datos agrupados, el tamaño efectivo es menor que el recuento nominal cuando las observaciones del mismo clúster son similares; una simulación de regresión logística multinivel recomendó al menos 10 EPV para modelos preespecificados y halló que podían necesitarse hasta 50 EPV con selección de variables — [Pavlou et al., *Journal of Clinical Epidemiology*, 2015, DOI 10.1016/j.jclinepi.2015.02.002](https://doi.org/10.1016/j.jclinepi.2015.02.002).
- TRIPOD-Cluster advierte que no existe guía formal sobre un número mínimo general de clústeres y que la similitud intra-clúster reduce el tamaño efectivo — [Debray et al., *BMJ*, 2023, TRIPOD-Cluster](https://pmc.ncbi.nlm.nih.gov/articles/PMC9903176/).
- Para validación externa binaria, reglas como 100 eventos y 100 no-eventos pueden no dar precisión suficiente, especialmente para calibración; el tamaño debe derivarse de anchos de intervalo deseados para O/E, pendiente de calibración, C-statistic y utilidad al umbral de decisión — [Riley et al., *Statistics in Medicine*, 2021, DOI 10.1002/sim.9025](https://doi.org/10.1002/sim.9025); [Snell et al., simulación de validación, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8352630/).

### Inferences
- Definir cuatro cantidades separadas: `event_rate_row` (positivos/fila-mes elegible), `n_positive_episodes` (rachas contiguas colapsadas según una regla de *washout*), `n_positive_entities` y `n_mature_positive_episodes`. La suficiencia se juzga por las tres últimas y por su dispersión temporal; la primera solo convierte un objetivo de episodios en volumen aproximado (`N ≈ E/prevalencia`).
- Usar un `episode_id` estable y una regla de colapso definida por negocio. Si seis meses consecutivos de una entidad describen un mismo caso, cuentan como un episodio para suficiencia y para intervalos por clúster, aunque puedan seguir siendo seis filas para entrenar una arquitectura secuencial.
- Registrar por corte: filas elegibles, entidades, episodios, positivos, positivos maduros, meses, entidades positivas, episodios por entidad, tasa de revisión de alertas y tasa de auditoría fuera de alertas. No sumar pseudo-etiquetas ni duplicados sintéticos al número de eventos reales.
- **Heurística operativa, no hecho universal:** no aprobar supervisión por alcanzar una prevalencia porcentual. Exigir a la vez volumen absoluto, independencia razonable, madurez, cobertura temporal y precisión de las métricas al presupuesto de alertas.

### Gaps
- No hay una fórmula publicada que convierta directamente filas `entity_id`-`codmes` correlacionadas en episodios independientes para esta problemática; el *washout* debe venir de la semántica del evento y validarse con expertos.
- No se conocen aún la prevalencia real, ICC/autocorrelación, número de entidades positivas, meses maduros ni parámetros candidatos del repositorio; sin ellos no se puede producir un cálculo formal de potencia/tamaño.

## ¿Cuántos eventos permiten explorar, pilotear o promover un modelo?

### Takeaway
La recomendación es una escalera de decisión basada en episodios positivos maduros e independientes, no un corte por tasa. Los números siguientes son **heurísticas conservadoras para planificación**, que deben reemplazarse por `pmsampsize`/simulación, curvas de aprendizaje e intervalos en cuanto existan los datos.

### Cited Findings
- Una referencia histórica de 10 EPV no debe tratarse como ley: los criterios contemporáneos calculan tamaño según prevalencia, parámetros y rendimiento anticipado, y las simulaciones muestran que el EPV aislado es insuficiente — [Riley et al., 2019](https://doi.org/10.1002/sim.7992); [van Smeden et al., 2019](https://doi.org/10.1177/0962280218784726).
- En un estudio empírico de conjuntos tabulares clínicos, los promedios requeridos para estabilidad fueron aproximadamente 11 eventos por variable para regresión logística, 205 para XGBoost, 231 para Random Forest y 342 para redes neuronales; sus autores enfatizan que son relaciones empíricas dependientes del dataset, no reglas universales — [Silvey & Liu, *JMIR*, 2024, DOI 10.2196/60231](https://doi.org/10.2196/60231).
- El mismo estudio encontró que las redes neuronales requirieron el mayor tamaño mediano (más de 12.000 observaciones) y fueron las más variables; el desequilibrio, fuerza/número de predictores y no linealidad cambiaron el tamaño requerido — [Silvey & Liu, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11688588/).
- El método de curvas de aprendizaje estima desempeño esperado frente al tamaño de entrenamiento y permite comparar estrategias; su extrapolación depende de que los datos iniciales sean representativos — [Dayimu et al., *Statistics in Medicine*, 2024, DOI 10.1002/sim.10121](https://doi.org/10.1002/sim.10121).

### Inferences
- **Matriz referencial propuesta (episodios positivos reales, maduros y colapsados):**

| Nivel | Condición indicativa | Uso permitido | No afirmar todavía |
|---|---:|---|---|
| 0. Sin base | `<30` episodios o `<10` entidades positivas | IF/VAE, descripción de errores, mejorar etiquetado | Comparación supervisada generalizable |
| 1. Exploración | `30–99` episodios, repartidos en varios meses/entidades | Regresión logística ridge/Firth de muy pocos grados de libertad; PU/semi-supervisado solo como experimento | Calibración fiable, selección amplia de variables o despliegue |
| 2. Piloto | `100–199` episodios y al menos `50` positivos *out-of-time* acumulados | Benchmark supervisado simple, boosting fuertemente restringido, curvas de aprendizaje | Validación confirmatoria si no hay unos `100` eventos OOS independientes |
| 3. Candidato | `≥200–400` episodios **y** desarrollo que cumple cálculo formal **y** `≥100–200` positivos OOS | Comparar ridge/hazard y boosting; calibrar y evaluar utilidad | Que árboles profundos o redes ya tengan datos suficientes |
| 4. Complejo | miles de episodios/entidades y curva de aprendizaje aún ascendente o estable, con múltiples regímenes temporales | Modelos temporales profundos/representaciones secuenciales como challenger | Superioridad por arquitectura sin prueba OOS |

- Los cortes `30/100/200` son **gates de gobernanza**, no resultados de una publicación. `100` se usa como alerta de que la validación sigue siendo imprecisa y `200` como punto desde el cual puede ser factible separar desarrollo y una validación con ~100 eventos; el cálculo de Riley puede exigir más.
- Traducción referencial a tasa: para reunir 200 episodios, una prevalencia de 0,1% exige aproximadamente 200.000 unidades independientes elegibles; 0,5%, 40.000; 1%, 20.000; 2%, 10.000; 5%, 4.000. En paneles, estas divisiones son solo límite optimista porque las filas no son independientes.
- Además del total, exigir como **heurística**: ningún fold usado para comparar modelos con menos de 20 episodios positivos; acumular al menos 50 positivos OOS para piloto y preferir 100–200 para conclusiones. Si un mes tiene menos de 20, reportarlo sin ranking y agrupar métricas OOS preespecificadas entre ventanas, conservando también el desglose mensual.
- Para regresión/hazard, contar grados de libertad reales: dummies, splines, interacciones y lags consumen parámetros. Usar shrinkage y preespecificación; no hacer selección exhaustiva con pocos eventos.

### Gaps
- La literatura no respalda un mínimo universal de positivos para XGBoost, Random Forest o redes profundas; los promedios empíricos de Silvey & Liu provienen de datos clínicos tabulares y no deben transferirse literalmente a este panel.
- `20` positivos por ventana y los niveles `30/100/200` son umbrales internos de seguridad propuestos, no puntos de corte validados externamente.

## ¿Qué familias de modelos son más plausibles para esta problemática?

### Takeaway
El orden recomendado es: baseline IF/VAE vigente; luego regresión penalizada o hazard discreto; después gradient boosting con complejidad acotada; semi-supervisión/PU como challenger cuando hay no revisados, nunca tratándolos automáticamente como negativos; y modelos temporales profundos solo con miles de episodios diversos y ganancia OOS demostrada.

### Cited Findings
- El aprendizaje semi-supervisado combina etiquetas y no etiquetados, pero depende de supuestos sobre suavidad, clústeres o distribución; violarlos puede degradar el rendimiento frente al baseline supervisado — [van Engelen & Hoos, *Machine Learning*, 2020, DOI 10.1007/s10994-019-05855-6](https://doi.org/10.1007/s10994-019-05855-6); [Li & Liang, *Frontiers of Computer Science*, 2019, DOI 10.1007/s11704-019-8452-2](https://doi.org/10.1007/s11704-019-8452-2).
- En aprendizaje positivo-no etiquetado (PU), los no etiquetados pueden contener positivos; aprender exige supuestos sobre el mecanismo de etiquetado o las distribuciones. El supuesto SCAR requiere que los positivos etiquetados sean una muestra uniforme de todos los positivos — [Bekker & Davis, *Machine Learning*, 2020, DOI 10.1007/s10994-020-05877-5](https://doi.org/10.1007/s10994-020-05877-5).
- La regresión de Firth reduce sesgo en eventos raros y resuelve separación, pero las probabilidades necesitan correcciones adicionales para mantener calibración cuando los eventos son raros — [Puhr et al., *Statistics in Medicine*, 2017, DOI 10.1002/sim.7273](https://doi.org/10.1002/sim.7273).
- En datos tabulares pequeños, gradient-boosted trees han sido baselines fuertes; la evidencia empírica citada arriba muestra mayor demanda de datos para RF/XGB/NN que para regresión, especialmente redes — [Silvey & Liu, 2024](https://doi.org/10.2196/60231); [Hollmann et al., *Nature*, 2025, DOI 10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6).

### Inferences
- **Primera opción supervisada:** regresión logística ridge/elastic net con lags predefinidos. Si el objetivo es `evento en los próximos h meses` y hay censura/madurez desigual, preferir hazard de tiempo discreto por entidad-mes. Firth queda como análisis de sensibilidad ante separación, no como solución automática de calibración.
- **Segunda opción:** LightGBM/XGBoost/CatBoost con profundidad baja, regularización, pesos y calibración OOS. Debe superar tanto a ridge/hazard como al ranking IF/VAE en `precision@K`, `recall@K`, calibración y estabilidad temporal.
- **Semi-supervisado recomendado:** usar embeddings/reconstruction de VAE como variables y labels humanos solo para una cabeza supervisada; o PU si `target=0` significa realmente “no revisado”. Mantener baseline supervisado con solo etiquetas confiables y exigir que el método semi-supervisado mejore OOS; no entrenar con pseudo-etiquetas como si fueran verdad.
- **Profundo temporal:** TCN/GRU/Transformer solamente como challenger con miles de episodios, suficientes entidades y regímenes, y aprendizaje incremental aún visible. Para panel mensual tabular y pocos positivos, lags + boosting/regresión son más auditables y plausibles.
- Mantener IF y VAE en paralelo para novedades no representadas por labels históricos y combinar mediante stacking/ranking solo después de una evaluación temporal sin fuga.

### Gaps
- No existe evidencia específica del repositorio que demuestre que la estructura latente del VAE cumple el supuesto de clúster requerido por semi-supervisión.
- La elección entre logística mensual y hazard depende de si `target` describe estado, inicio de episodio o evento dentro de un horizonte; esa definición aún debe fijarse.

## ¿Cómo debe evaluarse la suficiencia y el desempeño en el tiempo?

### Takeaway
La prueba debe imitar producción: cortes rolling-origin, features disponibles al corte, labels maduros y métricas al presupuesto real de revisión. Curvas PR y `precision/recall@K` son primarias; AUROC sola no basta, y toda cifra debe acompañarse de intervalos por bootstrap de entidad/episodio o simulación.

### Cited Findings
- Rolling-origin permite varios orígenes, distribuciones de error por horizonte y menor dependencia de un único periodo especial; varios periodos de prueba ayudan a cubrir fases distintas — [Tashman, *International Journal of Forecasting*, 2000, DOI 10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0).
- En series no estacionarias, evaluaciones fuera de muestra que preservan el orden temporal, repetidas en distintos periodos, estimaron mejor el rendimiento que particiones aleatorias — [Cerqueira et al., *Machine Learning*, 2020, DOI 10.1007/s10994-020-05910-7](https://doi.org/10.1007/s10994-020-05910-7).
- En clases desbalanceadas, la curva precision-recall informa mejor sobre recuperación de la clase positiva que ROC porque TN masivos pueden dominar la interpretación — [Saito & Rehmsmeier, *PLOS ONE*, 2015, DOI 10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).
- La validación debe cuantificar calibración, discriminación y utilidad/beneficio al umbral de decisión, con tamaño definido por la precisión deseada de esos estimadores — [Riley et al., 2021](https://doi.org/10.1002/sim.9025).

### Inferences
- Folds mensuales o trimestrales rolling-origin: entrenar solo hasta `t`, predecir `t+h`, esperar madurez del label y evaluar. Aplicar gap/purga igual al mayor horizonte de feature/label cuando ventanas se solapen.
- No dividir aleatoriamente filas. Bootstrapear por `entity_id` y, para sensibilidad, por bloques temporales; episodios repetidos de una entidad deben quedarse juntos dentro de cada remuestreo.
- Reportar por fold y agregado: prevalencia, episodios positivos, PR-AUC con baseline de prevalencia, `precision@K`, `recall@K`, lift@K, falsos positivos por 1.000, Brier/log-loss, calibración-in-the-large y pendiente. Definir K como la capacidad mensual real de revisión.
- Criterio de piloto: comparar IF, VAE, ridge/hazard, boosting y semi-supervisado con exactamente los mismos cortes y K. Promover solo si la mejora supera un margen predefinido y su intervalo por clúster no indica degradación material; revisar estabilidad por mes, segmento y novedad.
- Construir curvas de aprendizaje temporales por número de episodios/meses, no solo por filas: 25%, 50%, 75%, 100% del historial, conservando orden. Si la curva y los intervalos no se estabilizan, adquirir labels antes de aumentar complejidad.
- Para cada fold, calcular de antemano el ancho de IC aceptable para `precision@K`/sensibilidad. Con pocos positivos, reportar incertidumbre exacta/bootstrap y abstenerse de ordenar modelos; no “arreglar” un test pobre mediante oversampling.

### Gaps
- No existe un mínimo publicado universal de positivos por ventana para `precision@K`; depende de K, prevalencia, precisión esperada y ancho de IC deseado.
- Falta el presupuesto mensual de revisión K y el coste relativo FP/FN, necesarios para definir el punto operativo y hacer potencia sobre la métrica que importa.

## ¿Qué controles de labels y confiabilidad de fuentes deben exigirse?

### Takeaway
Antes de cambiar de paradigma, hay que probar que los labels no son selectivos, inmaduros ni confundidos con ausencia de revisión. La carpeta de labels debe permitir reconstruir qué se revisó, cuándo, por quién y bajo qué política; además se necesita una muestra aleatoria de no-alertas.

### Cited Findings
- El problema de etiquetas selectivas aparece cuando los outcomes observados dependen de decisiones previas: los casos etiquetados dejan de representar a la población y la evaluación del modelo se sesga — [Lakkaraju et al., KDD 2017, DOI 10.1145/3097983.3098066](https://doi.org/10.1145/3097983.3098066).
- PU learning requiere conocer o asumir cómo se seleccionan positivos; tratar todo no etiquetado como negativo es una suposición ingenua que el survey excluye como base fiable — [Bekker & Davis, 2020](https://doi.org/10.1007/s10994-020-05877-5).
- La validación temporal usa periodos no solapados y debe respetar el momento real de disponibilidad de datos; evaluar solo en un periodo elegido por conveniencia puede dar una imagen histórica o contemporánea incompleta — [Riley et al., *BMJ*, 2024, DOI 10.1136/bmj-2023-074819](https://doi.org/10.1136/bmj-2023-074819).

### Inferences
- Campos mínimos adicionales al trío `entity_id`, `codmes`, `target`: `episode_id`, `target_definition_version`, `label_status` (`positive/negative/uncertain/unreviewed`), `reviewed_at`, `label_available_at`, `maturity_date`, `reviewer_id_hash`, `review_policy`, `review_reason`, `source_model`, `source_score`, `alert_rank`, `sampling_probability`, `adjudication_status`, `evidence_ref`, `label_confidence`, `valid_from`, `supersedes_label_id`.
- Nunca codificar `unreviewed` como `0`. Separar negativo confirmado, incierto y no revisado. Entrenar supervisado solo con labels maduros; para PU conservar el mecanismo/probabilidad de selección.
- Reservar cada mes una fracción aleatoria de la capacidad para revisar no-alertas, estratificada por score/segmento, y guardar `sampling_probability`. Esto permite estimar falsos negativos y corregir sesgo de verificación; el porcentaje se define por potencia y presupuesto, no por costumbre.
- Congelar snapshots `as_of_date`: ninguna feature o label posterior al corte puede entrar a entrenamiento/evaluación. La madurez debe ser horizonte objetivo + retraso operativo observado, con buffer preespecificado.
- Medir acuerdo entre revisores en una muestra duplicada y adjudicar desacuerdos antes de que labels dudosos pasen a `gold`.

### Gaps
- No está documentado el retraso entre ocurrencia, revisión y confirmación del evento; no se puede fijar aún el buffer de madurez.
- No se conoce si hoy solo se revisan alertas de IF/VAE; si es así, una comparación supervisada sin auditoría aleatoria de no-alertas estaría afectada por etiquetas selectivas.

### Evaluación de confiabilidad de las fuentes usadas

| Fuente | Estado | Afirmación que respalda | Limitación de transferencia | Confianza |
|---|---|---|---|---|
| Riley et al. 2019, *Statistics in Medicine* | Artículo metodológico revisado por pares | Cálculo de muestra para desarrollo binario/supervivencia, más allá de EPV | Diseñado para modelos de predicción; requiere supuestos de R²/prevalencia | Alta |
| van Smeden et al. 2019, *Statistical Methods in Medical Research* | Simulación revisada por pares | EPV aislado es insuficiente; efecto de shrinkage/selección | Regresión logística, no redes temporales | Alta |
| Riley et al. 2021 / Snell et al. 2021 | Metodología y simulación revisadas por pares | Muestra de validación según precisión; 100/100 puede fallar | Marco clínico; debe adaptar utilidad a alertas | Alta |
| Pavlou et al. 2015 | Simulación revisada por pares | Clustering y EPV; riesgo de selección de variables | Clústeres tipo centros, no exactamente entidad-tiempo | Media-alta |
| Debray et al. 2023, TRIPOD-Cluster | Guía metodológica revisada por pares | Tamaño efectivo y reporte con clústeres | Orientación de reporte, no corte numérico | Alta |
| Silvey & Liu 2024, *JMIR* | Estudio empírico revisado por pares | Diferencia de demanda de datos LR/XGB/RF/NN | Datasets clínicos; promedios no universales ni temporales | Media |
| Dayimu et al. 2024 | Metodología revisada por pares | Curvas de aprendizaje para planificar muestra | Extrapolación sensible a representatividad/modelo de curva | Media-alta |
| Tashman 2000 | Revisión metodológica revisada por pares | Rolling-origin y múltiples periodos | Forecasting clásico, no clasificación rara directamente | Alta |
| Cerqueira et al. 2020 | Estudio empírico revisado por pares | Orden temporal y no estacionariedad | Forecasting; datasets distintos de panel de alertas | Media-alta |
| Saito & Rehmsmeier 2015 | Artículo metodológico revisado por pares | PR frente a ROC en desbalance | No define K, costes ni tamaño muestral | Alta |
| van Engelen & Hoos 2020 / Li & Liang 2019 | Surveys revisados por pares | Supuestos y riesgos de semi-supervisión | Evidencia amplia, no específica del dominio | Alta para riesgo; media para elección concreta |
| Bekker & Davis 2020 | Survey revisado por pares | Supuestos PU/SCAR y no etiquetado ≠ negativo | Requiere verificar mecanismo en este flujo | Alta |
| Puhr et al. 2017 | Simulación/metodología revisada por pares | Firth en eventos raros y calibración | Regresión, no boosting/deep | Alta |
| Lakkaraju et al. 2017, KDD | Conferencia selectiva revisada por pares | Problema de etiquetas selectivas | Dominio original distinto; mecanismo causal análogo | Alta |
| Riley et al. 2024, *BMJ* | Guía metodológica revisada por pares | Validación temporal y ciclo desarrollo-validación | Marco clínico | Alta |
| Hollmann et al. 2025, *Nature* | Artículo revisado por pares | Fuerza de métodos tabulares en muestra pequeña | TabPFN y benchmarks no temporales; no fija umbral | Media |

