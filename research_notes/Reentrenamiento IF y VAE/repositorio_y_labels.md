# Arquitectura del repositorio y contrato de labels para reentrenamiento IF–VAE

## ¿Dónde deben vivir las etiquetas revisadas?

### Takeaway
Las revisiones humanas deben planificarse como **datos fuente operacionales, durables y no reproducibles**, no como un artefacto de una corrida. Propongo `data/reviewed_labels/` en la raíz, con eventos append-only particionados y una vista materializada, y no `artifacts/data/`: hoy `artifacts/` se define como estado generado íntegramente inspeccionable o borrable y además Git lo ignora por completo.

### Cited Findings
- `src/utils/paths.py` es la fuente única de rutas y separa `artifacts/{data,logs,models,tuning,reports}`; su propio contrato dice que todo el estado generado puede borrarse en conjunto. — [Fuente local primaria: `src/utils/paths.py`, líneas 1–31 y 55–71](../../src/utils/paths.py#L1-L71)
- `.gitignore` excluye todo `artifacts/` porque se considera reproducible desde `python main.py`; una revisión humana no es reproducible y perderla impediría reconstruir el historial decisional. — [Fuente local primaria: `.gitignore`, líneas 1–5](../../.gitignore#L1-L5)
- El panel actual entra por `artifacts/data/data.csv` y el ground truth sintético por `artifacts/data/ground_truth.parquet`, pero este último es generado y oculto para evaluación; no equivale a feedback humano persistente. — [Fuente local primaria: `src/utils/paths.py`, líneas 73–76](../../src/utils/paths.py#L73-L76); [Fuente local primaria: `src/data/loader.py`, líneas 1–19](../../src/data/loader.py#L1-L19)
- W3C PROV modela procedencia mediante Entidades, Actividades y Agentes y relaciones como derivación, generación, uso y atribución; respalda conservar quién produjo una etiqueta y de qué evidencia derivó. Es una Recomendación normativa de interoperabilidad, no evidencia empírica de mejora predictiva. — [W3C PROV-O, Recomendación oficial, 2013](https://www.w3.org/TR/prov-o/)
- NIST AI RMF indica que mantener procedencia de datos de entrenamiento ayuda a transparencia y rendición de cuentas; exige gobernanza, monitoreo periódico, roles y documentación de supervisión humana. Es guía gubernamental voluntaria, no un requisito legal ni un experimento. — [NIST AI 100-1, estándar/guía oficial, DOI 10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1); [NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)

### Inferences
- Estructura futura propuesta, **sin crearla todavía**:
  - `data/reviewed_labels/README.md`: finalidad, taxonomía, responsables, retención y procedimiento de adjudicación.
  - `data/reviewed_labels/schema/reviewed_label.schema.json`: contrato versionado.
  - `data/reviewed_labels/events/year=YYYY/month=MM/*.parquet`: bitácora append-only de decisiones.
  - `data/reviewed_labels/current/reviewed_labels.parquet`: vista efectiva, recreable desde `events/`, una fila vigente por `(entity_id, codmes)`.
  - `data/reviewed_labels/manifests/*.json`: snapshots congelados que una corrida de reentrenamiento consumió, con hashes y corte temporal.
- Se debe ignorar en Git **solo el contenido sensible** de `events/`, `current/` y `manifests/`, pero versionar README/esquema/fixtures anonimizados. La ubicación real debe poder sobrescribirse por variable/CLI para producción; el default relativo debe añadirse a `src/utils/paths.py` para conservar la convención de rutas.
- Si la política organizacional obliga a que toda entrada viva fuera del repositorio, la misma interfaz debe aceptar un URI externo; el repositorio conservaría esquema y manifiestos sin PII. Esto no cambia el contrato lógico.

### Gaps
- No existe política local de retención, clasificación de datos, cifrado, control de acceso ni identificador de responsable; deben definirse con Seguridad/Legal antes de persistir comentarios libres o identidades de revisores.
- No hay hoy un directorio de datos fuente durable fuera de `artifacts/`; la propuesta introduce deliberadamente una nueva categoría y requiere acordarla en la decisión de arquitectura.

## ¿Cuál es el contrato real actual de entidad, tiempo, target, particiones, scores y reentrenamiento?

### Takeaway
El contrato actual es un panel por `(entity_id, period)` —en datos reales el tiempo puede inferirse como `codmes`—, etiquetas internas binarias `is_anomaly`, y cuatro bloques cronológicos `train | val | test | OOT`. IF y VAE siguen siendo detectores no supervisados: con `--supervised`, las etiquetas influyen en la selección de hiperparámetros y métricas, pero el `fit` final de ambos no recibe `y`.

### Cited Findings
- El loader infiere tiempo por nombres que contienen `period/date/time` y entidad por `entity/individual/id`; **`codmes` no está en esos hints**, por lo que solo podría detectarse mediante fallback estructural. El target inline se infiere por `target/ground_truth/groundtruth`. — [Fuente local primaria: `src/data/loader.py`, líneas 37–43 y 81–127](../../src/data/loader.py#L37-L127)
- `codmes` compacto `YYYYMM` sí tiene parser explícito, pero esto ocurre después de que se haya identificado la columna temporal; conviene configurar/mapejar el nombre y no confiar en inferencia. — [Fuente local primaria: `src/data/loader.py`, líneas 131–203](../../src/data/loader.py#L131-L203)
- El ground truth separado exige columnas con los nombres de entidad y tiempo inferidos más `is_anomaly`; normaliza etiquetas a 0/1 y actualmente deduplica por clave conservando silenciosamente la primera ocurrencia. Las filas no coincidentes se rellenan como 0. — [Fuente local primaria: `src/evaluation/labels.py`, líneas 32–35 y 81–140](../../src/evaluation/labels.py#L32-L140); [Fuente local primaria: `src/evaluation/labels.py`, líneas 143–176](../../src/evaluation/labels.py#L143-L176)
- En sintético, el contrato de referencia contiene `entity_id`, `period`, `is_anomaly`, `anomaly_type`; los tipos son `global`, `local`, `contextual`, `collective` y `none`. — [Fuente local primaria: `src/data/synthetic.py`, líneas 735–760 y 923–929](../../src/data/synthetic.py#L735-L760)
- `chronological_split` reserva bloques estrictamente ordenados: train ajusta transformaciones/modelos; validación selecciona hiperparámetros y calibra umbral; test se usa una vez para reporte; OOT queda después y alimenta el entregable de negocio. — [Fuente local primaria: `src/evaluation/splits.py`, líneas 100–145](../../src/evaluation/splits.py#L100-L145); [Fuente local primaria: `main.py`, líneas 573–613](../../main.py#L573-L613)
- La configuración actual usa por defecto 2 periodos de validación, 3 de test y 3 de OOT; la estrategia supervisada está apagada por defecto y requiere opt-in explícito. — [Fuente local primaria: `main.py`, líneas 64–84 y 124–156](../../main.py#L64-L156)
- El preprocesador trata entidad/tiempo como llaves, no como features, y ajusta las transformaciones estimadas solo a la ventana de train. — [Fuente local primaria: `src/preprocessing/pipeline.py`, líneas 888–948](../../src/preprocessing/pipeline.py#L888-L948)
- IF recibe `y` solo en `tune_iforest` cuando la corrida es supervisada; el ganador se reajusta sobre todo `X` mediante `fit(X)` y se persiste. VAE sigue el mismo patrón: `y` entra al tuner, mientras el detector ganador ejecuta `fit(X, valid_mask=...)`. — [Fuente local primaria: `main.py`, líneas 803–843](../../main.py#L803-L843); [Fuente local primaria: `src/models/iforest.py`, líneas 869–883](../../src/models/iforest.py#L869-L883); [Fuente local primaria: `main.py`, líneas 1013–1053](../../main.py#L1013-L1053); [Fuente local primaria: `src/models/vae.py`, líneas 1480–1497](../../src/models/vae.py#L1480-L1497)
- Los scores siguen la convención “mayor = más anómalo”; los umbrales se calibran en validación y el export OOT conserva por entidad el mes de máximo score, añadiendo banda percentil y opcionalmente `alert`. — [Fuente local primaria: `src/evaluation/oot_report.py`, líneas 156–213 y 217–305](../../src/evaluation/oot_report.py#L156-L305)
- El dashboard ya captura estados revisados en el navegador y descarga `casos_revisados.csv`, pero solo exporta entidad, identidad, estado y timestamps; no incluye `codmes`, target binario, modelo, score, umbral ni evidencia, y no tiene ingestión persistente. — [Fuente local primaria: `src/reporting/analyst_dashboard.py`, líneas 626–658](../../src/reporting/analyst_dashboard.py#L626-L658)
- Los YAML de tuning registran estudio, dirección, valor/trial ganador, modo objetivo, seed y parámetros; el de IF incluye contaminación/holdout y el de VAE epochs, pero no un hash de dataset/labels ni versión del contrato de revisión. — [Fuente local primaria: `src/models/iforest.py`, líneas 801–827](../../src/models/iforest.py#L801-L827); [Fuente local primaria: `src/models/vae.py`, líneas 1415–1438](../../src/models/vae.py#L1415-L1438)
- Isolation Forest original es no supervisado y aísla anomalías mediante longitudes cortas de ruta en árboles aleatorios; el VAE original maximiza una cota variacional sin una etiqueta de clase. — [Liu, Ting y Zhou, ICDM 2008, peer-reviewed, DOI 10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17); [Kingma y Welling, ICLR 2014, peer-reviewed](https://iclr.cc/archive/2014/conference-proceedings/)

### Inferences
- Debe existir un mapeo explícito configurable `review_entity_col=entity_id`, `review_time_col=codmes`, `review_target_col=target`; el loader de feedback normalizará internamente a las llaves del panel y a `is_anomaly`, sin renombrar físicamente el archivo del usuario.
- “Reentrenar con labels” tiene tres niveles que el plan debe separar:
  1. **Nivel A (compatible hoy):** labels solo para evaluación temporal, selección de hiperparámetros y/o calibración; detector final continúa no supervisado.
  2. **Nivel B (feedback directo IF):** adoptar/implementar una variante active anomaly detection que actualiza el ranking del bosque; es cambio algorítmico, no simple carga de CSV.
  3. **Nivel C (VAE semisupervisado):** añadir cabeza/clase u objetivo conjunto, con ablation contra VAE puramente no supervisado; también es cambio de arquitectura/función de pérdida.
- El primer release debe quedarse en Nivel A para reducir riesgo; niveles B/C requieren experimentos separados, aprobación y versionado mayor.

### Gaps
- No existe hoy `run_id`/`model_version` canónico compartido por modelos, reportes y dashboard; algunos artefactos usan nombres de archivo y otros el contexto de observabilidad.
- No existe un loader para `target` revisado ni semántica formal de `target=0`: podría significar “revisado normal” o “no se encontró evidencia”; esa definición de negocio debe cerrarse antes de entrenar.

## ¿Qué campos mínimos y recomendados deben añadirse?

### Takeaway
`entity_id`, `codmes` y `target` son necesarios pero insuficientes: sin procedencia, vigencia temporal, versión de política, evidencia del modelo y control de conflictos no puede reproducirse qué sabía una corrida ni distinguir una corrección humana de un negativo asumido.

### Cited Findings
- W3C PROV requiere poder relacionar la entidad (label/dataset), actividad (revisión/adjudicación) y agente (revisor), base conceptual de `review_event_id`, `reviewer_id`, timestamps y derivación. — [W3C PROV-O, Recomendación oficial](https://www.w3.org/TR/prov-o/)
- *Datasheets for Datasets* propone documentar motivación, composición, proceso de recolección y usos recomendados del dataset. Es artículo revisado/publicado en Communications of the ACM; aporta transparencia documental, no demuestra causalmente una ganancia de accuracy. — [Gebru et al., CACM 2021, peer-reviewed/editorial, DOI 10.1145/3458723](https://doi.org/10.1145/3458723)
- NIST AI RMF vincula trazabilidad de datos, documentación de roles, revisión humana y monitoreo con transparencia/accountability. Es marco voluntario y agnóstico de sector. — [NIST AI 100-1, guía oficial](https://doi.org/10.6028/NIST.AI.100-1)
- El trabajo clásico de Dawid–Skene modela tasas de error distintas por observador cuando no existe verdad directamente observable, mostrando por qué conservar identidad del revisor y votos individuales es superior a sobrescribirlos con un consenso opaco. Es artículo peer-reviewed; sus supuestos de clase latente e independencia no deben darse por válidos sin evaluación local. — [Dawid y Skene, JRSS C 1979, DOI 10.2307/2346806](https://doi.org/10.2307/2346806)
- La literatura contemporánea documenta que el desacuerdo puede provenir de ambigüedad, solapamiento de clases, subjetividad o error; por tanto, conflicto no equivale automáticamente a etiqueta mala. — [Uma, Almanea y Poesio, Frontiers in AI 2022, peer-reviewed, DOI 10.3389/frai.2022.818451](https://doi.org/10.3389/frai.2022.818451)
- ISO/IEC 5259-2:2024 define un modelo y medidas de calidad de datos para analítica/ML. Es estándar internacional reconocido, pero el texto completo es de pago y la página pública solo permite validar alcance, no controles específicos. — [ISO, ficha oficial ISO/IEC 5259-2:2024](https://www.iso.org/standard/81860.html)

### Inferences
- **Campos mínimos obligatorios para admitir una decisión como label vigente:**

| Campo | Tipo/regla | Propósito |
|---|---|---|
| `review_event_id` | UUID/string, único e inmutable | idempotencia y auditoría |
| `entity_id` | string no vacío | llave de entidad, conservar ceros a la izquierda |
| `codmes` | string canónico `YYYYMM`, válido | tiempo del hecho; convertir a datetime solo al unir |
| `target` | int8 `{0,1}` | `1=anomalía confirmada`, `0=normal confirmado`; nunca usar null como 0 |
| `review_status` | enum `reviewed`, `conflict`, `superseded`, `withdrawn` | distingue label utilizable de evento pendiente/conflictivo |
| `reviewed_at` | timestamp UTC con zona | cuándo se emitió la decisión |
| `label_available_at` | timestamp UTC con zona | desde cuándo podía usarla el pipeline; control point-in-time |
| `reviewer_id` | identificador seudonimizado | responsabilidad, calidad/acuerdo, sin PII directa |
| `label_policy_version` | semver/string | rúbrica que define qué significa target |
| `reason_code` | enum versionado | razón estructurada; evita depender de comentario libre |
| `source_run_id` | string | corrida que presentó el caso al analista |
| `source_model` | enum/lista (`iforest`,`vae`,`stacked`) | origen de selección |
| `source_model_version` | hash/versión | reproduce el ranking mostrado |
| `feature_snapshot_hash` | SHA-256 del registro as-of | prueba de identidad del ejemplo revisado |
| `schema_version` | semver/string | evolución del contrato |

- **Campos recomendados:** `reviewer_role`, `review_channel`, `confidence` (ordinal, nunca peso automático hasta calibrarlo), `adjudicator_id`, `adjudicated_at`, `supersedes_event_id`, `case_id`, `score_if`, `score_vae`, `threshold_if`, `threshold_vae`, `percentile_if`, `percentile_vae`, `selection_policy`, `source_split`, `source_dataset_hash`, `evidence_refs` (sin copiar PII), `comment_redacted`, `created_at`, `ingested_at`.
- `source_split` describe de dónde surgió el caso; **no** debe usarse como partición futura. La partición efectiva (`train/val/test/locked_oot`) debe vivir en el manifiesto inmutable de cada retraining, calculada por `codmes` y corte temporal.
- Para múltiples revisores se preserva un evento por voto; una adjudicación crea otro evento que referencia a los anteriores. Nunca se destruyen los votos ni se actualiza una fila in place.

### Gaps
- Faltan taxonomía y rúbrica de `reason_code` (fraude confirmado, falso positivo por estacionalidad, error de datos, caso inconcluso, etc.).
- Faltan base jurídica y minimización requerida para almacenar identificadores, comentarios y evidencia. Hasta definirlas, `comment_redacted` debe ser opcional y quedar fuera del dataset de entrenamiento.

## ¿Qué controles de unicidad, inmutabilidad, validación, privacidad, conflictos y point-in-time deben planificarse?

### Takeaway
El dataset debe ser un log de eventos inmutable más una vista efectiva derivada, con validación fail-closed. No se debe conservar “la primera” etiqueta duplicada ni imputar un caso sin match como negativo: ambas conductas actuales son aceptables para ground truth sintético auxiliar, pero peligrosas para feedback humano.

### Cited Findings
- El loader actual aplica `drop_duplicates(..., keep="first")` y rellena labels no emparejados con 0; esto ocultaría conflictos o faltantes si se reutilizara sin cambios para revisión humana. — [Fuente local primaria: `src/evaluation/labels.py`, líneas 113–139 y 164–175](../../src/evaluation/labels.py#L113-L175)
- El validador de panel ya trata llaves duplicadas `(entity,time)` y periodos no parseables como bloqueantes antes de split/preprocesamiento; el contrato de labels debe tener igual severidad. — [Fuente local primaria: `main.py`, líneas 560–571](../../main.py#L560-L571)
- El pipeline ya verifica que test y OOT no se solapen y que preprocessing solo aprenda de train; reutilizar esa separación evita que labels futuros informen decisiones pasadas. — [Fuente local primaria: `main.py`, líneas 573–613](../../main.py#L573-L613)
- Kaufman, Rosset y Perlich definen leakage como introducir información sobre el target que no estaría legítimamente disponible al predecir y recomiendan separación learn–predict. Es trabajo peer-reviewed KDD; su marco es general y no prescribe este esquema exacto. — [Kaufman et al., KDD 2011, DOI 10.1145/2020408.2020496](https://doi.org/10.1145/2020408.2020496)
- NIST AI RMF exige documentar revisión, roles, desempeño y riesgos a lo largo del ciclo de vida; W3C PROV soporta derivación/atribución de cada versión. — [NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/); [W3C PROV-O](https://www.w3.org/TR/prov-o/)

### Inferences
- **Unicidad:** `review_event_id` único global; el estado efectivo admite a lo sumo una adjudicación utilizable por `(entity_id,codmes,label_policy_version)`. Dos targets activos distintos bloquean el entrenamiento.
- **Inmutabilidad:** append-only; correcciones mediante `supersedes_event_id`. Manifest con hash SHA-256 de cada fichero, conteos, esquema, rango temporal, query de selección y commit de código.
- **Validación fail-closed:** tipos, `{0,1}`, formato/calendario de `codmes`, timestamps UTC, enums, claves presentes en el snapshot del panel, hashes válidos, no-null obligatorio, no duplicados, y ausencia de labels `conflict/withdrawn/pending` en entrenamiento. Rechazos a cuarentena con código de error; no coerción silenciosa.
- **Punto en el tiempo:** una corrida con `training_as_of=T` solo consume eventos con `label_available_at <= T` y features cuyo `feature_event_time <= codmes_end` y `feature_ingested_at <= T`. El OOT bloqueado nunca entra en fit/tuning/threshold; una vez reveladas sus etiquetas deja de ser el OOT virgen para la siguiente evaluación y debe rodarse el horizonte.
- **Deduplicación/conflicto:** repeticiones byte-idénticas se consideran retries idempotentes; mismas claves con targets distintos se marcan `conflict`; adjudicador independiente emite resolución. Reportar tasa de conflicto y acuerdo, estratificados por reason/reviewer/periodo.
- **Privacidad/acceso:** seudonimizar `reviewer_id`, cifrar en reposo y tránsito en almacenamiento real, mínimo privilegio (analista escribe eventos; pipeline lee labels aprobados; administrador adjudica), logs de acceso, retención/rectificación, y prohibición de texto libre sensible en features.
- **Sesgo de selección:** los casos revisados provienen mayormente de la cola P90/P95/P99; por ello no representan la prevalencia poblacional. Conservar `selection_policy`, scores y percentiles permite ponderar/diagnosticar, y se necesita una muestra aleatoria de control para estimar falsos negativos.

### Gaps
- No se encontró en el repositorio autenticación/autorización ni backend del dashboard: su estado parece cliente-local. La solución productiva requiere un sistema de escritura autenticado que está fuera del código actual.
- No hay decisión sobre si una entidad puede tener múltiples anomalías en el mismo `codmes` ni target multiclase; si el negocio necesita tipos, debe añadirse `target_type` separado del binario, con taxonomía versionada.

## ¿Qué debe crearse o cambiarse después, archivo por archivo, sin implementarlo ahora?

### Takeaway
El trabajo debe dividirse en contrato/gobernanza, ingestión y validación, snapshot temporal, integración de entrenamiento, evaluación bloqueada y operación. No se debe activar retraining hasta que los gates de calidad y leakage pasen y exista un baseline congelado.

### Cited Findings
- El proyecto centraliza rutas en `src/utils/paths.py`, configuración/orquestación en `main.py`, carga de labels en `src/evaluation/labels.py`, splits en `src/evaluation/splits.py`, validaciones en `src/utils/assumptions.py`, y documentación de corrida en reporting. — [Fuente local primaria: `src/utils/paths.py`](../../src/utils/paths.py); [Fuente local primaria: `main.py`](../../main.py); [Fuente local primaria: `src/evaluation/labels.py`](../../src/evaluation/labels.py); [Fuente local primaria: `src/evaluation/splits.py`](../../src/evaluation/splits.py); [Fuente local primaria: `src/utils/assumptions.py`](../../src/utils/assumptions.py); [Fuente local primaria: `src/reporting/report.py`](../../src/reporting/report.py)
- El feedback binario puede mejorar ranking de detectores basados en árboles en un loop analista–modelo, pero la evidencia IF-AAD hallada es workshop/arXiv y no una validación de este panel; debe tratarse como hipótesis experimental, no como garantía. — [Das et al., IDEA/KDD workshop 2017, no peer review de journal, arXiv 1708.09441](https://arxiv.org/abs/1708.09441)
- Los modelos generativos profundos semisupervisados son una arquitectura explícita diferente que combina datos etiquetados y no etiquetados; no basta con pasar `y` a un VAE estándar. — [Kingma et al., NeurIPS 2014, peer-reviewed](https://papers.nips.cc/paper_files/paper/2014/hash/6d42b1217a6996997ead5a8398c1f944-Abstract.html)
- Model Cards recomienda documentar uso previsto, datos, procedimientos de evaluación, desempeño por condiciones y limitaciones. Es paper peer-reviewed FAT* 2019; es marco de reporte, no control técnico suficiente por sí solo. — [Mitchell et al., FAT* 2019, DOI 10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596)

### Inferences
- **Fase 0 — decisión y rúbrica (antes de código):** aprobar semántica exacta de `target`, taxonomía de razones, quién puede revisar/adjudicar, SLA, retención, privacidad, corte point-in-time y criterio de rollback.
- **Fase 1 — contrato y almacenamiento:**
  - crear `data/reviewed_labels/{README.md,schema/,events/,current/,manifests/}`;
  - modificar `.gitignore` para excluir eventos/manifiestos sensibles, no esquema/documentación;
  - añadir en `src/utils/paths.py` `REVIEWED_LABELS_DIR`, `REVIEWED_LABELS_CURRENT`, `REVIEWED_LABELS_MANIFEST_DIR` y rutas de cuarentena;
  - añadir `configs/reviewed_labels.yaml` (nuevo) para nombres de columnas, enums, cutoff, política de conflictos y URI override.
- **Fase 2 — ingestión/QA:**
  - crear `src/data/reviewed_labels.py`: lectura CSV/Parquet, normalización `codmes`, validación de esquema, append idempotente, vista efectiva y manifiesto;
  - extender `src/utils/assumptions.py` con `validate_reviewed_labels` y controles de llave/panel/point-in-time;
  - no reutilizar `_join_ground_truth` sin separar su semántica tolerante; extender `src/evaluation/labels.py` con un join estricto o módulo independiente;
  - tests nuevos: `tests/test_reviewed_label_contract.py`, `tests/test_reviewed_label_conflicts.py`, `tests/test_label_point_in_time.py`, `tests/test_label_manifest.py`.
- **Fase 3 — captura:** extender `src/reporting/analyst_dashboard.py` para exportar/capturar `entity_id`, **mes concreto**, target, reason, model/version, scores/thresholds, review timestamps y event id. Un CSV descargado puede ser MVP, pero producción requiere API autenticada y backend append-only.
- **Fase 4 — integración Nivel A:**
  - extender `PipelineConfig`/CLI en `main.py` con `reviewed_labels_path`, `labels_as_of`, `label_policy_version`, `minimum_label_count`, `use_reviewed_labels_for={evaluation,tuning}`, nunca autoactivar supervised por mera presencia de archivo;
  - construir snapshot antes del split, pero asignar partición por `codmes`; impedir cualquier label posterior a cutoff;
  - anexar a tuning YAMLs de `src/models/iforest.py` y `src/models/vae.py`: `labels_manifest_hash`, `dataset_hash`, `code_commit`, periodos y conteos por target/reason, sin PII;
  - extender `src/reporting/report.py` y documentación de modelo con lineage, etiqueta policy, cobertura, desacuerdo, métricas temporales, limitaciones y comparación baseline.
- **Fase 5 — evaluación:** conservar test y OOT cerrados; reportar PR-AUC (prioritaria por rareza), ROC-AUC, precision@K/recall@K, workload/false-alert rate, calibración y métricas por periodo/segmento/reason. Comparar: baseline sin labels, Nivel A con labels, y solo después Nivel B IF-AAD / Nivel C semi-supervised VAE.
- **Fase 6 — gates de promoción:** volumen mínimo positivo/negativo, tasa de conflicto, cobertura temporal, ausencia de leakage, estabilidad por seeds, mejora con intervalo de confianza, no degradación material por segmento, reproducibilidad por hash y rollback probado. Si no se cumplen, mantener modelo anterior.
- **Definición de terminado:** una corrida puede reconstruirse exactamente desde commit + dataset snapshot hash + label manifest hash + config; el reporte muestra qué labels entraron y cuáles fueron excluidos; OOT permanece temporalmente posterior; y un auditor puede seguir cada target hasta la revisión/adjudicación que lo generó.

### Gaps
- Deben fijarse umbrales numéricos de promoción según costo de falsos positivos/falsos negativos; la literatura general no puede decidirlos por el negocio.
- No hay CI visible para tests de labels ni almacenamiento de secretos; el plan debe adaptarse al entorno de despliegue elegido.

## ¿Qué fuentes son confiables y cuáles son sus límites?

### Takeaway
La base recomendada combina código/tests locales como evidencia primaria de “qué hace hoy el sistema”, papers originales peer-reviewed para mecanismos y riesgos, y normas oficiales para gobernanza. Ninguna fuente aislada prueba que reentrenar este IF/VAE con estos labels mejorará producción; eso requiere evaluación prospectiva temporal propia.

### Cited Findings
- **Código y tests del repositorio — fuente primaria local, máxima relevancia arquitectónica, sin peer review externo.** Confirman contratos ejecutables actuales, pero pueden contener bugs y no demuestran validez científica. — [Repositorio local: `main.py`](../../main.py); [tests locales](../../tests)
- **Isolation Forest — paper original peer-reviewed (ICDM 2008), alta relevancia al IF; limitación: no estudia feedback humano ni este dominio.** — [DOI 10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)
- **Auto-Encoding Variational Bayes — paper original aceptado ICLR 2014, alta relevancia al VAE; limitación: objetivo generativo general, no protocolo de anomaly feedback.** — [Proceedings ICLR 2014](https://iclr.cc/archive/2014/conference-proceedings/); [preprint original](https://arxiv.org/abs/1312.6114)
- **Semi-supervised deep generative models — NeurIPS 2014 peer-reviewed, evidencia de factibilidad arquitectónica; limitación: benchmarks de clasificación, no valida la implementación propuesta ni anomalía bancaria.** — [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2014/hash/6d42b1217a6996997ead5a8398c1f944-Abstract.html)
- **IF-AAD — workshop/preprint, evidencia empírica directamente relevante al feedback sobre árboles; limitación fuerte: no es journal/main-conference paper y no debe elevarse a recomendación productiva sin réplica local.** — [arXiv 1708.09441](https://arxiv.org/abs/1708.09441); [paper del workshop IDEA’17](https://poloclub.gatech.edu/idea2017/papers/p25-das.pdf)
- **Leakage in Data Mining — KDD 2011 peer-reviewed, alta relevancia a separación temporal/learn–predict; limitación: marco general, no especifica ventanas del proyecto.** — [DOI 10.1145/2020408.2020496](https://doi.org/10.1145/2020408.2020496)
- **Dawid–Skene — JRSS C peer-reviewed, fundamento clásico para error por revisor; limitación: supuestos estadísticos deben comprobarse y el modelo no reemplaza adjudicación de expertos.** — [DOI 10.2307/2346806](https://doi.org/10.2307/2346806)
- **Datasheets y Model Cards — publicaciones ACM peer-reviewed, útiles para documentación de datasets/modelos; limitación: son marcos de transparencia, no controles de integridad ni evidencia causal de desempeño.** — [Datasheets, DOI 10.1145/3458723](https://doi.org/10.1145/3458723); [Model Cards, DOI 10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596)
- **W3C PROV-O — Recomendación normativa oficial para procedencia interoperable; limitación: no prescribe almacenamiento tabular ni gobierno sectorial.** — [W3C PROV-O](https://www.w3.org/TR/prov-o/)
- **NIST AI RMF 1.0 — guía gubernamental oficial, voluntaria y agnóstica de sector; limitación: está en proceso de revisión en 2026 y no es certificación ni ley.** — [NIST publicación oficial](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10); [DOI 10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1)
- **ISO/IEC 5259-2:2024 — estándar internacional oficial sobre medidas de calidad de datos para ML; limitación: acceso completo de pago, por lo que aquí solo se valida alcance desde la ficha ISO y no se atribuyen cláusulas no inspeccionadas.** — [Ficha ISO oficial](https://www.iso.org/standard/81860.html)

### Inferences
- Criterio de inclusión aplicado: fuente original o emisor normativo; identidad bibliográfica/DOI verificable; afirmación limitada a lo que la fuente realmente soporta; distinción explícita entre evidencia empírica, recomendación normativa y diseño inferido.
- Criterio de exclusión: blogs comerciales, agregadores sin texto primario, Wikipedia/Reddit, afirmaciones de vendors y papers recientes no revisados usados como autoridad. Se pueden usar solo para descubrir la fuente original, no para sostener decisiones.
- La validación exhaustiva debe continuar en la ejecución: registrar versión exacta de cada norma consultada y realizar revisión bibliográfica periódica; no congelar decisiones solo porque una fuente sea prestigiosa.

### Gaps
- ISO/IEC 5259-2 no pudo revisarse en texto completo por paywall; antes de declarar conformidad se necesita acceso institucional y lectura cláusula por cláusula.
- No se halló ensayo prospectivo peer-reviewed que demuestre directamente el beneficio de este loop específico IF + VAE + analistas en este dataset. La conclusión correcta es “hipótesis plausible que debe validarse”, no “mejora garantizada”.
