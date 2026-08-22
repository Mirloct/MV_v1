> Curated, paraphrased reference digest from GeeksforGeeks (geeksforgeeks.org), with sources cited per section. Maintained for use by this project's documentation. All explanations are summarized in our own words; consult the linked GfG articles for the full text and code examples.

# GeeksforGeeks Reference Notes

These notes map to the project's modules. Each section paraphrases one or more GeeksforGeeks articles and lists the exact source URL(s) used. Where a GfG statement was imprecise, a correction is noted inline.

---

## 1. Anomaly / Outlier Detection

Anomaly detection is the analytical technique of identifying data points, events, or behaviors that deviate from an expected pattern. The goal is to surface irregularities that may signal fraud, faults, intrusions, or other abnormal system behavior early enough to act on them. Approaches fall into statistical methods (e.g., flagging points far from the mean in terms of standard deviations), density-based methods (e.g., DBSCAN), and distance-based methods (e.g., k-NN). In machine-learning terms, detection can be supervised (needs labeled anomalies), unsupervised (no labels, the common case), or semi-supervised (trained mostly on normal data).

Anomalies are conventionally grouped into three types:

- **Point (global) anomalies** — a single observation that lies far from the rest of the data. Example: a credit-card charge much larger than an account's usual spending.
- **Contextual anomalies** — a point that is only anomalous within a particular context, while looking normal overall. Example: 85 F is normal in summer but anomalous in winter; a sales spike is normal on a holiday but suspicious on an ordinary day. Context is usually defined by conditional attributes such as time or location.
- **Collective anomalies** — a group of points that are each individually plausible but jointly anomalous. Example: a sudden burst of network traffic from one IP over a short window, which can indicate a denial-of-service attack.

Sources:
- https://www.geeksforgeeks.org/computer-networks/what-is-anomaly-detection/
- https://www.geeksforgeeks.org/data-analysis/types-of-outliers-in-data-mining/

---

## 2. Isolation Forest

Isolation Forest is an unsupervised, tree-based anomaly detector built on a different intuition than most methods: instead of modeling what "normal" looks like and measuring deviation from it, it directly *isolates* anomalies. Because anomalies are few and different, they are easy to separate from the bulk of the data with only a few random cuts.

How it works: the algorithm builds an ensemble of *isolation trees*. To grow each tree it recursively partitions the data by picking a feature at random and then a random split value within that feature's range, continuing until points are isolated. The key quantity is the **path length** — the number of splits needed to isolate a given point. Anomalies tend to be isolated near the root (short paths), while normal points require many more splits (long paths). Averaging path lengths across many trees yields an anomaly score; shorter average path length means a higher anomaly score and a greater likelihood of being an outlier. The method scales well to high-dimensional data and is relatively robust to noise.

Key parameters:
- **n_estimators** — the number of trees in the ensemble; more trees give more stable scores.
- **max_samples** — the number of samples drawn to build each tree (sub-sampling is a core part of the original algorithm; small samples actually help isolate anomalies).
- **contamination** — the assumed proportion of anomalies in the data, used to set the score threshold that separates outliers from inliers.
- **random_state** — fixes the randomness for reproducible results.

Sources:
- https://www.geeksforgeeks.org/machine-learning/what-is-isolation-forest/
- https://www.geeksforgeeks.org/machine-learning/anomaly-detection-using-isolation-forest/

---

## 3. Autoencoders and Variational Autoencoders (VAE)

An **autoencoder** is a neural network that learns to compress data and then reconstruct it. It has two parts: an *encoder* that maps the input to a compact latent (bottleneck) representation, and a *decoder* that reconstructs the original input from that representation. Training minimizes the **reconstruction error** (e.g., mean squared error) between input and output. This makes autoencoders useful for anomaly detection: a model trained mostly on normal data reconstructs normal inputs well but reconstructs unfamiliar/anomalous inputs poorly, so a high reconstruction error is used as an anomaly signal.

A **Variational Autoencoder (VAE)** is a generative extension. Instead of encoding each input to a single fixed latent point, the encoder outputs the parameters of a probability distribution over the latent space — a mean vector (mu) and a standard-deviation vector (sigma, in practice usually log-variance). A latent code is then *sampled* from this distribution and decoded. This yields a smooth, continuous latent space from which new, realistic samples can be generated.

Two ideas make the VAE trainable and well-behaved:
- **KL divergence regularization** — the loss adds a Kullback-Leibler divergence term that pushes each input's latent distribution toward a prior (typically a standard Gaussian). This keeps the latent space smooth and prevents it from collapsing to disconnected points. The full VAE objective is the **ELBO (Evidence Lower Bound)**: reconstruction log-likelihood minus KL divergence. Maximizing the ELBO (equivalently, minimizing reconstruction error plus the KL term) is a tractable proxy for maximizing the otherwise-intractable data likelihood.
- **Reparameterization trick** — you cannot backpropagate through a random sampling step directly. The trick rewrites the sample as z = mu + sigma * epsilon, where epsilon is drawn from a fixed standard normal. The randomness is moved into epsilon, so gradients flow through mu and sigma and the network trains with ordinary backpropagation.

(VAEs were introduced by Kingma and Welling in 2013.)

Sources:
- https://www.geeksforgeeks.org/machine-learning/variational-autoencoders/
- https://www.geeksforgeeks.org/machine-learning/role-of-kl-divergence-in-variational-autoencoders/
- https://www.geeksforgeeks.org/numpy/types-of-autoencoders/

---

## 4. Data Preprocessing / Feature Scaling

Data preprocessing turns raw data into a clean, consistent form so a model learns real patterns rather than noise or artifacts. Typical steps: inspect the data and its types, handle missing values, detect and treat outliers (e.g., the IQR rule flags points below Q1 - 1.5*IQR or above Q3 + 1.5*IQR), encode categorical variables, and scale numeric features.

**Feature scaling.** Many algorithms that rely on distances or gradients (k-NN, SVM, k-means, neural networks, and gradient-descent-based models) are biased toward features with numerically larger ranges when data is unscaled. Scaling puts features on comparable scales, speeds up gradient-descent convergence, and prevents any one feature from dominating. Note: tree-based models (decision trees, random forests, gradient boosting) are generally invariant to monotonic feature scaling and do not require it.
- **Normalization (Min-Max scaling)**: X' = (X - X_min) / (X_max - X_min), rescaling values into [0, 1]. It preserves the shape of the distribution but is sensitive to outliers (a single extreme value stretches the range). Good for bounded inputs and neural networks.
- **Standardization (Z-score)**: X' = (X - mu) / sigma, giving mean 0 and unit variance. It is a good default for most algorithms and is less distorted by outliers than min-max.
- **Robust scaling**: uses the median and interquartile range instead of mean/standard deviation, making it the best choice when outliers are heavy.
An important practice not always stressed in intro articles: fit the scaler on the training set only, then apply it to validation/test data, to avoid data leakage.

**Handling missing values.** Missingness is categorized as MCAR (completely at random), MAR (depends on other observed variables), or MNAR (depends on the missing value itself). Options include dropping rows/columns (simple but shrinks the sample and can bias results if data is not MCAR) and imputation: simple mean/median/mode fills, or more accurate model-based methods such as k-NN imputation, iterative/regression imputation, and multiple imputation.

**Encoding categorical variables.** Models need numeric input, so categories must be encoded. **One-hot encoding** creates one binary column per category (1 present, 0 absent) and suits *nominal* data with no inherent order (colors, countries); it pairs well with linear models, neural networks, and k-NN. **Label encoding** assigns each category an integer — compact, but it implies an ordering, so it is appropriate for *ordinal* data (Low/Medium/High) or for tree-based models (decision trees, XGBoost) that do not treat the integers as magnitudes.

Sources:
- https://www.geeksforgeeks.org/data-analysis/data-preprocessing-machine-learning-python/
- https://www.geeksforgeeks.org/machine-learning/feature-engineering-scaling-normalization-and-standardization/
- https://www.geeksforgeeks.org/machine-learning/normalization-vs-standardization/
- https://www.geeksforgeeks.org/data-analysis/handling-missing-values-machine-learning/
- https://www.geeksforgeeks.org/machine-learning/one-hot-encoding-vs-label-encoding/

---

## 5. Evaluation Metrics for (Imbalanced) Classification / Anomaly Detection

On imbalanced problems (anomaly detection is extreme: anomalies may be well under 1% of data), plain **accuracy** is misleading because always predicting the majority class scores highly. Metrics are built from the **confusion matrix** of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

- **Precision** = TP / (TP + FP) — of the points flagged as anomalies, how many really are. High precision means few false alarms.
- **Recall (Sensitivity, TPR)** = TP / (TP + FN) — of all true anomalies, how many were caught. High recall means few missed anomalies.
- **F1 score** = 2 * (Precision * Recall) / (Precision + Recall) — the harmonic mean of precision and recall. The harmonic mean penalizes imbalance between the two, so F1 is only high when both are high; it is preferred over accuracy on imbalanced data.
- **F-beta score** = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall) — a weighted generalization of F1. beta > 1 weights recall more (use when missing an anomaly is costly); beta < 1 weights precision more (use when false alarms are costly); beta = 1 recovers F1. (F-beta generalizes the F1 formula the GfG F1 article presents; included here for completeness.)
- **ROC curve / ROC-AUC** — plots TPR against False Positive Rate (FPR = FP / (FP + TN)) across all thresholds; AUC summarizes it in one number. AUC near 1.0 means strong separation between classes, 0.5 means random guessing. Caveat: on highly imbalanced data ROC-AUC can look overly optimistic because a large TN count suppresses FPR.
- **Precision-Recall AUC (PR-AUC / Average Precision)** — plots precision against recall across thresholds. Because it ignores true negatives and focuses on the positive (rare) class, it is the more informative summary than ROC-AUC for heavily imbalanced anomaly-detection tasks.
- **Precision@k / Recall@k** — when only the top-scoring candidates can be reviewed (a common operational constraint in anomaly detection), rank points by anomaly score and evaluate the top k: precision@k is the fraction of those top-k that are true anomalies; recall@k is the fraction of all true anomalies captured within the top-k. (GfG does not have a dedicated precision@k article; this is standard ranking-metric usage consistent with its precision/recall definitions.)

Sources:
- https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/
- https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/
- https://www.geeksforgeeks.org/machine-learning/evaluation-metrics-for-classification-model-in-python/
- https://www.geeksforgeeks.org/machine-learning/handling-imbalanced-data-for-classification/

---

## 6. Hyperparameter Tuning (Optuna)

Hyperparameters are settings configured *before* training that control the learning process (e.g., tree depth, learning rate, number of estimators, latent dimension). Unlike model *parameters* (weights), they are not learned from the data. Hyperparameter tuning searches for the values that maximize a validation objective and help avoid over/underfitting. Common strategies: **grid search** (exhaustive over a specified grid — thorough but expensive), **random search** (samples random combinations — often more efficient), and **Bayesian optimization** (builds a surrogate model of the objective and uses past trials to choose promising next configurations).

**Optuna** is an automatic hyperparameter-optimization framework with a define-by-run API: you write an *objective function* that suggests hyperparameter values and returns a score, and Optuna runs many *trials* to minimize or maximize it. Under the hood it typically uses Bayesian-style samplers (e.g., the Tree-structured Parzen Estimator) plus pruning to stop unpromising trials early, and it integrates with essentially any ML/DL framework.

Sources:
- https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning/
- https://www.geeksforgeeks.org/machine-learning/optuna/
- https://www.geeksforgeeks.org/machine-learning/hyperparameters-optimization-methods-ml/
