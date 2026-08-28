"""Preprocessing pipeline and transform diagnostics for the banking panel.

Two independent entry points live here:

* :func:`fit_transform_panel` -- the full sklearn pipeline used by ``main.py``
  (imputation, shape transforms, categorical encoding, panel features).
* :mod:`src.preprocessing.linear_scaling` -- stateless ``y = a*x + b``
  rescaling of the continuous block only, with no estimator objects. Useful
  standalone (EDA, a leaner serving path) and deliberately *not* wired into
  the pipeline above, which owns its own scaling.
"""

from src.preprocessing.linear_scaling import (
    SCALING_METHODS,
    apply_linear_scaling,
    minmax_scale,
    robust_scale,
    robust_scale_params,
    scaling_params,
    select_continuous_columns,
    standard_scale,
)
from src.preprocessing.pipeline import (
    CATEGORICAL_ENCODINGS,
    NUMERIC_TRANSFORMS,
    PanelFeatureEngineer,
    SignedLog1p,
    aggregate_attribution_by_source,
    build_preprocessing_pipeline,
    categorical_feature_mask,
    fit_transform_panel,
    group_name_by_source,
    make_numeric_transformer,
    split_matrix_for_model,
)
from src.preprocessing.statistics import (
    compute_transform_diagnostics,
    infer_numeric_features,
    plot_transform_diagnostics,
    recommend_transform,
)

__all__ = [
    "build_preprocessing_pipeline",
    "fit_transform_panel",
    "make_numeric_transformer",
    "PanelFeatureEngineer",
    "SignedLog1p",
    "NUMERIC_TRANSFORMS",
    "CATEGORICAL_ENCODINGS",
    "compute_transform_diagnostics",
    "recommend_transform",
    "plot_transform_diagnostics",
    "infer_numeric_features",
    "categorical_feature_mask",
    "split_matrix_for_model",
    "group_name_by_source",
    "aggregate_attribution_by_source",
    # -- stateless linear rescaling (src/preprocessing/linear_scaling.py) --
    "SCALING_METHODS",
    "select_continuous_columns",
    "robust_scale_params",
    "scaling_params",
    "apply_linear_scaling",
    "robust_scale",
    "standard_scale",
    "minmax_scale",
]
