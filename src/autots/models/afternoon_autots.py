"""
Custom AutoTS Configuration for VN30F1M Afternoon Sessions.
"""
import os
from autots import AutoTS


def build_afternoon_model(fast_mode: bool = False, template_path: str = None) -> AutoTS:
    """
    Initializes AutoTS with directional metrics and causal whitelist transformers.
    If template_path is provided and exists, imports the best model template (including Ensemble)
    for fast inference without running genetic search or single-bar re-selection.
    """
    financial_transformer_list = {
        "DifferencedTransformer": 0.30,
        "PctChangeTransformer": 0.20,
        "RobustScaler": 0.15,
        "StandardScaler": 0.10,
        "EWMAFilter": 0.15,
        "AlignLastValue": 0.10,
    }

    metric_weighting = {
        'dwae_weighting': 3.5,     # Directional Weighted Absolute Error
        'oda_weighting': 2.0,      # Origin Direction Accuracy
        'spl_weighting': 2.0,      # Scaled Pinball Loss (prediction interval coverage)
        'contour_weighting': 1.0,  # Slope and curvature
        'mae_weighting': 0.8,
        'rmse_weighting': 0.4,
        'smape_weighting': 0.2,
        'runtime_weighting': 0.05,
    }

    model_candidates = [
        "RollingRegression",
        "WindowRegression",
        "VAR",
        "VECM",
        "BallTreeMultivariateMotif",
    ]

    use_template = template_path is not None and os.path.exists(template_path)
    max_generations = 0 if use_template else (5 if fast_mode else 15)
    num_validations = 0 if use_template else (1 if fast_mode else 2)

    model = AutoTS(
        forecast_length=1,
        frequency="12h",
        prediction_interval=0.90,
        ensemble='simple',
        model_list=model_candidates,
        transformer_list=financial_transformer_list,
        transformer_max_depth=2,
        metric_weighting=metric_weighting,
        max_generations=max_generations,
        num_validations=num_validations,
        validation_method="backwards",
        n_jobs=1 if use_template else "auto",
        verbose=0,
    )

    if use_template:
        model.import_template(template_path, method="only", include_ensemble=True)
        model.import_best_model(template_path, include_ensemble=True)
        orig_fit = model.fit

        def _fast_template_fit(df, *args, **kwargs):
            if getattr(model, "best_model", None) is not None and not model.best_model.empty:
                df_slice = df.iloc[-801:] if len(df) > 801 else df
                return model.fit_data(df_slice, weights=kwargs.get("weights", {}))
            return orig_fit(df, *args, **kwargs)

        model.fit = _fast_template_fit

    return model
