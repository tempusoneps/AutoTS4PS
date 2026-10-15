"""
Custom AutoTS Configuration for VN30F1M Afternoon Sessions.
"""
from autots import AutoTS

def build_afternoon_model(fast_mode: bool = False) -> AutoTS:
    """
    Initializes AutoTS with directional metrics and causal whitelist transformers.
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

    max_generations = 5 if fast_mode else 15
    num_validations = 1 if fast_mode else 2

    return AutoTS(
        forecast_length=1,
        frequency="infer",
        prediction_interval=0.90,
        ensemble='simple',
        model_list=model_candidates,
        transformer_list=financial_transformer_list,
        transformer_max_depth=2,
        metric_weighting=metric_weighting,
        max_generations=max_generations,
        num_validations=num_validations,
        validation_method="backwards",
        n_jobs="auto",
        verbose=0
    )
