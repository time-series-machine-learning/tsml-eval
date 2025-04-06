import sys

sys.path.append("C:/Users/alexb/Documents/University/PhD/aeon/aeon")
sys.path.append("C:/Users/alexb/Documents/University/PhD/aeon/tsml-eval")

from tsml_eval.evaluation.multiple_estimator_evaluation import (
    evaluate_forecasters_by_problem,
)

if __name__ == "__main__":
    evaluate_forecasters_by_problem(
        "../results",
        [
            "RocketRegressor",
            "MultiRocketRegressor",
            "ResNetRegressor",
            "fpcregressor",
            "fpcr-b-spline",
            "TimeCNNRegressor",
            "FCNRegressor",
            "1nn-ed",
            "1nn-dtw",
            "5nn-ed",
            "5nn-dtw",
            "FreshPRINCERegressor",
            "TimeSeriesForestRegressor",
            "DrCIFRegressor",
            "Ridge",
            "SVR",
            "RandomForestRegressor",
            "RotationForestRegressor",
            "xgboost",
            "NaiveForecaster",
        ],
        "../aeon/aeon/datasets/local_data/forecasting/windowed_series.txt",
        "../results/plots",
        10,
        True,
        None,
        False,
    )
