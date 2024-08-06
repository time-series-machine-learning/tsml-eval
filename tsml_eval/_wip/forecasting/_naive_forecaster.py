from tsml_eval._wip.forecasting.base import BaseForecaster

class NaiveForecaster(BaseForecaster):
    """Naive forecaster that predicts the last value seen."""

    def __init__(self, axis=1, horizon=1):
        y_ = None   # Train series
        super().__init__(axis=axis, horizon=horizon)

    def _fit(self, y, X):
        self.y_ = y
        return self

    def _predict(self, y, X):
        return self.y_[-self.horizon:]