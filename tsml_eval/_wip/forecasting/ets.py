from tsml_eval._wip.forecasting.base import BaseForecaster


def ETSForecaster(BaseForecaster):


    def __init__(self, horizon=1, window=None):
        self.horizon = horizon
        self.window = window
        self._is_fitted = False
        super().__init__()

