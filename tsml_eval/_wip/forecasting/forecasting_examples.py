from tsml_eval._wip.forecasting.base import BaseForecaster
from tsml_eval._wip.forecasting._naive_forecaster import NaiveForecaster
from aeon.datasets import load_airline


y = load_airline()
y = y.to_numpy()
print(y)
naive = NaiveForecaster()
naive.fit(y)
print(naive.predict())

