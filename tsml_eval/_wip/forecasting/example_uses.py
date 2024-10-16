from aeon.datasets import load_airline
from tsml_eval._wip.forecasting.window_base import BaseWindowForecaster
#BaseWindowForecaster2
import numpy as np
y = load_airline()
x = y.to_numpy()

# Use case 1, have to slice the test data into instances externally
x_train = x[:100]
x_test = x[100:]
print(x_train.shape, "\n", x_test.shape)
forecaster = BaseWindowForecaster(horizon=1, window=10)
forecaster.fit(x_train)
#slice test into test instances
print(forecaster.predict(x_test))

# forecaster2 = BaseWindowForecaster2(horizon=1, window=10)
# forecaster.fit(x_train)
# print(forecaster.predict(x_test))
