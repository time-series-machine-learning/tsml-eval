from aeon.transformations.collection.feature_based import Catch22
import pycatch22
import numpy as np
from aeon.datasets import tsc_datasets, load_italy_power_demand

IPD_X_train, IPD_y_train = load_italy_power_demand(split="train")
aeon_c22 = Catch22(replace_nans=True)

results_aeon = aeon_c22.fit_transform(IPD_X_train)
results_pycatch22 = pycatch22.catch22_all(IPD_X_train[0][0])

print("Results of first data aeon     : ", results_aeon[0])
print("Results of first data pycatch22: ", np.array(results_pycatch22["values"]))