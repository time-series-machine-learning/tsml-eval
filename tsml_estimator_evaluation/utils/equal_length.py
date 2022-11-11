from scipy import signal
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_ts
from sktime.utils.data_io import write_dataframe_to_tsfile as write_ts
import numpy as np

dataset = "InsectWingBeat"
size = 20
class_label = ["Aedes_female", "Aedes_male", "Fruit_flies", "House_flies", "Quinx_female", "Quinx_male", "Stigma_female", "Stigma_male", "Tarsalis_female", "Tarsalis_male"]

X_train, y_train = load_ts("C:\\Users\\fbu19zru\\code\\Multivariate_ts\\" + dataset + "\\" + dataset + "_TRAIN.ts")
X_test, y_test = load_ts("C:\\Users\\fbu19zru\\code\\Multivariate_ts\\" + dataset + "\\" + dataset + "_TEST.ts")

for key, value in X_train.iterrows():
    # print(key, value)
    for key2, value2 in value.iteritems():
        #  print(key, key2)
        X_train.loc[key][key2] = np.round(signal.resample(X_train.iloc[key][key2], size), 6)

for key, value in X_test.iterrows():
    for key2, value2 in value.iteritems():
        X_test.loc[key][key2] = np.round(signal.resample(X_test.iloc[key][key2], size), 6)

# print(X_train)

write_ts(
    X_train,
    "C:\\Users\\fbu19zru\\code\\Multivariate_ts\\",
    dataset + "Eq",
    class_label,
    y_train,
    True,
    size,
    False, None,"_TRAIN")

write_ts(
    X_test,
    "C:\\Users\\fbu19zru\\code\\Multivariate_ts\\",
    dataset + "Eq",
    class_label,
    y_test,
    True,
    size,
    False, None,"_TEST")
