from tsml_eval.evaluation.multiple_estimator_evaluation import \
     evaluate_clusterers_by_problem, evaluate_classifiers_by_problem, evaluate_regressors_by_problem
from aeon.datasets.tsc_data_lists import univariate_equal_length
from aeon.benchmarking import plot_critical_difference
from aeon.benchmarking.results_loaders import get_estimator_results

import pandas as pd
import numpy as np

import numpy as np

from aeon.classification.tests.test_classification_reproduction import _print_results_for_classifier
from aeon.transformations.tests.test_transformer_reproduction import _print_results_for_transformer


from aeon.datasets.tsc_data_lists import univariate_equal_length

print(univariate_equal_length)
print(list(univariate_equal_length))

from aeon.benchmarking import get_estimator_results, get_estimator_results_as_array
est = ["STC","RDST","ROCKET","MrSQM"]
import pandas as pd

new_res = pd.read_csv("C:\\Temp\\Accuracy_mean.csv")
data_names = new_res.iloc[:,0].tolist()
cls_names = new_res.columns[1:].tolist()

print(cls_names)

new_results = new_res.iloc[:,3].values.tolist()

array = np.zeros((len(data_names),len(est)+1))
published_results = get_estimator_results(estimators=est, default_only=False)
for i in range(len(data_names)):
    for j in range(len(est)):
        array[i][j] =np.average(published_results[est[j]][data_names[i]])
    array[i][len(est)] = new_results[i]
from aeon.benchmarking import plot_critical_difference
est.append(cls_names[2])
cd = plot_critical_difference(array, est)
cd.show()

print(est)

print(np.mean(array,axis=0))


def compare_to_reference(new_path, est):
     new_res = pd.read_csv(new_path)
     data_names = new_res.iloc[:,0].tolist()
     cls_names = new_res.columns[1:].tolist()
     new_results = new_res.iloc[:,1:].values
     array = np.zeros((len(data_names),len(est)+len(cls_names)))
     published_results = get_estimator_results(estimators=est, default_only=False)
     for i in range(len(data_names)):
         for j in range(len(est)):
             array[i,j] =np.average(published_results[est[j]][data_names[i]])
         for j in range(len(cls_names)):
              array[i,len(est)+j] = new_results[i,j]

     for i in range(len(cls_names)):
         est.append(cls_names[i])

     cd = plot_critical_difference(array, est)
     cd.show()

#compare_to_reference("C:\\Temp\\testy.csv",est = ["STC","RDST","ROCKET","MrSQM"])
#
#
# resamples = 30
# results = "/gpfs/home/ajb/ResultsWorkingArea/STC_Tests/"
# names = ["MAIN", "STDEV", "REFERENCE"]
# locations = [results+names[0],results+names[1], results+names[2]]
#
# evaluate_classifiers_by_problem(
#      load_path=locations,
#      classifier_names=[("STC", "MAIN"), ("STC", "STDEV"), ("STC", "REFERENCE")],
#      dataset_names=univariate_equal_length,
#      resamples=30,
#      eval_name="stdev_test",
#      verify_results=False,
#      error_on_missing=False
# )
