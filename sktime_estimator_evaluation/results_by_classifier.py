import os
import numpy as np
import csv
from root import ROOT_DIR


def get_single_classifier_results(classifier, root=ROOT_DIR, package="tsml"):
    """Load the results for a single classifier on a single resample.

     Load from results into a dictionary of {problem_names: accuracy (numpy array)
    """
    print(root)
    cls_file = "results/"+package+"/ByClassifier/"+classifier+"_TESTFOLDS.csv"
    print(cls_file)
    abspath = os.path.join(root, cls_file)
    # Check file exists

    # Open and store in a dictionary
    with open(abspath, "r", encoding="utf-8") as file:
        header = file.readline()
        print(header)
        results ={}
        for line in file:
            all=line.split(",")

            print(all[0])
            res = np.array(all[1:]).astype(np.float)
#            results
            results[all[0]]=res
            print(res)
#            print(np.array(map(float, all[1:])))
    #return dictionary of problem/accurcacy
        return results


def get_classifier_results(classifiers, datasets, resample=0, root=ROOT_DIR,
                           package ="tsml"):
    """Collate results for classifiers over problems.

    given lists of n datasets and m classifiers, form an n by m array of accuracies,
    averaged over resamples. If not present, NaN is inserted

    Returns a len(problems) x len(classifiers) array. NaN returned if combination not found
    """
    all_results = []
    n_cls = len(classifiers)
    n_data = len(datasets)
    results = np.zeros(shape=(n_data, n_cls))
    print(n_cls," classifiers and ", n_data," datasets")
    for cls in classifiers:
        all_results.append(get_single_classifier_results(cls))
    prob_index = 0
    for pr in datasets:
        cls_index = 0
        for res in all_results:
            results[prob_index][cls_index] = np.NaN
            if pr in res and len(res[pr]) > resample: # resample present
                results[prob_index][cls_index] = res[pr][resample]
            cls_index = cls_index+1
        prob_index = prob_index+1
    return results

#r=get_single_classifier_results("ROCKET")
#print(r["ItalyPowerDemand"][0])
#print(r["Yoga"][0])
#cls = ["ROCKET", "Arsenal"]
#data = ["ItalyPowerDemand", "Yoga", "FRED"]
#res = get_classifier_results(cls, data)
#print(res)
