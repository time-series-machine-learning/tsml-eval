import os
import numpy as np
import csv
from root import ROOT_DIR

valid_uni_classifiers =[
    "Arsenal", "BOSS","Catch22","cBOSS","CIF","DrCIF","HC1","HC2","InceptionTime",
    "ProximityForest","ResNet","RISE","ROCKET","S-BOSS","STC","STSF","TDE",
    "TS-CHIEF","TSF","WEASEL"
]
valid_multi_classifiers = [
    "CBOSS", "CIF", "DTW_A", "DTW_D", "DTW_I", "gRSF", "InceptionTime","mrseql",
    "MUSE","ResNet","RISE","ROCKET","STC","TapNet","TSF"
]


def get_single_classifier_results(classifier, root=ROOT_DIR, package="tsml", type="Univariate"):
    """Load the results for a single classifier on a single resample from local storage.

    Parameters
    ----------
    classifier: string
    root: string default is <thispackagelocation>/estimator-evaluation/results/
    Returns
    ----------
    A dictionary of {problem_names: accuracies (numpy array)}
    """
    cls_file = "results/"+package+"/ByClassifier/"+type+"/"+classifier+"_TESTFOLDS.csv"
    abspath = os.path.join(root, cls_file)
    # Check file exists

    # Open and store in a dictionary
    with open(abspath, "r", encoding="utf-8") as file:
        header = file.readline()
        results = {}
        for line in file:
            all = line.split(",")
            res = np.array(all[1:]).astype(np.float)
#            results
            results[all[0]] = res
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


def get_single_classifier_results_from_web(classifier, type="Univariate"):
    """Load the results for a single classifier on a single resample.

     Load from results into a dictionary of {problem_names: accuracy (numpy array)}.

     classifier: one of X
     type: string, either "Univariate" or "Multivariate"
    """
    if type == "Univariate":
        if not classifier in valid_uni_classifiers:
            raise Exception("Error, classifier ", classifier, "not in univariate set")
    elif type == "Multivariate":
        if not classifier in valid_multi_classifiers:
            raise Exception("Error, classifier ", classifier, "not in multivariate set")
    else:
        raise Exception("Type must be Univariate or Multivariate, you set it to ",type)

    url = "https://timeseriesclassification.com/results/ResultsByClassifier/"+type\
          +"/"+ classifier

    url = url+"_TESTFOLDS.csv"
    import requests
    response = requests.get(url)
    data = response.text
    split = data.split('\n')
    results = {}
    for i, line in enumerate(split):
        if len(line) > 0 and i > 0:
            all = line.split(",")
            res = np.array(all[1:]).astype(float)
            results[all[0]] = res
    return results

def get_averaged_results_from_web(datasets, classifiers, start=0, end=1,
                             type="Multivariate"):
    """Extracts all results for UCR/UEA datasets on tsc.com for classifiers,
    then formats them into an array size n_datasets x n_classifiers.
    """
    if end<start:
        raise Exception("End resample smaller than start resample")
    results = np.zeros(shape=(len(datasets),len(classifiers)))
    cls_index = 0
    for cls in classifiers:
        selected = {}
        # Get all the results
        full_results = get_single_classifier_results_from_web(cls, type=type)
        # Extract the required ones
        data_index = 0
        for d in datasets:
            results[data_index][cls_index] = np.NaN
            if d in full_results:
                all_resamples = full_results[d]
                if len(all_resamples) >= end: # Average here
                    mean = all_resamples[start]
                    for i in range(start+1,end):
                        mean = mean+all_resamples[i]
                    results[data_index][cls_index] =mean/(end-start)
            data_index = data_index + 1
        cls_index = cls_index + 1
    return results
