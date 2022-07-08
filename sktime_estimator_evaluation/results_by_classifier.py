import os
import numpy as np
import csv
from root import ROOT_DIR


def get_single_classifier_results(classifier, root =ROOT_DIR, package = "tsml"):
    """Load the results for a single classifier.

     Load from results into a dictionary of {problem_names: accuracy (numpy array)
    """
    print(root)
    cls_file = "results/"+package+"/ByClassifier/"+classifier+"_TESTFOLDS.csv"
    print(cls_file)
    abspath = os.path.join(root, cls_file)
    file = open(abspath, "r", encoding="utf-8")
    #Read and discard header line

    #Look for results file locally

    #Look on tsc.com?

    #load from CSV

    #return array


def get_classifier_results(classifiers, problems, resamples = 30, root =ROOT_DIR, package = "tsml"):
    """Collate results for classifiers over problems.

    given lists of n problems and m classifiers, form an n by m array of accuracies, averaged over resamples

    Returns a len(problems) x len(classifiers) array. NaN returned if combination not found
    """