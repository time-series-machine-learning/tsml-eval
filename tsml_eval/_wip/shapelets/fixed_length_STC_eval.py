from tsml_eval.evaluation import evaluate_classifiers_by_problem
import os

classifiers = ["stc", "fixedlengthshapelettransformclassifier", "notfixedlengthshapelettransformclassifier"]

flstc_dir = "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/fixedlengthshapelettransformclassifier/Predictions/"
stc_dir = "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/stc/Predictions/"
nflstc_dir = "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/notfixedlengthshapelettransformclassifier/Predictions/"
def find_common_folders(directory1, directory2):

    folders1 = set(get_folder_names(directory1))
    folders2 = set(get_folder_names(directory2))
    # Find the common datasets
    common_folders = folders1.intersection(folders2)
    return list(common_folders) 

def get_folder_names(directory):

    items = os.listdir(directory)
    # Filter the list to include dataset names
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    return folders

def main():
    datasets = find_common_folders(nflstc_dir,find_common_folders(stc_dir,flstc_dir))

    evaluate_classifiers_by_problem(
    "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/",
    classifiers,
    datasets,
    "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/evaluated_results/",
    resamples=1,
    eval_name="FixedLengthEval",
)
    
if __name__ == "__main__":
    main()