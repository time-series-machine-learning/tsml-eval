import os
from tsml_eval.evaluation import evaluate_classifiers_by_problem

classifiers = [
    "stc", 
    "fixedlengthshapelettransformclassifier", 
    "notfixedlengthshapelettransformclassifier",
    "dilatedshapelettransformclassifier"
]

flstc_dir = "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/fixedlengthshapelettransformclassifier/Predictions/"
dstc_dir = "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/dilatedshapelettransformclassifier/Predictions/"
stc_dir = "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/stc/Predictions/"
nflstc_dir = "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/notfixedlengthshapelettransformclassifier/Predictions/"

def get_folder_names(directory):
    items = os.listdir(directory)
    # Filter the list to include dataset names (directories only)
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    return folders

def find_common_folders(*directories):
    # Start with folder names in the first directory
    common_folders = set(get_folder_names(directories[0]))
    # Intersect with folders in all other directories
    for directory in directories[1:]:
        common_folders.intersection_update(get_folder_names(directory))
    return list(common_folders) 

def main():
    # Find datasets that are common across all directories
    datasets = find_common_folders(stc_dir, flstc_dir, nflstc_dir,dstc_dir)

    evaluate_classifiers_by_problem(
        "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/results/",
        classifiers,
        datasets,
        "/mainfs/lyceum/ik2g21/aeon/ClassificationResults/evaluated_results/",
        resamples=1,
        eval_name="DilatedFixedLengthEval",
    )
    
if __name__ == "__main__":
    main()
