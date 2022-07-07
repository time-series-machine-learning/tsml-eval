
def evaluate_cluster_results(path: str, output_dir: str):
    pass

if __name__ == '__main__':

    # Take raw data and derive metrics

    # Resolve relative directory
    evaluate_cluster_results('../evaluation/dummy_data/cluster_results/')

    # Resolve relative file
    evaluate_cluster_results('../evaluation/dummy_data/cluster_results/dummy_data.csv')

    # Resolve absolute directory
    evaluate_cluster_results('/home/chris/Documents/sktime_evaluation/evaluation/dummy_data/cluster_results/')

    # Resolve relative file
    evaluate_cluster_results('/home/chris/Documents/sktime_evaluation/evaluation/dummy_data/cluster_results/')
