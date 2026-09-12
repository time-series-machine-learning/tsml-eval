#!/bin/bash
# Generate the missing DrCIF-500 results for the 112 UCR datasets, with train files.
#
# HC2 builds its interval component as DrCIFClassifier(n_estimators=500), but the stored
# UCR DrCIF results use the class default of 200, so an ensemble rebuilt from them is not
# HC2. See run_ucr_component_experiments.sh for the general case.

exec "$(dirname "$0")/run_ucr_component_experiments.sh" DrCIF-500 IntervalBased "$@"
