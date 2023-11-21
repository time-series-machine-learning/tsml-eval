from sklearn.utils.estimator_checks import parametrize_with_checks

from tsml_eval.estimators.clustering.consensus.ivc import IterativeVotingClustering
from tsml_eval.estimators.clustering.consensus.simple_vote import SimpleVote


@parametrize_with_checks([SimpleVote(n_clusters=2), IterativeVotingClustering(n_clusters=2, max_iterations=5)])
def test_simple_vote(estimator, check):
    check(estimator)
