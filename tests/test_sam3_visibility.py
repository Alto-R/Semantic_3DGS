import numpy as np
import pytest

from scripts.task1.sam3.visibility import informative_observations
from scripts.task1.sam3.instance_membership import accumulate_membership


def test_visibility_requires_absolute_and_relative_mass():
    # A tiny tail and an absolutely tiny splat are not reliable observations.
    got = informative_observations(np.arange(4), np.array([.04, .004, .05, .2]),
                                  np.array([1., .01, 1., 2.]), .01, .05)
    assert got.tolist() == [2, 3]


def test_visibility_gates_both_votes_and_denominator():
    peak = np.array([1.])
    events, observed = [], np.zeros(1, np.uint16)
    for mass, detected in [(1., True), (.8, True), (.001, True), (.002, False)]:
        seen = informative_observations(np.array([0]), np.array([mass]), peak, .01, .05)
        observed[seen] += 1
        if detected:
            events.append((seen, np.ones(len(seen), np.uint16)))
    result = accumulate_membership(events, observed, 1)
    assert result.support_counts.tolist() == [2]
    assert result.observe_counts.tolist() == [2]
    assert result.status.tolist() == [1]


def test_relaxed_consensus_keeps_minimum_two_cameras_and_strict_boundary():
    events = [(np.array([0, 1, 2]), np.array([1, 1, 1])),
              (np.array([0, 1]), np.array([1, 1]))]
    result = accumulate_membership(events, np.array([5, 4, 1]), 3, .4)
    assert result.status.tolist() == [3, 1, 2]


@pytest.mark.parametrize('floor', [-.1, 1.1, float('nan')])
def test_invalid_relative_floor(floor):
    with pytest.raises(ValueError):
        informative_observations(np.array([0]), np.array([1.]), np.array([1.]), 0, floor)


def test_invalid_consensus_threshold():
    with pytest.raises(ValueError):
        accumulate_membership([], np.array([1]), 1, float('nan'))


def test_soft_weights_preserve_weak_votes_and_downweight_weak_misses(tmp_path):
    from scripts.task1.sam3.visibility import observation_reliability
    from scripts.task1.sam3.instance_membership import save_membership, load_membership
    events, weights = [], []
    total = np.zeros(1)
    for mass, detected in [(1.,True),(.01,True),(.002,False),(.001,False)]:
        w = observation_reliability(np.array([0]),np.array([mass]),np.array([1.]),.01,.05)
        total += w
        if detected:
            events.append((np.array([0]),np.array([1])))
            weights.append(w)
    result = accumulate_membership(events,np.array([4]),1,.5,weights,total)
    assert result.support_counts.tolist() == [2]
    assert result.status.tolist() == [1]
    np.testing.assert_allclose(result.scores,[1.2/1.26])
    save_membership(tmp_path/'m.npz',result)
    loaded = load_membership(tmp_path/'m.npz')
    np.testing.assert_allclose(loaded.observe_weights,total)
    np.testing.assert_allclose(loaded.support_weights,[1.2])


def test_soft_weighted_support_cannot_exceed_denominator():
    with pytest.raises(RuntimeError):
        accumulate_membership([(np.array([0]),np.array([1]))],np.array([1]),1,.5,
                              [np.array([1.])],np.array([.1]))
