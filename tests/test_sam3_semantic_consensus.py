import numpy as np
from scripts.task1.sam3.instance_membership import accumulate_membership
from scripts.task1.sam3.semantic_consensus import pool_concepts,project_labels


def test_fragmented_instance_ids_still_support_one_class():
    events=[(np.array([0]),np.array([i])) for i in [1,1,2,2]]
    source=accumulate_membership(events,np.array([6]),1,.4)
    assert source.status.tolist()==[3,3]
    pooled=pool_concepts(source,np.array([0,1,1]),.4)
    assert pooled.support_counts.tolist()==[4]
    assert pooled.status.tolist()==[1]
    assert project_labels(pooled).tolist()==[1]
    # The semantic operation does not upgrade either unresolved instance.
    assert source.status.tolist()==[3,3]


def test_pooling_different_classes_never_combines_support():
    events=[(np.array([0]),np.array([i])) for i in [1,2]]
    source=accumulate_membership(events,np.array([2]),1,.1)
    pooled=pool_concepts(source,np.array([0,1,2]),.1)
    assert pooled.status.tolist()==[2,2]
    assert project_labels(pooled).tolist()==[0]


def test_object_display_priority_does_not_create_membership():
    events=[(np.array([0,0]),np.array([1,2])),(np.array([0,0]),np.array([1,2])),
            (np.array([0]),np.array([1]))]
    source=accumulate_membership(events,np.array([3]),1,.4)
    pooled=pool_concepts(source,np.array([0,1,2]),.4)
    assert project_labels(pooled).tolist()==[1]
    assert project_labels(pooled,[2]).tolist()==[2]
    assert project_labels(pooled,[3]).tolist()==[1]
