from __future__ import annotations
import numpy as np
import pytest
from scripts.run_cascade_pre_response_merit import allocate_depths

def test_allocate_depths_matches_total_and_bounds():
    ids=[f'q{i}' for i in range(7)]
    depths=allocate_depths(19,ids,'fixed-salt',5)
    assert int(depths.sum())==19
    assert depths.min()>=1
    assert depths.max()<=5
    assert sorted(depths.tolist())==[2,2,3,3,3,3,3]

def test_allocate_depths_is_deterministic_and_query_keyed():
    ids=[f'q{i}' for i in range(10)]
    first=allocate_depths(23,ids,'salt',4)
    second=allocate_depths(23,ids,'salt',4)
    assert np.array_equal(first,second)
    mapping=dict(zip(ids,first))
    reversed_depths=allocate_depths(23,list(reversed(ids)),'salt',4)
    assert mapping==dict(zip(reversed(ids),reversed_depths))

def test_allocate_depths_rejects_invalid_budget():
    with pytest.raises(ValueError):
        allocate_depths(2,['a','b','c'],'salt',4)
    with pytest.raises(ValueError):
        allocate_depths(13,['a','b','c'],'salt',4)
