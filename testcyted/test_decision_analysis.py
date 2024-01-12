#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Optional, Union
import numpy as np
from cyted.utils.decision_analysis import DecisionTree, compile_decisions


def decision_fn(a: bool, b: bool, c: bool) -> bool:
    return a and (b or c)


def test_compile() -> None:
    nargs = 3
    decisions, used = compile_decisions(decision_fn)
    assert set(decisions.keys()) == set(used.keys())

    num_input_combinations = 2**nargs
    assert len(decisions) == num_input_combinations

    assert decisions[False, False, False] is False
    for inputs in decisions:
        assert decisions[inputs] == decision_fn(*inputs)

    assert all(len(used[inputs]) == nargs for inputs in used)
    for a, b, c in used:
        assert used[a, b, c] == (True, a, a and not b)


def validate_bool_array(x: np.ndarray, shape: tuple[int, ...], value: Optional[Union[bool, np.ndarray]] = None) -> None:
    assert isinstance(x, np.ndarray)
    assert x.dtype == bool
    assert x.shape == shape
    if value is not None:
        assert (x == value).all()


def test_decision_tree_scalar() -> None:
    all_inputs: list[tuple[bool, bool]] = [(False, False), (False, True), (True, False), (True, True)]
    expected_decisions = {(a, b): a and b for a, b in all_inputs}
    expected_used = {(a, b): (True, a) for a, b in all_inputs}

    tree = DecisionTree(lambda a, b: a and b)
    assert tree._decisions_dict == expected_decisions
    assert tree._used_dict == expected_used

    for inputs in all_inputs:
        assert tree.decide(*inputs) == expected_decisions[inputs]
        assert tree.were_used(*inputs) == expected_used[inputs]


def test_decision_tree_vectorised() -> None:
    tree = DecisionTree(lambda a, b: a and b)

    num_a, num_b = 5, 3
    a = np.random.rand(num_a, 1) > 0.5
    b = np.random.rand(1, num_b) > 0.3

    decisions = tree.decide(a.flatten(), a.flatten())
    validate_bool_array(decisions, shape=(num_a,), value=a.flatten())

    decisions = tree.decide(a, b)
    validate_bool_array(decisions, shape=(num_a, num_b), value=a & b)

    were_used = tree.were_used(a, b)
    assert isinstance(were_used, tuple)
    assert len(were_used) == 2

    a_used, b_used = were_used
    validate_bool_array(a_used, shape=(num_a, num_b), value=True)
    validate_bool_array(b_used, shape=(num_a, num_b), value=a)
