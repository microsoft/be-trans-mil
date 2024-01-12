#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import inspect
import itertools
from typing import Callable, Union

import numpy as np

BoolOrArray = Union[bool, np.ndarray]


def compile_decisions(
    decision_fn: Callable[..., bool]
) -> tuple[dict[tuple[bool, ...], bool], dict[tuple[bool, ...], tuple[bool, ...]]]:
    """Enumerate all decisions and whether each variable was evaluated by a Boolean decision function.

    :param decision_fn: A function taking any number of Boolean arguments and returning a Boolean.
    :return: A tuple of two dictionaries mapping all possible combinations of Boolean inputs to: (1) the function's
        output decisions and (2) to a tuple of Boolean values indicating which inputs were evaluated.
    """

    class _Node:
        def __init__(self, value: bool) -> None:
            self.value = value
            self.evaluated = False

        def __bool__(self) -> bool:
            # The object will be converted to bool only if needed,
            # due to logical short-circuiting during evaluation.
            self.evaluated = True
            return self.value

    args = inspect.getfullargspec(decision_fn).args

    decisions = {}
    used = {}
    for inputs in itertools.product([False, True], repeat=len(args)):
        nodes = [_Node(value) for value in inputs]
        decisions[inputs] = bool(decision_fn(*nodes))
        used[inputs] = tuple(node.evaluated for node in nodes)

    return decisions, used


class DecisionTree:
    """A utility class representing the evaluation of a Boolean decision function.

    The object has two vecorised methods that accept and return Boolean NumPy arrays:
    - `decide`: returns the function's Boolean decisions.
    - `were_used`: returns whether each of the input variables had to be evaluated by the function.
    """

    def __init__(self, fn: Callable[..., bool]):
        """
        :param fn: A function taking any number of Boolean arguments and returning a Boolean.
        """
        self._decisions_dict, self._used_dict = compile_decisions(fn)
        self.decide = np.vectorize(lambda *args: self._decisions_dict[args])
        self.were_used = np.vectorize(lambda *args: self._used_dict[args])
