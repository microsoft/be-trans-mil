#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Any, Callable, Sequence, TypeVar
import numpy as np
import pandas as pd

K = TypeVar("K")
V = TypeVar("V")


def draw_bootstrap_samples(df: pd.DataFrame, num_samples: int, random_state: Any = None) -> list[pd.DataFrame]:
    """Generate bootstrap replicates of a given dataframe, sampling rows with replacement.

    :param df: Input dataframe.
    :param num_samples: How many bootstrap samples to draw (typically in the 100s).
    :param random_state: Any object that can be passed to NumPy to seed a random number generator,
        including integers. In `None`, fresh samples will be drawn each time.
    :return: A `num_samples`-long list of resampled dataframes with same size and structure as `df`.
    """
    if num_samples < 1:
        raise ValueError("Number of bootstrap samples must be at least 1")
    return [df.sample(len(df), replace=True, axis='index', random_state=random_state) for _ in range(num_samples)]


def collate_list_of_dicts(dicts: list[dict[K, V]]) -> dict[K, list[V]]:
    """Convert a list of dictionaries into a dictionary of lists, preserving order."""
    keys = dicts[0].keys()
    if not all(elem.keys() == keys for elem in dicts[1:]):
        raise ValueError("Every dict in the list should have the same keys")
    return {key: [elem[key] for elem in dicts] for key in keys}


def compute_bootstrap_quantiles(
    df: pd.DataFrame,
    statistics_fn: Callable[..., dict[K, Any]],
    quantiles: Sequence[float],
    num_samples: int,
    random_state: Any = None,
    **fn_kwargs: Any,
) -> dict[K, np.ndarray]:
    """Compute quantiles of arbitrary statistics using bootstrapping.

    :param df: Input dataframe.
    :param statistics_fn: A function taking as first positional input a dataframe similar to `df`
        and producing a dictionary of scalar or array-like statistics.
    :param quantiles: Which quantiles to compute, between 0 and 1 (e.g. `[0.05, 0.95]`).
    :param num_samples: How many bootstrap samples to draw (typically in the 100s).
    :param random_state: Any object that can be passed to NumPy to seed a random number generator,
        including integers. In `None`, fresh samples will be drawn each time.
    :param fn_kwargs: Additional keyword arguments to pass to `statistics_fn`.
    :return: A dictionary with the same keys as returned by `statistics_fn`, and array values whose
        first dimension corresponds to each of the requested quantiles.
    """
    boot_dfs = draw_bootstrap_samples(df, num_samples, random_state=random_state)
    boot_stats_dicts = [statistics_fn(boot_df, **fn_kwargs) for boot_df in boot_dfs]
    collated_stats_dict = collate_list_of_dicts(boot_stats_dicts)
    quantiles_dict: dict[K, np.ndarray] = {
        key: np.nanquantile(values, quantiles, axis=0) for key, values in collated_stats_dict.items()
    }
    return quantiles_dict
