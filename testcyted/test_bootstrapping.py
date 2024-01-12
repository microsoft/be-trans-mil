#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Union
import numpy as np
import pandas as pd
import pytest
from cyted.utils import bootstrapping
from pandas.testing import assert_frame_equal


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])


def test_get_bootstrap_samples(df: pd.DataFrame) -> None:
    num_samples = 10
    boot_dfs = bootstrapping.draw_bootstrap_samples(df, num_samples)

    assert len(boot_dfs) == num_samples
    for boot_df in boot_dfs:
        assert isinstance(boot_df, pd.DataFrame)
        assert len(boot_df) == len(df)
        assert list(boot_df.columns) == list(df.columns)
        assert boot_df.index.isin(df.index).all()


def test_get_bootstrap_samples_seeding(df: pd.DataFrame) -> None:
    num_samples = 3
    seed = 42
    boot_dfs = bootstrapping.draw_bootstrap_samples(df, num_samples, random_state=seed)
    boot_dfs_same_seed = bootstrapping.draw_bootstrap_samples(df, num_samples, random_state=seed)
    boot_dfs_diff_seed = bootstrapping.draw_bootstrap_samples(df, num_samples, random_state=seed + 1)

    for boot_df, boot_df_same_seed, boot_df_diff_seed in zip(boot_dfs, boot_dfs_same_seed, boot_dfs_diff_seed):
        assert_frame_equal(boot_df_same_seed, boot_df, check_exact=True)
        with pytest.raises(AssertionError):
            assert_frame_equal(boot_df_diff_seed, boot_df, check_exact=True)


def test_collate_list_of_dicts() -> None:
    dicts = [
        {'a': 0, 'b': 'foo'},
        {'a': 1, 'b': 'bar'},
    ]
    collated_dict = bootstrapping.collate_list_of_dicts(dicts)
    assert collated_dict == {'a': [0, 1], 'b': ['foo', 'bar']}


def test_compute_bootstrap_quantiles(df: pd.DataFrame) -> None:
    def get_statistics(df: pd.DataFrame) -> dict[str, Union[float, np.ndarray]]:
        return {
            'scalar': df['a'].mean(),  # shape: ()
            'array': np.quantile(df['c'], [0.1, 0.9]),  # shape: (2,)
        }

    quantiles = [0.1, 0.5, 0.9]
    quantiles_dict = bootstrapping.compute_bootstrap_quantiles(df, get_statistics, quantiles=quantiles, num_samples=10)

    assert list(quantiles_dict.keys()) == ['scalar', 'array']
    assert all(isinstance(values, np.ndarray) for values in quantiles_dict.values())
    assert quantiles_dict['scalar'].shape == (len(quantiles),)
    assert quantiles_dict['array'].shape == (len(quantiles), 2)
