#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pytest

from cyted.utils.split_utils import DEFAULT_TEST_SPLIT_LABEL, DEFAULT_TRAIN_SPLIT_LABEL, split_dataframe

_GROUP_COLUMN = "group_id"
_STRAT1_COLUMN = "strat1"
_STRAT2_COLUMN = "strat2"


@pytest.fixture
def mock_df() -> pd.DataFrame:
    num_instances = 1000
    num_groups = 250

    rng = np.random.RandomState(seed=0)
    groups = rng.randint(num_groups, size=num_instances)
    strat1 = rng.randint(3, size=num_groups)
    strat2 = rng.randint(2, size=num_groups)

    df = pd.DataFrame(
        {
            _GROUP_COLUMN: groups,
            _STRAT1_COLUMN: strat1[groups],
            _STRAT2_COLUMN: strat2[groups],
        }
    )
    return df


def validate_splits(splits: pd.Series, df: pd.DataFrame) -> None:
    assert splits.isna().sum() == 0
    assert splits.isin([DEFAULT_TRAIN_SPLIT_LABEL, DEFAULT_TEST_SPLIT_LABEL]).all()
    assert splits.index.equals(df.index)


def validate_split_counts(splits: pd.Series, test_frac: float, num_total: int) -> None:
    num_test = (splits == DEFAULT_TEST_SPLIT_LABEL).sum()
    assert 0 < num_test < num_total
    assert abs(num_test - test_frac * num_total) < 1


def validate_grouped_split_counts(splits: pd.Series, groups: pd.Series, test_frac: float, num_groups: int) -> None:
    grouped_splits = splits.groupby(groups)
    assert (grouped_splits.nunique() == 1).all()
    validate_split_counts(grouped_splits.first(), test_frac, num_groups)


def test_split_dataframe(mock_df: pd.DataFrame) -> None:
    num_total = len(mock_df)
    test_frac = 0.3

    splits = split_dataframe(mock_df, test_frac=test_frac)
    validate_splits(splits, mock_df)
    validate_split_counts(splits, test_frac, num_total)

    strat_splits = split_dataframe(mock_df, test_frac=test_frac, stratify=_STRAT1_COLUMN)
    validate_splits(strat_splits, mock_df)
    validate_split_counts(splits, test_frac, num_total)

    strat_splits = split_dataframe(mock_df, test_frac=test_frac, stratify=[_STRAT1_COLUMN, _STRAT2_COLUMN])
    validate_splits(strat_splits, mock_df)
    validate_split_counts(splits, test_frac, num_total)


def test_split_dataframe_grouped(mock_df: pd.DataFrame) -> None:
    groups = mock_df[_GROUP_COLUMN]
    num_groups = groups.nunique()
    test_frac = 0.3

    splits = split_dataframe(mock_df, test_frac=test_frac, group=_GROUP_COLUMN)
    validate_splits(splits, mock_df)
    validate_grouped_split_counts(splits, groups, test_frac, num_groups)

    strat_splits = split_dataframe(mock_df, test_frac=test_frac, stratify=_STRAT1_COLUMN, group=_GROUP_COLUMN)
    validate_splits(strat_splits, mock_df)
    validate_grouped_split_counts(strat_splits, groups, test_frac, num_groups)

    strat_splits = split_dataframe(
        mock_df, test_frac=test_frac, stratify=[_STRAT1_COLUMN, _STRAT2_COLUMN], group=_GROUP_COLUMN
    )
    validate_splits(strat_splits, mock_df)
    validate_grouped_split_counts(strat_splits, groups, test_frac, num_groups)


@pytest.mark.flaky(reruns=3)  # Parts of this test have a very small chance of failing at random
def test_split_dataframe_seeding(mock_df: pd.DataFrame) -> None:
    test_frac = 0.3

    unseeded_splits = split_dataframe(mock_df, test_frac=test_frac)
    other_unseeded_splits = split_dataframe(mock_df, test_frac=test_frac)
    error_message = (
        "Consecutive random unseeded splits are identical. This can happen at random "
        "with a very small probability; please try re-running this test."
    )
    assert not unseeded_splits.equals(other_unseeded_splits), error_message

    seeded_splits = split_dataframe(mock_df, test_frac=test_frac, seed=0)
    assert not seeded_splits.equals(unseeded_splits), "Seeded split is identical to random unseeded split"

    seeded_splits_same = split_dataframe(mock_df, test_frac=test_frac, seed=0)
    assert seeded_splits.equals(seeded_splits_same), "Splits with the same seed are different"

    seeded_splits_diff = split_dataframe(mock_df, test_frac=test_frac, seed=1)
    assert not seeded_splits.equals(seeded_splits_diff), "Splits with different seeds are identical"
