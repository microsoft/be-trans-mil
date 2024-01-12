#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import functools
import os
from pathlib import Path
from typing import Optional, Union
from unittest import mock

import pandas as pd
import pytest
from _pytest.mark import MarkDecorator
from torch import cuda

PathOrString = Union[str, Path]


def tests_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the tests.
    If a relative path is provided then concatenate it with the absolute path
    to the repository root.

    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path(os.path.realpath(__file__)).parent
    return root / path if path else root


def skipif_dataset_unavailable(dataset_dir: PathOrString, dataset_name: str) -> MarkDecorator:
    """Convenience for pytest.mark.skipif() in case a dataset is not found at the expected location.

    :param dataset_dir: The expected dataset directory.
    :param dataset_name: A dataset label to show in the testing logs.
    :return: A Pytest skipif mark decorator.
    """
    return pytest.mark.skipif(
        not Path(dataset_dir).is_dir(), reason=f"{dataset_name} dataset not available at {dataset_dir}"
    )


def skipif_no_gpu() -> MarkDecorator:
    """Convenience for pytest.mark.skipif() in case no GPU is available.

    :return: A Pytest skipif mark decorator.
    """
    has_gpu = cuda.is_available() and cuda.device_count() > 0
    return pytest.mark.skipif(not has_gpu, reason="No GPU available")


def mock_cached_read_csv() -> mock._patch:
    """Mocks pandas.read_csv() to use a cache to prevent re-reading large CSV files.

    :return: A unittest mock patch object for the cached pandas.read_csv() function.
    """
    cached_read_csv = functools.lru_cache(maxsize=None)(pd.read_csv)
    return mock.patch("pandas.read_csv", side_effect=cached_read_csv)


RELATIVE_TEST_OUTPUTS_PATH = "test_outputs"
TEST_OUTPUTS_PATH = tests_root_directory().parent / RELATIVE_TEST_OUTPUTS_PATH
