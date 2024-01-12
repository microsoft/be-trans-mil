#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Union
import pytest
from cyted.configs.DeepSMILESlidesCyted import TFF3_HECytedMIL
from health_cpath.datamodules.base_module import HistoDataModule
from health_cpath.datasets.base_dataset import SlidesDataset, TilesDataset

from testcyted.utils_for_tests import mock_cached_read_csv, skipif_dataset_unavailable
from cyted.datasets.cyted_module import CytedSlidesDataModule
from cyted.cyted_schema import CytedSchema
from cyted.datasets.cyted_slides_dataset import CytedSlidesDataset
from cyted.data_paths import (
    CYTED_RAW_DATASET_DIR,
    CYTED_RAW_DATASET_ID,
    CYTED_SPLITS_CSV_FILENAME,
    get_cyted_dataset_dir,
    CYTED_DATA_SPLITS_DIR,
)
from cyted.utils.cyted_utils import CytedParams
from health_cpath.preprocessing.loading import LoadingParams
from health_cpath.utils.wsi_utils import TilingParams


def validate_splits(datamodule: HistoDataModule, full_dataset: Union[TilesDataset, SlidesDataset]) -> None:
    for split_dataset in [datamodule.train_dataset, datamodule.val_dataset, datamodule.test_dataset]:
        assert isinstance(split_dataset, type(full_dataset))

    train_ids = set(datamodule.train_dataset.dataset_df.index)
    val_ids = set(datamodule.val_dataset.dataset_df.index)
    test_ids = set(datamodule.test_dataset.dataset_df.index)

    assert len(train_ids) > 0
    assert len(val_ids) > 0
    assert len(test_ids) > 0

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    all_ids = set(full_dataset.dataset_df.index)

    assert (train_ids | val_ids | test_ids) == all_ids


@skipif_dataset_unavailable(CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID)
@pytest.mark.parametrize(
    "image_column, label_column",
    [
        (CytedSchema.HEImage, CytedSchema.TFF3Positive),
        (CytedSchema.TFF3Image, CytedSchema.TFF3Positive),
        # Run these tests once we generate splits for P53/Atypia
        # (CytedSchema.P53Image, CytedSchema.P53Positive),
        # (CytedSchema.HEImage, CytedSchema.Atypia),
        # (CytedSchema.HEImage, CytedSchema.P53Positive),
    ],
)
def test_cyted_slides_datamodule(image_column: str, label_column: str) -> None:
    with mock_cached_read_csv():  # Prevent repeatedly re-reading large CSVs, for speed
        root_path = get_cyted_dataset_dir(image_column=image_column)
        splits_csv = Path(CYTED_DATA_SPLITS_DIR) / CYTED_SPLITS_CSV_FILENAME[label_column]
        full_dataset = CytedSlidesDataset(root=root_path, image_column=image_column, label_column=label_column)

        # Test prescribed splits:
        datamodule = CytedSlidesDataModule(
            root_path=root_path,
            cyted_params=CytedParams(image_column=image_column, label_column=label_column),
            splits_csv=splits_csv,
            loading_params=LoadingParams(),
            tiling_params=TilingParams(),
        )

        validate_splits(datamodule, full_dataset)

        # Test cross-validation splits:
        cv_datamodule = CytedSlidesDataModule(
            root_path=root_path,
            cyted_params=CytedParams(image_column=image_column, label_column=label_column),
            splits_csv=splits_csv,
            loading_params=LoadingParams(),
            tiling_params=TilingParams(),
            crossval_index=0,
            crossval_count=5,
        )

        validate_splits(cv_datamodule, full_dataset)

        # Check that test set is the same:
        test_ids = set(datamodule.test_dataset.dataset_df.index)
        cv_test_ids = set(cv_datamodule.test_dataset.dataset_df.index)
        assert cv_test_ids == test_ids


@skipif_dataset_unavailable(CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID)
def test_eval_data_module() -> None:
    """Test if the evaluation data module can be created, and if it contains all rows."""
    # Create an instance of the model. Create the data module, check that it contains all rows.
    model = TFF3_HECytedMIL()
    eval_module = model.get_eval_data_module()
    test_loader = eval_module.test_dataloader()
    expected_len = 1141
    assert len(test_loader) == expected_len
    assert len(eval_module.train_dataset) == 0
    assert len(eval_module.val_dataset) == 0
    assert len(eval_module.test_dataset) == expected_len
    with pytest.raises(NotImplementedError, match="train_dataloader should not be called when running evaluation"):
        eval_module.train_dataloader()
    with pytest.raises(NotImplementedError, match="val_dataloader should not be called when running evaluation"):
        eval_module.val_dataloader()


@skipif_dataset_unavailable(CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID)
def test_data_module() -> None:
    """Test if the training data module can be created and contains the right metadata columns."""
    model = TFF3_HECytedMIL()
    train_module = model.get_data_module()
    print(f"Columns found: {train_module.train_dataset.dataset_df.columns}")
    for column in CytedSchema.metadata_columns():
        assert column in train_module.train_dataset.dataset_df.columns


@skipif_dataset_unavailable(CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID)
def test_data_module_with_background_keys() -> None:
    """Test if the training data module can be created and contains metadata columns for background normalization."""
    background_keys = CytedSchema.background_columns()
    model = TFF3_HECytedMIL(background_keys=background_keys)
    train_module = model.get_data_module()
    print(f"Columns found: {train_module.train_dataset.dataset_df.columns}")
    for column in CytedSchema.metadata_columns():
        assert column in train_module.train_dataset.dataset_df.columns
    for column in background_keys:
        assert column in train_module.train_dataset.dataset_df.columns
