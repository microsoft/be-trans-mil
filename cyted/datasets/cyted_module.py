#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader
from cyted.cyted_schema import CytedSchema
from cyted.datasets.cyted_slides_dataset import CytedSlidesDataset
from cyted.fixed_paths import PathOrString
from cyted.utils.cyted_utils import CytedParams
from cyted.utils.split_utils import SplitsMixin, StratifyType
from health_cpath.datamodules.base_module import SlidesDataModule
from health_cpath.models.transforms import Subsampled
from health_cpath.utils.naming import ModelKey, TileKey


class CytedSlidesDataModule(SplitsMixin[CytedSlidesDataset], SlidesDataModule):
    """CytedSlidesDataModule is the child class of SlidesDataModule specific to Cyted datasets.

    Method get_splits() returns the train, val, test splits from the selected Cyted dataset.
    """

    def __init__(
        self,
        cyted_params: CytedParams,
        excluded_slides_csv: Optional[PathOrString] = None,
        splits_csv: Optional[PathOrString] = None,
        column_filter: Optional[Dict[str, str]] = None,
        stratify_by: StratifyType = "",
        metadata_columns: Optional[Iterable] = None,
        **kwargs: Any,
    ) -> None:
        self.cyted_params = cyted_params
        self.splits_csv = splits_csv
        self.excluded_slides_csv = excluded_slides_csv
        self.column_filter = column_filter
        self.metadata_columns = set(
            metadata_columns if metadata_columns is not None else CytedSchema.metadata_columns()
        )
        super().__init__(stratify_by=stratify_by, **kwargs)

    def _get_dataset(
        self, dataset_df: Optional[pd.DataFrame] = None, train: Optional[bool] = None
    ) -> CytedSlidesDataset:
        return CytedSlidesDataset(
            root=self.root_path,
            image_column=self.cyted_params.image_column,
            label_column=self.cyted_params.label_column,
            splits_csv=self.splits_csv,
            dataset_df=dataset_df,
            train=train,
            excluded_slides_csv=self.excluded_slides_csv,
            dataframe_kwargs=self.dataframe_kwargs,
            column_filter=self.column_filter,
            metadata_columns=self.metadata_columns,
        )

    def get_shuffle_transform(self, stage: ModelKey) -> List[Callable]:
        shuffle_keys = [
            TileKey.IMAGE,
            TileKey.TILE_ID,
            TileKey.TILE_TOP,
            TileKey.TILE_BOTTOM,
            TileKey.TILE_LEFT,
            TileKey.TILE_RIGHT,
        ]
        if self.bag_sizes[stage] > 0:
            max_size = self.bag_sizes[stage] if stage != ModelKey.TRAIN else self.cyted_params.bag_size_subsample
            return [Subsampled(keys=shuffle_keys, max_size=max_size)]
        return []  # When bag_size is 0 (whole slide), we don't shuffle as max_size needs to be infered dynamically

    def get_tiling_transforms(self, stage: ModelKey) -> List[Callable]:
        """
        This method returns the transforms that are applied to the raw whole slide images.
        First, the image is loaded, then it is tiled on the fly, next we extract tiles coordinates from
        the metadata, then we shuffle the tiles to avoid intensity ordering bias, and finally we split the
        tiles into individual images to be able to apply tile wise augmentations and processings.
        The tiles will later on be grouped into bags by a collate function.
        """
        return [
            self.loading_params.get_load_roid_transform(),
            self.tiling_params.get_tiling_transform(bag_size=self.bag_sizes[stage], stage=stage),
            self.tiling_params.get_extract_coordinates_transform(),
            *self.get_shuffle_transform(stage),
            self.tiling_params.get_split_transform(),
        ]


class CytedSlidesEvalModule(CytedSlidesDataModule):
    """This class implements a data module where all data is returned in the test data loader, possibly using
    exclusion lists.
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        # We need to supply the class weights here, even if empty, otherwise the super class
        # would try to infer the weights from the empty training dataset and fail
        super().__init__(class_weights=torch.tensor([]), **kwargs)

    def get_splits(self) -> Tuple[CytedSlidesDataset, CytedSlidesDataset, CytedSlidesDataset]:
        """Creates an empty training and validation dataset, and a test dataset that contains all the data.

        :return: A tuple of (train, val, test) datasets.
        """
        required_columns = [
            self.cyted_params.image_column,
            self.cyted_params.label_column,
            CytedSchema.CytedID,
            CytedSchema.Year,
            *self.metadata_columns,
        ]
        empty = CytedSlidesDataset(
            root=self.root_path,
            image_column=self.cyted_params.image_column,
            label_column=self.cyted_params.label_column,
            dataset_df=pd.DataFrame({col: [] for col in required_columns}),
        )
        test_dataset = CytedSlidesDataset(
            root=self.root_path,
            image_column=self.cyted_params.image_column,
            label_column=self.cyted_params.label_column,
            excluded_slides_csv=self.excluded_slides_csv,
            dataframe_kwargs=self.dataframe_kwargs,
            column_filter=self.column_filter,
        )
        return empty, empty, test_dataset

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError("train_dataloader should not be called when running evaluation")

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError("val_dataloader should not be called when running evaluation")
