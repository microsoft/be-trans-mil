#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
from enum import Enum
import logging
import pandas as pd
import param

from pathlib import Path
from typing import Any, List, Optional, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from cyted.cyted_schema import CytedSchema
from cyted.data_paths import CYTED_BOUNDING_BOXES_JSON
from cyted.datasets.cyted_slides_dataset import CytedSlidesDataset
from cyted.utils.azure_run_config import CytedAzureRunConfig
from cyted.utils.azure_utils import replace_ampersand_in_filename
from cyted.utils.crop_utils import CropAndConvertWSIToTiffd
from cyted.utils.registration_utils import RegisterCropConvertWSIToTiffd, RegistrationMetrics
from health_azure import DatasetConfig
from health_cpath.utils.naming import SlideKey
from health_cpath.utils.tiff_conversion_config import TiffConversionConfig


class DatasetTypes(str, Enum):
    FIXED = "fixed_dataset"
    FIXED_MASK = "fixed_mask_dataset"
    PREPROCESSED_MOVING = "preprocessed_moving_dataset"
    INPUT_TRANSFORMS = "input_transforms_dataset"
    OUTPUT_DIFF = "output_diff_dataset"
    OUTPUT_TRANSFORMS = "output_transforms_dataset"


class CytedTiffConversionConfig(TiffConversionConfig, CytedAzureRunConfig):
    """Cyted configuration for converting raw data into tiff files."""

    wsi_subfolder: str = param.String(
        default="",
        doc="The name of the subfolder where the WSI images are stored. If empty, the WSI images will be stored in "
        "the root of the output dataset.",
    )
    automatic_output_name: bool = param.Boolean(
        default=True,
        doc="If true, the output dataset will be named automatically as "
        "'preprocessed_<image_column>_<target_magnification>'.",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.automatic_output_name:
            self.output_dataset += self.get_auto_output_name()

    def get_auto_output_name(self) -> str:
        """Returns an automatic name of the dataset to be created."""
        target_magnification = self.target_magnifications[0] if self.target_magnifications else ""
        image_str = replace_ampersand_in_filename(self.image_column)
        return f"preprocessed_{image_str.lower()}_{target_magnification}x"

    def process_dataset(self, input_folders: List[Path], output_folders: List[Path]) -> None:
        """Converts the raw data into tiff files."""
        dataset = self.get_slides_dataset(input_folders[0])
        self.run(dataset, output_folders[0], self.wsi_subfolder)


class CytedCroppingAndTiffConversionConfig(CytedTiffConversionConfig):
    """Cyted configuration for cropping and converting raw data into tiff files. If a mask dataset is provided, the
    background will be set to white.
    Otherwise, foreground masks will be generated on the fly and background will be hardcoded from the masks."""

    bounding_box_path: Path = param.ClassSelector(
        class_=Path,
        default=CYTED_BOUNDING_BOXES_JSON[CytedSchema.TFF3Image],
        doc="The path to the json file containing the bounding boxes for the slides.",
    )
    mask_dataset: Optional[Path] = param.ClassSelector(
        default=None,
        class_=Path,
        doc="The dataset id of the dataset containing the foreground masks for hardcoding background to white."
        "If running locally, this should be the name of the directory. If running on Azure, this is the dataset "
        "name to be retrieved. If empty, masks will be generated on the fly "
        "and background will be hardcoded from the generated masks.",
    )
    output_mask_dataset: Optional[str] = param.String(
        default=None,
        doc="The name of the output masks dataset where generated masks will be stored.",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.automatic_output_name and self.output_mask_dataset:
            self.output_mask_dataset += self.get_auto_output_name()

    def get_input_datasets(self) -> List[DatasetConfig]:
        """Creates the input datasets configs for the AzureML run."""
        input_datasets = super().get_input_datasets()
        if self.mask_dataset:
            logging.info(
                f"In AzureML use mounted masks dataset '{self.mask_dataset}' in datastore {self.datastore}"
                "as it was provided as an argument in mask_dataset."
            )
            input_datasets.append(
                DatasetConfig(name=str(self.mask_dataset), datastore=self.datastore, use_mounting=True)
            )
        return input_datasets

    def get_input_folders_for_local_run(self) -> List[Path]:
        """Returns the input datasets folders for local runs."""
        input_folders = super().get_input_folders_for_local_run()
        if self.mask_dataset:
            input_folders.append(self.mask_dataset)
        return input_folders

    def get_output_datasets(self) -> Optional[List[DatasetConfig]]:
        """Creates the output dataset config for the AzureML run."""
        output_datasets = super().get_output_datasets()
        assert isinstance(output_datasets, list)
        # Add masks dataset if not already available
        if not self.mask_dataset and self.output_mask_dataset:
            logging.info(
                f"In AzureML create output masks dataset '{self.output_mask_dataset}'"
                f"in datastore {self.datastore}, as no mask_dataset was provided."
            )
            output_datasets.append(
                DatasetConfig(name=str(self.output_mask_dataset), datastore=self.datastore, use_mounting=True)
            )
        return output_datasets

    def get_output_folders_from_local_run(self) -> List[Path]:
        """Returns the output dataset folders for local runs."""
        output_folders = super().get_output_folders_from_local_run()
        # Add masks dataset if not already available
        if not self.mask_dataset and self.output_mask_dataset:
            output_mask_folder = self.create_folder_for_output_dataset(self.output_mask_dataset)
            output_folders.append(output_mask_folder)
        return output_folders

    def get_transform(self, output_folder: Path) -> CropAndConvertWSIToTiffd:
        return CropAndConvertWSIToTiffd(
            bounding_box_path=self.bounding_box_path,
            masks_path=Path(self.mask_dataset) if self.mask_dataset else None,
            plot_path=self.plot_path,
            output_folder=output_folder,
            image_key=self.image_key,
            target_magnifications=self.target_magnifications,
            add_lowest_magnification=self.add_lowest_magnification,
            default_base_objective_power=self.default_base_objective_power,
            replace_ampersand_by=self.replace_ampersand_by,
            compression=self.compression,
            tile_size=self.tile_size,
            output_masks_path=Path(self.output_mask_dataset) if not self.mask_dataset else None,  # type: ignore
        )

    def add_background_values_to_dataset(self, background_value_dict: Dict[str, List[int]]) -> None:
        """Adds the background values to the dataset.

        :param background_value_dict: A dictionary mapping the slide id to the background value.
        """
        background_value_df = pd.DataFrame.from_dict(
            background_value_dict, orient="index", columns=list(CytedSchema.background_columns())
        )
        assert isinstance(self.slides_dataset, CytedSlidesDataset), "Expected dataset to be CytedSlidesDataset."
        self.slides_dataset.dataset_df = self.slides_dataset.dataset_df.merge(
            background_value_df, left_index=True, right_index=True
        )
        self.slides_dataset.dataset_df.index.name = CytedSchema.CytedID

    def __call__(self, dataloader: DataLoader) -> None:
        background_value_dict = {}
        for data in tqdm(dataloader, total=len(dataloader)):
            background_value_dict[data[SlideKey.SLIDE_ID][0]] = [pd.NA, pd.NA, pd.NA]
            if CytedSchema.Background in data:
                # Data loaders convert the background values from tuples to tensor, convert back to tuple
                background_value_dict[data[SlideKey.SLIDE_ID][0]] = [int(x) for x in data[CytedSchema.Background]]
        self.add_background_values_to_dataset(background_value_dict)

    def process_dataset(self, input_folders: List[Path], output_folders: List[Path]) -> None:
        """Crop and convert the raw data into tiff files."""
        self.mask_dataset = input_folders[1] if self.mask_dataset else None
        self.output_mask_dataset = str(output_folders[1]) if self.output_mask_dataset else None
        super().process_dataset(input_folders, output_folders)


class CytedRegistrationConfig(CytedCroppingAndTiffConversionConfig):
    """Cyted configuration to perform end to end registration of TFF3 images to H&E images.
    The registration is performed using downsampled arrays of original TFF3 and H&E slides (at target magnification).
    In the registered crops, background is optionally hardcoded. Result is a registered, (background hardcoded)
    and stitched TFF3 dataset. Registration transforms are stored in pickle files in an output folder (if provided).
    Difference between the H&E and TFF3 masks is stored in a separate output folder (if provided)."""

    fixed_dataset: Path = param.ClassSelector(
        class_=Path,
        doc="The dataset id of the dataset containing the fixed or reference images (e.g. H&E images)."
        "If running locally, this should be the name of the directory. If running on Azure, this is the H&E dataset "
        "name to be retrieved",
    )
    fixed_mask_dataset: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        doc="The dataset id of the H&E mask dataset. If H&E masks are available, these will be used for"
        "stain matrix estimation using Macenko method in the registration pipeline.",
    )
    preprocessed_moving_dataset: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        doc="The dataset id of a preprocessed moving (e.g. TFF3). If available, registration is performed only on"
        "positive slides, and negative preprocessed slides are included in the output dataset without registration.",
    )
    input_transforms_dataset: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        doc="The dataset id of the input transforms dataset."
        "If transforms are already stored, these would be used for registration.",
    )
    bounding_box_path_fixed: Path = param.ClassSelector(
        class_=Path,
        default=CYTED_BOUNDING_BOXES_JSON[CytedSchema.HEImage],
        doc="The path to the json file containing the bounding boxes for the fixed (H&E) slides.",
    )
    output_diff_dataset: Optional[str] = param.String(
        default=None,
        doc="The name of the output dataset where difference between the registered H&E and TFF3 masks will be stored.",
    )
    output_transforms_dataset: Optional[str] = param.String(
        default=None,
        doc="The name of the output dataset where the registration transforms are stored in the pickle files.",
    )
    hardcode_background: Optional[bool] = param.Boolean(
        default=True, doc="Flag if background should be hardcoded for registered dataset."
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.automatic_output_name and self.output_diff_dataset:
            self.output_diff_dataset += self.get_auto_output_name() + "_diff"
        if self.automatic_output_name and self.output_transforms_dataset:
            self.output_transforms_dataset += self.get_auto_output_name() + "_transforms"
        self.input_dataset_idx = {
            DatasetTypes.FIXED: 1000,  # set a high initial index (for debugging only)
            DatasetTypes.FIXED_MASK: 1000,
            DatasetTypes.INPUT_TRANSFORMS: 1000,
            DatasetTypes.PREPROCESSED_MOVING: 1000,
        }
        self.output_dataset_idx = {
            DatasetTypes.OUTPUT_DIFF: 1000,
            DatasetTypes.OUTPUT_TRANSFORMS: 1000,
        }

    def get_auto_output_name(self) -> str:
        """Returns an automatic name of the dataset to be created."""
        target_magnification = self.target_magnifications[0] if self.target_magnifications else ""
        image_str = replace_ampersand_in_filename(self.image_column)
        return f"registered_{image_str.lower()}_{target_magnification}x"

    def set_input_dataset_and_append(
        self, dataset: Optional[Path], dataset_key: DatasetTypes, input_datasets: List[DatasetConfig]
    ) -> List[DatasetConfig]:
        """Set the input dataset indices and append to the list of input datasets."""
        logging.info(
            f"In AzureML use mounted {dataset_key} '{dataset}' in "
            f"datastore {self.datastore} as it was provided as an argument in {dataset_key}."
        )
        input_datasets.append(DatasetConfig(name=str(dataset), datastore=self.datastore, use_mounting=True))
        self.input_dataset_idx[dataset_key] = len(input_datasets) - 1
        return input_datasets

    def get_input_datasets(self) -> List[DatasetConfig]:
        """Creates the input datasets configs for the AzureML run."""
        input_datasets = super().get_input_datasets()
        if self.fixed_dataset:
            input_datasets = self.set_input_dataset_and_append(self.fixed_dataset, DatasetTypes.FIXED, input_datasets)
        if self.fixed_mask_dataset:
            input_datasets = self.set_input_dataset_and_append(
                self.fixed_mask_dataset, DatasetTypes.FIXED_MASK, input_datasets
            )
        if self.input_transforms_dataset:
            input_datasets = self.set_input_dataset_and_append(
                self.input_transforms_dataset, DatasetTypes.INPUT_TRANSFORMS, input_datasets
            )
        if self.preprocessed_moving_dataset:
            input_datasets = self.set_input_dataset_and_append(
                self.preprocessed_moving_dataset, DatasetTypes.PREPROCESSED_MOVING, input_datasets
            )
        return input_datasets

    def get_input_folders_for_local_run(self) -> List[Path]:
        """Returns the input datasets folders for local runs."""
        input_folders = super().get_input_folders_for_local_run()
        input_folders.append(self.fixed_dataset)
        self.input_dataset_idx[DatasetTypes.FIXED] = len(input_folders) - 1
        if self.fixed_mask_dataset:
            input_folders.append(self.fixed_mask_dataset)
            self.input_dataset_idx[DatasetTypes.FIXED_MASK] = len(input_folders) - 1
        if self.input_transforms_dataset:
            input_folders.append(self.input_transforms_dataset)
            self.input_dataset_idx[DatasetTypes.INPUT_TRANSFORMS] = len(input_folders) - 1
        if self.preprocessed_moving_dataset:
            input_folders.append(self.preprocessed_moving_dataset)
            self.input_dataset_idx[DatasetTypes.PREPROCESSED_MOVING] = len(input_folders) - 1
        return input_folders

    def set_output_dataset_and_append(
        self, dataset: Optional[str], dataset_key: DatasetTypes, output_datasets: List[DatasetConfig]
    ) -> List[DatasetConfig]:
        """Set the output dataset indices and append to the list of output datasets."""
        logging.info(
            f"In AzureML create {dataset_key} '{dataset}'"
            f"in datastore {self.datastore} as it was provided as an argument in {dataset_key}."
        )
        output_datasets.append(DatasetConfig(name=str(dataset), datastore=self.datastore, use_mounting=True))
        self.output_dataset_idx[dataset_key] = len(output_datasets) - 1
        return output_datasets

    def get_output_datasets(self) -> List[DatasetConfig]:
        """Creates the output dataset config for the AzureML run."""
        output_datasets = super().get_output_datasets()
        assert output_datasets is not None
        # Add output datasets if specified in the command line arguments
        if self.output_diff_dataset:
            output_datasets = self.set_output_dataset_and_append(
                self.output_diff_dataset, DatasetTypes.OUTPUT_DIFF, output_datasets
            )
        if self.output_transforms_dataset:
            output_datasets = self.set_output_dataset_and_append(
                self.output_transforms_dataset, DatasetTypes.OUTPUT_TRANSFORMS, output_datasets
            )
        return output_datasets

    def get_output_folders_from_local_run(self) -> List[Path]:
        """Returns the output dataset folders for local runs."""
        output_folders = super().get_output_folders_from_local_run()
        # Add output datasets if specified in the command line arguments
        if self.output_diff_dataset:
            output_diff_folder = self.create_folder_for_output_dataset(self.output_diff_dataset)
            output_folders.append(output_diff_folder)
            self.output_dataset_idx[DatasetTypes.OUTPUT_DIFF] = len(output_folders) - 1
        if self.output_transforms_dataset:
            output_transforms_folder = self.create_folder_for_output_dataset(self.output_transforms_dataset)
            output_folders.append(output_transforms_folder)
            self.output_dataset_idx[DatasetTypes.OUTPUT_TRANSFORMS] = len(output_folders) - 1
        return output_folders

    def get_transform(self, output_folder: Path) -> RegisterCropConvertWSIToTiffd:
        return RegisterCropConvertWSIToTiffd(
            bounding_box_path=self.bounding_box_path,
            fixed_dataset_path=Path(self.fixed_dataset),
            bb_path_fixed=self.bounding_box_path_fixed,
            label_column=self.label_column,
            fixed_masks_path=Path(self.fixed_mask_dataset) if self.fixed_mask_dataset else None,
            input_transforms_path=Path(self.input_transforms_dataset) if self.input_transforms_dataset else None,
            output_folder=output_folder,
            image_key=self.image_key,
            target_magnifications=self.target_magnifications,
            default_base_objective_power=self.default_base_objective_power,
            replace_ampersand_by=self.replace_ampersand_by,
            compression=self.compression,
            tile_size=self.tile_size,
            output_diff_path=Path(self.output_diff_dataset) if self.output_diff_dataset else None,  # type: ignore
            output_transforms_path=Path(self.output_transforms_dataset) if self.output_transforms_dataset else None,
            hardcode_background=self.hardcode_background,
            dataset_csv=self.dataset_csv,
            preprocessed_path=Path(self.preprocessed_moving_dataset) if self.preprocessed_moving_dataset else None,
        )

    def process_dataset(self, input_folders: List[Path], output_folders: List[Path]) -> None:
        """Crop and convert the raw data into tiff files."""
        self.fixed_dataset = input_folders[self.input_dataset_idx[DatasetTypes.FIXED]]
        self.fixed_mask_dataset = (
            input_folders[self.input_dataset_idx[DatasetTypes.FIXED_MASK]] if self.fixed_mask_dataset else None
        )
        self.input_transforms_dataset = (
            input_folders[self.input_dataset_idx[DatasetTypes.INPUT_TRANSFORMS]]
            if self.input_transforms_dataset
            else None
        )
        self.preprocessed_moving_dataset = (
            input_folders[self.input_dataset_idx[DatasetTypes.PREPROCESSED_MOVING]]
            if self.preprocessed_moving_dataset
            else None
        )
        self.output_diff_dataset = (
            str(output_folders[self.output_dataset_idx[DatasetTypes.OUTPUT_DIFF]]) if self.output_diff_dataset else None
        )
        self.output_transforms_dataset = (
            str(output_folders[self.output_dataset_idx[DatasetTypes.OUTPUT_TRANSFORMS]])
            if self.output_transforms_dataset
            else None
        )
        super().process_dataset(input_folders, output_folders)

    def add_metric_values_to_dataset(self, metric_value_dict: Dict[str, List[int]]) -> None:
        """Adds the registration metric values to the dataset.

        :param metric_value_dict: A dictionary mapping the slide id to the registration metric values.
        """
        metric_value_df = pd.DataFrame.from_dict(
            metric_value_dict, orient="index", columns=[RegistrationMetrics.MI_BEFORE, RegistrationMetrics.MI_AFTER]
        )
        assert isinstance(self.slides_dataset, CytedSlidesDataset), "Expected dataset to be CytedSlidesDataset."
        self.slides_dataset.dataset_df = self.slides_dataset.dataset_df.merge(
            metric_value_df, left_index=True, right_index=True
        )
        self.slides_dataset.dataset_df.index.name = CytedSchema.CytedID

    def __call__(self, dataloader: DataLoader) -> None:
        metric_value_dict = {}
        background_value_dict = {}
        for data in tqdm(dataloader, total=len(dataloader)):
            metric_value_dict[data[SlideKey.SLIDE_ID][0]] = [pd.NA, pd.NA]
            background_value_dict[data[SlideKey.SLIDE_ID][0]] = [pd.NA, pd.NA, pd.NA]
            if CytedSchema.Background in data:
                # Data loaders convert the background values from tuples to tensor, convert back to tuple
                background_value_dict[data[SlideKey.SLIDE_ID][0]] = [int(x) for x in data[CytedSchema.Background]]
            if RegistrationMetrics.MI_BEFORE in data:
                metric_value_dict[data[SlideKey.SLIDE_ID][0]] = [
                    float(data[RegistrationMetrics.MI_BEFORE]),
                    float(data[RegistrationMetrics.MI_AFTER]),
                ]
        self.add_background_values_to_dataset(background_value_dict)
        self.add_metric_values_to_dataset(metric_value_dict)
