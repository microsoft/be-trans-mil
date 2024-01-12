#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import logging
from pathlib import Path
import pandas as pd
import param
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from monai.transforms import Compose
from monai.data.dataset import Dataset
from typing import Any, Dict, List, Optional

from health_azure import DatasetConfig
from health_azure.logging import logging_section
from health_cpath.preprocessing.loading import LoadingParams
from health_cpath.utils.naming import ModelKey, SlideKey, TileKey
from health_cpath.utils.wsi_utils import TilingParams
from cyted.analysis.slide_analysis_transforms import (
    AnalysisMetadata,
    BrownStainRatiod,
    TilesBrownStainRatiod,
    TilesCountd,
)
from cyted.utils.azure_run_config import CytedAzureRunConfig


class AnalysisConfig(CytedAzureRunConfig, LoadingParams, TilingParams):
    """Abstract class for analysis configurations. This is used to analyse a dataset and save the results in a csv file.
    It inherits from CytedAzureRunConfig to allow the use of AzureML to run the analysis and LoadingParams and
    TilingParams for loading and tiling the whole slides.
    """

    image_key: SlideKey = param.ClassSelector(
        class_=SlideKey,
        default=SlideKey.IMAGE,
        doc="The key of the image in the batch dictionary.",
    )
    id_key: SlideKey = param.ClassSelector(
        class_=SlideKey,
        default=SlideKey.SLIDE_ID,
        doc="The key of the slide id in the batch dictionary.",
    )
    num_workers: int = param.Integer(
        default=1,
        doc="The number of workers used for analysis.",
    )

    def get_output_datasets(self) -> Optional[List[DatasetConfig]]:
        return None

    def get_transforms(self) -> Compose:
        raise NotImplementedError("get_transforms() no implemented in AnalysisConfig abstract class.")

    def get_analysis_metadata_keys(self) -> List[AnalysisMetadata]:
        raise NotImplementedError("get_analysis_metadata_keys() no implemented in AnalysisConfig abstract class.")

    def save_analysis_metadata(self, metadata_dict: Dict, output_folder: Path, filename: str) -> None:
        """Saves the analysis metadata to a csv file.

        :param metadata_dict: The metadata dictionary to save.
        :param output_folder: The folder to save the csv file to.
        :param filename: The name of the csv file.
        """
        df = pd.DataFrame.from_dict(metadata_dict, orient="columns")
        df.to_csv(output_folder / filename, index=False, index_label=SlideKey.SLIDE_ID)

    def __call__(self, dataloader: DataLoader, output_folder: Path) -> None:
        metadata_keys = self.get_analysis_metadata_keys() + [SlideKey.LABEL, SlideKey.SLIDE_ID]
        metadata_dict: Dict[str, List] = {key: [] for key in metadata_keys}

        for data in tqdm(dataloader, total=len(dataloader)):
            for key in metadata_keys:
                metadata_dict[key].append(data[key][0].item() if isinstance(data[key][0], Tensor) else data[key][0])

        logging.info(f"Saving analysis metadata to {output_folder}")
        self.save_analysis_metadata(metadata_dict, output_folder, "analysis_outputs.csv")

    def analyse_dataset(self, input_folders: List[Path], output_folder: Path) -> None:
        """Analyse a dataset and save the results in a csv file.

        :param input_folders: The input folders containing the dataset to analyse.
        :param output_folder: The output folder to save the csv file to.
        """
        dataset = self.get_slides_dataset(input_folders[0])
        transformed_dataset = Dataset(dataset, self.get_transforms())  # type: ignore
        dataloader = DataLoader(transformed_dataset, num_workers=self.num_workers, batch_size=1)
        metadata_keys = [meta_key.value for meta_key in self.get_analysis_metadata_keys()]
        with logging_section(f"Analysis of dataset {input_folders[0]} using metadata keys {', '.join(metadata_keys)}"):
            self(dataloader, output_folder)


class TilesCountConfig(AnalysisConfig):
    """Analysis configuration to count the number of tiles in a whole slide image."""

    def get_transforms(self) -> Compose:
        return Compose(
            [
                self.get_load_roid_transform(),
                self.get_tiling_transform(bag_size=0, stage=ModelKey.TEST),  # whole slide tiling
                TilesCountd(self.image_key),
            ]
        )

    def get_analysis_metadata_keys(self) -> List[AnalysisMetadata]:
        return [AnalysisMetadata.TILES_COUNT]


class BrownStainSlidesConfig(AnalysisConfig):
    """Analysis configuration to estimate the brown stain ratio in a whole slide image. Additionally, tiles count per
    slide is also estimated."""

    downsample_factor: int = param.Integer(
        default=1,
        doc="The downsample scale factor to use for the brown stain estimation. Defaults to 1 for no downsampling."
        "If set to 10, the image will be downsampled by a factor of 10.",
    )
    background_val: int = param.Integer(
        default=255,
        allow_None=True,
        doc="The background value to use to mask background pixels when estimating the brown stain. If None, it will be"
        " inferred as the 80th percentile of the wsi pixel values.",
    )
    area_threshold: int = param.Integer(
        default=64,
        doc="The maximum area, in pixels, of a contiguous hole that will be filled in the brown stain mask. "
        "Defaults to 64.",
    )
    variance_threshold: int = param.Integer(
        default=500000,
        doc="The variance threshold to use for considering a stain as brown. Consider all cells are normal if variance"
        " is too small. Defaults to 500000.",
    )

    def get_transforms(self) -> Compose:
        return Compose(
            [
                self.get_load_roid_transform(),
                BrownStainRatiod(
                    image_key=self.image_key,
                    id_key=self.id_key,
                    downsample_factor=self.downsample_factor,
                    background_val=self.background_val,
                    area_threshold=self.area_threshold,
                    variance_threshold=self.variance_threshold,
                    plot_path=self.plot_path,
                ),
                self.get_tiling_transform(bag_size=0, stage=ModelKey.TEST),  # whole slide tiling
                TilesCountd(self.image_key),
            ]
        )

    def get_analysis_metadata_keys(self) -> List[AnalysisMetadata]:
        return [
            AnalysisMetadata.BROWN_STAIN_RATIO,
            *AnalysisMetadata.get_pixel_count_keys(),
            *AnalysisMetadata.get_otsu_keys(),
            AnalysisMetadata.TILES_COUNT,
        ]


def get_value_from_listlike(var: Any) -> Any:
    """Returns the value of a list-like variable.

    :param var: The list-like variable, e.g. multi-dimensional tensor, tuple, list, etc.
    :return: The value of the variable to be returned.
    """
    if isinstance(var, Tensor):
        return var.item()
    elif isinstance(var, (tuple, list)):
        return var[0]
    else:
        return var


class BrownStainTilesConfig(BrownStainSlidesConfig):
    """
    Analysis configuration to estimate the brown stain ratio in tiles, given the slide level mask of brown stain.
    """

    def get_transforms(self) -> Compose:
        return Compose(
            [
                self.get_load_roid_transform(),
                BrownStainRatiod(  # slide level brown stain estimation
                    image_key=self.image_key,
                    id_key=self.id_key,
                    downsample_factor=self.downsample_factor,
                    background_val=self.background_val,
                    area_threshold=self.area_threshold,
                    variance_threshold=self.variance_threshold,
                    plot_path=self.plot_path,
                    save_binary_mask=True,  # save the binary mask
                ),
                self.get_tiling_transform(bag_size=0, stage=ModelKey.TEST),  # whole slide tiling
                TilesCountd(self.image_key),
                self.get_extract_coordinates_transform(),
                TilesBrownStainRatiod(image_key=self.image_key),  # tile level brown stain estimation from mask
            ]
        )

    def tile_level_analysis(self, data: Dict[str, Any], output_folder: Path) -> None:
        metadata_tile_keys = [
            TileKey.TILE_ID,
            TileKey.TILE_LEFT,
            TileKey.TILE_RIGHT,
            TileKey.TILE_TOP,
            TileKey.TILE_BOTTOM,
        ] + self.get_analysis_metadata_tile_keys()
        metadata_tiles_dict: Dict[str, List] = {key: [] for key in metadata_tile_keys}
        for i in range(len(data[TileKey.TILE_ID])):
            for key in metadata_tile_keys:
                tile_data = data[key].squeeze(0) if isinstance(data[key], Tensor) else data[key]
                assert len(tile_data) == len(
                    data[TileKey.TILE_ID]
                ), f"Tile data in {key} doesn't have the same length as the number of tiles."
                metadata_tiles_dict[key].append(get_value_from_listlike(tile_data[i]))
        logging.info(f"Saving tiles analysis metadata for slide {data[SlideKey.SLIDE_ID][0][0]} to {output_folder}")
        self.save_analysis_metadata(
            metadata_tiles_dict,
            output_folder,
            f"analysis_outputs_tiles_{data[SlideKey.SLIDE_ID][0][0]}.csv",
        )

    def __call__(self, dataloader: DataLoader, output_folder: Path) -> None:
        metadata_slide_keys = [SlideKey.SLIDE_ID, SlideKey.LABEL] + self.get_analysis_metadata_keys()
        metadata_dict: Dict[str, List] = {key: [] for key in metadata_slide_keys}

        for data in tqdm(dataloader, total=len(dataloader)):
            # store slide level data in slide-level CSV
            for key in metadata_slide_keys:  # type: ignore
                metadata_dict[key].append(get_value_from_listlike(data[key][0]))
            # store tile level data in tile-level CSVs
            self.tile_level_analysis(data, output_folder)

        logging.info(f"Saving slides analysis metadata to {output_folder}")
        self.save_analysis_metadata(metadata_dict, output_folder, "analysis_outputs_slides.csv")

    def get_analysis_metadata_tile_keys(self) -> List[AnalysisMetadata]:
        return [
            AnalysisMetadata.BROWN_PIXELS_TILES,
            AnalysisMetadata.FOREGROUND_PIXELS_TILES,
            AnalysisMetadata.BROWN_STAIN_RATIO_TILES,
        ]

    def analyse_dataset(self, input_folders: List[Path], output_folder: Path) -> None:
        """Analyse a dataset and save the results in tiles and slides csv files.

        :param input_folders: The input folders containing the dataset to analyse.
        :param output_folder: The output folder to save the csv file to.
        """
        dataset = self.get_slides_dataset(input_folders[0])
        transformed_dataset = Dataset(dataset, self.get_transforms())  # type: ignore
        dataloader = DataLoader(transformed_dataset, num_workers=self.num_workers, batch_size=1)
        metadata_keys = [meta_key.value for meta_key in self.get_analysis_metadata_keys()]
        metadata_keys += [meta_key.value for meta_key in self.get_analysis_metadata_tile_keys()]
        with logging_section(f"Analysis of dataset {input_folders[0]} using metadata keys {', '.join(metadata_keys)}"):
            self(dataloader, output_folder)
