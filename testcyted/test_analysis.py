import numpy as np
import pandas as pd
from pathlib import Path

import pytest
from cyted.analysis.analysis_configs import BrownStainSlidesConfig, BrownStainTilesConfig, TilesCountConfig
from cyted.analysis.slide_analysis_transforms import AnalysisMetadata, BrownStainRatiod
from cyted.cyted_schema import CytedSchema
from cyted.data_paths import (
    CYTED_DATASET_TSV,
    CYTED_DATASET_ID,
    CYTED_DEFAULT_DATASET_LOCATION,
    CYTED_EXCLUSION_LIST_CSV,
    get_cyted_dataset_dir,
)
from health_cpath.utils.naming import SlideKey, TileKey
from testcyted.utils_for_tests import skipif_dataset_unavailable, skipif_no_gpu


def test_get_hem_binary() -> None:
    transform = BrownStainRatiod()
    wsi_hed = np.zeros((5, 5, 3))
    wsi_hed[1:3, 1:3] = 1
    hem_binary = transform.get_hem_binary(wsi_hed)
    assert np.all(hem_binary == (1 - wsi_hed[..., 2]))


def test_get_masked_dab_values() -> None:
    transform = BrownStainRatiod()
    wsi_hed = np.ones((5, 5, 3))
    wsi_hed[1:3, 1:3, 2] = -5
    hem_binary = np.ones((5, 5))
    hem_binary[1:3, 1:3] = 0
    dab_values = transform.get_masked_dab_values(wsi_hed, hem_binary)
    assert dab_values.shape == (4,)
    assert np.all(dab_values == 5)


def test_brown_stain_ratio_transform() -> None:
    transform = BrownStainRatiod()
    binary_image = np.zeros((5, 5))
    binary_image[1:3, 1:3] = 1
    dab_binary = np.ones((5, 5))
    dab_binary[1, 1] = 0
    brown_stain_ratio, f_pixels, brown_pixels = transform.get_brown_stain_ratio(binary_image, dab_binary)
    assert brown_stain_ratio == 0.25
    assert f_pixels == 4
    assert brown_pixels == 1


@pytest.mark.gpu
@skipif_no_gpu()  # This test does not need a GPU, but needs a lot of memory and fails on the Github agents
@skipif_dataset_unavailable(CYTED_DEFAULT_DATASET_LOCATION, CYTED_DATASET_ID[CytedSchema.TFF3Image])
def test_brown_stain_ratio_slides_config(tmp_path: Path) -> None:
    image_col = CytedSchema.TFF3Image
    dataset_csv = CYTED_DATASET_TSV
    data_root = get_cyted_dataset_dir(image_col)
    limit = 1
    config = BrownStainSlidesConfig(
        dataset=str(data_root),
        dataset_csv=dataset_csv,
        image_column=image_col,
        label_column=CytedSchema.TFF3Positive,
        num_workers=1,
        level=0,
        limit=limit,
        background_val=255,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[image_col],
    )
    assert config.get_analysis_metadata_keys() == [
        AnalysisMetadata.BROWN_STAIN_RATIO,
        *AnalysisMetadata.get_pixel_count_keys(),
        *AnalysisMetadata.get_otsu_keys(),
        AnalysisMetadata.TILES_COUNT,
    ]
    config.analyse_dataset([data_root], tmp_path)
    assert (tmp_path / "analysis_outputs.csv").exists()
    df = pd.read_csv(tmp_path / "analysis_outputs.csv")
    assert len(df) == limit
    assert df[AnalysisMetadata.TILES_COUNT].to_list() == [4315]
    assert df[AnalysisMetadata.BROWN_STAIN_RATIO].to_list() == [0.1595081156159299]
    assert df.columns.tolist() == config.get_analysis_metadata_keys() + [SlideKey.LABEL, SlideKey.SLIDE_ID]


@pytest.mark.gpu
@skipif_no_gpu()  # This test does not need a GPU, but needs a lot of memory and fails on the Github agents
@skipif_dataset_unavailable(CYTED_DEFAULT_DATASET_LOCATION, CYTED_DATASET_ID[CytedSchema.TFF3Image])
def test_tiles_count_config(tmp_path: Path) -> None:
    image_col = CytedSchema.TFF3Image
    dataset_csv = CYTED_DATASET_TSV
    data_root = get_cyted_dataset_dir(image_col)
    limit = 1
    config = TilesCountConfig(
        dataset=str(data_root),
        dataset_csv=dataset_csv,
        image_column=image_col,
        label_column=CytedSchema.TFF3Positive,
        num_workers=1,
        level=0,
        limit=limit,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[image_col],
    )
    assert config.get_analysis_metadata_keys() == [AnalysisMetadata.TILES_COUNT]
    config.analyse_dataset([data_root], tmp_path)
    assert (tmp_path / "analysis_outputs.csv").exists()
    df = pd.read_csv(tmp_path / "analysis_outputs.csv")
    assert len(df) == limit
    assert df[AnalysisMetadata.TILES_COUNT].to_list() == [4315]
    assert df.columns.tolist() == config.get_analysis_metadata_keys() + [SlideKey.LABEL, SlideKey.SLIDE_ID]


@pytest.mark.gpu
@skipif_no_gpu()  # This test does not need a GPU, but needs a lot of memory and fails on the Github agents
@skipif_dataset_unavailable(CYTED_DEFAULT_DATASET_LOCATION, CYTED_DATASET_ID[CytedSchema.TFF3Image])
def test_brown_stain_ratio_tiles_config(tmp_path: Path) -> None:
    image_col = CytedSchema.TFF3Image
    dataset_csv = CYTED_DATASET_TSV
    data_root = get_cyted_dataset_dir(image_col)
    limit = 1
    config = BrownStainTilesConfig(
        dataset=str(data_root),
        dataset_csv=dataset_csv,
        image_column=image_col,
        label_column=CytedSchema.TFF3Positive,
        tile_size=224,
        num_workers=1,
        level=0,
        limit=limit,
        background_val=255,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[image_col],
    )
    assert config.get_analysis_metadata_keys() == [
        AnalysisMetadata.BROWN_STAIN_RATIO,
        *AnalysisMetadata.get_pixel_count_keys(),
        *AnalysisMetadata.get_otsu_keys(),
        AnalysisMetadata.TILES_COUNT,
    ]
    assert config.get_analysis_metadata_tile_keys() == [
        AnalysisMetadata.BROWN_PIXELS_TILES,
        AnalysisMetadata.FOREGROUND_PIXELS_TILES,
        AnalysisMetadata.BROWN_STAIN_RATIO_TILES,
    ]
    config.analyse_dataset([data_root], tmp_path)

    assert (tmp_path / "analysis_outputs_slides.csv").exists()
    df_slides = pd.read_csv(tmp_path / "analysis_outputs_slides.csv")
    assert len(df_slides) == limit
    assert df_slides[AnalysisMetadata.TILES_COUNT].to_list() == [4315]
    assert df_slides[AnalysisMetadata.BROWN_STAIN_RATIO].to_list() == [0.1595081156159299]
    assert df_slides.columns.tolist() == [SlideKey.SLIDE_ID, SlideKey.LABEL] + config.get_analysis_metadata_keys()

    tile_csv_path = tmp_path / f"analysis_outputs_tiles_{df_slides[SlideKey.SLIDE_ID][0]}.csv"
    assert tile_csv_path.exists()
    df_tiles = pd.read_csv(tile_csv_path)
    assert len(df_tiles) == 4315
    assert (
        df_tiles.columns.tolist()
        == [
            TileKey.TILE_ID,
            TileKey.TILE_LEFT,
            TileKey.TILE_RIGHT,
            TileKey.TILE_TOP,
            TileKey.TILE_BOTTOM,
        ]
        + config.get_analysis_metadata_tile_keys()
    )
    assert df_tiles[AnalysisMetadata.BROWN_STAIN_RATIO_TILES].to_list()[0:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
