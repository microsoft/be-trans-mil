#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import shutil
from pathlib import Path
from typing import Dict, Generator, Tuple

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from monai.data.wsi_reader import WSIReader

from health_cpath.preprocessing.loading import WSIBackend
from health_cpath.preprocessing.tiff_conversion import ResolutionUnit
from testhisto.mocks.base_data_generator import MockHistoDataType
from testhisto.mocks.slides_generator import MockPandaSlidesGenerator, TilesPositioningType
from testhisto.utils.utils_testhisto import assert_binary_files_match

from cyted.fixed_paths import repository_root_directory
from cyted.utils.viz_utils import format_rounded_length, load_thumbnail, load_tile, mpp_to_power, plot_multi_scale_tiles


def expected_results_folder() -> Path:
    """Gets the path to the folder where the expected results are stored.

    :return: The path to the folder where the expected results are stored.
    """
    return repository_root_directory() / "testcyted" / "testdata" / "plots"


def test_format_rounded_length() -> None:
    cases: Dict[float, Tuple[float, str]] = {
        1.0: (1.0, "1µm"),
        2.0: (2.0, "2µm"),
        4.0: (2.0, "2µm"),
        8.0: (5.0, "5µm"),
        16.0: (10.0, "10µm"),
        32.0: (20.0, "20µm"),
        64.0: (50.0, "50µm"),
        128.0: (100.0, "100µm"),
        256.0: (200.0, "200µm"),
        512.0: (500.0, "500µm"),
        1024.0: (1000.0, "1mm"),
        2048.0: (2000.0, "2mm"),
        4096.0: (2000.0, "2mm"),
        8192.0: (5000.0, "5mm"),
        16384.0: (10000.0, "10mm"),
    }
    for raw_value, (expected_rounded_value, expected_label) in cases.items():
        rounded_value, label = format_rounded_length(raw_value)
        assert rounded_value <= raw_value
        assert rounded_value == expected_rounded_value
        assert label == expected_label


def test_mpp_to_power() -> None:
    cases: Dict[float, float] = {
        0.12: 80.0,
        0.23: 40.0,
        0.25: 40.0,
        0.30: 40.0,
        0.46: 20.0,
        0.92: 10.0,
        1.00: 10.0,
        1.84: 5.0,
        2.00: 5.0,
        15.0: 0.625,
    }
    for mpp, expected_power in cases.items():
        power = mpp_to_power(mpp)
        assert power == expected_power


_TILE_SIZE = 28
_N_LEVELS = 4
_N_TILES = 2**_N_LEVELS
_WSI_SIZE = _TILE_SIZE * _N_TILES


@pytest.fixture(scope="session")
def wsi_path(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    tmp_root_dir = tmp_path_factory.mktemp("mock_slides")
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_root_dir,
        mock_type=MockHistoDataType.FAKE,
        tiles_pos_type=TilesPositioningType.RANDOM,
        resultion_unit=ResolutionUnit.MICROMETER,
        n_repeat_diag=_N_TILES,
        n_repeat_tile=1,
        n_tiles=_N_TILES**2 // 2,
        n_slides=6,
        n_channels=3,
        n_levels=_N_LEVELS,
        tile_size=_TILE_SIZE,
        background_val=255,
    )
    wsi_generator.generate_mock_histo_data()
    img_path = wsi_generator.dest_data_path / "train_images/_0.tiff"
    assert img_path.is_file()
    yield img_path
    shutil.rmtree(tmp_root_dir)


def test_load_thumbnail(wsi_path: Path) -> None:
    reader = WSIReader(WSIBackend.CUCIM)
    slide = reader.read(str(wsi_path))

    thumb = load_thumbnail(reader, slide)
    assert isinstance(thumb, np.ndarray)

    scale = 2 ** (_N_LEVELS - 1)
    expected_size = _WSI_SIZE // scale
    assert thumb.shape == (3, expected_size, expected_size)


def test_load_tile(wsi_path: Path) -> None:
    reader = WSIReader(WSIBackend.CUCIM)
    slide = reader.read(str(wsi_path))
    tile_size = 10
    xy = (_WSI_SIZE // 2, _WSI_SIZE // 2)

    for level in range(_N_LEVELS):
        tile = load_tile(reader, slide, level=level, xy=xy, tile_size=tile_size)
        assert isinstance(tile, np.ndarray)
        assert tile.shape == (3, tile_size, tile_size)


def test_plot_multi_scale_tiles(wsi_path: Path, tmp_path: Path) -> None:
    reader = WSIReader(WSIBackend.CUCIM)
    slide = reader.read(str(wsi_path))
    tile_size = _TILE_SIZE
    xy = (_WSI_SIZE // 2, _WSI_SIZE // 2)

    levels = list(range(_N_LEVELS))

    fig, axs = plot_multi_scale_tiles(reader, slide, xy=xy, levels=levels, tile_size=tile_size)
    assert isinstance(fig, Figure)
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (2, len(levels))
    assert isinstance(axs[0, 0], Axes)

    fig_filename = "multi_scale_tiles.png"
    fig_path = tmp_path / fig_filename
    fig.savefig(str(fig_path), bbox_inches="tight")

    expected_fig_path = expected_results_folder() / fig_filename
    # To update the reference image, uncomment the line below:
    # fig.savefig(str(expected_fig_path), bbox_inches='tight')

    assert_binary_files_match(fig_path, expected_fig_path)
