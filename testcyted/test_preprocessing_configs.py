#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import os
import pickle
import pytest
from pathlib import Path
import SimpleITK as sitk
import pandas as pd
from cyted.utils.registration_utils import RegistrationMetrics

from health_cpath.preprocessing.loading import WSIBackend
from health_cpath.utils.naming import SlideKey

from cyted.cyted_schema import CytedSchema
from cyted.datasets.cyted_slides_dataset import CytedSlidesDataset
from cyted.data_paths import (
    CYTED_BOUNDING_BOXES_JSON,
    CYTED_DATASET_TSV,
    CYTED_EXCLUSION_LIST_CSV,
    CYTED_RAW_DATASET_DIR,
    CYTED_RAW_DATASET_ID,
    CYTED_CSV_FILENAME,
    get_cyted_mask_dataset_dir,
)

from cyted.utils.preprocessing_configs import (
    CytedCroppingAndTiffConversionConfig,
    CytedTiffConversionConfig,
    CytedRegistrationConfig,
)
from testcyted.utils_for_tests import skipif_dataset_unavailable, skipif_no_gpu
from testhisto.preprocessing.test_tiff_conversion import validate_tiff_conversion


@pytest.mark.gpu
@skipif_no_gpu()  # This test does not need a GPU, but needs a lot of memory and fails on the Github agents
@skipif_dataset_unavailable(CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID)
@pytest.mark.parametrize(
    "image_col, label_col",
    [
        (CytedSchema.HEImage, CytedSchema.TFF3Positive),
        (CytedSchema.TFF3Image, CytedSchema.TFF3Positive),
        (CytedSchema.P53Image, CytedSchema.P53Positive),
    ],
)
def test_convert_cyted_ndpi_to_tiff(image_col: str, label_col: str, tmp_path: Path) -> None:
    target_magnification = 1.25  # 32x downsampled -> 2^5 for level 5
    limit = 2
    dataset_csv = CYTED_CSV_FILENAME
    data_root = CYTED_RAW_DATASET_DIR
    converted_dataset_csv = CYTED_DATASET_TSV
    conversion_config = CytedTiffConversionConfig(
        output_dataset=str(tmp_path),
        converted_dataset_csv=converted_dataset_csv,
        dataset=data_root,
        dataset_csv=dataset_csv,
        image_column=image_col,
        label_column=label_col,
        target_magnifications=[target_magnification],
        add_lowest_magnification=True,
        num_workers=1,
        limit=limit,
    )
    dataset = conversion_config.get_slides_dataset(dataset_root=Path(data_root))
    conversion_config.run(dataset=dataset, output_folder=tmp_path)

    # Test that the original dataset is not modified
    original_dataset = CytedSlidesDataset(
        root=data_root, dataset_csv=Path(data_root) / dataset_csv, image_column=image_col, label_column=label_col
    )
    assert original_dataset.dataset_df.iloc[:limit].equals(dataset.dataset_df)

    # Validate that the new dataset has the correct number of rows
    new_dataset = CytedSlidesDataset(
        root=tmp_path, dataset_csv=tmp_path / converted_dataset_csv, image_column=image_col, label_column=label_col
    )
    assert new_dataset.dataset_df.shape[0] == dataset.dataset_df.shape[0]
    assert new_dataset.dataset_df.index.equals(dataset.dataset_df.index)

    original_files = [Path(dataset[i][SlideKey.IMAGE]) for i in range(limit)]
    converted_files = [Path(new_dataset[i][SlideKey.IMAGE]) for i in range(limit)]
    transform = conversion_config.get_transform(tmp_path)
    # Test that the converted files are valid using OpenSlide and CuCIM
    validate_tiff_conversion(converted_files, original_files, transform, same_format=False, backend=WSIBackend.CUCIM)
    validate_tiff_conversion(
        converted_files, original_files, transform, same_format=False, backend=WSIBackend.OPENSLIDE
    )


@pytest.mark.gpu
@skipif_no_gpu()  # This test does not need a GPU, but needs a lot of memory and fails on the Github agents
@skipif_dataset_unavailable(CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID)
@pytest.mark.parametrize(
    "image_col, label_col",
    [
        (CytedSchema.HEImage, CytedSchema.TFF3Positive),
        (CytedSchema.TFF3Image, CytedSchema.TFF3Positive),
    ],
)
def test_crop_and_convert_cyted_ndpi_to_tiff(image_col: str, label_col: str, tmp_path: Path) -> None:
    target_magnification = 1.25  # 32x downsampled -> 2^5 for level 5
    limit = 2
    dataset_csv = CYTED_CSV_FILENAME
    data_root = CYTED_RAW_DATASET_DIR
    bbox_path = CYTED_BOUNDING_BOXES_JSON[image_col]
    mask_dataset = get_cyted_mask_dataset_dir(image_col) if image_col == CytedSchema.HEImage else None
    converted_dataset_csv = CYTED_DATASET_TSV
    conversion_config = CytedCroppingAndTiffConversionConfig(
        output_dataset=str(tmp_path),
        output_mask_dataset=str(tmp_path),
        converted_dataset_csv=converted_dataset_csv,
        dataset=data_root,
        dataset_csv=dataset_csv,
        image_column=image_col,
        label_column=label_col,
        bounding_box_path=bbox_path,
        mask_dataset=mask_dataset,
        target_magnifications=[target_magnification],
        add_lowest_magnification=False,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[image_col],
        num_workers=1,
        limit=limit,
        automatic_output_name=False,
    )
    dataset = conversion_config.get_slides_dataset(dataset_root=Path(data_root))
    conversion_config.run(dataset=dataset, output_folder=tmp_path)

    original_dataset = CytedSlidesDataset(
        root=data_root,
        dataset_csv=Path(data_root) / dataset_csv,
        image_column=image_col,
        label_column=label_col,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[image_col],
    )
    assert original_dataset.dataset_df.iloc[:limit].index.equals(dataset.dataset_df.index)

    # Validate that the new dataset has the correct number of rows
    new_dataset = CytedSlidesDataset(
        root=tmp_path,
        dataset_csv=tmp_path / converted_dataset_csv,
        image_column=image_col,
        label_column=label_col,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[image_col],
    )
    assert new_dataset.dataset_df.shape[0] == dataset.dataset_df.shape[0]
    assert new_dataset.dataset_df.index.equals(dataset.dataset_df.index)

    original_files = [Path(dataset[i][SlideKey.IMAGE]) for i in range(limit)]
    converted_files = [Path(new_dataset[i][SlideKey.IMAGE]) for i in range(limit)]
    transform = conversion_config.get_transform(tmp_path)
    # Test that the converted files are valid using OpenSlide and CuCIM
    # The data is not expected to be equal because the cropping
    validate_tiff_conversion(
        converted_files,
        original_files,
        transform,
        same_format=False,
        backend=WSIBackend.CUCIM,
        data_is_equal=False,
    )
    validate_tiff_conversion(
        converted_files, original_files, transform, same_format=False, backend=WSIBackend.OPENSLIDE, data_is_equal=False
    )
    # Tests if output masks are created and have size > 0B
    if image_col == CytedSchema.TFF3Image:
        output_mask_dataset = str(tmp_path)
        assert len(os.listdir(output_mask_dataset)) == 5
        for mask_file in os.listdir(output_mask_dataset):
            if mask_file.endswith(".png"):
                assert (tmp_path / mask_file).stat().st_size > 0


@pytest.mark.gpu
@skipif_no_gpu()  # This test does not need a GPU, but needs a lot of memory and fails on the Github agents
@skipif_dataset_unavailable(CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID)
def test_register_crop_convert(tmp_path: Path) -> None:
    target_magnification = 1.25  # 32x downsampled -> 2^5 for level 5
    limit = 2
    dataset_csv = CYTED_CSV_FILENAME
    data_root = CYTED_RAW_DATASET_DIR
    bbox_path = CYTED_BOUNDING_BOXES_JSON[CytedSchema.TFF3Image]
    bbox_path_fixed = CYTED_BOUNDING_BOXES_JSON[CytedSchema.HEImage]
    fixed_mask_dataset = get_cyted_mask_dataset_dir(CytedSchema.HEImage)
    converted_dataset_csv = CYTED_DATASET_TSV

    conversion_config = CytedRegistrationConfig(
        fixed_dataset=Path(data_root),
        output_dataset=str(tmp_path),
        converted_dataset_csv=converted_dataset_csv,
        output_transforms_dataset=str(tmp_path),
        output_diff_dataset=str(tmp_path),
        dataset=data_root,
        dataset_csv=dataset_csv,
        image_column=CytedSchema.TFF3Image,
        bounding_box_path=Path(bbox_path),
        bounding_box_path_fixed=Path(bbox_path_fixed),
        fixed_mask_dataset=Path(fixed_mask_dataset),
        target_magnifications=[target_magnification],
        add_lowest_magnification=False,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[CytedSchema.TFF3Image],
        num_workers=1,
        limit=limit,
        hardcode_background=False,
        automatic_output_name=False,
    )
    dataset = conversion_config.get_slides_dataset(dataset_root=Path(data_root))
    conversion_config.run(dataset=dataset, output_folder=tmp_path)

    original_dataset = CytedSlidesDataset(
        root=data_root,
        dataset_csv=Path(data_root) / dataset_csv,
        image_column=CytedSchema.TFF3Image,
        label_column=CytedSchema.TFF3Positive,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[CytedSchema.TFF3Image],
    )
    assert original_dataset.dataset_df.iloc[:limit].index.equals(dataset.dataset_df.index)

    # Validate that the new dataset has the correct number of rows
    new_dataset = CytedSlidesDataset(
        root=tmp_path,
        dataset_csv=tmp_path / converted_dataset_csv,
        image_column=CytedSchema.TFF3Image,
        label_column=CytedSchema.TFF3Positive,
        excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[CytedSchema.TFF3Image],
    )
    assert new_dataset.dataset_df.shape[0] == dataset.dataset_df.shape[0]
    assert new_dataset.dataset_df.index.equals(dataset.dataset_df.index)

    original_files = [Path(dataset[i][SlideKey.IMAGE]) for i in range(limit)]
    converted_files = [Path(new_dataset[i][SlideKey.IMAGE]) for i in range(limit)]
    transform = conversion_config.get_transform(tmp_path)
    # Test that the converted files are valid using OpenSlide and CuCIM
    # The data is not expected to be equal because the registration and cropping
    validate_tiff_conversion(
        converted_files,
        original_files,
        transform,
        same_format=False,
        backend=WSIBackend.CUCIM,
        data_is_equal=False,
    )
    validate_tiff_conversion(
        converted_files, original_files, transform, same_format=False, backend=WSIBackend.OPENSLIDE, data_is_equal=False
    )
    # Tests if output diff and transforms are created and have size > 0B
    outputs_folder = str(tmp_path)
    assert len(os.listdir(outputs_folder)) == 7
    for file in os.listdir(outputs_folder):
        if file.endswith(".png") or file.endswith(".pickle"):
            assert (tmp_path / file).stat().st_size > 0

    # Check contents of output transform file
    for file in os.listdir(outputs_folder):
        if file.endswith(".pickle"):
            with open(tmp_path / file, "rb") as handle:
                transforms_dict = pickle.load(handle)
            assert len(transforms_dict.keys()) == 4
            assert transforms_dict["magnification"] == 1.25
            assert transforms_dict[RegistrationMetrics.MI_BEFORE] >= 0.0
            assert transforms_dict[RegistrationMetrics.MI_AFTER] >= 0.0
            for key in transforms_dict.keys():
                if "TFF3" in key:
                    assert len(transforms_dict[key]) == 2  # two transforms for two crops
                    assert isinstance(transforms_dict[key][0], sitk.Transform)
        if file.endswith(".tsv") or file.endswith(".csv"):
            output_df = pd.read_csv(tmp_path / file, sep="\t")
            assert RegistrationMetrics.MI_BEFORE in list(output_df.columns)
            assert RegistrationMetrics.MI_AFTER in list(output_df.columns)
            assert isinstance(output_df[RegistrationMetrics.MI_BEFORE][0], float)
            assert isinstance(output_df[RegistrationMetrics.MI_AFTER][0], float)
