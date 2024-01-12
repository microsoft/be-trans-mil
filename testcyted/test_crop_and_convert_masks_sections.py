#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import os
from pathlib import Path

import numpy as np
from PIL import Image
from cyted.cyted_schema import CytedSchema
from cyted.data_paths import (
    CYTED_BOUNDING_BOXES_JSON,
    CYTED_HE_HISTOQC_OUTPUTS_DATASET_DIR,
    CYTED_HE_HISTOQC_OUTPUTS_DATASET_ID,
)
from cyted.preproc.crop_and_convert_binary_masks_to_lr_sections import (
    generate_cropped_section_masks,
    sort_bboxes_left_to_right,
)
from cyted.utils.azure_utils import replace_ampersand_in_filename
from health_ml.utils.box_utils import Box
from testcyted.utils_for_tests import skipif_dataset_unavailable


@skipif_dataset_unavailable(CYTED_HE_HISTOQC_OUTPUTS_DATASET_DIR, CYTED_HE_HISTOQC_OUTPUTS_DATASET_ID)
def test_generate_cropped_section_masks(tmp_path: Path) -> None:
    source_magnification = 1.25
    target_magnification = 1.25
    limit = 2

    generate_cropped_section_masks(
        src_dir=Path(CYTED_HE_HISTOQC_OUTPUTS_DATASET_DIR),
        dest_dir=tmp_path,
        bounding_box_path=CYTED_BOUNDING_BOXES_JSON[CytedSchema.HEImage],
        source_magnification=source_magnification,
        target_magnification=target_magnification,
        limit=limit,
    )

    # Test the number and name of masks in the limit slides
    mask_files = [name for name in os.listdir(tmp_path)]
    assert len(mask_files) == limit

    expected_mask_names = []
    expected_input_masks = os.listdir(CYTED_HE_HISTOQC_OUTPUTS_DATASET_DIR)[:limit]
    for slide_name in expected_input_masks:
        expected_mask_names.append(replace_ampersand_in_filename(slide_name) + "_mask_use.png")

    assert set(expected_mask_names) == set(mask_files)
    assert len(expected_mask_names) == len(mask_files)

    for mask_path in mask_files:
        mask_image = np.array(Image.open(tmp_path / mask_path))

        # Test if mask files are ternary
        assert np.unique(mask_image.all()) in [0, 128, 255]

        # Test if left section is 128 and right section is 255
        x_128 = np.where(mask_image[:, :, 0] == 128)[1]  # find xs where the 2D image is 128: left section
        x_255 = np.where(mask_image[:, :, 0] == 255)[1]  # find xs where the 2D image is 255: right section
        assert x_128.all() < x_255.all()


def test_sort_bboxes_left_to_right() -> None:
    list_unsorted = [Box(1, 2, 3, 4), Box(9, 4, 5, 6), Box(0, 1, 2, 3), Box(2, 4, 6, 1)]
    expected_list_sorted = [Box(0, 1, 2, 3), Box(1, 2, 3, 4), Box(2, 4, 6, 1), Box(9, 4, 5, 6)]
    assert expected_list_sorted == sort_bboxes_left_to_right(list_unsorted)
