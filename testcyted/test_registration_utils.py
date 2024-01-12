#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
from cyted.cyted_schema import CytedLabel, CytedSchema
from cyted.data_paths import CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID
from cyted.utils.coco_utils import CocoSchema
import SimpleITK as sitk

from cyted.utils.registration_utils import (
    get_background_colour,
    get_cropped_images,
    register_crops,
    join_hardcode_background,
    find_reference_image_and_label_from_csv,
    find_reference_he_images_from_folder,
)
from testcyted.utils_for_tests import skipif_dataset_unavailable

# TODO: Tests for plots (used in registration.ipynb notebook)


def test_get_background_colour() -> None:
    rgb_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    image = rgb_image.transpose(1, 2, 0)
    bg_colour = get_background_colour(image, q=0.8)
    assert bg_colour.all() <= 255
    assert bg_colour.all() >= 0
    assert bg_colour.shape == (3,)


def test_get_cropped_images() -> None:
    image_array = np.ones((3, 1000, 1000), dtype="uint8") * 255
    image_array[:, 100:300, 100:300] = 128
    image_array[:, 500:600, 500:600] = 0
    dummy_path = Path("x/y.tiff")

    bounding_box_data = {
        CocoSchema.IMAGES: [
            {
                CocoSchema.ID: 1,
                CocoSchema.WIDTH: image_array.shape[2],
                CocoSchema.HEIGHT: image_array.shape[1],
                CocoSchema.FILENAME: dummy_path.name,
            }
        ],
        CocoSchema.ANNOTATIONS: [
            {
                CocoSchema.ID: 1,
                CocoSchema.CATEGORY_ID: 1,
                CocoSchema.IMAGE_ID: 1,
                CocoSchema.AREA: 0.15,
                CocoSchema.BBOX: [0.05, 0.09, 0.19, 0.82],
            },
            {
                CocoSchema.ID: 2,
                CocoSchema.CATEGORY_ID: 1,
                CocoSchema.IMAGE_ID: 1,
                CocoSchema.AREA: 0.20,
                CocoSchema.BBOX: [0.4, 0.3, 0.56, 0.24],
            },
        ],
    }
    image_df = pd.DataFrame(bounding_box_data[CocoSchema.IMAGES])
    bb_df = pd.DataFrame(bounding_box_data[CocoSchema.ANNOTATIONS])

    crops, bg_color = get_cropped_images(image_path=dummy_path, image_array=image_array, image_df=image_df, bb_df=bb_df)

    assert len(crops) == 2
    assert len(bg_color) == 3
    assert 0 <= bg_color.all() <= 255


def test_register_crops() -> None:
    dummy_array1 = np.ones((3, 200, 200), dtype="uint8") * 255
    dummy_array2 = np.ones((3, 250, 250), dtype="uint8") * 255
    dummy_array1[:, 40:100, 40:100] = 128
    dummy_array2[:, 100:160, 100:160] = 128
    dummy_crops_he = [dummy_array1, dummy_array2]
    dummy_crops_tff3 = [dummy_array2, dummy_array1]
    dummy_path = Path("x/y.tiff")
    dummy_bg_color = np.array([255.0, 255.0, 255.0])

    tff3_crops_registered, transform_crops, background_tff3, initial_metric_value, final_metric_value = register_crops(
        he_crops=dummy_crops_he,
        tff3_crops=dummy_crops_tff3,
        he_mask_crops_resized=[],
        he_bg_color=dummy_bg_color,
        tff3_bg_color=dummy_bg_color,
        tff3_image_path=dummy_path,
    )

    assert len(dummy_crops_he) == len(dummy_crops_tff3) == len(tff3_crops_registered) == 2

    # resulting transformed image should be the same size as the original image
    for i in range(len(dummy_crops_he)):
        assert dummy_crops_he[i].shape == tff3_crops_registered[i].shape

    assert len(transform_crops) == 2
    for i in range(len(transform_crops)):
        assert isinstance(transform_crops[i], sitk.Transform)

    assert len(background_tff3) == 3
    assert isinstance(background_tff3, tuple)
    assert tuple(dummy_bg_color.astype(int).squeeze()) == background_tff3

    assert initial_metric_value >= 0
    assert final_metric_value >= 0


def test_join_hardcode_background() -> None:
    dummy_array1 = np.ones((3, 200, 200), dtype="uint8") * 255
    dummy_array2 = np.ones((3, 250, 250), dtype="uint8") * 255
    dummy_array1[:, 40:100, 40:100] = 128
    dummy_array2[:, 100:160, 100:160] = 128
    dummy_crops_tff3 = [dummy_array1, dummy_array2]
    dummy_bg = (255, 255, 255)

    cropped_slide, tff3_mask_image = join_hardcode_background(
        tff3_crops_registered=dummy_crops_tff3,
        background_tff3=dummy_bg,
        target_magnification=2.5,
        hardcode_background=True,
    )

    assert cropped_slide.transpose(2, 0, 1).shape[0] == dummy_array1.shape[0]
    assert cropped_slide.transpose(2, 0, 1).shape[1] == max(dummy_array1.shape[1], dummy_array2.shape[1])
    assert cropped_slide.transpose(2, 0, 1).shape[2] >= dummy_array1.shape[2] + dummy_array2.shape[2]

    assert np.array(tff3_mask_image.convert("RGB")).shape == cropped_slide.shape


def test_find_reference_image_and_label_from_csv(tmp_path: Path) -> None:
    data_file_snippet = (
        "CYT full ID	TFF3 positive	P53 positive	Atypia	H&E\n"
        "21CYT03122	Y	N	N	21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi\n"
        "22CYT01935	Y	Y	N	22CYT01935 20P01895 A1 HE - 2022-04-06 12.29.55.ndpi\n"
        "22CYT02716	N	NA	N	22CYT02716 A1 HE 22P00480 - 2022-05-12 14.54.40.ndpi\n"
    )

    dataset_tsv = tmp_path / "dataset.tsv"
    dataset_tsv.write_text(data_file_snippet)
    src_path, src_label = find_reference_image_and_label_from_csv(
        dataset_csv_path=dataset_tsv,
        src_id="21CYT03122",
        image_column=CytedSchema.HEImage,
        label_column=CytedSchema.TFF3Positive,
    )
    assert src_path == "21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi"
    assert src_label == CytedLabel.Yes


@skipif_dataset_unavailable(CYTED_RAW_DATASET_DIR, CYTED_RAW_DATASET_ID)
def test_find_reference_he_images_from_folder() -> None:
    src_id = "21CYT03122"
    src_path = find_reference_he_images_from_folder(fixed_dataset_path=Path(CYTED_RAW_DATASET_DIR), src_id=src_id)
    assert src_path[0] == "21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi"
