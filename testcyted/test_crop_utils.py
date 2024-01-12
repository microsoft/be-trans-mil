#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pathlib import Path
from cyted.utils.coco_utils import CocoSchema
from health_ml.utils.box_utils import Box

from cyted.cyted_schema import CytedSchema
from cyted.data_paths import CYTED_BOUNDING_BOXES_JSON
from cyted.utils.crop_utils import (
    crop_inclusion_boxes,
    get_absolute_bounding_boxes,
    get_exclusion_and_inclusion_bounding_boxes,
    join_images,
    preprocess_bounding_boxes,
    replace_exclusion_boxes_with_background,
    hardcode_background_from_mask,
    resize_and_convert_mask,
    resize_image_array,
    resize_image_array_to_scale_factor,
    segment_and_post_process,
    sort_bboxes_left_to_right,
    create_new_slide_from_given_mask,
    create_new_slide_by_generating_mask,
    segment_foreground_quantile,
)


def test_get_exclusion_and_inclusion_bounding_boxes() -> None:
    images_df, bboxes_df = preprocess_bounding_boxes(CYTED_BOUNDING_BOXES_JSON[CytedSchema.TFF3Image])

    # Slide with known 1 exclusion and 2 inclusion boxes
    slide_filename = "21CYT00291 20P01898 TFF3 A1 - 2021-02-23 12.36.18.ndpi"
    width, height = 3180, 1332
    exclusion_bounding_boxes, inclusion_bounding_boxes = get_exclusion_and_inclusion_bounding_boxes(
        image_path=Path(slide_filename),
        image_df=images_df,
        bounding_box_df=bboxes_df,
        image_width=width,
        image_height=height,
    )
    assert len(exclusion_bounding_boxes) == 1
    assert len(inclusion_bounding_boxes) == 2

    # Slide with known 0 exclusion and 2 inclusion boxes
    slide_filename = "22CYT03035 A1 TFF3 22P02056 - 2022-05-26 13.14.01.ndpi"
    width, height = 2820, 936  # at 0.625x, from AML COCO JSON
    exclusion_bounding_boxes, inclusion_bounding_boxes = get_exclusion_and_inclusion_bounding_boxes(
        image_path=Path(slide_filename),
        image_df=images_df,
        bounding_box_df=bboxes_df,
        image_width=width,
        image_height=height,
    )
    assert len(exclusion_bounding_boxes) == 0
    assert len(inclusion_bounding_boxes) == 2


@pytest.mark.parametrize("bounding_box_coordinates_list", [[[0.05, 0.4, 0.6, 0.7], [0.1, 0.2, 0.3, 0.4]]])
@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("height", [2, 3])
def test_get_absolute_bounding_boxes(bounding_box_coordinates_list: list, width: int, height: int) -> None:
    absolute_bounding_boxes = get_absolute_bounding_boxes(
        bounding_box_coordinates_list=bounding_box_coordinates_list, width=width, height=height
    )

    for abs_bbox, rel_coords in zip(absolute_bounding_boxes, bounding_box_coordinates_list):
        assert abs_bbox.x == round(rel_coords[0] * width)
        assert abs_bbox.y == round(rel_coords[1] * height)
        assert abs_bbox.w == round(rel_coords[2] * width)
        assert abs_bbox.h == round(rel_coords[3] * height)


def generate_dummy_array_with_boxes() -> tuple[np.ndarray, list[Box]]:
    dummy_image_array = np.ones((3, 6, 6)) * 255
    # 255, 255, 255, 255, 255, 255
    # 255, 255, 255, 255, 255, 255
    # 255, 255, 255, 255, 255, 255
    # 255, 255, 255, 255, 255, 255
    # 255, 255, 255, 255, 255, 255

    dummy_bounding_boxes = [Box(x=0, y=0, w=4, h=2), Box(x=2, y=3, w=3, h=3)]
    for box in dummy_bounding_boxes:
        y_slice, x_slice = box.to_slices()
        dummy_image_array[:, y_slice, x_slice] = 121

    # 121, 121, 121, 121, 255, 255
    # 121, 121, 121, 121, 255, 255
    # 255, 255, 255, 255, 255, 255
    # 255, 255, 121, 121, 121, 255
    # 255, 255, 121, 121, 121, 255
    # 255, 255, 121, 121, 121, 255

    return dummy_image_array, dummy_bounding_boxes


def test_replace_exclusion_boxes_with_background() -> None:
    dummy_image_array, dummy_bounding_boxes = generate_dummy_array_with_boxes()
    image_array, background = replace_exclusion_boxes_with_background(dummy_image_array, dummy_bounding_boxes)
    # We expect that all 121s are gone
    np.testing.assert_allclose(image_array, np.ones((3, 6, 6)) * 255)
    np.testing.assert_allclose(background, [255, 255, 255])


def test_crop_inclusion_boxes_and_join_images() -> None:
    background = (255, 255, 255)
    dummy_image_array, dummy_bounding_boxes = generate_dummy_array_with_boxes()
    inclusion_image_list = crop_inclusion_boxes(dummy_image_array, dummy_bounding_boxes)
    np.testing.assert_allclose(inclusion_image_list[0], np.ones((3, 2, 4)) * 121)
    np.testing.assert_allclose(inclusion_image_list[1], np.ones((3, 3, 3)) * 121)

    joined_image = join_images(inclusion_image_list, background)
    # 121, 121, 255
    # 121, 121, 255
    # 121, 121, 255
    # 121, 121, 255
    # 121, 121, 121
    # 121, 121, 121
    # 121, 121, 121
    expected_output = np.array(
        [
            [121, 121, 255],
            [121, 121, 255],
            [121, 121, 255],
            [121, 121, 255],
            [121, 121, 121],
            [121, 121, 121],
            [121, 121, 121],
        ]
    )
    expected_output = np.expand_dims(expected_output, 0)
    expected_output = np.tile(expected_output, (3, 1, 1))
    np.testing.assert_allclose(joined_image.transpose(2, 1, 0), expected_output)


def test_hardcode_background_from_mask() -> None:
    dummy_mask_array = np.zeros((4, 4))
    dummy_mask_array[1:2, 1:2] = 255
    dummy_mask_image = Image.fromarray(dummy_mask_array, mode="L")
    dummy_image_array = np.ones((3, 6, 6)) * 128
    channels, height, width = dummy_image_array.shape

    generated_masked_array = hardcode_background_from_mask(dummy_image_array, dummy_mask_image)
    assert np.max(generated_masked_array) == 255
    assert generated_masked_array.shape == dummy_image_array.shape

    expected_mask_array = resize_and_convert_mask(mask_image=dummy_mask_image, width=width, height=height)
    assert expected_mask_array.shape[0] == generated_masked_array.shape[1]
    assert expected_mask_array.shape[1] == generated_masked_array.shape[2]

    for i in range(channels):
        assert set((generated_masked_array[i][np.where(~expected_mask_array)])) == {255}


def test_sort_bboxes_left_to_right() -> None:
    list_unsorted = [Box(1, 2, 3, 4), Box(9, 4, 5, 6), Box(0, 1, 2, 3), Box(2, 4, 6, 1)]
    expected_list_sorted = [Box(0, 1, 2, 3), Box(1, 2, 3, 4), Box(2, 4, 6, 1), Box(9, 4, 5, 6)]
    assert expected_list_sorted == sort_bboxes_left_to_right(list_unsorted)


def test_resize_image_array_to_scale_factor() -> None:
    test_array = np.zeros((3, 16, 16), dtype="uint8")
    factor = 1/4
    resized_test_array = resize_image_array_to_scale_factor(test_array, scale_factor=factor)
    assert resized_test_array.shape == (int(test_array.shape[1] * factor), int(test_array.shape[2] * factor), 3)


def test_resize_image_array() -> None:
    test_array = np.zeros((3, 1000, 1000), dtype="uint8")
    source_magnification = 40.0
    target_magnification = 1.25
    factor = target_magnification / source_magnification
    resized_test_array = resize_image_array(test_array, source_magnification, target_magnification)
    assert resized_test_array.shape == (3, int(test_array.shape[1] * factor), int(test_array.shape[2] * factor))


def test_segment_and_post_process() -> None:
    test_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    image = test_image.transpose(1, 2, 0)
    mask = image.mean(2) < 200
    bg_colour = np.median(image[~mask], axis=0)
    test_list = [test_image, test_image]
    mask_list = segment_and_post_process(test_list, bg_colour, se1_size=2, se2_size=4)
    assert np.unique(mask_list[0].all()) in [0, 255]
    assert np.unique(mask_list[1].all()) in [0, 255]
    assert mask_list[0].shape == test_list[0].shape
    assert len(mask_list) == len(test_list)


def test_create_new_slide_from_given_mask() -> None:
    dummy_image_array, dummy_bounding_boxes = generate_dummy_array_with_boxes()
    dummy_mask_array = np.zeros((6, 6))
    dummy_mask_array[1:2, 1:2] = 255
    dummy_mask_image = Image.fromarray(dummy_mask_array, mode="L")
    dummy_result = create_new_slide_from_given_mask(
        image_array=dummy_image_array,
        mask_image=dummy_mask_image,
        inclusion_bboxes=dummy_bounding_boxes,
        image_path=Path(""),
    )
    background: np.ndarray = np.median(dummy_result, axis=(1, 2), keepdims=True)

    # check if background is hardcoded
    np.allclose(background, (255, 255, 255))

    # check if H of the result image are less or equal to H of image array, and C are equal
    assert dummy_result.transpose(2, 0, 1).shape[1] <= dummy_image_array.shape[1]
    assert dummy_result.transpose(2, 0, 1).shape[0] == dummy_image_array.shape[0]

    # check if the mask is binary
    assert np.unique(np.array(dummy_mask_image).astype(float)) in [0, 255]


def test_create_new_slide_by_generating_mask() -> None:
    image_array = np.ones((3, 1000, 1000), dtype="uint8") * 255
    image_array[:, 100:100, 100:100] = 128
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

    _, inclusion_image_list = get_exclusion_and_inclusion_bounding_boxes(dummy_path, image_df, bb_df, 1000, 1000)
    background_im = tuple(np.median(image_array, axis=(1, 2), keepdims=True).astype(int).squeeze())

    dummy_result = create_new_slide_by_generating_mask(
        image_array, dummy_path, inclusion_image_list, image_df, bb_df, background_im
    )

    # check if background is hardcoded
    background = np.median(dummy_result, axis=(1, 2), keepdims=True)
    assert np.allclose(background, (255, 255, 255))

    # check if H of the result image are less or equal to H of image array, and C are equal
    assert dummy_result.transpose(2, 0, 1).shape[1] <= image_array.shape[1]
    assert dummy_result.transpose(2, 0, 1).shape[0] == image_array.shape[0]


def test_segment_foreground_quantile() -> None:
    h_image = np.random.randint(0, 255, size=(1000, 1000), dtype=np.uint8)  # hematoxylin channel only
    mask = segment_foreground_quantile(x=h_image)
    assert mask.shape == h_image.shape
    assert mask.all() in [True, False]
