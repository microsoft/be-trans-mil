#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Iterable

import numpy as np
import PIL.Image

from health_ml.utils.box_utils import Box

from cyted.utils.roi_utils import get_left_right_roi_bboxes

CocoDict = dict[str, list[dict]]


class CocoSchema:
    ID = "id"

    IMAGES = "images"
    IMAGE_ID = "image_id"
    FILENAME = "file_name"
    WIDTH = "width"
    HEIGHT = "height"

    ANNOTATIONS = "annotations"
    AREA = "area"
    BBOX = "bbox"

    CATEGORIES = "categories"
    CATEGORY_ID = "category_id"
    NAME = "name"


DEFAULT_CATEGORIES = [
    {CocoSchema.ID: 1, CocoSchema.NAME: "Tissue"},
    {CocoSchema.ID: 2, CocoSchema.NAME: "Exclusion"},
]


def _mask_to_coco_image_dict(mask: np.ndarray, filename: str, image_id: int) -> dict:
    return {
        CocoSchema.ID: image_id,
        CocoSchema.WIDTH: float(mask.shape[1]),
        CocoSchema.HEIGHT: float(mask.shape[0]),
        CocoSchema.FILENAME: filename,
    }


def _box_to_coco_annotation_dict(box: Box, width: int, height: int, annotation_id: int, image_id: int) -> dict:
    normalised_coords = [
        box.x / width,
        box.y / height,
        box.w / width,
        box.h / height,
    ]
    area = normalised_coords[2] * normalised_coords[3]
    return {
        CocoSchema.ID: annotation_id,
        CocoSchema.CATEGORY_ID: 1,  # Tissue
        CocoSchema.IMAGE_ID: image_id,
        CocoSchema.AREA: area,
        CocoSchema.BBOX: normalised_coords,
    }


def mask_to_coco_dict(dataset_dir: Path, filename: str, image_id: int) -> CocoDict:
    """Compute bounding boxes for the given mask and output a COCO dictionary.

    :param dataset_dir: The base directory from which to load the mask.
    :param filename: The mask image's filename (.png) relative to `dataset_dir`.
        This will be included in the output COCO metadata.
    :param image_id: A numeric ID to link image metadata and the bounding boxes.
    :return: A COCO-format dictionary containing:

        - `'images'`: a list of dictionaries with image metadata (here, a singleton)
        - `'annotations'`: a list of bounding boxes, including normalised coordinates
        - `'categories'`: the annotation categories, here "Tissue" (1) and "Exclusion" (2)

        This is serialisable as a valid COCO JSON file, in the same format as
        those produced by the Azure ML labeller.
    """
    mask_path = dataset_dir / filename
    mask = np.array(PIL.Image.open(mask_path)).astype(bool)
    height, width = mask.shape

    left_bbox, right_bbox = get_left_right_roi_bboxes(mask)

    image_dict = _mask_to_coco_image_dict(mask, filename=filename, image_id=image_id)

    # Annotation IDs are defined like this because we're always saving 2 boxes per image
    left_bbox_dict = _box_to_coco_annotation_dict(
        left_bbox,
        width=width,
        height=height,
        annotation_id=2 * image_id - 1,
        image_id=image_id,
    )

    right_bbox_dict = _box_to_coco_annotation_dict(
        right_bbox,
        width=width,
        height=height,
        annotation_id=2 * image_id,
        image_id=image_id,
    )

    return {
        CocoSchema.IMAGES: [image_dict],
        CocoSchema.ANNOTATIONS: [left_bbox_dict, right_bbox_dict],
        CocoSchema.CATEGORIES: DEFAULT_CATEGORIES,
    }


def merge_coco_dicts(dicts: Iterable[CocoDict]) -> CocoDict:
    """Combine a collection of COCO dictionaries into a single one."""
    merged_dict: CocoDict = {
        CocoSchema.IMAGES: [],
        CocoSchema.ANNOTATIONS: [],
        CocoSchema.CATEGORIES: DEFAULT_CATEGORIES,
    }
    for coco_dict in dicts:
        merged_dict[CocoSchema.IMAGES].extend(coco_dict[CocoSchema.IMAGES])
        merged_dict[CocoSchema.ANNOTATIONS].extend(coco_dict[CocoSchema.ANNOTATIONS])
    return merged_dict
