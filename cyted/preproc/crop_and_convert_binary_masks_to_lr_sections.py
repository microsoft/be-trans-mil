#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""
Script to convert binary masks (non-cropped dataset) to
separate Left and Right (L-R) tissue sections after cropping and stitching (cropped dataset).
Requires bounding boxes in COCO format to extract the L-R tissue sections.
Left section will be labelled with pixel values 128.
Right section will be labelled with pixel values 255.
Background will be labelled with pixel values 0.
"""
import glob
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from cyted.utils.azure_utils import replace_ampersand_in_filename

cpath_root = Path(__file__).resolve().parents[2]
sys.path.append(str(cpath_root))

from cyted import fixed_paths  # noqa: E402

fixed_paths.add_submodules_to_path()

from cyted.cyted_schema import CytedSchema  # noqa: E402
from cyted.data_paths import CYTED_BOUNDING_BOXES_JSON, CYTED_HE_HISTOQC_OUTPUTS_DATASET_DIR  # noqa: E402
from cyted.utils.crop_utils import (  # noqa: E402
    crop_inclusion_boxes,
    get_exclusion_and_inclusion_bounding_boxes,
    join_images,
    preprocess_bounding_boxes,
    replace_exclusion_boxes_with_background,
    sort_bboxes_left_to_right,
    resize_image_array,
)

CYTED_SRC = CYTED_HE_HISTOQC_OUTPUTS_DATASET_DIR
CYTED_DEST = "/cyted_drive/cyted-he-masks-cropped-sections-dataset-1.25x"


def generate_cropped_section_masks(
    src_dir: Path,
    bounding_box_path: Path,
    dest_dir: Path,
    source_magnification: float = 1.25,
    target_magnification: float = 1.25,
    limit: Optional[int] = None,
) -> None:
    """
    Generate cropped section masks (with L-R sections) from non-cropped masks.
    :param src_dir: Directory containing the original (non-cropped) masks, with prefix `_mask_use.png`.
    :param dest_dir: Destination directory to save the mask files.
    :param source_magnification: Magnification of the input masks (e.g. 40. for 40x) (default=1.25).
    :param target_magnification: Magnification of the output masks (default=1.25).
    :param limit: Number of masks to generate. Default is None.
    """
    src_paths = glob.glob(str(src_dir) + "/**/*_mask_use.png")
    if len(src_paths) < 1:
        logging.error(f"No PNG masks with suffix `_mask_use.png` found in directory {src_dir}. Exiting...")

    if limit is not None:
        src_paths = src_paths[:limit]

    logging.info(f"Starting binary mask cropping and sectioning of {len(src_paths)} masks to {dest_dir}.")
    image_df, bounding_box_df = preprocess_bounding_boxes(bounding_box_path)

    for src_path in tqdm(src_paths, total=len(src_paths)):
        dest_path = dest_dir / replace_ampersand_in_filename(Path(src_path).name)
        if not os.path.isfile(dest_path):
            mask_array = np.asarray(cv2.imread(src_path)).transpose(2, 0, 1)
            mask_array_resized = resize_image_array(
                image_array=mask_array,
                source_magnification=source_magnification,
                target_magnification=target_magnification,
            )
            _, height, width = mask_array_resized.shape  # CHW format
            exclusion_bboxes, inclusion_bboxes = get_exclusion_and_inclusion_bounding_boxes(
                Path(src_path), image_df, bounding_box_df, image_width=width, image_height=height
            )
            image_array, _ = replace_exclusion_boxes_with_background(mask_array_resized, exclusion_bboxes)

            # sort inclusion_boxes such that first box is left most
            inclusion_bboxes = sort_bboxes_left_to_right(inclusion_bboxes)
            inclusion_image_list = crop_inclusion_boxes(image_array, inclusion_bboxes)

            # hardcode left section to 128
            assert len(inclusion_image_list) == 2
            left_inclusion_image = np.copy(inclusion_image_list[0])
            np.place(left_inclusion_image, left_inclusion_image == 255, 128)
            inclusion_image_list[0] = left_inclusion_image

            # join inclusion images, background should be black (0,0,0)
            joined_image = join_images(inclusion_image_list, background=(0, 0, 0))
            cv2.imwrite(str(dest_path), joined_image)

    logging.info(f"Cropped masks with L-R sections now available at {dest_dir}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default=CYTED_SRC,
        help="Directory containing the original (non-cropped) masks, with prefix `_mask_use.png`.",
    )
    parser.add_argument(
        "--bounding_box_path",
        type=Path,
        default=CYTED_BOUNDING_BOXES_JSON[CytedSchema.HEImage],
        help="Path for AML bounding boxes.",
    )
    parser.add_argument(
        "--source_magnification",
        type=float,
        default=1.25,
        help="Magnification of the input masks.",
    )
    parser.add_argument(
        "--target_magnification",
        type=float,
        default=1.25,
        help="Magnification at which the masks need to be generated.",
    )
    parser.add_argument(
        "--dest_dir",
        type=Path,
        default=CYTED_DEST,
        help="Destination directory to save the mask files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of masks to generate. Default is None.",
    )
    args = parser.parse_args()

    Path(args.dest_dir).mkdir(exist_ok=True, parents=True)

    generate_cropped_section_masks(
        src_dir=args.src_dir,
        bounding_box_path=args.bounding_box_path,
        dest_dir=Path(args.dest_dir),
        source_magnification=args.source_magnification,
        target_magnification=args.target_magnification,
        limit=args.limit,
    )
