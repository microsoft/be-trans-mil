#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""
Script to copy all slides that are marked as positive for TFF3 from the original dataset to a new folder.
"""
import logging
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

cpath_root = Path(__file__).resolve().parents[2]
sys.path.append(str(cpath_root))

from cyted import fixed_paths  # noqa: E402

fixed_paths.add_submodules_to_path()

from cyted.cyted_schema import CytedLabel, CytedSchema  # noqa: E402
from cyted.data_paths import CYTED_CSV_FILENAME, CYTED_RAW_DATASET_DIR, CYTED_DEFAULT_DATASET_LOCATION  # noqa: E402
from cyted.datasets.cyted_slides_dataset import CytedSlidesDataset  # noqa: E402

SRC_DATASET_PATH = CYTED_DEFAULT_DATASET_LOCATION + "cyted-TFF3-histoqc-thumbnails-0.625x"
DEST_DATASET_PATH = CYTED_DEFAULT_DATASET_LOCATION + "cyted-TFF3-histoqc-thumbnails-0.625x-positives-only"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dataset",
        type=str,
        default=SRC_DATASET_PATH,
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default=CytedSchema.TFF3Image,
        help="Name of the column containing the image path",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=CytedSchema.TFF3Positive,
        help="Name of the column containing the label",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default=DEST_DATASET_PATH,
        help="Destination directory to save the files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit mumber of slides to copy. Default is None which copies all slides",
    )
    args = parser.parse_args()
    Path(args.dest_dir).mkdir(parents=True, exist_ok=True)

    cyted_dataset = CytedSlidesDataset(
        root=args.root_dataset,
        dataset_csv=Path(CYTED_RAW_DATASET_DIR) / CYTED_CSV_FILENAME,
        image_column=args.image_column,
        label_column=args.label_column,
        column_filter={args.label_column: CytedLabel.Yes},
    )
    if args.limit is not None:
        cyted_dataset.dataset_df = cyted_dataset.dataset_df.iloc[: args.limit]

    slides = cyted_dataset.dataset_df[args.image_column].tolist()
    slides = [f"{slide}_thumb.png" for slide in slides]
    logging.info(f"Found {len(slides)} slides")
    logging.info(f"Copying {len(slides)} slides from {args.root_dataset} to {args.dest_dir}")
    # copy the files to the destination directory
    for slide in tqdm(slides):
        shutil.copyfile(os.path.join(args.root_dataset, slide), os.path.join(args.dest_dir, slide))
