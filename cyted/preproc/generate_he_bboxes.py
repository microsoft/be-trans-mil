#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import functools
import json
import logging
import multiprocessing
import os
from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Any, Callable, Generator, Optional, Sequence, TypeVar

from tqdm import tqdm

cpath_root = Path(__file__).resolve().parents[2]
sys.path.append(str(cpath_root))

from cyted import fixed_paths  # noqa: E402

fixed_paths.add_submodules_to_path()

from cyted.cyted_schema import CytedSchema  # noqa: E402
from cyted.data_paths import CYTED_BOUNDING_BOXES_JSON  # noqa: E402
from cyted.utils.coco_utils import CocoDict, mask_to_coco_dict, merge_coco_dicts  # noqa: E402

T = TypeVar("T")


def map_parallel(
    func: Callable[..., T], iterable: Sequence, num_workers: Optional[int] = None, **kwargs: Any
) -> Generator[T, None, None]:
    partial_func = functools.partial(func, **kwargs)

    if num_workers == 0:
        pool = None
        map_func = map  # type: ignore
    else:
        pool = multiprocessing.Pool(num_workers)
        map_func = pool.imap_unordered  # type: ignore

    yield from tqdm(map_func(partial_func, iterable), total=len(iterable))

    if pool is not None:
        pool.close()


def mask_entry_to_coco_dict(mask_entry: dict) -> CocoDict:
    coco_dict = mask_to_coco_dict(
        dataset_dir=mask_entry["dataset_dir"], filename=mask_entry["filename"], image_id=mask_entry["image_id"]
    )
    return coco_dict


def main(masks_dir: Path, bbox_json_path: Path, num_workers: Optional[int] = None, limit: int = 0) -> None:
    root_dir, masks_dirname = masks_dir.parent, masks_dir.name
    assert (root_dir / masks_dirname) == masks_dir

    mask_filenames = [
        f"{masks_dirname}/{wsi_filename}/{wsi_filename}_mask_use.png"
        for wsi_filename in os.listdir(masks_dir)
        if os.path.isdir(masks_dir / wsi_filename)
    ]

    if limit > 0:
        mask_filenames = mask_filenames[:limit]

    mask_items = [
        {"dataset_dir": root_dir, "filename": filename, "image_id": image_index + 1}
        for image_index, filename in enumerate(mask_filenames)
    ]

    logging.info(f"Starting conversion of {len(mask_items)} masks to COCO JSON bounding boxes...")

    coco_dicts = list(map_parallel(mask_entry_to_coco_dict, mask_items, num_workers=num_workers))
    full_coco_dict = merge_coco_dicts(coco_dicts)

    bbox_json_path.resolve().parent.mkdir(exist_ok=True, parents=True)
    with open(bbox_json_path, "w") as f:
        json.dump(full_coco_dict, f, indent=2)

    logging.info(f"Bounding boxes saved to {bbox_json_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--masks_dir",
        type=Path,
        required=True,
        help="Directory containing HistoQC outputs, including binary masks in PNG format.",
    )
    parser.add_argument(
        "--bbox_json",
        type=Path,
        default=CYTED_BOUNDING_BOXES_JSON[CytedSchema.HEImage],
        help="Path for generated bounding boxes in COCO JSON format.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers to use for processing (default: CPU count).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of slides to convert (default: 0, i.e. all).",
    )
    args = parser.parse_args()

    main(
        masks_dir=args.masks_dir,
        bbox_json_path=args.bbox_json,
        num_workers=args.num_workers,
        limit=args.limit,
    )
