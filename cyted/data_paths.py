#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from cyted.cyted_schema import CytedSchema
from cyted.fixed_paths import repository_root_directory


CYTED_DEFAULT_DATASET_LOCATION = "/cyted/"
CYTED_RAW_CSV_FILENAME = "delta_slides_msr_2022-11-11.tsv"
CYTED_CSV_FILENAME_BEFORE_REVIEW = "exclude_wrong_stains_2022-11-22.tsv"
CYTED_CSV_FILENAME = "reviewed_dataset_2023-02-10.tsv"
CYTED_RAW_DATASET_ID = "cyted-raw-20221102"
CYTED_RAW_DATASET_DIR = CYTED_DEFAULT_DATASET_LOCATION + CYTED_RAW_DATASET_ID
# We added data_splits to the Cyted workspace to store the data splits separately. This avoids downloading the whole
# raw dataset when we download the datsets on AML to limit throuput overheads due to concurrent mounting from multiple
# AML runs.
CYTED_DATA_SPLITS_ID = "data_splits"
CYTED_DATA_SPLITS_DIR = CYTED_DEFAULT_DATASET_LOCATION + CYTED_DATA_SPLITS_ID
CYTED_CROPPED_TFF3_DATASET_ID = "preprocessed_TFF3_10x"  # "crop_and_convert_tff3_10x_to_tiff_2022_12_06"
CYTED_CROPPED_TFF3_DATASET_DIR = CYTED_DEFAULT_DATASET_LOCATION + CYTED_CROPPED_TFF3_DATASET_ID
CYTED_REGISTERED_TFF3_DATASET_ID = "hardcode_registered_tff3_2.5x"
CYTED_REGISTERED_TFF3_DATASET_DIR = CYTED_DEFAULT_DATASET_LOCATION + CYTED_REGISTERED_TFF3_DATASET_ID
CYTED_CROPPED_TFF3_10X_TILES_DATASET_ID = "crop_and_convert_tff3_10X_tiles_20221219_115956_level0_224"
CYTED_CROPPED_TFF3_10X_TILES_DATASET_DIR = CYTED_DEFAULT_DATASET_LOCATION + CYTED_CROPPED_TFF3_10X_TILES_DATASET_ID
CYTED_CROPPED_HE_DATASET_ID = "crop_and_convert_he_10x_to_tiff_2022_12_14"
CYTED_CROPPED_HE_DATASET_DIR = CYTED_DEFAULT_DATASET_LOCATION + CYTED_CROPPED_HE_DATASET_ID
CYTED_CROPPED_HE_255_BACKGROUND_DATASET_ID = "crop_and_convert_he_10x_to_tiff_no_hardcoded_bckd_2023_01_17"
CYTED_CROPPED_HE_255_BACKGROUND_DATASET_DIR = (
    CYTED_DEFAULT_DATASET_LOCATION + CYTED_CROPPED_HE_255_BACKGROUND_DATASET_ID
)

CYTED_CROPPED_HE_10X_TILES_DATASET_ID = "crop_and_convert_he_10X_tiles_20230119_154902_level0_224"
CYTED_CROPPED_HE_10X_TILES_DATASET_DIR = CYTED_DEFAULT_DATASET_LOCATION + CYTED_CROPPED_HE_10X_TILES_DATASET_ID
CYTED_DATASET_TSV = "dataset.tsv"
CYTED_DATASET_ID = {
    CytedSchema.HEImage: CYTED_CROPPED_HE_255_BACKGROUND_DATASET_ID,
    CytedSchema.TFF3Image: CYTED_CROPPED_TFF3_DATASET_ID,
    CytedSchema.P53Image: CYTED_RAW_DATASET_ID,  # TODO: add converted p53 dataset id
}
CYTED_SPLITS_CSV_BEFORE_REVIEW = "splits_20221124_085458.csv"
CYTED_SPLITS_CSV_FILENAME = {
    CytedSchema.TFF3Positive: "splits_20221124_085458_after_review.csv",
    CytedSchema.P53Positive: "on hold",
    CytedSchema.Atypia: "on hold",
}
CYTED_PREPROCESSING_DIR = repository_root_directory("cyted/preproc/files")
CYTED_EXCLUSION_LIST_CSV = {
    CytedSchema.HEImage: None,
    CytedSchema.TFF3Image: CYTED_PREPROCESSING_DIR / "tff3_exclusion_list_20221216_165203.csv",
    CytedSchema.P53Image: None,
}
CYTED_BOUNDING_BOXES_JSON = {
    CytedSchema.HEImage: CYTED_PREPROCESSING_DIR / "he_bboxes_from_histoqc_masks_2022-12-14.json",
    CytedSchema.TFF3Image: CYTED_PREPROCESSING_DIR / "tff3_control_tissue_exclusion_via_aml_labelling.json",
}
CYTED_TFF3_LABELS_REVIEW_CSV = CYTED_PREPROCESSING_DIR / "tff3_labels_review_2023-02-10.csv"

CYTED_HE_HISTOQC_OUTPUTS_DATASET_ID = "cyted-HE-histoqc-v2.1-20221125"
CYTED_HE_HISTOQC_OUTPUTS_DATASET_DIR = CYTED_DEFAULT_DATASET_LOCATION + CYTED_HE_HISTOQC_OUTPUTS_DATASET_ID

CYTED_FOREGROUND_MASKS_DATASET_ID = {
    # Foreground masks for the raw H&E dataset using histoqc
    CytedSchema.HEImage: CYTED_HE_HISTOQC_OUTPUTS_DATASET_ID,
    # Foreground masks for the cropped tff3 dataset using aml labelling to exclude control tissue
    CytedSchema.TFF3Image: "cyted-tff3-masks-cropped-dataset-1.25x",
}


def get_cyted_dataset_dir(image_column: str) -> Path:
    return Path(CYTED_DEFAULT_DATASET_LOCATION + CYTED_DATASET_ID[image_column])


def get_cyted_mask_dataset_dir(image_column: str) -> Path:
    return Path(CYTED_DEFAULT_DATASET_LOCATION + CYTED_FOREGROUND_MASKS_DATASET_ID[image_column])
