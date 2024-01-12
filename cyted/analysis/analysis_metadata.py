#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from enum import Enum
from typing import List


class AnalysisMetadata(str, Enum):
    """Metadata keys for analysis results."""

    BROWN_STAIN_RATIO = "brown_stain_ratio_slide"
    BROWN_STAIN_BINARY_MASK = "brown_stain_binary_mask_slide"
    OTSU_CLASS_VAR = "otsu_class_variance_slide"
    DAB_THRESHOLD = "dab_threshold_slide"
    TILES_COUNT = "tiles_count"
    FOREGROUND_PIXELS = "foreground_pixels_slide"
    FOREGROUND_HEM_PIXELS = "hem_pixels_slide"
    BROWN_PIXELS = "brown_pixels_slide"
    FOREGROUND_MASK = "foreground_mask_slide"
    BROWN_STAIN_RATIO_TILES = "brown_stain_ratio_tiles"
    FOREGROUND_PIXELS_TILES = "foreground_pixels_tiles"
    BROWN_PIXELS_TILES = "brown_pixels_tiles"

    @staticmethod
    def get_brown_stain_keys() -> List["AnalysisMetadata"]:
        return [AnalysisMetadata.BROWN_STAIN_RATIO, AnalysisMetadata.BROWN_STAIN_BINARY_MASK]

    @staticmethod
    def get_otsu_keys() -> List["AnalysisMetadata"]:
        return [AnalysisMetadata.OTSU_CLASS_VAR, AnalysisMetadata.DAB_THRESHOLD]

    @staticmethod
    def get_pixel_count_keys() -> List["AnalysisMetadata"]:
        return [
            AnalysisMetadata.FOREGROUND_PIXELS,
            AnalysisMetadata.FOREGROUND_HEM_PIXELS,
            AnalysisMetadata.BROWN_PIXELS,
        ]
