#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import List

from health_cpath.datasets.base_dataset import TilesDataset


def tiles_per_slide(dataset: TilesDataset) -> List[int]:
    tiles_per_slide_series = dataset.dataset_df.groupby([TilesDataset.SLIDE_ID_COLUMN]).size()
    dist = tiles_per_slide_series.tolist()
    return dist
