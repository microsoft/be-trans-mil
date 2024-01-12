#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Dict
from health_cpath.configs.classification.BaseMIL import BaseMILSlides, BaseMIL


def get_common_aml_run_tags(container: BaseMIL) -> Dict[str, str]:
    return {
        "split_seed": "1",
        "sampling_seed": str(container.get_effective_random_seed()),
        # Memory parameters
        "batch_size": str(container.batch_size),
        "batch_size_inf": str(container.batch_size_inf),
        "max_bag_size": str(container.max_bag_size),
        "max_bag_size_inf": str(container.max_bag_size_inf),
        "encoding_chunk_size": str(container.encoding_chunk_size),
        # Encoder parameters
        "encoder_type": container.encoder_type,
        "tune_encoder": str(container.tune_encoder),
        "pretrained_encoder": str(container.pretrained_encoder),
        "projection_dim": str(container.projection_dim),
        # Pooling parameters
        "pool_type": container.pool_type,
        "transformer_pool_heads": str(container.num_transformer_pool_heads),
        "transformer_pool_layers": str(container.num_transformer_pool_layers),
        "transformer_dropout": str(container.transformer_dropout),
        "tune_pooling": str(container.tune_pooling),
        "pretrained_pooling": str(container.pretrained_pooling),
        # Classifier parameters
        "dropout_rate": str(container.dropout_rate),
        "tune_classifier": str(container.tune_classifier),
        "pretrained_classifier": str(container.pretrained_classifier),
        # Optimizer parameters
        "l_rate": str(container.l_rate),
        "weight_decay": str(container.weight_decay),
        "primary_val_metric": container.primary_val_metric,
        # WSI Loading parameters
        "level": str(container.level),
        "backend": container.backend,
        "margin": str(container.margin),
        "roi_type": container.roi_type,
        "foreground_threshold": str(container.foreground_threshold),
        # Tiling on the fly parameters
        "tile_size": str(container.tile_size),
    }


def get_tiling_on_the_fly_aml_run_tags(container: BaseMILSlides) -> Dict[str, str]:
    return {
        "tile_overlap": str(container.tile_overlap),
        "tile_sort_fn": str(container.tile_sort_fn),
        "tile_pad_mode": str(container.tile_pad_mode),
        "intensity_threshold": str(container.intensity_threshold),
        "background_val": str(container.background_val),
        "rand_min_offset": str(container.rand_min_offset),
        "rand_max_offset": str(container.rand_max_offset),
        "inf_offset": str(container.inf_offset),
    }
