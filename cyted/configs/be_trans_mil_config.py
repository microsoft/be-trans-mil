#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Union, Callable
from monai.transforms import Compose, ScaleIntensityRanged, RandRotate90d, RandFlipd
import numpy as np
import param
from cyted.datasets.cyted_module import CytedSlidesDataModule, CytedSlidesEvalModule
from cyted.cyted_schema import CytedSchema
from cyted.datasets.cyted_slides_dataset import CytedSlidesDataset
from cyted.data_paths import (
    CYTED_DATASET_ID,
    CYTED_REGISTERED_TFF3_DATASET_ID,
    CYTED_SPLITS_CSV_FILENAME,
    CYTED_EXCLUSION_LIST_CSV,
    get_cyted_dataset_dir,
)
from cyted.fixed_paths import repository_root_directory
from cyted.utils.cyted_utils import CytedParams
from cyted.utils.mil_configs_utils import get_common_aml_run_tags, get_tiling_on_the_fly_aml_run_tags

from health_azure.utils import create_from_matching_params, is_running_in_azure_ml
from health_cpath.configs.classification.BaseMIL import BaseMILSlides
from health_cpath.datamodules.base_module import SlidesDataModule
from health_cpath.models.encoders import (
    DenseNet121_NoPreproc,
    Resnet18_NoPreproc,
    Resnet50_NoPreproc,
    SwinTransformer_NoPreproc,
)
from health_cpath.models.transforms import MetaTensorToTensord, StainNormMacenkod, transform_dict_adaptor
from health_cpath.preprocessing.loading import LoadingParams, ROIType, WSIBackend
from health_cpath.utils.wsi_utils import TilingParams
from health_cpath.utils.naming import MetricsKey, ModelKey, PlotOption, SlideKey
from health_ml.networks.layers.attention_layers import TransformerPoolingBenchmark
from health_ml.utils.data_augmentations import StainNormalization


class StainNormMethod:
    """Contains constants for stain normalization method."""

    Macenko = "Macenko"
    Reinhard = "Reinhard"


class CytedBaseMIL(BaseMILSlides, CytedParams):
    smoke_run: bool = param.Boolean(
        False, doc="Whether this is a smoke run. This is useful to disable whole slide inference for smoke tests."
    )
    dataset_suffix: str = param.String(
        default="",
        doc="Suffix to add to the dataset name. Useful to workaround the aml sdk issues that force us to "
        "create new data assets in some workspaces like innereye4ws. Make sure to use --dataset_suffix=_v1 when using "
        "compute attached to innereye4ws workspace.",
    )
    stain_norm_method: Optional[str] = param.String(
        default=None,
        doc="Stain normalization method to use (Macenko, Reinhard, None). If None, no stain normalization is applied.",
    )

    def __init__(self, column_filter: Optional[Dict[str, str]] = None, **kwargs: Any) -> None:
        """Base config for Cyted dataset."""
        self.column_filter = column_filter
        default_kwargs = dict(
            # WSI loading params
            level=0,  # 10x magnification for converted/cropped tiff files
            backend=WSIBackend.CUCIM,
            roi_type=ROIType.WHOLE,
            image_key=SlideKey.IMAGE,
            margin=0,
            # Tiling on the fly params
            tile_size=224,
            background_val=255,
            intensity_threshold_scale=0.9,
            # background_keys=CytedSchema.background_columns(), # uncomment to use tile wise background normalization
            # dataset parameters
            azure_datasets=[f"{CYTED_DATASET_ID[self.image_column]}{self.dataset_suffix}"],
            local_datasets=[get_cyted_dataset_dir(self.image_column)],
            # Resnet18 is used by default, but can be changed via commandline
            encoder_type=Resnet18_NoPreproc.__name__,
            tune_encoder=True,
            # pooling and classifier are tuned by default
            pool_type=TransformerPoolingBenchmark.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=8,
            dropout_rate=None,
            pool_hidden_dim=2048,
            # tiles per slide ~ 3700
            encoding_chunk_size=2300,  # might need to be decreased for 16 gb gpus to 250
            bag_size_subsample=2300,
            max_bag_size=2300,  # for 8 gpus of 32 gb each, use 250 for 4 gpus of 16 gb each
            max_bag_size_inf=4600,  # Use double the training bag size for inference to get closer to WS inference
            batch_size=1,
            batch_size_inf=1,
            max_epochs=50,
            # learning rate and weight decay from DeepSMILE paper
            l_rate=3e-5,
            weight_decay=0.1,
            adam_betas=(0.9, 0.99),
            primary_val_metric=MetricsKey.AUROC,
            pl_log_every_n_steps=20,
            save_intermediate_outputs=False,
            pl_progress_bar_refresh_rate=10,
            pl_deterministic=True,
            max_num_workers=os.cpu_count(),
            # we can revisit this value with the new compute clusters
            stratify_by=[self.label_column, CytedSchema.PatientPathway, CytedSchema.Year],
            stratify_plots_by=CytedSchema.PatientPathway,
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        self.metadata_columns = CytedSchema.metadata_columns()
        if self.background_keys is not None:
            self.metadata_columns = self.metadata_columns.union(self.background_keys)

        if not is_running_in_azure_ml():
            self.max_epochs = 2

    def validate(self) -> None:
        BaseMILSlides.validate(self)
        CytedParams.validate(self)

    def get_preprocessing_transforms(self, image_key: str) -> Sequence[Callable]:
        return [
            *self.get_background_normalization_transform(),
            ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0),
            MetaTensorToTensord(keys=image_key),
        ]

    def get_geometric_augmentation_transforms(self, image_key: str) -> Sequence[Callable]:
        return [
            RandFlipd(keys=image_key, spatial_axis=0, prob=0.5),
            RandFlipd(keys=image_key, spatial_axis=1, prob=0.5),
            RandRotate90d(keys=image_key, prob=0.5),
        ]

    def get_stain_augmentation_transforms(self, image_key: str) -> Sequence[Callable]:
        return []  # Add stain augmentation transforms here: HEDJitter or Gaussian Blur

    def get_stain_normalization_transforms(self, image_key: str) -> Sequence[Callable]:
        if self.stain_norm_method == StainNormMethod.Macenko:
            stain_norm_transform: Sequence[Callable] = [StainNormMacenkod(SlideKey.IMAGE)]
            logging.info(f"Using {self.stain_norm_method} stain normalization.")
        elif self.stain_norm_method == StainNormMethod.Reinhard:
            stain_norm_transform = [transform_dict_adaptor(StainNormalization(), image_key, image_key)]
            logging.info(f"Using {self.stain_norm_method} stain normalization.")
        else:
            stain_norm_transform = []
            logging.info("Not using any stain normalization.")
        return stain_norm_transform

    def get_transforms_dict(self, image_key: str) -> Dict[ModelKey, Union[Callable, None]]:
        preprocessing_transforms = self.get_preprocessing_transforms(image_key)
        geo_augmentations = self.get_geometric_augmentation_transforms(image_key)
        stain_augmentations = self.get_stain_augmentation_transforms(image_key)
        # Geometric augmentations go first to be applied on uint8 data before normalization to float in range 0 1
        transform_train = Compose([*geo_augmentations, *preprocessing_transforms, *stain_augmentations])
        transform_inf = Compose(preprocessing_transforms)
        return {ModelKey.TRAIN: transform_train, ModelKey.VAL: transform_inf, ModelKey.TEST: transform_inf}

    def get_dataframe_kwargs(self, image_column: str) -> Dict[str, Any]:
        datatypes: Dict[str, Any] = {
            CytedSchema.CytedID: str,
            CytedSchema.QCReport: str,
            image_column: str,
            CytedSchema.PatientPathway: str,
        }
        if self.background_keys is not None:
            for key in self.background_keys:
                datatypes[key] = np.uint8
        return dict(dtype=datatypes)

    def get_dataloader_kwargs(self) -> dict:
        default_dataloader_kwargs = super().get_dataloader_kwargs()
        default_dataloader_kwargs["pin_memory"] = False  # Fixes High CPU usage and increased training time on Azure
        return default_dataloader_kwargs

    def get_data_module(self) -> SlidesDataModule:
        self.print_effective_batch_size()
        # Train/test splits are stored in the repository so that they can be shared more easily with Cyted
        splits_folder = repository_root_directory() / "cyted" / "preproc" / "files"
        if not splits_folder.is_dir():
            raise FileNotFoundError(f"Could not find the splits folder {splits_folder}")
        return CytedSlidesDataModule(
            root_path=self.local_datasets[0],
            cyted_params=create_from_matching_params(self, CytedParams),
            excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[self.image_column],
            splits_csv=splits_folder / CYTED_SPLITS_CSV_FILENAME[self.label_column],
            column_filter=self.column_filter,
            # loading, tiling on the fly and transforms params
            loading_params=create_from_matching_params(self, LoadingParams),
            tiling_params=create_from_matching_params(self, TilingParams),
            transforms_dict=self.get_transforms_dict(SlideKey.IMAGE),
            # batch / bag sizes
            batch_size=self.batch_size,
            batch_size_inf=self.batch_size_inf,
            max_bag_size=self.max_bag_size,
            max_bag_size_inf=self.max_bag_size_inf,
            # cross validation / splits params
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            stratify_by=self.stratify_by or [self.label_column],
            seed=self.get_effective_random_seed(),
            dataloader_kwargs=self.get_dataloader_kwargs(),
            dataframe_kwargs=self.get_dataframe_kwargs(self.image_column),
            pl_replace_sampler_ddp=self.pl_replace_sampler_ddp,
            metadata_columns=self.metadata_columns,
        )

    def get_eval_data_module(self) -> SlidesDataModule:
        return CytedSlidesEvalModule(
            root_path=self.local_datasets[0],
            cyted_params=create_from_matching_params(self, CytedParams),
            excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[self.image_column],
            column_filter=self.column_filter,
            # loading, tiling on the fly and transforms params
            loading_params=create_from_matching_params(self, LoadingParams),
            tiling_params=create_from_matching_params(self, TilingParams),
            transforms_dict=self.get_transforms_dict(SlideKey.IMAGE),
            # batch / bag sizes
            batch_size=self.batch_size,
            batch_size_inf=self.batch_size_inf,
            max_bag_size=self.max_bag_size,
            max_bag_size_inf=self.max_bag_size_inf,
            dataloader_kwargs=self.get_dataloader_kwargs(),
            dataframe_kwargs=self.get_dataframe_kwargs(self.image_column),
            pl_replace_sampler_ddp=self.pl_replace_sampler_ddp,
            metadata_columns=self.metadata_columns,
        )

    def get_slides_dataset(self) -> CytedSlidesDataset:
        return CytedSlidesDataset(
            root=self.local_datasets[0],
            label_column=self.label_column,
            image_column=self.image_column,
            dataframe_kwargs=self.get_dataframe_kwargs(image_column=self.image_column),
        )

    def get_test_plot_options(self) -> Set[PlotOption]:
        plot_options = super().get_test_plot_options()
        plot_options.update([PlotOption.ATTENTION_HEATMAP, PlotOption.PR_CURVE, PlotOption.ROC_CURVE])
        return plot_options

    def get_additional_aml_run_tags(self) -> Dict[str, str]:
        common_tags = get_common_aml_run_tags(self)
        tiling_tags = get_tiling_on_the_fly_aml_run_tags(self)
        cyted_tags = self.get_cyted_params_aml_tags()
        return {**common_tags, **tiling_tags, **cyted_tags}

    def get_additional_environment_variables(self) -> Dict[str, str]:
        return {"DATASET_MOUNT_CACHE_SIZE": "1"}

    def on_run_extra_validation_epoch(self) -> None:
        super().on_run_extra_validation_epoch()
        if not self.smoke_run:
            self.max_bag_size_inf = 0
            self.batch_size_inf = 1

    def set_bag_sizes(self, bag_size: int) -> None:
        """Set training related parameters to the given bag size and inference as double the bag size."""
        self.encoding_chunk_size = bag_size
        self.bag_size_subsample = bag_size
        self.max_bag_size = bag_size
        self.max_bag_size_inf = 2 * bag_size

    def print_effective_batch_size(self) -> None:
        effective_batch_size = self.max_num_gpus * self.pl_accumulate_grad_batches * self.batch_size
        logging.info(f"Effective batch size: {effective_batch_size}")


class TFF3CytedMIL(CytedBaseMIL):
    """TFF3 Cyted MIL model. Uses the TransMIL model on TFF3 stain to predict TFF3 label."""

    def __init__(self, **kwargs: Any) -> None:
        self.image_column = CytedSchema.TFF3Image
        self.label_column = CytedSchema.TFF3Positive
        super().__init__(**kwargs)
        self.intensity_threshold_scale = 0.999


class HECytedMIL(CytedBaseMIL):
    """Base class for HE Cyted MIL models. Uses the TransMIL model on HE stain."""

    def __init__(self, **kwargs: Any) -> None:
        self.image_column = CytedSchema.HEImage
        super().__init__(**kwargs)
        self.local_datasets.append(get_cyted_dataset_dir(CytedSchema.TFF3Image))
        self.azure_datasets.append(f"{CYTED_REGISTERED_TFF3_DATASET_ID}{self.dataset_suffix}")
        self.stain_norm_method = None

    def get_preprocessing_transforms(self, image_key: str) -> List[Callable]:
        return [
            *super().get_preprocessing_transforms(image_key),
            *super().get_stain_normalization_transforms(image_key),
        ]

    def get_extra_slides_dataset_for_plotting(self) -> Optional[CytedSlidesDataset]:
        if len(self.local_datasets) > 1:
            return CytedSlidesDataset(
                root=self.local_datasets[1],
                label_column=self.label_column,
                image_column=CytedSchema.TFF3Image,
                dataframe_kwargs=self.get_dataframe_kwargs(image_column=CytedSchema.TFF3Image),
                excluded_slides_csv=CYTED_EXCLUSION_LIST_CSV[self.image_column],
            )
        return None


class TFF3_HECytedMIL(HECytedMIL):
    """TFF3 HE Cyted MIL model. Uses the TransMIL model on HE stain to predict TFF3 label."""

    def __init__(self, **kwargs: Any) -> None:
        # self.dataset_suffix = "_cytedeastus2"  # uncomment this line when running in innereye4ws workspace
        self.label_column = CytedSchema.TFF3Positive
        super().__init__(**kwargs)

    def set_model_variant(self, variant_name: str) -> None:
        if variant_name == "cyted":
            self.azure_datasets = ["preprocessed_h_e_10.0x"]
            self.local_datasets = []
            # Adjust effective batch size to 8
            self.pl_accumulate_grad_batches = 2
            # Reduce number of workers to avoid dataloader errors
            self.max_num_workers = self.max_num_workers // 2
            # Set croosvalidation index to 1 to compare with expected results in documentation he_workflow.md
            self.crossval_count = 5
            self.crossval_index = 1


class TFF3_HECytedMIL_Resnet50(TFF3_HECytedMIL):
    """TFF3 HE Cyted MIL model using Resnet50 encoder."""

    def __init__(self, **kwargs: Any) -> None:
        self.dataset_suffix = "_cytedeastus2"  # comment this line when running outside of innereye4ws workspace
        default_kwargs = dict(
            encoder_type=Resnet50_NoPreproc.__name__,
            use_encoder_checkpointing=True,
            batchnorm_momentum=0.1,
            pl_static_graph=True,
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def set_model_variant(self, variant_name: str) -> None:
        super().set_model_variant(variant_name)
        if variant_name == "8a100_80g_bag_1700":
            # Params for 8 x A100 80 GB: increase docker_shm_size: --cluster=ND96v4-80GB --docker_shm_size=1750g
            self.set_bag_sizes(1700)
            self.use_encoder_checkpointing = False
        elif variant_name == "8a100_40g_bag_800":
            # Params for 8 x A100 40 GB: increase the docker_shm_size: --cluster=ND96v4 --docker_shm_size=1750g
            self.set_bag_sizes(800)
            self.use_encoder_checkpointing = False
        elif variant_name == "8v100_bag_600":
            # Params for 8 x V100 32GB: --cluster=v100x8-westus2
            self.set_bag_sizes(600)
            self.use_encoder_checkpointing = False
        elif variant_name == "8v100_bag_1200":
            # Params for 8 x V100 32GB with gradients accumulation: --cluster=v100x8-westus2
            self.set_bag_sizes(1200)
            self.use_encoder_checkpointing = True


class TFF3_HECytedMIL_SwinT(TFF3_HECytedMIL):
    """TFF3 HE Cyted MIL model using SwinTransformer encoder."""

    def __init__(self, **kwargs: Any) -> None:
        self.dataset_suffix = "_cytedeastus2"  # comment this line when running outside of innereye4ws workspace
        default_kwargs = dict(
            encoder_type=SwinTransformer_NoPreproc.__name__,
            # Params for A100 80 GB: increase the docker_shm_size to 850g
            # --cluster=ND96v4-80GB-EastUS2 --workspace_config=config.json --docker_shm_size=1850g
            use_encoder_checkpointing=True,
            batchnorm_momentum=0.1,
            pl_static_graph=True,
            # adjust the learning rate and weight decay for SwinT
            l_rate=1e-5,
            weight_decay=1e-2,
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        self.set_bag_sizes(1100)


class TFF3_HECytedMIL_Densenet(TFF3_HECytedMIL):
    """TFF3 HE Cyted MIL model using DenseNet encoder."""

    def __init__(self, **kwargs: Any) -> None:
        self.dataset_suffix = "_v1"  # comment this line when running outside of innereye4ws workspace
        default_kwargs = dict(
            encoder_type=DenseNet121_NoPreproc.__name__,
            use_encoder_checkpointing=True,
            batchnorm_momentum=0.1,
            pl_static_graph=True,
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        self.set_bag_sizes(700)
