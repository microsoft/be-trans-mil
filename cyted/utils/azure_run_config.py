#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import logging
import param
from pathlib import Path
from typing import List, Optional, Dict

from cyted.cyted_schema import CytedSchema
from cyted.data_paths import CYTED_CSV_FILENAME
from cyted.datasets.cyted_slides_dataset import CytedSlidesDataset
from health_azure import AzureRunInfo, DatasetConfig, submit_to_azure_if_needed
from health_cpath.utils.montage_config import AzureRunConfig
from health_azure.argparsing import ListOrDictParam


class CytedAzureRunConfig(AzureRunConfig):
    """Cyted dataset base configuration for preprocessing raw data into tiff files."""

    output_dataset: str = param.String(
        default="",
        doc="The name of the dataset that will be created in the destination directory. If running locally, "
        "this should be the name of the directory. If running on Azure, this is the dataset name to be created.",
    )
    dataset_csv: Optional[str] = param.String(
        default=CYTED_CSV_FILENAME, doc="The name of the dataset csv file if not using the default name."
    )
    label_column: str = param.String(
        default=CytedSchema.TFF3Positive, doc=f"The label column to use. Can be one of {CytedSchema.label_columns()}"
    )
    image_column: str = param.String(
        default=CytedSchema.HEImage, doc=f"The image column to use. Can be one of {CytedSchema.image_columns()}"
    )
    excluded_slides_csv: Optional[Path] = param.ClassSelector(
        default=None, class_=Path, doc="A list of image names to exclude from the dataset."
    )
    column_filter: Optional[Dict[str, str]] = ListOrDictParam(
        default=None, doc="A dictionary of column names and values to filter the dataset by."
    )
    limit: Optional[int] = param.Integer(
        default=None, doc="The number of images to process. If None, all images will be processed."
    )
    slide_ids: List[str] = param.List(
        default=[],
        class_=str,
        doc="A list of slide ids to process. If None, all images will be processed. This option is exclusive with "
        "--limit flag. If both are specified, --limit will take precedence. The slide ids should be separated by commas"
        "(i.e., --slide_ids=id1,id2,id3).",
    )
    plot_path: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        doc="An optional path where to save intermediate plots. This is useful for debugging purposes. If None, no "
        "plots will be saved.",
    )

    def get_slides_dataset(self, dataset_root: Path) -> CytedSlidesDataset:
        """Returns the cyted dataset with the specified parameters.

        :param dataset_root: The root directory of the dataset.
        :return: The cyted dataset limited to the specified number of images.
        """
        dataset_csv = dataset_root / self.dataset_csv if self.dataset_csv else None
        dataset = CytedSlidesDataset(
            root=dataset_root,
            dataset_csv=dataset_csv,
            image_column=self.image_column,
            label_column=self.label_column,
            excluded_slides_csv=self.excluded_slides_csv,
            column_filter=self.column_filter,
        )
        if self.limit is not None:
            dataset.dataset_df = dataset.dataset_df.iloc[: self.limit]
        elif self.slide_ids:
            dataset.dataset_df = dataset.dataset_df[dataset.dataset_df.index.isin(self.slide_ids)]
        return dataset

    def get_input_datasets(self) -> List[DatasetConfig]:
        """Creates the input datasets configs for the AzureML run."""
        logging.info(f"In AzureML use mounted dataset '{self.dataset}' in datastore {self.datastore}")
        return [DatasetConfig(name=self.dataset, datastore=self.datastore, use_mounting=True)]

    def get_input_folders_for_local_run(self) -> List[Path]:
        """Returns the input datasets folders for local runs."""
        return [Path(self.dataset)]

    @staticmethod
    def create_folder_for_output_dataset(output_dataset: str) -> Path:
        """Creates folder for output dataset"""
        output_folder = Path(output_dataset)
        output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder

    @staticmethod
    def get_input_folders_from_aml_run(run_info: AzureRunInfo) -> List[Path]:
        """Returns the input datasets folders for AzureML runs."""
        input_folders: List[Path] = []
        for i, input_dataset in enumerate(run_info.input_datasets):
            assert input_dataset is not None, f"Input dataset folder {i} is None for AzureML run"
            input_folders.append(input_dataset)
        return input_folders

    def get_output_datasets(self) -> Optional[List[DatasetConfig]]:
        """Creates the output dataset config for the AzureML run."""
        logging.info(f"In AzureML create output dataset '{self.output_dataset}' in datastore {self.datastore}")
        return [DatasetConfig(name=self.output_dataset, datastore=self.datastore, use_mounting=True)]

    def get_output_folders_from_local_run(self) -> List[Path]:
        """Returns the output dataset folders for local runs."""
        output_folders = [self.create_folder_for_output_dataset(self.output_dataset)]
        return output_folders

    @staticmethod
    def get_output_folders_from_aml_run(run_info: AzureRunInfo) -> List[Path]:
        """Returns the output dataset folders for AzureML runs."""
        output_datasets: List[Path] = []
        for i, output_dataset in enumerate(run_info.output_datasets):
            assert output_dataset is not None, f"Output dataset folder {i} is None for AzureML run"
            output_datasets.append(output_dataset)
        return output_datasets

    def set_plot_path_for_local_run(self) -> None:
        """Sets the plot path to a local folder if it is not None."""
        if self.plot_path:
            self.plot_path.mkdir(parents=True, exist_ok=True)

    def set_plot_path_for_aml_run(config, run_info: AzureRunInfo) -> None:
        """Sets the plot path to a folder in the output folder of the Azureml run if it is not None."""
        if config.plot_path:
            assert run_info.output_folder is not None
            config.plot_path = run_info.output_folder / config.plot_path
            config.plot_path.mkdir(parents=True, exist_ok=True)

    def submit_to_azureml_if_needed(
        self, current_file: Path, repository_root: Path, submit_to_azureml: bool
    ) -> AzureRunInfo:
        """Submits the current script to AzureML if needed.

        :param current_file: The path to the current file.
        :param repository_root: The root directory of the repository.
        :param submit_to_azureml: A flag indicating whether to submit the script to AzureML.
        :return: An AzureRunInfo object.
        """
        logging.info(f"Submitting to AzureML, running on cluster {self.cluster}")
        return submit_to_azure_if_needed(
            entry_script=current_file,
            snapshot_root_directory=repository_root,
            compute_cluster_name=self.cluster,
            conda_environment_file=self.conda_env,
            submit_to_azureml=submit_to_azureml,
            input_datasets=self.get_input_datasets(),  # type: ignore
            output_datasets=self.get_output_datasets(),  # type: ignore
            strictly_aml_v1=True,
            docker_shm_size=self.docker_shm_size,
            wait_for_completion=self.wait_for_completion,
            workspace_config_file=self.workspace_config_path,
            display_name=self.display_name,
        )
