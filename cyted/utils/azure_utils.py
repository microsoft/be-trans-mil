#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
from collections import defaultdict
from pathlib import Path
import re
from typing import DefaultDict, Dict, List, Optional

import pandas as pd

from azureml.core.run import Run
from azureml.core import Workspace
from azureml._restclient.constants import RunStatus
from health_azure import get_workspace
from health_azure.utils import download_file_if_necessary, get_aml_run_from_run_id
from health_cpath.utils.output_utils import AML_TEST_OUTPUTS_CSV, AML_VAL_OUTPUTS_CSV, OUTPUTS_CSV_FILENAME
from health_ml.utils.checkpoint_utils import AZUREML_RETRY_PREFIX
from health_ml.utils.common_utils import DEFAULT_AML_UPLOAD_DIR, df_to_json

TEST = "test"
EXTRA_VAL = "extra_val"
AML_REMOTE_FILENAMES = {TEST: AML_TEST_OUTPUTS_CSV, EXTRA_VAL: AML_VAL_OUTPUTS_CSV.replace("val", EXTRA_VAL)}
DictPathsType = Dict[str, List[Path]]


def get_input_dataset_ids_from_run(run: Run) -> List[str]:
    input_datasets_info = run.get_details()["inputDatasets"]
    return [dataset_info["dataset"].name for dataset_info in input_datasets_info]


def get_target_runs(
    parent_run_id: str,
    child_runs_status: Optional[RunStatus] = None,
    additional_run_ids: Optional[List[str]] = None,
    workspace_config: Optional[Path] = None,
) -> List[Run]:
    """Get the children runs of the parent run with the given run id and any additional runs with the given run ids.

    :param parent_run_id: The run id of the parent Hyperdrive run.
    :param child_runs_status: The status of the child runs to include in the aggregation. Defaults to None which means
        all child runs will be included. Available statuses are: 'RunStatus.COMPLETED', 'RunStatus.FAILED',
        'RunStatus.RUNNING'. Use 'Completed' to include only the successful runs.
    :param additional_run_ids: Any additional run ids to include in the aggregation. Defaults to None.
    :param workspace_config: The path to the workspace config file. Defaults to None.
    :return: List of the runs to include in the aggregation.
    """
    aml_workspace = get_workspace(workspace_config_path=workspace_config)
    parent_run = get_aml_run_from_run_id(parent_run_id, aml_workspace=aml_workspace)
    runs = [child for child in parent_run.get_children(status=child_runs_status)]
    if additional_run_ids:
        return runs + [get_aml_run_from_run_id(run_id) for run_id in additional_run_ids]
    return runs


def _get_remote_output_file(files: List[str], stage: str) -> str:
    """Filters a list of files in an AzureML run to only retain those that could be outputs files.
    This takes the folder structures for retries into account, where files are written into subfolders like
    `retry_001`.

    :param files: The list of file names to check.
    :param stage: The stage to check for (e.g. 'test' or 'extra_val').
    :return: A string with the name of the file that is the outputs file for the given stage.
    """
    file_pattern = (
        f"{DEFAULT_AML_UPLOAD_DIR}(/{AZUREML_RETRY_PREFIX}[0-9]" + "{3})?" + f"/{stage}/{OUTPUTS_CSV_FILENAME}"
    )
    regex = re.compile(file_pattern)
    matches = [f for f in files if regex.match(f)]
    matches.sort()
    return matches[-1]


def download_outputs_csv_from_runs(
    download_dir: Path, runs: List[Run], overwrite: bool = False, stages: List[str] = [TEST, EXTRA_VAL]
) -> DictPathsType:
    """Download the outputs csv files from the given runs.

    :param download_dir: The directory to save the downloaded files to.
    :param runs: The runs to download the outputs from.
    :param overwrite: A flag to indicate whether to overwrite the downloaded files if they already exist. Defaults to
        False.
    :param stages: A list of the stages to download the outputs for. Defaults to ['test', 'extra_val'].
    :return: A dictionary mapping the stage to a list of the downloaded outputs csv files.
    """
    downloaded_outputs_csv_paths: DictPathsType = {stage: [] for stage in stages}
    for stage in stages:
        logging.info(f"Downloading {stage} outputs files..")
        for i, run in enumerate(runs):
            output_filename = download_dir / f"{i}_{stage}.csv"
            remote_filename = _get_remote_output_file(run.get_file_names(), stage)
            download_file_if_necessary(run, remote_filename, output_filename, overwrite=overwrite)
            downloaded_outputs_csv_paths[stage].append(output_filename)
    return downloaded_outputs_csv_paths


def download_run_metrics_if_required(
    run_id: str,
    download_dir: Path,
    aml_workspace: Workspace,
    workspace_config_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Fetch metrics logged to Azure ML from a run.
    Will only download the metrics if they do not already exist locally

    :param run_id: Azure ML run ID for the run.
    :param download_dir: Directory where to save the downloaded metrics as `aml_metrics_{run_id}.json`.
    :param aml_workspace: Azure ML workspace in which the runs were executed.
    :param workspace_config_path: If not provided with an AzureML Workspace, then load one given the information in this
        config
    :param overwrite: Whether to force the download even if metrics are already saved locally.
    :return: The path of the downloaded json file.
    """
    metrics_json = download_dir / f"aml_metrics_{run_id}.json"
    if not overwrite and metrics_json.is_file():
        print(f"AML metrics file already exists at {metrics_json}")
    else:
        assert run_id is not None, "Run_id must be provided"
        workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
        run = get_aml_run_from_run_id(run_id, aml_workspace=workspace)
        metrics: DefaultDict = defaultdict()
        run_metrics = run.get_metrics()
        keep_metrics = run_metrics.keys()
        run_tag = run_id
        for metric_name, metric_val in run_metrics.items():
            if metric_name in keep_metrics:
                if metric_name not in metrics:
                    metrics[metric_name] = {}
                metrics[metric_name][run_tag] = metric_val
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing additional run AML metrics file to {metrics_json}")
        df_to_json(metrics_df, metrics_json)
    return metrics_json


def replace_ampersand_in_filename(filename: str, replace_by: str = "_") -> str:
    """Replace the ampersand in the given filename by a replacement string.

    :param filename: The filename to replace the ampersand in.
    :param replace_by: The string to replace the ampersand with. Defaults to '_'.
    :return: The filename with the ampersand replaced.
    """
    return filename.replace("&", replace_by)
