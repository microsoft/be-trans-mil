#! /usr/bin/env python

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path
import time

import pytest
import param
from _pytest.main import ExitCode
from azureml._restclient.constants import RunStatus
from azureml.core import Run

from cyted import fixed_paths
from cyted.data_paths import CYTED_DEFAULT_DATASET_LOCATION

repo_root_folder = Path(__file__).resolve().parent
print(f"Repository root folder: {repo_root_folder}")
fixed_paths.add_submodules_to_path()

from health_azure import DatasetConfig, submit_to_azure_if_needed  # noqa: E402
from health_azure.himl import DEFAULT_DOCKER_BASE_IMAGE, OUTPUT_FOLDER  # noqa: E402
from health_azure.logging import logging_to_stdout  # noqa: E402
from health_azure.argparsing import (  # noqa: E402
    create_argparser,
    parse_arguments,
)
from health_ml.utils.common_utils import DEFAULT_AML_UPLOAD_DIR  # noqa: E402

PYTEST_RESULTS_FILE = "pytest_results.xml"
PYTEST_GPU_COVERAGE_FILE = "pytest_gpu_coverage.xml"


class RunPytestConfig(param.Parameterized):
    mark: str = param.String(default="", doc="The value to pass to pytest for the -m (mark) argument.")
    folder: str = param.String(
        default="",
        doc="The file or folder of tests that should be run. This value is used as the first argument to start "
        "pytest, so it can also be a specific test like 'my_test.py::any_test'",
    )
    coverage_module: str = param.String(
        default="",
        doc="This value is used as an argument to --cov of pytest to collect code coverage for the specified pyhton "
        "module. For example, in the subfolder hi-ml-cpath, one can collect code coverage for the "
        "histopathology module by setting `module=histopathology`. If set to '' (default), no coverage is collected.",
    )
    cluster: str = param.String(default="", doc="The name of the AzureML compute cluster where the script should run.")
    conda_env: Path = param.ClassSelector(
        class_=Path,
        default=repo_root_folder / "environment.yml",
        doc="The path to the Conda environment file that should be used when starting pytest in AzureML.",
    )
    experiment: str = param.String(
        default="run_pytest", doc="The name of the AzureML experiment where the run should start."
    )
    max_run_duration: str = param.String(
        default="30m",
        doc="The maximum runtime that is allowed for this job in AzureML. This is given as a floating"
        "point number with a string suffix s, m, h, d for seconds, minutes, hours, day. Examples: '3.5h', '2d'",
    )
    strictly_aml_v1: bool = param.Boolean(
        default=True, doc="If True (default), use AzureML v1 SDK. If False, use the v2 of the SDK"
    )


def run_pytest(folder_to_test: str, pytest_mark: str, coverage_module: str) -> None:
    """
    Runs pytest on a given folder, restricting to the tests that have the given pytest mark.
    If pytest finds no tests, or any of the tests fail, this function raises a ValueError. When run inside
    AzureML, this will make the job fail.

    :param pytest_mark: The pytest mark to use for filtering out the tests to run.
    :param folder_to_test: The folder with tests that should be run.
    :param coverage_module: The module for which test code coverage should be collected. When set to empty string '', no
        code coverage is collected.
    """
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / PYTEST_RESULTS_FILE
    pytest_args = [folder_to_test, f"--junitxml={str(results_file)}"]

    if coverage_module:
        pytest_args += [
            f"--cov={coverage_module}",
            "--cov-branch",
            "--cov-report=html",
            f"--cov-report=xml:{OUTPUT_FOLDER}/{PYTEST_GPU_COVERAGE_FILE}",
            "--cov-report=term-missing",
            "--cov-config=.coveragerc",
        ]
    if pytest_mark:
        pytest_args += ["-m", pytest_mark]
    logging.info(f"Starting pytest with these args: {pytest_args}")
    status_code = pytest.main(pytest_args)
    if status_code == ExitCode.NO_TESTS_COLLECTED:
        raise ValueError(f"PyTest did not find any tests to run, when restricting with this mark: {pytest_mark}")
    if status_code != ExitCode.OK:
        raise ValueError(f"PyTest failed with exit code: {status_code}")


def download_run_output_file(blob_path: Path, destination: Path, run: Run) -> Path:
    """
    Downloads a single file from the run's default output directory: ("outputs").

    :param blob_path: The relative path to the file to download. For example, if blobs_path = "foo/bar.csv", then the
        run result file "outputs/foo/bar.csv" will be downloaded to <destination>/bar.csv (the directory will be
        stripped off).
    :param run: The AzureML run to download the files from.
    :param destination: Local path to save the downloaded blob to.
    :return: Destination path to the downloaded file(s).
    """
    blobs_prefix = str((DEFAULT_AML_UPLOAD_DIR / blob_path).as_posix())
    destination = destination / blob_path.name
    logging.info(f"Downloading single file from run {run.id}: {blobs_prefix} -> {str(destination)}")
    try:
        run.download_file(blobs_prefix, str(destination), _validate_checksum=True)
    except Exception as ex:
        raise ValueError(f"Unable to download file '{blobs_prefix}' from run {run.id}") from ex
    return destination


def download_pytest_coverage_result(run: Run, destination_folder: Path = Path.cwd()) -> Path:
    """
    Downloads the pytest result file that is stored in the output folder of the given AzureML run.
    If there is no pytest result file, throw an Exception.
    :param run: The run from which the files should be read.
    :param destination_folder: The folder into which the pytest result file is downloaded.
    :return: The path (folder and filename) of the downloaded file.
    """
    logging.info(f"Downloading pytest gpu coverage file: {PYTEST_GPU_COVERAGE_FILE}")
    try:
        return download_run_output_file(Path(PYTEST_GPU_COVERAGE_FILE), destination=destination_folder, run=run)
    except Exception as ex:
        raise ValueError(f"No pytest result file {PYTEST_GPU_COVERAGE_FILE} was found for run {run.id}") from ex


def pytest_after_submission_hook(azure_run: Run) -> None:
    """A hook that will be called right after pytest gpu tests submission."""
    # We want the job output to be visible on the console. Do not exit yet if the job fails, because we
    # may need to download the pytest result file.
    azure_run.wait_for_completion(show_output=True, raise_on_error=False)
    # The AzureML job can optionally run pytest. Attempt to download it to the current directory.
    # A build step will pick up that file and publish it to Azure DevOps.
    # If pytest_mark is set, this file must exist.
    logging.info("Downloading pytest result file.")
    download_pytest_coverage_result(azure_run)
    if azure_run.status == RunStatus.FAILED:
        raise ValueError(f"The AzureML run failed. Please check this URL for details: " f"{azure_run.get_portal_url()}")


if __name__ == "__main__":
    config = RunPytestConfig()

    parser = create_argparser(
        config,
        description="Invoke pytest either locally or inside of an AzureML run. The value of the '--folder' option is "
        "becoming the first argument to pytest.To run on AzureML, provide the '--cluster' option.",
    )
    parser_results = parse_arguments(parser, fail_on_unknown_args=True)
    config = RunPytestConfig(**parser_results.args)
    logging_to_stdout()
    submit_to_azureml = config.cluster != ""
    dataset_config = DatasetConfig(
        name="cyted_full_container",
        target_folder=CYTED_DEFAULT_DATASET_LOCATION,
        use_mounting=True,
    )
    submit_to_azure_if_needed(
        compute_cluster_name=config.cluster,
        submit_to_azureml=submit_to_azureml,
        wait_for_completion=True,
        snapshot_root_directory=repo_root_folder,
        conda_environment_file=config.conda_env,
        experiment_name=config.experiment,
        max_run_duration=config.max_run_duration,
        after_submission=pytest_after_submission_hook,
        docker_base_image=DEFAULT_DOCKER_BASE_IMAGE,
        docker_shm_size="40g",
        strictly_aml_v1=config.strictly_aml_v1,
        input_datasets=[dataset_config],
    )
    run_pytest(folder_to_test=config.folder, pytest_mark=config.mark, coverage_module=config.coverage_module)
    time.sleep(10)  # Give the AzureML job time to finish uploading the pytest result file.
