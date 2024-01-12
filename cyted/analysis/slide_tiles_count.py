#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
"""
This script can be used to count the number of tiles in a slide. It can be run locally or on AzureML.

For a full documentation of the parameters, run `python cyted/analysis/slide_tiles_count.py --help`
"""
from argparse import ArgumentParser
import sys
from pathlib import Path

cpath_root = Path(__file__).resolve().parents[2]
sys.path.append(str(cpath_root))

from cyted.fixed_paths import add_submodules_to_path, repository_root_directory  # noqa: E402

add_submodules_to_path()

from cyted.analysis.analysis_configs import TilesCountConfig  # noqa: E402
from health_azure.logging import logging_to_stdout  # noqa: E402
from health_azure.argparsing import apply_overrides, create_argparser, parse_arguments  # noqa: E402


def create_tiles_count_argparser() -> ArgumentParser:
    """Creates the argument parser for the Cyted cropping and tiff conversion script."""
    return create_argparser(
        TilesCountConfig(),
        usage="python cyted/analysis/slide_tiles_count.py --dataset <dataset> --dataset_csv <dataset_csv>"
        "--image_column <image_column> --label_column <label_column> --num_workers <num_workers>",
        description="Counts the number of tiles in a slide.",
    )


def create_config_from_args() -> TilesCountConfig:
    """Creates the config for cropping and tiff conversion from the command line arguments."""
    parser = create_tiles_count_argparser()
    config = TilesCountConfig()
    parser_results = parse_arguments(parser, args=sys.argv[1:], fail_on_unknown_args=True)
    _ = apply_overrides(config, parser_results.args)
    return config


def main() -> None:
    config = create_config_from_args()
    logging_to_stdout()
    submit_to_azureml = config.cluster != ""

    if config.dataset.strip() == "":
        raise ValueError("Please provide a dataset name via --dataset")

    elif config.dataset.startswith("/"):
        if submit_to_azureml:
            raise ValueError("Cannot submit to AzureML if dataset is a local folder")

        input_folders = config.get_input_folders_for_local_run()
        output_folder = repository_root_directory("outputs/tiles_count")
        output_folder.mkdir(parents=True, exist_ok=True)

    else:
        current_file = Path(__file__).absolute()
        repository_root = current_file.parent.parent.parent
        run_info = config.submit_to_azureml_if_needed(current_file, repository_root, submit_to_azureml)

        input_folders = config.get_input_folders_from_aml_run(run_info)
        assert run_info.output_folder is not None
        output_folder = run_info.output_folder

    config.analyse_dataset(input_folders, output_folder)


if __name__ == "__main__":
    main()
