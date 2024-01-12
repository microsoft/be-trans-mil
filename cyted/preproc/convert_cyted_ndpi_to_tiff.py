#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
"""
This script can be used to convert NDPI slides to TIFF images.

For a full documentation of the parameters, run `python cyted/preproc/convert_cyted_ndpi_to_tiff.py --help`
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

cpath_root = Path(__file__).resolve().parents[2]
sys.path.append(str(cpath_root))

from cyted.fixed_paths import add_submodules_to_path  # noqa: E402

add_submodules_to_path()

from health_azure.logging import logging_to_stdout  # noqa: E402
from cyted.utils.preprocessing_configs import CytedTiffConversionConfig  # noqa: E402
from health_azure.argparsing import apply_overrides, create_argparser, parse_arguments  # noqa: E402


def create_tiff_conversion_argparser() -> ArgumentParser:
    """Creates the argument parser for the Cyted tiff conversion script."""
    return create_argparser(
        CytedTiffConversionConfig(),
        usage="python convert_cyted_ndpi_to_tiff.py --dataset <dataset> --dataset_csv <dataset_csv>"
        "--image_column <image_column> --label_column <label_column> --output_dataset <output_dataset> "
        "--target_magnifications <target_magnifications> --num_workers <num_workers>",
        description="Converts the slides dataset from ndpi to tiff format.",
    )


def create_config_from_args() -> CytedTiffConversionConfig:
    """Creates the tiff conversion config from the command line arguments."""
    parser = create_tiff_conversion_argparser()
    config = CytedTiffConversionConfig()
    parser_results = parse_arguments(parser, args=sys.argv[1:], fail_on_unknown_args=True)
    _ = apply_overrides(config, parser_results.args)
    return config


def main() -> None:
    config = create_config_from_args()
    logging_to_stdout()
    submit_to_azureml = config.cluster != ""

    if config.dataset.strip() == "":
        raise ValueError("Please provide a dataset name via --dataset")

    elif config.dataset.startswith("/") or config.output_dataset.startswith("/"):
        if submit_to_azureml:
            raise ValueError("Cannot submit to AzureML if dataset is a local folder")

        input_folders = config.get_input_folders_for_local_run()
        output_folders = config.get_output_folders_from_local_run()

    else:
        current_file = Path(__file__).absolute()
        repository_root = current_file.parent.parent.parent
        run_info = config.submit_to_azureml_if_needed(current_file, repository_root, submit_to_azureml)

        input_folders = config.get_input_folders_from_aml_run(run_info)
        output_folders = config.get_output_folders_from_aml_run(run_info)

    config.process_dataset(input_folders, output_folders)


if __name__ == "__main__":
    main()
