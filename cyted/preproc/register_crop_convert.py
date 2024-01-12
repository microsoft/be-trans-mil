#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
"""
This script can be used to register TFF3 slides to H&E slides, and crop and convert NDPI slides to TIFF images.

For a full documentation of the parameters, run `python cyted/preproc/register_crop_convert.py --help`
"""
from argparse import ArgumentParser
import sys
from pathlib import Path
from cyted.fixed_paths import add_submodules_to_path

add_submodules_to_path()
from cyted.utils.preprocessing_configs import CytedRegistrationConfig  # noqa
from health_azure.logging import logging_to_stdout  # noqa
from health_azure.argparsing import apply_overrides, create_argparser, parse_arguments  # noqa


def create_tiff_register_crop_argparser() -> ArgumentParser:
    """Creates the argument parser for the Cyted cropping and tiff conversion script."""
    return create_argparser(
        CytedRegistrationConfig(),
        usage="python register_crop_convert.py --dataset <fixed_dataset> --dataset_csv <dataset_csv>"
        "--image_column <image_column> --label_column <label_column> "
        "--fixed_dataset <fixed_dataset> --fixed_mask_dataset <fixed_mask_dataset>"
        "--input_transforms_dataset <input_transforms_dataset> --output_dataset <output_dataset>"
        "--output_diff_dataset <output_diff_dataset> --output_transforms_dataset <output_transforms_dataset>"
        "--target_magnifications <target_magnifications> --num_workers <num_workers>"
        "--bounding_box_path <bb_path_moving> --bounding_box_path_fixed <bb_path_fixed>"
        "--hardcode_background <hardcode_background>",
        description="Registers, converts and crops the TFF3 slides dataset.",
    )


def create_config_from_args() -> CytedRegistrationConfig:
    """Creates the config for cropping and tiff conversion from the command line arguments."""
    parser = create_tiff_register_crop_argparser()
    config = CytedRegistrationConfig()
    parser_results = parse_arguments(parser, args=sys.argv[1:], fail_on_unknown_args=True)
    _ = apply_overrides(config, parser_results.args)
    if config.automatic_output_name:
        # apply overrides resets the output dataset name, so we need to add it back
        config.output_dataset += config.get_auto_output_name()
        if config.output_diff_dataset:
            config.output_diff_dataset += config.get_auto_output_name() + "_diff"
        if config.output_transforms_dataset:
            config.output_transforms_dataset += config.get_auto_output_name() + "_transforms"
    return config


def main() -> None:
    config = create_config_from_args()
    logging_to_stdout()
    submit_to_azureml = config.cluster != ""

    if config.dataset.strip() == "":
        raise ValueError("Please provide a moving (e.g. TFF3) dataset name via --dataset")
    elif str(config.fixed_dataset).strip() == "":
        raise ValueError("Please provide a reference or fixed (e.g. H&E) dataset name via --fixed_dataset")
    elif (
        config.dataset.startswith("/")
        or str(config.fixed_dataset).startswith("/")
        or config.output_dataset.startswith("/")
        or (config.fixed_mask_dataset and config.fixed_mask_dataset.is_absolute())
    ):
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
