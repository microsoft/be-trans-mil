#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union

PathOrString = Union[Path, str]


def repository_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the present repository.

    :param: path: if provided, a relative path to append to the absolute path to the repository root.
    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path(__file__).resolve().parent.parent
    if path:
        return root / path
    else:
        return root


def get_hi_ml_submodule_relative_paths() -> List[Tuple[Path, str]]:
    """
    Returns the paths relative to the repository root where the submodules for hi-ml and hi-ml-azure are expected.
    It returns a list with a tuple (folder name, expected subfolder in that folder)
    """
    return [
        (Path("hi-ml") / "hi-ml-azure" / "src", "health_azure"),
        (Path("hi-ml") / "hi-ml" / "src", "health_ml"),
        (Path("hi-ml") / "hi-ml-cpath" / "src", "health_cpath"),
    ]


def add_to_path(path: PathOrString) -> None:
    """
    Adds the given path to sys.path if it is not already there.

    :param path: The path to add to sys.path
    """
    path_str = str(path)
    if path_str not in sys.path:
        logging.info(f"Adding folder {path} to sys.path")
        sys.path.insert(0, path_str)
    else:
        logging.debug(f"Not adding folder {path} because it is already in sys.path")


def add_submodules_to_path() -> None:
    """
    This function adds all submodules that the code uses to sys.path and to the environment variables. This is
    necessary to make the code work without any further changes when switching from/to using hi-ml as a package
    or as a submodule for development.
    It also adds the InnerEye root folder to sys.path. The latter is necessary to make AzureML and Pytorch Lightning
    work together: When spawning additional processes for DDP, the working directory is not correctly picked
    up in sys.path.
    """
    root = repository_root_directory()
    for folder_suffix, subfolder_that_must_exist in get_hi_ml_submodule_relative_paths():
        folder = root / folder_suffix
        if (folder / subfolder_that_must_exist).is_dir():
            add_to_path(folder)
        else:
            logging.debug(f"Not adding folder {folder} because it does not have subfolder {subfolder_that_must_exist}")
