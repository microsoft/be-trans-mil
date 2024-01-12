#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import sys
from pathlib import Path

cpath_root = Path(__file__).resolve().parent
sys.path.append(str(cpath_root))

from cyted import fixed_paths  # noqa: E402

fixed_paths.add_submodules_to_path()

from health_ml.runner import Runner  # noqa: E402


def main() -> None:
    runner = Runner(project_root=fixed_paths.repository_root_directory())
    runner.run()


if __name__ == "__main__":
    main()
