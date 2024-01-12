#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""
This script creates a train/validation/test split for the cyted dataset, stratified within each of the 2 cohorts
in the Cyted datasets.
"""
import sys
import time
from pathlib import Path

import pandas as pd

cpath_root = Path(__file__).resolve().parents[2]
sys.path.append(str(cpath_root))

from cyted import fixed_paths  # noqa: E402

fixed_paths.add_submodules_to_path()

from cyted.cyted_schema import CytedPathway, CytedSchema  # noqa: E402
from cyted.preproc.generate_dataset_splits import main  # noqa: E402
from cyted.utils.cyted_utils import CLINICAL_RISK_COLUMN, preprocess_cyted_df  # noqa: E402


if __name__ == "__main__":
    # You NEED to have cytedWESTUS3 mounted in order to directly upload the file
    UPLOAD_TO_BLOB = False

    # Init folders and files
    output_folder = Path("train_split_outputs")
    output_folder.mkdir(exist_ok=True)

    cyted_path = Path("/cyted/cyted-raw-20221102")
    raw_path = cyted_path / "exclude_wrong_stains_2022-11-22.tsv"

    surveillance_path = output_folder / "surveillance.csv"
    surveillance_split_path = output_folder / "surveillance_split.csv"

    screening_path = output_folder / "screening.csv"
    screening_split_path = output_folder / "screening_split.csv"

    timestr = time.strftime("%Y%m%d_%H%M%S")
    merged_split_name = Path("splits_" + timestr + ".csv")
    merged_split_path = output_folder / merged_split_name

    # Load raw metadata
    raw_df = pd.read_csv(raw_path, sep="\t", header=0)

    # Create risk columns
    raw_df = preprocess_cyted_df(raw_df)

    # Split into patient pathways and save
    surveillance_df = raw_df.loc[raw_df[CytedSchema.PatientPathway] == CytedPathway.Barretts]
    screening_df = raw_df.loc[raw_df[CytedSchema.PatientPathway] == CytedPathway.Screening]

    surveillance_df.to_csv(surveillance_path)
    screening_df.to_csv(screening_path)

    # Stratify first cohort (=Barrett's Surveillance)
    main(
        input_csv_path=surveillance_path,
        output_csv_path=surveillance_split_path,
        test_frac=0.2,
        index=CytedSchema.CytedID,
        group_column=None,
        strata_columns=[CytedSchema.TFF3Positive, CLINICAL_RISK_COLUMN],
        print_columns=[
            CytedSchema.TFF3Positive,
            CLINICAL_RISK_COLUMN,
            CytedSchema.Sex,
            CytedSchema.Age,
            CytedSchema.Smoking,
            CytedSchema.PragueC,
            CytedSchema.PragueM,
            CytedSchema.P53Positive,
            CytedSchema.Atypia,
        ],
        interactive=False,
        seed=0,
        exclusion_csv_path=None,
        partition_column=None,
    )

    # Stratify second cohort (=Reflux screening)
    main(
        input_csv_path=screening_path,
        output_csv_path=screening_split_path,
        test_frac=0.2,
        index=CytedSchema.CytedID,
        group_column=None,
        strata_columns=[CytedSchema.TFF3Positive],
        print_columns=[
            CytedSchema.Sex,
            CytedSchema.Age,
            CytedSchema.Smoking,
            CytedSchema.P53Positive,
            CytedSchema.Atypia,
        ],
        interactive=False,
        seed=0,
        exclusion_csv_path=None,
        partition_column=None,
    )

    # Concatenate tables again
    surveillance_split_df = pd.read_csv(surveillance_split_path)
    print(list(surveillance_split_df.columns.values))
    screening_split_df = pd.read_csv(screening_split_path)
    print(list(screening_split_df.columns.values))
    merged_df = pd.concat([surveillance_split_df, screening_split_df], ignore_index=True)
    print(list(merged_df.columns.values))
    merged_df.to_csv(merged_split_path, index=False)

    # Sanity check
    splits_df = pd.read_csv(merged_split_path)
    print(list(splits_df.columns.values))
    assert len(splits_df.columns) == 2

    # Upload to blob
    if UPLOAD_TO_BLOB:
        splits_df.to_csv(cyted_path / merged_split_name, index=False)
