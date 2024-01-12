#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""
Creates a train/val/test split from a dataset CSV file.
Splits are stratified by the specified columns. In addition, an exclusion list can be provided.
Statistics about specific columns are printed to verify the results of the stratification.
"""
import argparse
from pathlib import Path
import sys
from typing import List, Optional

import pandas as pd

cpath_root = Path(__file__).resolve().parents[2]
sys.path.append(str(cpath_root))

from cyted import fixed_paths  # noqa: E402

fixed_paths.add_submodules_to_path()

from cyted.utils.split_utils import split_dataframe, ungroup_series  # noqa: E402

DEFAULT_SPLIT_COLUMN = "split"
DEFAULT_SPLIT_FILENAME = "splits.csv"
IS_EXCLUDED_COLUMN = "is_excluded"
IS_GROUP_EXCLUDED_COLUMN = "is_group_excluded"


def main(
    input_csv_path: Path,
    output_csv_path: Path,
    test_frac: float,
    index: str,
    strata_columns: Optional[List[str]],
    group_column: Optional[str],
    print_columns: List[str],
    interactive: bool,
    delimiter: Optional[str] = ",",
    exclusion_csv_path: Optional[Path] = None,
    partition_column: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    dataset_df = pd.read_csv(input_csv_path, delimiter=delimiter)

    print(f"All available columns: {', '.join(dataset_df.columns)}\n")

    strata_columns = strata_columns or []

    if exclusion_csv_path is not None:
        excluded_values: pd.Series = pd.read_csv(exclusion_csv_path).squeeze("columns")
        exclusion_column = excluded_values.name
        assert isinstance(exclusion_column, str)
        is_excluded = dataset_df[exclusion_column].isin(excluded_values)
        dataset_df[IS_EXCLUDED_COLUMN] = is_excluded
        exclusion_indicator_column_to_stratify = IS_EXCLUDED_COLUMN

        if group_column:
            groups = dataset_df[group_column]
            # A group is excluded only if all its elements are excluded
            is_group_excluded = is_excluded.groupby(groups).all()
            dataset_df[IS_GROUP_EXCLUDED_COLUMN] = ungroup_series(is_group_excluded, groups=groups)
            exclusion_indicator_column_to_stratify = IS_GROUP_EXCLUDED_COLUMN

        strata_columns.append(exclusion_indicator_column_to_stratify)

    # Check for NaNs in strata columns, remove row if there is a NaN
    print(f"Number of rows before removing NaNs: {len(dataset_df)}\n")
    dataset_df = dataset_df.dropna(subset=strata_columns)
    print(f"Number of rows after removing NaNs: {len(dataset_df)}\n")

    # Set the index after exclusion in case we need to select on the index column
    dataset_df.set_index(index, inplace=True)

    splits_column = split_dataframe(
        dataset_df, test_frac=test_frac, stratify=strata_columns, group=group_column or "", seed=seed
    )
    splits_column.name = DEFAULT_SPLIT_COLUMN
    dataset_df[DEFAULT_SPLIT_COLUMN] = splits_column

    if print_columns:
        print_column_distributions(
            dataset_df,
            print_columns,
            group_column,
            with_exclusion=exclusion_csv_path is not None,
            partition=partition_column,
        )

    should_save = True
    if interactive:
        response = input(f'Would you like to save this split to "{output_csv_path.resolve()}"? [y/N] ')
        should_save = response.lower() == "y"

    if should_save:
        splits_column.to_csv(output_csv_path)
    else:
        print("Split discarded, nothing saved to disk.")


def print_column_distributions(
    df: pd.DataFrame,
    columns: List[str],
    group: Optional[str] = None,
    with_exclusion: bool = False,
    partition: Optional[str] = None,
) -> None:
    def format_fraction(count: int, total: int) -> str:
        return f"{count} ({100 * count / total:5.1f}%)"

    def format_counts(counts: pd.Series, exclude_last: bool) -> pd.Series:
        total = (counts[:-1] if exclude_last else counts).sum()
        return counts.map(lambda x: format_fraction(x, total))

    def get_count_table(df_: pd.DataFrame, column: str) -> pd.DataFrame:
        split_columns = [DEFAULT_SPLIT_COLUMN]
        if partition:
            split_columns.append(partition)

        # Compute individual counts
        counts_df = pd.crosstab([df_[col] for col in split_columns], df_[column], margins=True)
        if group:
            # Compute grouped counts
            group_counts_df = df_.pivot_table(
                index=split_columns, columns=column, values=group, aggfunc=pd.Series.nunique, fill_value=0, margins=True
            )
            # Juxtapose individual and grouped counts tables (vertically, but later transposed)
            counts_df = pd.concat(
                {f"ungrouped ({df_.index.name})": counts_df, f"grouped ({group})": group_counts_df}, axis="index"
            )
        return counts_df.apply(format_counts, axis="columns", exclude_last=True).T

    def append_empty_row(df_: pd.DataFrame) -> pd.DataFrame:
        new_row = pd.Series("", index=df_.columns, name="").to_frame().T
        new_row.index.name = df_.index.name
        return pd.concat([df_, new_row], axis="index")

    filtered_df = df[~df[IS_EXCLUDED_COLUMN]] if with_exclusion else None

    for column in columns:
        full_counts_df = get_count_table(df, column)
        if filtered_df is not None:
            filtered_counts_df = get_count_table(filtered_df, column)
            # Vertically juxtapose full and filtered counts tables with blank like in-between
            full_counts_df = append_empty_row(full_counts_df)
            full_counts_df = pd.concat({"full": full_counts_df, "filtered": filtered_counts_df}, axis="index")
        print(full_counts_df, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a train/test split for an existing dataset CSV file. "
        f'The output will be a CSV file with an index column and a "{DEFAULT_SPLIT_COLUMN}" column '
        'containing "train"/"test" labels.',
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the input dataset CSV.",
    )

    parser.add_argument(
        "test_frac",
        type=float,
        help="Fraction of the dataset to reserve for testing, between 0 and 1.",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        type=Path,
        help=f'Output path for the splits CSV. Defaults to "<input_dir>/{DEFAULT_SPLIT_FILENAME}".',
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Column name to use for indexing (e.g. image ID).",
    )
    parser.add_argument(
        "--group",
        help="Column name to use for grouping (e.g. subject ID) to prevent data leakage, if applicable.",
    )
    parser.add_argument(
        "--stratify",
        type=lambda s: s.split(","),
        default=[],
        help='Comma-separated list of column names to use for stratification (e.g. "label,important_metadata"), '
        "assuming discrete values. The split will be such that their joint distributions are approximately equal. "
        "When multiple columns are given, it may fail if there aren't enough samples for every combination.",
    )
    parser.add_argument(
        "--interactive",
        default=True,
        help="If true (default), will ask for user confirmation before writing the output CSV.",
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="Delimiter in input CSV.",
    )

    parser.add_argument(
        "--distribution",
        type=lambda s: s.split(","),
        default=[],
        help="Comma-separated list of additional columns whose distributions to print for verification "
        '(e.g. "other_label,other_metadata"). This is useful to check that unstratified attributes are '
        "well balanced across splits.",
    )
    parser.add_argument(
        "--partition",
        help="Name of a secondary column to use when visualising distributions (e.g. data subsets used separately "
        "downstream). Partitions by this column will be displayed within the train and test splits for "
        "inspection, but will not affect the splitting itself.",
    )
    parser.add_argument(
        "--exclusion-csv",
        type=Path,
        help="Path to a single-column CSV containing values to exclude from the input dataset CSV (e.g. image IDs). "
        "Its header should match a column in the dataset CSV, and any rows whose value in this column in found in "
        "the exclusion CSV will be removed before the split is computed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The random seed to use in generating the splits, for reproducibility. "
        "If omitted, will produce different random splits every time by default.",
    )

    args = parser.parse_args()

    if args.output_csv is None:
        assert isinstance(args.input_csv, Path)
        args.output_csv = args.input_csv.parent / DEFAULT_SPLIT_FILENAME

    for column in args.stratify:
        if column not in args.distribution:
            args.distribution.append(column)

    main(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        delimiter=args.delimiter,
        test_frac=args.test_frac,
        index=args.index,
        group_column=args.group,
        strata_columns=args.stratify,
        print_columns=args.distribution,
        interactive=args.interactive,
        seed=args.seed,
        exclusion_csv_path=args.exclusion_csv,
        partition_column=args.partition,
    )
