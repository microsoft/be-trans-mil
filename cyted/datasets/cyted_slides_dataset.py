#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd
from cyted.data_paths import CYTED_DATASET_TSV, CYTED_SPLITS_CSV_FILENAME
from cyted.utils.split_utils import split_dataframe_using_splits_csv

from health_cpath.datasets.base_dataset import SlidesDataset
from cyted.cyted_schema import CytedSchema, CytedLabel, QC_PASS
from cyted.utils.cyted_utils import sanitise_prague_columns

PathOrString = Union[Path, str]


def filter_dataframe(
    df: pd.DataFrame, image_column: str, label_column: str, column_filter: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Applies a set of filtering operations to sanitize the Cyted dataset: Removing all rows where the
    pathologist's QC failed, removing all rows where the label is unknown.

    All label columns (TFF3 and P53 positive) are mapped from Y/N/NA to 1/0/nan.
    The Prague measurements have the "<1" values replaced with 0.5.

    :param df: The dataframe to filter.
    :param label_column: CSV column name for slide label.
    :param column_filter: A dictionary mapping column names to values. Only rows with the given values are kept.
    """
    if CytedSchema.QCReport in df:
        logging.info(f"Removing all rows that did not pass pathology QC. Before filtering: {len(df)} rows")
        df = df[df[CytedSchema.QCReport] == QC_PASS]

    logging.info(f"Filtering for valid labels. Before filtering: {len(df)} rows")
    if df[label_column].dtype == object:  # Only filter if the column is not already numeric
        df = df[df[label_column].isin(CytedLabel.yes_or_no())]

    logging.info(f"Filtering for valid image path. Before filtering: {len(df)} rows")
    # Missing images are indicated by NA. Dataset reading already maps that to NaN, but filter twice to be sure.
    df = df[~df[image_column].isna()]
    df = df[~df[image_column].isin((CytedLabel.Unknown, ""))]

    if column_filter is not None:
        logging.info(f"Applying column filter to dataset. Before filtering: {len(df)} rows")
        for column, value in column_filter.items():
            df = df[df[column] == value]

    # Replace "Y" and "N" with 1 and 0 in all label columns if necessary
    if df[label_column].dtype == object:
        labels = set(df[label_column].unique())
        allowed_labels = CytedLabel.yes_or_no()
        if len(labels - allowed_labels) != 0:
            raise ValueError(f"Expected labels to be one of {CytedLabel.yes_or_no()}, but got {labels}")
        for column in CytedSchema.label_columns():
            if column in df:  # Not all datasets have all label columns, e.g. when using a subset of the columns
                df[column] = df[column].replace({CytedLabel.Yes: 1, CytedLabel.No: 0, CytedLabel.Unknown: pd.NA})

    # Filter out slides where labels are not reported (NA)
    logging.info(f"Filtering for valid labels. Before filtering: {len(df)} rows")
    df = df[~df[label_column].isna()]
    logging.info(f"After applying all filters: {len(df)} rows")

    if CytedSchema.PragueC in df and CytedSchema.PragueM in df:
        sanitise_prague_columns(df)

    # Replace all columns that contain the string "NA" with NaN.
    df = df.replace(CytedLabel.Unknown, pd.NA)

    return df


class CytedSlidesDataset(SlidesDataset):
    """Base dataset class for loading Cyted slides.

    Iterating over this dataset returns a dictionary following the `SlideKey` schema plus meta-data
    from the Cyted metadat.
    """

    def __init__(
        self,
        root: PathOrString,
        image_column: str,
        label_column: str,
        dataset_csv: Optional[PathOrString] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        train: Optional[bool] = None,
        splits_csv: Optional[PathOrString] = None,
        excluded_slides_csv: Optional[PathOrString] = None,
        dataframe_kwargs: Optional[Dict[str, Any]] = None,
        column_filter: Optional[Dict[str, str]] = None,
        metadata_columns: Optional[Iterable] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param root: Root directory of the dataset.
        :param image_column: CSV column name for slide image path.
        :param label_column: CSV column name for slide label.
        :param dataset_csv: Full path to a dataset CSV file.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
            from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        :param train: If given, only returns the train or test set. If None, returns the full dataset.
        :param splits_csv: Full path to a splits CSV file, containing `SLIDE_ID_COLUMN` (index) and `SPLIT_COLUMN`.
            If omitted, the CSV will be read from `"{root}/{DEFAULT_SPLITS_CSV_FILENAME}"`.
            Ignored unless `train` is specified.
        :param excluded_slides_csv: Full path to a CSV file containing a list of slides to exclude from the dataset.
        :param dataframe_kwargs: Keyword arguments to pass to `pd.read_csv()` when loading the dataset CSV.
        :param column_filter: A dictionary mapping column names to values. Only rows with the given values are kept.
            This is not applied if `dataset_df` is given.
        :param metadata_columns: Additional metadata columns to add to the dataset. By default, the columns in
            `CytedSchema.metadata_columns()` are added.
        """
        metadata_columns = metadata_columns if metadata_columns is not None else CytedSchema.metadata_columns()
        dataframe_kwargs = dataframe_kwargs or {}
        dataframe_kwargs = {**dataframe_kwargs, "sep": "\t"}
        super().__init__(
            root=root,
            dataset_csv=dataset_csv,
            dataset_df=dataset_df,
            default_csv_filename=CYTED_DATASET_TSV,
            label_column=label_column,
            n_classes=1,
            validate_columns=False,
            dataframe_kwargs=dataframe_kwargs,
            slide_id_column=CytedSchema.CytedID,
            metadata_columns=tuple(metadata_columns),
            image_column=image_column,
            **kwargs,
        )
        assert isinstance(self.dataset_df, pd.DataFrame)
        # Apply QC filter only if reading from file. When passing in a dataframe, the user is responsible for filtering.
        if dataset_df is None:
            self.dataset_df = filter_dataframe(self.dataset_df, image_column, label_column, column_filter)

        # Remove excluded slides
        if excluded_slides_csv is not None:
            excluded_slide_ids: pd.Series = pd.read_csv(excluded_slides_csv).squeeze("columns")
            self.dataset_df = self.dataset_df[~self.dataset_df.index.isin(excluded_slide_ids)]

        # Add year column as first 2 characters of index column (CytedID)
        if CytedSchema.Year not in self.dataset_df:
            self.dataset_df.insert(0, CytedSchema.Year, self.dataset_df.index.str[:2])

        # Split into train and test
        if train is not None:
            splits_csv = splits_csv or self.root_dir / CYTED_SPLITS_CSV_FILENAME[self.label_column]
            self.dataset_df = split_dataframe_using_splits_csv(
                self.dataset_df, splits_csv, train=train, index_col=self.slide_id_column
            )

        self.validate_columns()
