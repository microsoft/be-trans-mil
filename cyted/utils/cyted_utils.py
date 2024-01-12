#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Callable, Dict, List, Optional
import pandas as pd
import param

from cyted.cyted_schema import CytedSchema, CytedLabel, CytedPathway, QC_PASS

from cyted.utils.split_utils import StratifyType
from health_cpath.models.transforms import NormalizeBackgroundd
from health_cpath.utils.naming import SlideKey

BIOMARKER_COLUMN = "Biomarker"
CLINICAL_RISK_COLUMN = "Clinical risk"
RISK_GROUP_COLUMN = "Risk group"
QC_PASS_COLUMN = "QC pass"
IS_SURVEILLANCE_COLUMN = "Surveillance case"


def get_cyted_df_as_str(path: Path) -> pd.DataFrame:
    """Reads the Cyted dataset in its original format, with string values, not making any attempts to parse."""
    return pd.read_csv(path, sep="\t", dtype=str, na_values=(), keep_default_na=False, index_col=CytedSchema.CytedID)


def sanitise_prague_columns(df: pd.DataFrame) -> None:
    """Parse Prague measurements in a Cyted dataframe from string to float (with "<1"→0.5)."""

    def _parse_prague_measurement(x: str) -> float:
        return 0.5 if x == "<1" else float(x)

    prague_columns = [CytedSchema.PragueM, CytedSchema.PragueC]
    for column in prague_columns:
        if df[column].dtype == object:  # Only parse if not already parsed (e.g. column is string)
            df[column] = df[column].map(_parse_prague_measurement)


def sanitise_label_columns(df: pd.DataFrame) -> None:
    """Parse "Y"/"N" labels in a Cyted dataframe as Boolean."""
    for column in CytedSchema.label_columns():
        df[column] = df[column].map({CytedLabel.Yes: True, CytedLabel.No: False})


def add_risk_columns(df: pd.DataFrame) -> None:
    """Add risk stratification columns to a Cyted dataframe in-place.

    The added columns (`Biomarker`, `Clinical risk`, and `Risk group`) are as defined in:

    * Pilonis, Killcoyne, et al. (2022). Lancet Oncology, 23(2):270-278.
      https://doi.org/10.1016/S1470-2045(21)00667-7
    """

    def _get_clinical_risk_factor(df: pd.DataFrame) -> pd.Series:
        prague_m, prague_c = df[CytedSchema.PragueM], df[CytedSchema.PragueC]
        ultralong_segment = (prague_m > 10) | (prague_c > 6)
        long_segment = (prague_m > 5) | (prague_c >= 3)
        demographic_risk = (df[CytedSchema.Sex] == "Male") | (df[CytedSchema.Age] > 60)
        return ultralong_segment | (long_segment & demographic_risk)

    def _get_risk_group(biomarker: pd.Series, clinical_risk: pd.Series) -> pd.Series:
        risk_group = pd.Series("high", index=biomarker.index, name=RISK_GROUP_COLUMN)
        risk_group[~biomarker & clinical_risk] = "moderate"
        risk_group[~biomarker & ~clinical_risk] = "low"
        return risk_group

    df[BIOMARKER_COLUMN] = df[CytedSchema.Atypia] | df[CytedSchema.P53Positive]
    df[CLINICAL_RISK_COLUMN] = _get_clinical_risk_factor(df)
    df[RISK_GROUP_COLUMN] = _get_risk_group(biomarker=df[BIOMARKER_COLUMN], clinical_risk=df[CLINICAL_RISK_COLUMN])


def preprocess_cyted_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse numerical/Boolean columns of a Cyted dataframe and add risk group columns.

    :param df: Cyted dataframe.
    :return: Cleaned copy of the dataframe, with parsed numerical `Prague (C/M)` columns and
        Boolean labels (`TFF3 positive`, `P53 positive`, and `Atypia`). Also includes risk
        columns (`Biomarker`, `Clinical risk`, and `Risk group`, as defined in Pilonis, Killcoyne,
        et al. 2022) and convenience Boolean columns `Surveillance case` and `QC pass`.
    """
    df = df.copy()

    sanitise_prague_columns(df)
    sanitise_label_columns(df)
    add_risk_columns(df)

    df[IS_SURVEILLANCE_COLUMN] = df[CytedSchema.PatientPathway] == CytedPathway.Barretts
    df[QC_PASS_COLUMN] = df[CytedSchema.QCReport] == QC_PASS

    return df


class CytedParams(param.Parameterized):
    label_column: str = param.String(
        default=CytedSchema.TFF3Positive, doc=f"The label column to use. Can be one of {CytedSchema.label_columns()}"
    )
    image_column: str = param.String(
        default=CytedSchema.HEImage, doc=f"The image column to use. Can be one of {CytedSchema.image_columns()}"
    )
    stratify_by: Optional[StratifyType] = param.List(
        default=None,
        doc=f"Columns to stratify by. Can be any of the Cyted metadata columns {CytedSchema.metadata_columns()}.",
    )
    # Set either `background_q_percentile` or `background_keys` to normalize images' background before tiling
    background_q_percentile: Optional[int] = param.Integer(
        default=None, allow_None=True, doc="Percentile of background pixel intensities to use for normalization."
    )
    background_keys: Optional[List[str]] = param.List(
        default=None, allow_None=True, doc="Keys to use for background normalization. If empty, use all images."
    )
    bag_size_subsample: int = param.Integer(default=600, doc="Number of images to subsample from each bag.")

    def validate(self) -> None:
        if self.label_column not in CytedSchema.label_columns():
            raise ValueError(f"Invalid label column: {self.label_column}. Choose one of {CytedSchema.label_columns()}")
        if self.image_column not in CytedSchema.image_columns():
            raise ValueError(f"Invalid image column: {self.image_column}. Choose one of {CytedSchema.image_columns()}")

    def get_cyted_params_aml_tags(self) -> Dict[str, str]:
        """Get AML tags for the current parameter values."""
        return {
            "label_column": self.label_column,
            "image_column": self.image_column,
            "stratify_by": str(self.stratify_by),
            "background_q_percentile": str(self.background_q_percentile),
            "background_keys": str(self.background_keys),
        }

    def get_background_normalization_transform(self) -> List[Callable]:
        if self.background_keys or self.background_q_percentile:
            return [NormalizeBackgroundd(SlideKey.IMAGE, self.background_keys, self.background_q_percentile)]
        else:
            return []
