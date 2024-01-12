#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import List, Set


class CytedSchema:
    """Contains constants for the names of columns in the Cyted dataset."""

    CytedID = "CYT full ID"
    HEImage = "H&E"
    TFF3Image = "TFF3"
    P53Image = "P53"
    P53Positive = "P53 positive"
    TFF3Positive = "TFF3 positive"
    Atypia = "Atypia"
    QCReport = "QC Report"
    PatientPathway = "Patient pathway"
    Sex = "Sex"
    Age = "Age"
    Smoking = "Smoking"
    PragueC = "Prague (C)"
    PragueM = "Prague (M)"
    Background = "background"
    Background_r = "background_r"
    Background_g = "background_g"
    Background_b = "background_b"
    Year = "year"

    @staticmethod
    def label_columns() -> Set[str]:
        """Returns all columns that contain labels."""
        return {CytedSchema.TFF3Positive, CytedSchema.P53Positive, CytedSchema.Atypia}

    @staticmethod
    def image_columns() -> Set[str]:
        """Returns all columns that contain images."""
        return {CytedSchema.HEImage, CytedSchema.TFF3Image, CytedSchema.P53Image}

    @staticmethod
    def background_columns() -> List[str]:
        return [
            CytedSchema.Background_r,
            CytedSchema.Background_g,
            CytedSchema.Background_b,
        ]

    @staticmethod
    def metadata_columns() -> Set[str]:
        """Returns all columns that contain metadata (largely, anything but the image columns)."""
        return {
            CytedSchema.P53Positive,
            CytedSchema.TFF3Positive,
            CytedSchema.Atypia,
            CytedSchema.PatientPathway,
            CytedSchema.QCReport,
            CytedSchema.Sex,
            CytedSchema.Age,
            CytedSchema.Smoking,
            CytedSchema.PragueC,
            CytedSchema.PragueM,
            CytedSchema.Year,
        }


class CytedPathway:
    """Contains constants for the different patient pathways in the Cyted dataset."""

    Barretts = "Barrett's Surveillance"
    Screening = "Reflux screening"


class CytedLabel:
    """Contains constants for the values of the label columns in the Cyted dataset."""

    No = "N"
    Yes = "Y"
    Unknown = "NA"

    @staticmethod
    def yes_or_no() -> Set[str]:
        """Returns all valid values for a label: 'Y' or 'N'. 'NA' is not a valid value."""
        return {
            CytedLabel.Yes,
            CytedLabel.No,
        }


QC_PASS = "pass"
