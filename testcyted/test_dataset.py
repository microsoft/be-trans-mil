#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from cyted.cyted_schema import CytedSchema, CytedPathway
from cyted.datasets.cyted_slides_dataset import CytedSlidesDataset
from health_cpath.utils.naming import SlideKey


def _write_dataset_snippet_and_load(
    tmp_path: Path, data_file_snippet: str, label_column: str, **kwargs: Any
) -> CytedSlidesDataset:
    dataset_tsv = tmp_path / "dataset.tsv"
    dataset_tsv.write_text(data_file_snippet)
    dataset = CytedSlidesDataset(
        root=tmp_path, dataset_csv=dataset_tsv, image_column=CytedSchema.HEImage, label_column=label_column, **kwargs
    )
    return dataset


# Disable check for long lines in the whole file for simplicity.
# flake8: noqa: E501


def test_cyted_slides_qc_filtering(tmp_path: Path) -> None:
    """Tests if the dataset loading applies the QC filter correctly, and returns the right images."""
    data_file_snippet = """CYT full ID	Pot ID	Patient pathway	QC Report	Sex	Age	Smoking	Prague (C)	Prague (M)	TFF3 positive	P53 positive	Atypia	H&E	TFF3	P53	NEG	Unknown	Extra	Rescan	Respin
21CYT03122	21P01143	Barrett's Surveillance	pass	Female	74.3	Past	0	3	Y	N	N	21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi	21CYT03122 21P01143 A1 TFF3 - 2021-09-13 16.11.39.ndpi	21CYT03122 21P01143 A1 P53 - 2021-09-13 16.13.40.ndpi	NA	NA	NA	NA	NA
22CYT01935	20P01895	Barrett's Surveillance	failed	Male	79.3	NA	0	3	Y	N	N	22CYT01935 20P01895 A1 HE - 2022-04-06 12.29.55.ndpi	22CYT01935 20P01895 A1 TFF3 - 2022-04-06 12.31.44.ndpi	22CYT01935 20P01895 A1 P53 - 2022-04-06 12.34.09.ndpi	NA	NA	NA	NA	NA
22CYT02716	22P00480	Reflux screening	pass	Female	77.3	Past	NA	NA	N	N	N	22CYT02716 A1 HE 22P00480 - 2022-05-12 14.54.40.ndpi	22CYT02716 A1 TFF3 22P00480 - 2022-05-12 14.55.57.ndpi	22CYT02716 A1 P53 22P00480 - 2022-05-12 14.57.54.ndpi	NA	NA	NA	NA	NA
"""
    dataset = _write_dataset_snippet_and_load(tmp_path, data_file_snippet, label_column=CytedSchema.TFF3Positive)
    records = [dataset[i] for i in range(len(dataset))]
    # We have 3 lines in the dataframe, but one of them has "failed" QC.
    assert len(records) == 2
    # The IDs of the first and last records: Those are the ones that have passed QC.
    assert records[0][SlideKey.SLIDE_ID] == "21CYT03122"
    assert records[1][SlideKey.SLIDE_ID] == "22CYT02716"
    assert records[0][SlideKey.IMAGE].endswith("21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi")
    assert records[1][SlideKey.IMAGE].endswith("22CYT02716 A1 HE 22P00480 - 2022-05-12 14.54.40.ndpi")
    assert records[0][SlideKey.LABEL] == 1
    assert records[1][SlideKey.LABEL] == 0


def test_cyted_slides_label_filtering(tmp_path: Path) -> None:
    """Tests if the dataset loading removes all lines where the label is NA."""
    # Test data with NA in the P53 column, line 4
    data_file_snippet = """CYT full ID	Patient pathway	QC Report	Sex	Age	Smoking	Prague (C)	Prague (M)	TFF3 positive	P53 positive	Atypia	H&E
21CYT03122	Barrett's Surveillance	pass	Female	74.3	Past	0	3	Y	N	N	21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi
22CYT01935	Barrett's Surveillance	pass	Male	79.3	NA	0	3	Y	Y	N	22CYT01935 20P01895 A1 HE - 2022-04-06 12.29.55.ndpi
22CYT02716	Reflux screening	pass	Female	77.3	Past	NA	NA	N	NA	N	22CYT02716 A1 HE 22P00480 - 2022-05-12 14.54.40.ndpi
"""
    dataset = _write_dataset_snippet_and_load(tmp_path, data_file_snippet, CytedSchema.P53Positive)

    records = [dataset[i] for i in range(len(dataset))]
    # We have 3 lines in the dataframe, but only 2 of them have a valid label.
    assert len(records) == 2
    assert records[0][SlideKey.SLIDE_ID] == "21CYT03122"
    assert records[1][SlideKey.SLIDE_ID] == "22CYT01935"
    assert records[0][SlideKey.LABEL] == 0
    assert records[1][SlideKey.LABEL] == 1


def test_cyted_image_filtering(tmp_path: Path) -> None:
    """Tests if the dataset loading removes all lines where the image column is NA."""
    data_file_snippet = """CYT full ID	Pot ID	Patient pathway	QC Report	Sex	Age	Smoking	Prague (C)	Prague (M)	TFF3 positive	P53 positive	Atypia	H&E	TFF3
21CYT03122	21P01143	Barrett's Surveillance	pass	Female	74.3	Past	0	3	Y	N	N	NA	21CYT03122 21P01143 A1 TFF3 - 2021-09-13 16.11.39.ndpi
22CYT01935	20P01895	Barrett's Surveillance	pass	Male	79.3	NA	0	3	Y	N	N		22CYT01935 20P01895 A1 TFF3 - 2022-04-06 12.31.44.ndpi
22CYT02716	22P00480	Reflux screening	pass	Female	77.3	Past	NA	NA	N	N	N	22CYT02716 A1 HE 22P00480 - 2022-05-12 14.54.40.ndpi	22CYT02716 A1 TFF3 22P00480
"""
    dataset = _write_dataset_snippet_and_load(tmp_path, data_file_snippet, CytedSchema.P53Positive)
    assert len(dataset) == 1

    records = [dataset[i] for i in range(len(dataset))]
    # We have 3 lines in the dataframe, but only 2 of them have a valid label.
    assert len(records) == 1
    assert records[0][SlideKey.SLIDE_ID] == "22CYT02716"


def test_cyted_slides_column_filtering(tmp_path: Path) -> None:
    """Tests if the dataset loading applies extra filtering based on column name and value."""
    data_file_snippet = """CYT full ID	Patient pathway	QC Report	Sex	Age	Smoking	Prague (C)	Prague (M)	TFF3 positive	P53 positive	Atypia	H&E
21CYT03122	Barrett's Surveillance	pass	Female	74.3	Past	0	3	Y	N	N	21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi
22CYT01935	Barrett's Surveillance	pass	Male	79.3	NA	0	3	Y	Y	N	22CYT01935 20P01895 A1 HE - 2022-04-06 12.29.55.ndpi
22CYT02716	Reflux screening	pass	Female	77.3	Past	NA	NA	N	N	N	22CYT02716 A1 HE 22P00480 - 2022-05-12 14.54.40.ndpi
"""
    dataset = _write_dataset_snippet_and_load(
        tmp_path,
        data_file_snippet,
        label_column=CytedSchema.TFF3Positive,
        column_filter={CytedSchema.PatientPathway: CytedPathway.Screening},
    )
    # We have 3 lines in the dataframe, but only 1 of them has the correct value in the "Patient pathway" column.
    records = [dataset[i] for i in range(len(dataset))]
    assert len(records) == 1
    assert records[0][SlideKey.SLIDE_ID] == "22CYT02716"

    # Test that the inverse filtering also works: Barrett's Surveillance has a ' character in it, is that correctly set?
    dataset2 = _write_dataset_snippet_and_load(
        tmp_path,
        data_file_snippet,
        label_column=CytedSchema.TFF3Positive,
        column_filter={CytedSchema.PatientPathway: CytedPathway.Barretts},
    )
    assert len(dataset2) == 2


def test_cyted_metadata(tmp_path: Path) -> None:
    """Tests if the dataset loading correctly returns all metadata"""
    data_file_snippet = """CYT full ID	Pot ID	Patient pathway	QC Report	Sex	Age	Smoking	Prague (C)	Prague (M)	TFF3 positive	P53 positive	Atypia	H&E	TFF3	P53	NEG	Unknown	Extra	Rescan	Respin
21CYT03122	21P01143	Barrett's Surveillance	pass	Female	74.3	Past	0	3	Y	N	N	21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi	21CYT03122 21P01143 A1 TFF3 - 2021-09-13 16.11.39.ndpi	21CYT03122 21P01143 A1 P53 - 2021-09-13 16.13.40.ndpi	NA	NA	NA	NA	NA
"""
    dataset = _write_dataset_snippet_and_load(tmp_path, data_file_snippet, label_column=CytedSchema.TFF3Positive)
    records = [dataset[i] for i in range(len(dataset))]
    assert len(records) == 1
    # The IDs of the first and last records: Those are the ones that have passed QC.
    assert records[0][SlideKey.SLIDE_ID] == "21CYT03122"
    assert records[0][SlideKey.LABEL] == 1
    assert records[0][SlideKey.METADATA] == {
        CytedSchema.QCReport: "pass",
        CytedSchema.PatientPathway: CytedPathway.Barretts,
        CytedSchema.Sex: "Female",
        CytedSchema.Age: 74.3,
        CytedSchema.Smoking: "Past",
        CytedSchema.PragueC: 0,
        CytedSchema.PragueM: 3,
        CytedSchema.P53Positive: 0,
        CytedSchema.TFF3Positive: 1,
        CytedSchema.Atypia: 0,
        CytedSchema.Year: "21",
    }


def test_cyted_metadata_special(tmp_path: Path) -> None:
    """Tests if the dataset loading correctly maps missing values for the label column to NaN, and that the Prague
    measurements are mapped to floating point values."""
    data_file_snippet = """CYT full ID	Pot ID	Patient pathway	QC Report	Sex	Age	Smoking	Prague (C)	Prague (M)	TFF3 positive	P53 positive	Atypia	H&E	TFF3	P53	NEG	Unknown	Extra	Rescan	Respin
21CYT03122	21P01143	Barrett's Surveillance	pass	Female	74.3	Past	<1	3	Y	N	NA	21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi	21CYT03122 21P01143 A1 TFF3 - 2021-09-13 16.11.39.ndpi	21CYT03122 21P01143 A1 P53 - 2021-09-13 16.13.40.ndpi	NA	NA	NA	NA	NA
"""
    dataset = _write_dataset_snippet_and_load(tmp_path, data_file_snippet, label_column=CytedSchema.TFF3Positive)
    records = [dataset[i] for i in range(len(dataset))]
    assert len(records) == 1
    assert math.isnan(records[0][SlideKey.METADATA][CytedSchema.Atypia])
    assert records[0][SlideKey.METADATA][CytedSchema.PragueC] == 0.5


def test_cyted_metadata_missing(tmp_path: Path) -> None:
    """Tests if the dataset loading replaces all NA values with NaN"""
    data_file_snippet = """CYT full ID	Patient pathway	QC Report	Sex	Age	Smoking	Prague (C)	Prague (M)	TFF3 positive	P53 positive	Atypia	H&E
22CYT02716	Reflux screening	pass	Female	77.3	Past	NA	NA	N	N	N	22CYT02716 A1 HE 22P00480 - 2022-05-12 14.54.40.ndpi
"""
    dataset = _write_dataset_snippet_and_load(tmp_path, data_file_snippet, label_column=CytedSchema.P53Positive)
    records = [dataset[i] for i in range(len(dataset))]
    assert len(records) == 1
    assert math.isnan(records[0][SlideKey.METADATA][CytedSchema.PragueC])
    assert math.isnan(records[0][SlideKey.METADATA][CytedSchema.PragueM])


def test_cyted_background_metadata(tmp_path: Path) -> None:
    """Tests if the dataset loading correctly returns background metadata"""
    data_file_snippet = """CYT full ID	Pot ID	Patient pathway	QC Report	Sex	Age	Smoking	Prague (C)	Prague (M)	TFF3 positive	P53 positive	Atypia	H&E	TFF3	P53	NEG	Unknown	Extra	Rescan	Respin	background_r	background_g	background_b
21CYT03122	21P01143	Barrett's Surveillance	pass	Female	74.3	Past	0	3	Y	N	N	21CYT03122 21P01143 A1 H&E  - 2021-09-13 16.09.51.ndpi	21CYT03122 21P01143 A1 TFF3 - 2021-09-13 16.11.39.ndpi	21CYT03122 21P01143 A1 P53 - 2021-09-13 16.13.40.ndpi	NA	NA	NA	NA	NA	225	223	235
"""
    metadata_columns = {*CytedSchema.metadata_columns(), *CytedSchema.background_columns()}
    dataset = _write_dataset_snippet_and_load(
        tmp_path, data_file_snippet, label_column=CytedSchema.TFF3Positive, metadata_columns=metadata_columns
    )
    records = [dataset[i] for i in range(len(dataset))]
    assert len(records) == 1
    assert records[0][SlideKey.METADATA] == {
        CytedSchema.QCReport: "pass",
        CytedSchema.PatientPathway: CytedPathway.Barretts,
        CytedSchema.Sex: "Female",
        CytedSchema.Age: 74.3,
        CytedSchema.Smoking: "Past",
        CytedSchema.PragueC: 0,
        CytedSchema.PragueM: 3,
        CytedSchema.P53Positive: 0,
        CytedSchema.TFF3Positive: 1,
        CytedSchema.Atypia: 0,
        CytedSchema.Background_b: 235,
        CytedSchema.Background_g: 223,
        CytedSchema.Background_r: 225,
        CytedSchema.Year: "21",
    }
