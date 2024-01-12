#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import numpy as np
import pytest
from cyted.utils.array_utils import is_sorted_1d

from cyted.utils.conf_matrix import ConfusionMatrix, ConfusionMatrixSweep


@pytest.fixture
def conf_matrix() -> ConfusionMatrixSweep:
    num_total = 100
    true_labels = np.random.randint(2, size=num_total)
    num_positives = true_labels.sum()
    num_negatives = num_total - num_positives
    pred_scores = np.zeros(num_total)
    pred_scores[true_labels == 1] = np.random.randn(num_positives) + 1
    pred_scores[true_labels == 0] = np.random.randn(num_negatives) - 1
    return ConfusionMatrix.from_labels_and_scores(true_labels, pred_scores)


def test_confusion_matrix_from_labels_and_scores(conf_matrix: ConfusionMatrixSweep) -> None:
    assert is_sorted_1d(conf_matrix.thresholds, ascending=False)
    assert is_sorted_1d(conf_matrix.true_positives, ascending=True)
    assert is_sorted_1d(conf_matrix.false_positives, ascending=True)


def test_confusion_matrix_levels(conf_matrix: ConfusionMatrixSweep) -> None:
    level = 0.9

    cm_at_thr, thr_at_thr = conf_matrix.at_threshold(level)  # Just check that it succeeds
    assert isinstance(cm_at_thr, ConfusionMatrix)
    assert isinstance(thr_at_thr, float)

    cm_at_sens, thr_at_sens = conf_matrix.at_sensitivity(level)
    assert isinstance(cm_at_sens, ConfusionMatrix)
    assert isinstance(thr_at_sens, float)
    sens_tol = 1 / conf_matrix.num_positives
    assert 0 <= cm_at_sens.sensitivity - level < sens_tol

    cm_at_spec, thr_at_spec = conf_matrix.at_specificity(level)
    assert isinstance(cm_at_spec, ConfusionMatrix)
    assert isinstance(thr_at_spec, float)
    spec_tol = 1 / conf_matrix.num_negatives
    # This test fails occasionally when not adding the 1e-6 tolerance
    assert 0 <= cm_at_spec.specificity - level <= (spec_tol + 1e-6)


NON_INDEXABLE_ATTRS = [
    "num_total",
    "num_positives",
    "num_negatives",
    "prevalence",
]
INDEXABLE_ATTRS = [
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "pred_positives",
    "pred_negatives",
    "sensitivity",
    "specificity",
    "pos_pred_value",
    "neg_pred_value",
]


def test_confusion_matrix_int_indexing(conf_matrix: ConfusionMatrix) -> None:
    arbitrary_index = 42
    indexed_conf_matrix = conf_matrix[arbitrary_index]
    for attr in NON_INDEXABLE_ATTRS:
        assert getattr(indexed_conf_matrix, attr) == getattr(conf_matrix, attr)
    for attr in INDEXABLE_ATTRS:
        assert getattr(indexed_conf_matrix, attr) == getattr(conf_matrix, attr)[arbitrary_index]


def test_confusion_matrix_slice_indexing(conf_matrix: ConfusionMatrix) -> None:
    arbitrary_slice = slice(24, 42, 2)
    indexed_conf_matrix = conf_matrix[arbitrary_slice]
    assert isinstance(indexed_conf_matrix, ConfusionMatrixSweep)
    for attr in NON_INDEXABLE_ATTRS:
        assert getattr(indexed_conf_matrix, attr) == getattr(conf_matrix, attr)
    for attr in INDEXABLE_ATTRS:
        assert np.all(getattr(indexed_conf_matrix, attr) == getattr(conf_matrix, attr)[arbitrary_slice])


def test_confusion_matrix_auc(conf_matrix: ConfusionMatrixSweep) -> None:
    def _test_auc(auc: float) -> None:
        assert isinstance(auc, float)
        assert 0 <= auc <= 1

    _test_auc(conf_matrix.get_area_under_roc_curve())
    _test_auc(conf_matrix.get_area_under_pr_curve())


def test_confusion_matrix_resample(conf_matrix: ConfusionMatrixSweep) -> None:
    grid = np.linspace(0, 1, 10)

    def _test_interp(values: np.ndarray) -> None:
        assert isinstance(values, np.ndarray)
        assert values.shape == grid.shape

    interp_sens = conf_matrix.resample_roc_curve(specificities=grid)
    interp_ppv = conf_matrix.resample_pr_curve(sensitivities=grid)
    _test_interp(interp_sens)
    _test_interp(interp_ppv)

    interp_sens_rev = conf_matrix.resample_roc_curve(specificities=grid[::-1])
    interp_ppv_rev = conf_matrix.resample_pr_curve(sensitivities=grid[::-1])
    assert np.allclose(interp_sens_rev, interp_sens[::-1])
    assert np.allclose(interp_ppv_rev, interp_ppv[::-1])
