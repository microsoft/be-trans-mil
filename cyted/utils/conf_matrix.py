#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from dataclasses import dataclass, InitVar
from typing import Union, overload

import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics._ranking import _binary_clf_curve  # type: ignore

from cyted.utils.array_utils import is_sorted_1d, search_1d


@dataclass(frozen=True)
class ConfusionMatrix:
    """Utility class for computing confusion matrices and related metrics."""

    num_total: int
    num_positives: int

    true_positives: np.ndarray
    false_positives: np.ndarray

    validate_args: InitVar[bool] = True

    def __post_init__(self, validate_args: bool) -> None:
        if validate_args:
            self.validate()

    def validate(self) -> None:
        # For some analyses, num_total and/or num_positives may actually be arrays,
        # so we use np.all() to support that.
        if np.all(self.num_total < 0):
            raise ValueError(f"num_total must be > 0, got {self.num_total}")
        if np.all(self.num_positives < 0):
            raise ValueError(f"num_positives must be > 0, got {self.num_positives}")
        if np.all(self.num_positives > self.num_total):
            raise ValueError(f"num_positives must be <= num_total, got {self.num_positives} > {self.num_total}")

        if np.any(self.true_positives > self.num_positives):
            raise ValueError("true_positives must be <= num_positives")
        if np.any(self.false_positives > self.num_negatives):
            raise ValueError("false_positives must be <= num_negatives")

    @staticmethod
    def from_labels_and_scores(
        true_labels: Union[np.ndarray, pd.Series], pred_scores: Union[np.ndarray, pd.Series]
    ) -> "ConfusionMatrixSweep":
        """Construct a `ConfusionMatrixSweep` from true binary labels and predicted scores.

        :param true_labels: `(N,)` array of true binary labels.
        :param pred_scores: `(N,)` array of predicted scores for the target class.
        :return: A confusion matrix sweep for the range of thresholds in `pred_scores`.
        """
        true_labels = np.asarray(true_labels)
        pred_scores = np.asarray(pred_scores)

        num_total: int = true_labels.shape[0]
        num_positives: int = np.sum(true_labels, axis=-1)

        # Use scikit-learn's internal function to sweep over thresholds,
        # returning the corresponding numbers of false and true positives.
        false_positives, true_positives, thresholds = _binary_clf_curve(true_labels, pred_scores)

        return ConfusionMatrixSweep(
            num_total=num_total,
            num_positives=num_positives,
            true_positives=true_positives,
            false_positives=false_positives,
            thresholds=thresholds,
        )

    @staticmethod
    def from_predicted_labels(
        true_labels: Union[np.ndarray, pd.Series], pred_labels: Union[np.ndarray, pd.Series]
    ) -> "ConfusionMatrix":
        """Construct a `ConfusionMatrix` from true and predicted binary labels.

        The shapes of `true_labels` and `pred_labels` must be broadcastable.

        :param true_labels: `(..., N)` array of true binary labels.
        :param pred_labels: `(..., N)` array of predicted binary labels.
        :return: A confusion matrix with the broadcasted shape of `true_labels` and `pred_labels`,
            minus the last dimension.
        """
        true_labels = np.asarray(true_labels)
        pred_labels = np.asarray(pred_labels)

        num_total: int = true_labels.shape[0]
        num_positives: int = true_labels.sum(-1)

        true_positives = (true_labels * pred_labels).sum(-1)
        false_positives = ((1 - true_labels) * pred_labels).sum(-1)

        return ConfusionMatrix(
            num_total=num_total,
            num_positives=num_positives,
            true_positives=true_positives,
            false_positives=false_positives,
        )

    @property
    def num_negatives(self) -> int:
        return self.num_total - self.num_positives

    @property
    def false_negatives(self) -> np.ndarray:
        return self.num_positives - self.true_positives

    @property
    def true_negatives(self) -> np.ndarray:
        return self.num_negatives - self.false_positives

    @property
    def pred_positives(self) -> np.ndarray:
        return self.true_positives + self.false_positives

    @property
    def pred_negatives(self) -> np.ndarray:
        return self.num_total - self.pred_positives

    @property
    def prevalence(self) -> float:
        return self.num_positives / self.num_total

    @property
    def sensitivity(self) -> np.ndarray:
        return self.true_positives / self.num_positives

    @property
    def specificity(self) -> np.ndarray:
        return self.true_negatives / self.num_negatives

    @property
    def pos_pred_value(self) -> np.ndarray:
        return self.true_positives / self.pred_positives

    @property
    def neg_pred_value(self) -> np.ndarray:
        return self.true_negatives / self.pred_negatives

    @overload
    def __getitem__(self, index_or_slice: slice) -> "ConfusionMatrix":
        ...

    @overload
    def __getitem__(self, index_or_slice: int) -> "ConfusionMatrix":
        ...

    def __getitem__(self, index_or_slice: Union[int, slice]) -> "ConfusionMatrix":
        return ConfusionMatrix(
            num_total=self.num_total,
            num_positives=self.num_positives,
            true_positives=self.true_positives[index_or_slice],
            false_positives=self.false_positives[index_or_slice],
        )


@dataclass(frozen=True)
class ConfusionMatrixSweep(ConfusionMatrix):
    """A 1D confusion matrix for a sweep of thresholds, as in a ROC or PR curve."""

    thresholds: np.ndarray = np.array([])

    def validate(self) -> None:
        super().validate()

        # Additional checks for array-valued fields
        num_thresholds = len(self.thresholds)
        expected_shape = (num_thresholds,)
        if self.thresholds.shape != expected_shape:
            raise ValueError(f"Expected thresholds with shape {expected_shape}, got {self.thresholds.shape}")
        if self.true_positives.shape != expected_shape:
            raise ValueError(f"Expected true_positives with shape {expected_shape}, got {self.true_positives.shape}")
        if self.false_positives.shape != expected_shape:
            raise ValueError(f"Expected false_positives with shape {expected_shape}, got {self.false_positives.shape}")

        if not is_sorted_1d(self.thresholds, ascending=False):
            raise ValueError("thresholds must be in descending order")
        if not is_sorted_1d(self.true_positives, ascending=True):
            raise ValueError("true_positives must be in ascending order")
        if not is_sorted_1d(self.false_positives, ascending=True):
            raise ValueError("false_positives must be in ascending order")

    @overload
    def __getitem__(self, index_or_slice: slice) -> "ConfusionMatrixSweep":
        ...

    @overload
    def __getitem__(self, index_or_slice: int) -> ConfusionMatrix:
        ...

    def __getitem__(self, index_or_slice: Union[int, slice]) -> ConfusionMatrix:
        if isinstance(index_or_slice, slice):
            return ConfusionMatrixSweep(
                num_total=self.num_total,
                num_positives=self.num_positives,
                true_positives=self.true_positives[index_or_slice],
                false_positives=self.false_positives[index_or_slice],
                thresholds=self.thresholds[index_or_slice],
            )
        return super().__getitem__(index_or_slice)

    def at_threshold(self, threshold: float) -> tuple[ConfusionMatrix, float]:
        """Returns the confusion matrix at the given threshold, and the threshold itself."""
        thr_index = search_1d(self.thresholds, threshold, ascending=False, first=True)
        own_threshold = self.thresholds[thr_index]
        return self[thr_index], own_threshold

    def at_sensitivity(self, sens_level: float) -> tuple[ConfusionMatrix, float]:
        """Returns the confusion matrix at the given sensitivity, and the corresponding threshold."""
        sens_index = search_1d(self.sensitivity, sens_level, ascending=True, first=True)
        threshold = self.thresholds[sens_index]
        return self[sens_index], threshold

    def at_specificity(self, spec_level: float) -> tuple[ConfusionMatrix, float]:
        """Returns the confusion matrix at the given specificity, and the corresponding threshold."""
        # Specificities are in decreasing order. Finds first index smaller than spec_level,
        # then we subtract 1 for least upper bound.
        spec_index = search_1d(self.specificity, spec_level, ascending=False, first=True) - 1
        spec_index = max(0, spec_index)  # handle edge case
        threshold = self.thresholds[spec_index]
        return self[spec_index], threshold

    def get_area_under_roc_curve(self) -> float:
        if self.thresholds.ndim != 1:
            raise NotImplementedError("AUC is only available for 1D confusion matrices")
        return float(auc(self.specificity, self.sensitivity))

    def get_area_under_pr_curve(self) -> float:
        if self.thresholds.ndim != 1:
            raise NotImplementedError("AUC is only available for 1D confusion matrices")
        return float(auc(self.sensitivity, self.pos_pred_value))

    def resample_roc_curve(self, specificities: np.ndarray) -> np.ndarray:
        """Interpolate sensitivities for the given specificities."""
        return np.interp(specificities, self.specificity[::-1], self.sensitivity[::-1])

    def resample_pr_curve(self, sensitivities: np.ndarray) -> np.ndarray:
        """Interpolate PPVs (precisions) for the given sensitivities (recalls)."""
        return np.interp(sensitivities, self.sensitivity, self.pos_pred_value)
