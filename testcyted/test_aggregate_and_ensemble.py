from pathlib import Path
import pytest
import torch
from torch import allclose
import numpy as np
import pandas as pd
from testhisto.utils.utils_testhisto import assert_binary_files_match

from cyted.scripts.aggregate_and_ensemble_results import (
    compute_metrics_with_pred_labels,
    compute_metrics_with_probs,
    get_ensembled_predictions,
    EnsembleMethod,
    plot_aggregated_auc_curves,
)
from health_cpath.utils.naming import ResultsKey, MetricsKey

dummy_true_labels = torch.Tensor([0, 1, 0, 1])
wrong_shape_dummy_true_labels = torch.Tensor([[0, 1, 0, 1], [0, 1, 0, 1]])
dummy_pred_labels = torch.Tensor([0, 1, 1, 0])
wrong_shape_dummy_pred_labels = torch.Tensor([[0, 1, 1, 0], [0, 1, 1, 0]])
dummy_pred_probs = torch.Tensor([0.1, 0.9, 0.8, 0.2])
wrong_shape_dummy_pred_probs = torch.Tensor([[0.1, 0.9, 0.8, 0.2], [0.1, 0.9, 0.8, 0.2]])
dummy_dict_1 = {
    ResultsKey.TRUE_LABEL: pd.Series([1, 1, 1, 0]),
    ResultsKey.PRED_LABEL: pd.Series([0, 1, 0, 1]),
    f"{ResultsKey.CLASS_PROBS}1": pd.Series([0.11, 0.91, 0.21, 0.81]),
}
dummy_dict_2 = {
    ResultsKey.TRUE_LABEL: pd.Series([0, 1, 1, 1]),
    ResultsKey.PRED_LABEL: pd.Series([1, 0, 1, 0]),
    f"{ResultsKey.CLASS_PROBS}1": pd.Series([0.95, 0.75, 0.85, 0.25]),
}
dummy_dict_3 = {
    ResultsKey.TRUE_LABEL: pd.Series([1, 0, 0, 1]),
    ResultsKey.PRED_LABEL: pd.Series([1, 1, 0, 0]),
    f"{ResultsKey.CLASS_PROBS}1": pd.Series([0.99, 0.49, 0.19, 0.29]),
}
dummy_dict_list = [dummy_dict_1, dummy_dict_2, dummy_dict_3]
expected_fig_path = Path("testcyted/testdata/plots/cross_val_auc_curves.png")


def test_compute_metrics_with_probs() -> None:
    auc_avg_prec_dict = compute_metrics_with_probs(dummy_true_labels, dummy_pred_probs)

    assert allclose(auc_avg_prec_dict[MetricsKey.AUROC], torch.tensor(0.75))
    assert allclose(auc_avg_prec_dict[MetricsKey.AVERAGE_PRECISION], torch.tensor(0.833333333))

    with pytest.raises(ValueError, match="probs should be a 1D tensor"):
        compute_metrics_with_probs(dummy_true_labels, wrong_shape_dummy_pred_probs)
    with pytest.raises(ValueError, match="true_labels should be a 1D tensor"):
        compute_metrics_with_probs(wrong_shape_dummy_true_labels, dummy_pred_probs)


def test_compute_metrics_with_pred_labels() -> None:
    metrics_dict = compute_metrics_with_pred_labels(
        dummy_true_labels, dummy_pred_labels
    )
    assert allclose(metrics_dict[MetricsKey.ACC], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.RECALL], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.PRECISION], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.SPECIFICITY], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.F1], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.COHENKAPPA], torch.tensor(0.0))
    assert allclose(metrics_dict[MetricsKey.CONF_MATRIX], torch.tensor([[1, 1], [1, 1]]))
    assert allclose(metrics_dict[MetricsKey.CONF_MATRIX_N], torch.tensor([[0.5, 0.5], [0.5, 0.5]]))

    metrics_dict = compute_metrics_with_pred_labels(
        dummy_true_labels, dummy_pred_probs
    )
    assert allclose(metrics_dict[MetricsKey.ACC], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.RECALL], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.PRECISION], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.SPECIFICITY], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.F1], torch.tensor(0.5))
    assert allclose(metrics_dict[MetricsKey.COHENKAPPA], torch.tensor(0.0))
    assert allclose(metrics_dict[MetricsKey.CONF_MATRIX], torch.tensor([[1, 1], [1, 1]]))
    assert allclose(metrics_dict[MetricsKey.CONF_MATRIX_N], torch.tensor([[0.5, 0.5], [0.5, 0.5]]))

    with pytest.raises(ValueError, match="pred_labels should be a 1D tensor"):
        compute_metrics_with_pred_labels(dummy_true_labels, wrong_shape_dummy_pred_probs)
    with pytest.raises(ValueError, match="true_labels should be a 1D tensor"):
        compute_metrics_with_pred_labels(wrong_shape_dummy_true_labels, dummy_pred_probs)


def test_get_ensembled_predictions() -> None:
    ensemble_preds = get_ensembled_predictions(dummy_dict_list, EnsembleMethod.MAJ_VOTE)
    assert np.allclose(ensemble_preds, np.array([1, 1, 0, 0]))

    ensemble_preds = get_ensembled_predictions(dummy_dict_list, EnsembleMethod.MEAN_PROB)
    assert np.allclose(ensemble_preds, np.array([0.6833333333333332, 0.7166666666666668, 0.4166666666666667, 0.45]))


def test_plot_aggregated_auc_curves(tmp_path: Path) -> None:
    plot_aggregated_auc_curves(dummy_dict_list, tmp_path)
    assert_binary_files_match(tmp_path / "cross_val_auc_curves.png", expected_fig_path)
