#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""
This script can be used to combine test predictions from multiple Cyted AML runs. The test_output.csv file
is first downloaded for each run, then predictions are combined via majority voting or averaging probabilites,
lastly metrics are calculated for the combined predictions.
Usage example from CLI:
    python aggregate_and_ensemble_results.py \
    --run_id <insert AML parent run ID here> \
    --workspace_config cyted_config.json \
    --ensemble_method majority_vote
"""

from enum import Enum
from pathlib import Path
from collections import Counter
from typing import Any, Optional, List, Dict, Sequence
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import pickle

from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_recall,
    binary_precision,
    binary_specificity,
    binary_auroc,
    binary_average_precision,
    binary_confusion_matrix,
    binary_cohen_kappa,
)
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from health_cpath.utils.naming import ResultsKey, MetricsKey
from cyted.fixed_paths import repository_root_directory
from cyted.utils.azure_utils import (
    get_target_runs,
    download_outputs_csv_from_runs,
    TEST,
    EXTRA_VAL,
    download_run_metrics_if_required,
    get_workspace,
)
from cyted.utils.op_point_plots import (
    save_plot_calibration,
    save_plot_roc_and_pr_curves,
    save_plot_screening_curve,
)
from cyted.utils.conf_matrix import ConfusionMatrix
from health_cpath.utils.report_utils import collect_hyperdrive_metrics, download_hyperdrive_metrics_if_required


class EnsembleMethod(str, Enum):
    MAJ_VOTE = "majority_vote"
    MEAN_PROB = "mean_probs"


def collect_output_dicts(
    data_dir: Path,
    run_id: str,
    additional_run_ids: Optional[List[str]],
    workspace_config_path: Optional[Path],
    stage: str,
    overwrite: bool = False,
) -> List[Dict]:
    """Collects output dictionaries from completed child runs with the specified run_id and additional_run_ids.

    :param data_dir (Path): Directory path to download outputs CSV files.
    :param run_id (str): ID of the parent run.
    :param additional_run_ids (List[str]): List of IDs of additional runs to include in the ensemble.
    :param workspace_config_path (Path): File path to the workspace configuration file.
    :param stage (str): Name of the stage to collect the output dictionaries from.
    :param overwrite (bool): Whether to overwrite existing output files.
    :return: List[Dict]: List of output dictionaries, where each dictionary contains information about the slide IDs and
             predicted class probabilities for each tile in a slide, as well as the true class labels.
    """
    runs = get_target_runs(
        parent_run_id=run_id,
        child_runs_status="Completed",
        additional_run_ids=additional_run_ids,
        workspace_config=workspace_config_path,
    )

    # Download test_output.csv from the specified runs, if required. Can take several seconds for each child run.
    outputs_csv_paths = download_outputs_csv_from_runs(data_dir, runs=runs, overwrite=overwrite, stages=[stage])

    # Create ensemble output
    list_output_dicts = []
    column_keys = [
        ResultsKey.TRUE_LABEL,
        ResultsKey.PRED_LABEL,
        ResultsKey.SLIDE_ID,
        f"{ResultsKey.CLASS_PROBS}0",
        f"{ResultsKey.CLASS_PROBS}1",
    ]

    for i, output_csv_path in enumerate(outputs_csv_paths[stage]):
        output_dict = {}
        tiles_df = pd.read_csv(output_csv_path)
        for column_key in column_keys:
            output_dict[column_key] = group_dataframe(
                df=tiles_df, group_by=ResultsKey.SLIDE_ID, return_column=column_key
            )

        slide_ids = group_dataframe(df=tiles_df, group_by=ResultsKey.SLIDE_ID, return_column=ResultsKey.SLIDE_ID)
        output_dict[ResultsKey.SLIDE_ID] = list(slide_ids.values)
        output_dict["run_id"] = runs[i].id
        list_output_dicts.append(output_dict)

    # Check if all output dicts have the same slide IDs
    if stage == TEST:
        slide_ids = list_output_dicts[0][ResultsKey.SLIDE_ID]
        for i, output_dict in enumerate(list_output_dicts):
            if output_dict[ResultsKey.SLIDE_ID] != slide_ids:
                raise ValueError(
                    r"Slide IDs in run %s is %s. Expected %s" % runs[i].id,
                    len(output_dict[ResultsKey.SLIDE_ID]),
                    len(slide_ids),
                )

    return list_output_dicts


def compute_metrics_with_pred_labels(
    true_labels: torch.Tensor, pred_labels: torch.Tensor, threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute torch metrics given true labels, predictions and threshold.

    :param true_labels: Tensor of ground truth labels.
    :param pred_labels: Tensor of predicted labels or probabilities (binary).
    :param threshold: Threshold for converting probabilities to binary predictions.
    :return: A dict of metrics: accuracy, recall, precision, specificity, f1, cohen kappa, confusion matrices
    """
    # Check dimensions of true_labels and pred_labels raise error if multiclasses
    if len(true_labels.shape) > 1:
        raise ValueError("true_labels should be a 1D tensor")
    if len(pred_labels.shape) > 1:
        raise ValueError("pred_labels should be a 1D tensor")

    acc = binary_accuracy(pred_labels, true_labels, threshold=threshold)
    rec = binary_recall(pred_labels, true_labels, threshold=threshold)
    prec = binary_precision(pred_labels, true_labels, threshold=threshold)
    spec = binary_specificity(pred_labels, true_labels, threshold=threshold)
    f1 = (2 * (prec * rec)) / (prec + rec)
    kappa = binary_cohen_kappa(pred_labels, true_labels, threshold=threshold)
    cf_mat = binary_confusion_matrix(preds=pred_labels, target=true_labels, threshold=threshold)
    cf_mat_n = binary_confusion_matrix(preds=pred_labels, target=true_labels, normalize="true", threshold=threshold)

    return {
        MetricsKey.ACC: acc,
        MetricsKey.RECALL: rec,
        MetricsKey.PRECISION: prec,
        MetricsKey.SPECIFICITY: spec,
        MetricsKey.F1: f1,
        MetricsKey.COHENKAPPA: kappa,
        MetricsKey.CONF_MATRIX: cf_mat,
        MetricsKey.CONF_MATRIX_N: cf_mat_n,
    }


def compute_metrics_with_probs(true_labels: torch.Tensor, probs: torch.Tensor) -> Dict[str, Any]:
    """Compute torch metrics given true labels and predictions.

    :param true_labels: List of ground truth labels.
    :param predictions: List of probabilities (binary).
    :return: a dict with auroc, average precision (area under the pr curve).
    """
    # Check dimensions of true_labels and pred_labels raise error if multiclasses
    if len(true_labels.shape) > 1:
        raise ValueError("true_labels should be a 1D tensor")
    if len(probs.shape) > 1:
        raise ValueError("probs should be a 1D tensor")

    auc = binary_auroc(probs, true_labels)
    avg_prec = binary_average_precision(probs, true_labels)

    # We need the ignore because of a bug in torchmetrics, the return type of binary_auroc Tuple[Tensor, Tensor, Tensor]
    return {MetricsKey.AUROC: auc, MetricsKey.AVERAGE_PRECISION: avg_prec}  # type: ignore


def plot_aggregated_auc_curves(list_output_dicts: List[Dict[str, pd.Series]], data_dir: Path) -> None:
    """The ROC curves are aggregated over all output dictionaries and a mean curve is plotted.

    :param list_output_dicts (List[Dict]): A list of output dictionaries. Each dictionary represents the output from
           a run and must contain the following keys: 'TrueLabel', 'PredLabel', and keys beginning with 'ClassProbs'.
           'TrueLabel' is a list of true labels for each sample, 'PredLabel' is a list of predicted labels for each
           sample, and keys beginning with 'ClassProbs' are lists of predicted probabilities for each sample.
    :param data_dir (str): A string specifying the directory where the ROC curve plot will be saved.
    """
    # For loop over list_output_dicts, get probs and true labels, compute confusion matrix,
    cm_list = []
    for output_dict in list_output_dicts:
        probs = output_dict[f"{ResultsKey.CLASS_PROBS}1"].to_numpy()
        true_labels = output_dict[ResultsKey.TRUE_LABEL].to_numpy()

        cm = ConfusionMatrix.from_labels_and_scores(true_labels=true_labels, pred_scores=probs)
        cm_list.append(cm)

    # Following this exmple https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    # It is easier to work with false positive rates since they are increasing: tpr = sensitivity, fpr = 1 - specificity
    mean_fpr = np.linspace(0, 1, 100)
    sens = []
    aucs = []
    fprs = []

    fig, ax = plt.subplots(figsize=(4, 4))
    for fold, cm in enumerate(cm_list):
        fpr = 1 - cm.specificity
        fprs.append(fpr)
        interp_sens = np.interp(mean_fpr, fpr, cm.sensitivity)
        interp_sens[0] = 0.0
        sens.append(interp_sens)
        auc_val = auc(fpr, cm.sensitivity)
        aucs.append(auc_val)
        ax.plot(1 - fpr, cm.sensitivity, label=r"Fold %i AUC = %0.2f" % (fold, auc_val), alpha=0.3, lw=1)

    mean_sens = np.mean(sens, axis=0)
    mean_sens[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_sens)
    std_auc = np.std(aucs)

    ax.plot(
        1 - mean_fpr,
        mean_sens,
        color="b",
        label=r"Mean AUC = %0.2f" % (mean_auc),
        lw=2,
        alpha=0.8,
    )

    std_sens = np.std(sens, axis=0)
    tprs_upper = np.minimum(mean_sens + std_sens, 1)
    tprs_lower = np.maximum(mean_sens - std_sens, 0)
    ax.fill_between(
        mean_fpr,
        np.flip(tprs_lower),
        np.flip(tprs_upper),
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 STD DEV = %0.2f" % (std_auc),
    )

    ax.set(xlabel="Specificity", ylabel="Sensitivity")
    ax.grid(color="0.9")
    ax.axis("square")
    ax.legend(loc="lower right")
    ax.axline((1, 0), (0, 1), ls="--", c="grey", label="AUC = 0.5")
    # Have to flip x axis since specificity is decreasing
    plt.gca().invert_xaxis()
    fig.savefig(data_dir / Path("cross_val_auc_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def group_dataframe(df: pd.DataFrame, group_by: str, return_column: str) -> pd.DataFrame:
    """
    Function to group a dataframe by specified column and return desired columns from the dataframe.

    :param df: Dataframe to perform the grouping.
    :param group_by: Name of the column to group by.
    :param return_column: Name of the column to return from the dataframe.
    :return: A dataframe with the grouped column and the desired column.
    """
    df_groupby = df.groupby(group_by)
    grouped_column = df_groupby[return_column].first()
    return grouped_column


def get_ensembled_predictions(test_dicts: List[Dict[str, Any]], ensemble_method: str) -> np.ndarray:
    """Get ensembled predictions based on ensemble_method (majority vote or mean of probabilities).

    :param test_dicts: A list of dictionaries, each dictionary representing one run (one AML model).
    :param ensemble_method: Method to perform ensembling, supported types: majority_vote and mean_probs.
        (default: majority_vote).
    :return: An 1D numpy array of ensembled predicted labels.
    """
    if ensemble_method == EnsembleMethod.MAJ_VOTE:
        pred_labels = []
        for test_dict in test_dicts:
            pred_labels.append(test_dict[ResultsKey.PRED_LABEL])
        pred_labels_array = np.transpose(np.array(pred_labels))
        preds_maj_vote = []
        for i in range(len(pred_labels_array)):
            x = Counter(pred_labels_array[i])
            most_common = x.most_common(1)[0][0]
            preds_maj_vote.append(most_common)
        return np.array(preds_maj_vote)
    else:
        preds_from_probs = np.array([test_dict[f"{ResultsKey.CLASS_PROBS}1"] for test_dict in test_dicts])
        return preds_from_probs.mean(axis=0)


def aggregate_extra_val_and_test_results(
    run_id: str,
    data_dir: Path,
    aml_workspace: Any,
    workspace_config_path: Optional[Path],
    overwrite: bool = True,
    hyperdrive_arg_name: str = "crossval_index",
    additional_run_ids: Optional[List[str]] = None,
) -> None:
    """Aggregate the extra validation and test results of a hyperdrive run and print the mean and standard deviation
    of several metrics.

    :param run_id: A string representing the ID of the hyperdrive run to aggregate the results from.
    :param data_dir: A Path object representing the directory to download the metrics files to.
    :param aml_workspace: An object representing the Azure Machine Learning workspace.
    :param workspace_config_path: An optional Path object representing the path to the workspace config file.
    :param overwrite: A boolean indicating whether to overwrite existing metrics files.
    :param hyperdrive_arg_name: A string representing the name of the hyperdrive run ID argument.
    :param additional_run_ids: An optional list of strings representing additional hyperdrive run IDs to aggregate
                               results from.
    """

    metrics_json = download_hyperdrive_metrics_if_required(
        run_id, data_dir, aml_workspace, overwrite=overwrite, hyperdrive_arg_name=hyperdrive_arg_name
    )

    metrics_json_additional: List[Any] = []
    if additional_run_ids is not None:
        for add_run_id in additional_run_ids:
            metrics_json_additional += [
                download_run_metrics_if_required(
                    run_id=add_run_id,
                    download_dir=data_dir,
                    aml_workspace=aml_workspace,
                    workspace_config_path=workspace_config_path,
                    overwrite=overwrite,
                )
            ]  # type:ignore

    # Get metrics dataframe from the downloaded json file
    metrics_df = collect_hyperdrive_metrics(metrics_json=metrics_json)

    # Get metrics dataframes from additional json files
    for additional_json in metrics_json_additional:
        metrics_df_additional = collect_hyperdrive_metrics(metrics_json=additional_json)
        metrics_df = pd.concat([metrics_df, metrics_df_additional], axis=1)

    metrics_list = [
        MetricsKey.ACC,
        MetricsKey.RECALL,
        MetricsKey.PRECISION,
        MetricsKey.SPECIFICITY,
        MetricsKey.F1,
        MetricsKey.COHENKAPPA,
        MetricsKey.AUROC,
        MetricsKey.AVERAGE_PRECISION,
    ]

    stages = [EXTRA_VAL, TEST]
    print(f"\n\nAggregate results from hyperdrive run {run_id}:")
    for stage in stages:
        stage = stage + "/"
        for metric in metrics_list:
            row_name = stage + metric
            cv_metrics = np.array(metrics_df.loc[row_name])

            # Find None and nan values, raise warning if there are any nans
            none_columns = []
            nan_columns = []
            cv_metrics_idx = []
            for i in range(len(cv_metrics)):
                if cv_metrics[i] is None:
                    none_columns.append(metrics_df.columns[i])
                elif np.isnan(cv_metrics[i]):
                    nan_columns.append(metrics_df.columns[i])
                else:
                    cv_metrics_idx.append(i)

            if len(nan_columns) > 0:
                print(f"Warning: {row_name} contains nan values in columns {nan_columns}")

            # Filter None and nan values
            cv_metrics = cv_metrics[cv_metrics_idx]

            # With sdk2 some metrics where duplicated so we had to take the first metric
            cv_metrics = [x if not isinstance(x, List) else x[0] for x in cv_metrics]  # type:ignore

            mean = np.mean(cv_metrics)
            std = np.std(cv_metrics)
            print(f"{row_name}: {round(mean,4)} ± {round(std,4)}")


def plot_combined_confusion_matrices(
    cm: np.ndarray,
    cm_n: np.ndarray,
    class_names: Sequence[str],
) -> plt.Figure:
    """Plots a normalized and non-normalized confusion matrix and returns the figure.

    :param cm: Non normalized confusion matrix to be plotted.
    :param cm_n: Normalized confusion matrix to be plotted.
    :param class_names: List of class names.

    return fig: Figure object containing the confusion matrix plot.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        # annot=cm,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    for i in range(cm_n.shape[0]):
        for j in range(cm_n.shape[1]):
            if i == j:
                if cm_n[i][j] > 0.5:
                    color = "white"
            else:
                color = "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{cm[i][j]} ({cm_n[i][j]:.2%})",
                ha="center",
                va="center",
                color=color,
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    return fig


def ensemble_test_results(
    list_output_dicts: List[Dict[str, pd.Series]],
    data_dir: Path,
    ensemble_method: str = EnsembleMethod.MEAN_PROB,
    add_best2_operating_point: bool = False,
) -> None:
    """Ensembles the results of multiple runs for the test stage and computes evaluation metrics and plots.

    :param list_output_dicts: A list of dictionaries containing the results of multiple test runs. Each dictionary
                              should contain a DataFrame with columns "slide_id", "true_label", and "prob",
                              representing the slide ID, true label, and predicted probability for each slide,
                              respectively.
    :param data_dir: The directory to save the output files.
    :param ensemble_method: The method to use for ensembling the results. Default is EnsembleMethod.MEAN_PROB.
    :param add_best2_operating_point: Whether to add the best2 operating point to the ROC curve plot.
    """
    ensemble_preds = get_ensembled_predictions(list_output_dicts, ensemble_method)  # type: ignore

    # save ensemble_prediction_labels, true labels
    ensemble_slide_ids = list_output_dicts[0][ResultsKey.TRUE_LABEL].index.to_list()
    ensemble_true_labels = list_output_dicts[0][ResultsKey.TRUE_LABEL].to_list()
    ensemble_df = pd.DataFrame(
        {
            ResultsKey.SLIDE_ID: ensemble_slide_ids,
            ResultsKey.TRUE_LABEL: ensemble_true_labels,
            ResultsKey.PROB: ensemble_preds.tolist(),
        }
    ).set_index(ResultsKey.SLIDE_ID)

    f = open(data_dir / Path("ensemble_probs.pkl"), "wb")
    pickle.dump(ensemble_df, f)
    f.close()

    metrics_dict = compute_metrics_with_pred_labels(
        torch.tensor(ensemble_true_labels).to(torch.int32), torch.tensor(ensemble_preds)
    )
    for key in metrics_dict:
        if (key != MetricsKey.CONF_MATRIX) and (key != MetricsKey.CONF_MATRIX_N):
            print(f"Ensemble test {key}:", round(float(metrics_dict[key]), 4))

    fig = plot_combined_confusion_matrices(
        cm=metrics_dict[MetricsKey.CONF_MATRIX].numpy(),
        cm_n=metrics_dict[MetricsKey.CONF_MATRIX_N].numpy(),
        class_names=["TFF3 negative", "TFF3 positive"],
    )
    plt.tight_layout()
    fig.savefig(data_dir / "ensemble_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if ensemble_method == EnsembleMethod.MEAN_PROB:
        auc_avg_prec_dict = compute_metrics_with_probs(
            torch.tensor(list_output_dicts[0][ResultsKey.TRUE_LABEL]).to(torch.int32), torch.tensor(ensemble_preds)
        )
        for key in auc_avg_prec_dict:
            print(f"Ensemble test {key}:", round(float(auc_avg_prec_dict[key]), 4))

        # Create operating point plots
        cm = ConfusionMatrix.from_labels_and_scores(
            true_labels=np.array(list_output_dicts[0][ResultsKey.TRUE_LABEL]), pred_scores=np.array(ensemble_preds)
        )

        save_plot_roc_and_pr_curves(cm, add_best2_operating_point, data_dir)
        save_plot_screening_curve(cm, data_dir)
        save_plot_calibration(ensemble_preds, list_output_dicts, data_dir)

    else:
        print(
            f"Warning: {ensemble_method} does not output probabilities. Therefore auroc and average precision is not"
            " applicable."
        )


def aggregate_and_ensemble_extra_val_and_test_results(
    run_id: str,
    output_dir: Path,
    workspace_config_path: Optional[Path] = None,
    additional_run_ids: Optional[List[str]] = None,
    overwrite: bool = True,
    hyperdrive_arg_name: str = "crossval_index",
    ensemble_method: str = EnsembleMethod.MEAN_PROB,
    add_best2_operating_point: bool = False,
) -> None:
    """
    Function to aggregate test results and compute metrics for an ensemble from AML runs.

    :param run_id: Run ID of a hyperdrive run.
    :param output_dir: Directory where to download Azure ML data and save plots.
    :param workspace_config_path: Path to Azure ML workspace config.json file.
        If omitted, will try to load default workspace.
    :param additional_run_ids: A list of additional run IDs (separated by comma).
        For example, when a child run fails and needs to be re-submitted. Default None.
    :param overwrite: Forces (re)download of metrics and output files, even if they already exist locally.
    :param hyperdrive_arg_name: Name of the hyperdrive argument to use for grouping runs.
    :param ensemble_method: Method to perform ensembling, supported types: majority_vote and mean_probs.
        (default: mean_probs).
    """
    data_dir = output_dir / f"aggregate_and_ensemble_test_results_{run_id}"
    data_dir.mkdir(parents=True, exist_ok=True)
    aml_workspace = get_workspace(workspace_config_path=workspace_config_path)

    # TODO: Can be possibly refactored by using only completed runs returned by get_target_runs
    aggregate_extra_val_and_test_results(
        run_id=run_id,
        data_dir=data_dir,
        aml_workspace=aml_workspace,
        workspace_config_path=workspace_config_path,
        overwrite=overwrite,
        hyperdrive_arg_name=hyperdrive_arg_name,
        additional_run_ids=additional_run_ids,
    )

    # Get output dicts of all completed runs
    list_output_dicts = collect_output_dicts(
        data_dir=data_dir,
        run_id=run_id,
        additional_run_ids=additional_run_ids,
        workspace_config_path=workspace_config_path,
        stage=TEST,
        overwrite=overwrite,
    )
    # Plot and save aggregated_auc_pr_curves
    plot_aggregated_auc_curves(list_output_dicts=list_output_dicts, data_dir=data_dir)

    # Plot and save ensemble test results
    ensemble_test_results(
        list_output_dicts=list_output_dicts,
        data_dir=data_dir,
        ensemble_method=ensemble_method,
        add_best2_operating_point=add_best2_operating_point,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--run_id", help="The parent Hyperdrive run ID. Only provide parent run id for hyperdrive runs."
    )
    parser.add_argument(
        "--workspace_config",
        help="Path to Azure ML workspace config.json file. If omitted, will try to load default workspace.",
    )
    parser.add_argument("--output_dir", help="Directory where to download Azure ML data and save plots.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Forces (re)download of metrics and output files, even if they already exist locally.",
    )
    parser.add_argument("--additional_run_ids", default=None, help="List of run ids, separated by comma.")
    parser.add_argument(
        "--ensemble_method",
        default=EnsembleMethod.MEAN_PROB,
        choices=[EnsembleMethod.MAJ_VOTE, EnsembleMethod.MEAN_PROB],
        help="Method to perform ensembling. supported types: majority_vote and mean_probs (default: mean_probs).",
    )
    parser.add_argument(
        "--hyperdrive_arg_name",
        default="crossval_index",
        help="Name of the argument used for grouping child runs in a hyperdrive (sweep) run (default: crossval_index).",
    )
    parser.add_argument(
        "--add_best2_operating_point",
        default=False,
        help="Add BEST2 human pathologist manual review operating point to ROC and PR curves.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = repository_root_directory("outputs")
    workspace_config = Path(args.workspace_config).resolve() if args.workspace_config else None

    print(f"Output dir: {Path(args.output_dir).resolve()}")

    aggregate_and_ensemble_extra_val_and_test_results(
        run_id=args.run_id,
        workspace_config_path=workspace_config,
        output_dir=Path(args.output_dir),
        additional_run_ids=args.additional_run_ids.split(",") if args.additional_run_ids else None,
        overwrite=args.overwrite,
        ensemble_method=args.ensemble_method,
        hyperdrive_arg_name=args.hyperdrive_arg_name,
        add_best2_operating_point=args.add_best2_operating_point,
    )
