#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from collections import defaultdict
from typing import Any, Callable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from scipy.stats import binned_statistic

from cyted.utils import op_point_plots
from cyted.utils.array_utils import search_1d
from cyted.utils.bootstrapping import compute_bootstrap_quantiles
from cyted.utils.conf_matrix import ConfusionMatrix, ConfusionMatrixSweep
from cyted.utils.decision_analysis import DecisionTree


def plot_bootstrap_roc_and_pr_curves(
    data: pd.DataFrame,
    roc_ax: plt.Axes,
    pr_ax: plt.Axes,
    label_column: str = "true_label",
    prob_column: str = "prob_class1",
) -> dict[str, tuple[Any, str]]:
    """Plot ROC and PR curves with bootstrap confidence intervals.

    :param data: Dataframe containing labels and predicted probabilities.
    :param roc_ax: Axes onto which to plot ROC curve.
    :param pr_ax: Axes onto which to plot ROC curve.
    :param label_column: Column of `data` containing true labels.
    :param prob_column: Column of `data` containing predicted probabilities.
    :return: Dictionary mapping `plot_type: (handle, label)` for creating a legend,
        where `plot_type` is one of `'roc'` or `'pr'`.
    """

    def compute_roc_and_pr_stats(
        sample: pd.DataFrame, label_column: str, prob_column: str, grid: np.ndarray
    ) -> dict[str, Union[float, np.ndarray]]:
        cm = ConfusionMatrix.from_labels_and_scores(true_labels=sample[label_column], pred_scores=sample[prob_column])
        return {
            "interp_sens": cm.resample_roc_curve(grid),
            "interp_ppv": cm.resample_pr_curve(grid),
            "auroc": cm.get_area_under_roc_curve(),
            "auprc": cm.get_area_under_pr_curve(),
        }

    def format_decimal(x: float) -> str:
        return f"{x:.3f}".lstrip("0")

    def format_auc_ci(mid: float, low: float, high: float) -> str:
        return f"{format_decimal(mid)} [{format_decimal(low)}–{format_decimal(high)}]"

    full_cm = ConfusionMatrix.from_labels_and_scores(true_labels=data["true_label"], pred_scores=data["prob_class1"])

    grid = np.linspace(0, 1, 101)
    full_stats = compute_roc_and_pr_stats(data, label_column, prob_column, grid)
    boot_quantiles = compute_bootstrap_quantiles(
        data,
        compute_roc_and_pr_stats,
        quantiles=(0.025, 0.975),
        num_samples=500,
        label_column=label_column,
        prob_column=prob_column,
        grid=grid,
    )

    sens_ci_fill = roc_ax.fill_between(grid, *boot_quantiles["interp_sens"], alpha=0.3)
    (full_sens_line,) = roc_ax.plot(full_cm.specificity, full_cm.sensitivity)
    op_point_plots.format_sens_spec_axes(roc_ax)
    roc_handle = (full_sens_line, sens_ci_fill)
    roc_label = f"AUC: {format_auc_ci(full_stats['auroc'], *boot_quantiles['auroc'])}"  # type: ignore

    ppv_ci_fill = pr_ax.fill_between(grid, *boot_quantiles["interp_ppv"], alpha=0.3)
    (ppv_line,) = pr_ax.plot(full_cm.sensitivity, full_cm.pos_pred_value)
    op_point_plots.format_pr_or_roc_axes("pr", pr_ax)
    pr_handle = (ppv_line, ppv_ci_fill)
    pr_label = f"AUC: {format_auc_ci(full_stats['auprc'], *boot_quantiles['auprc'])}"  # type: ignore

    roc_ax.axline((1, 0), (0, 1), ls="--", c="grey")
    pr_ax.axhline(full_cm.prevalence, ls="--", c="grey")

    return {
        "roc": (roc_handle, roc_label),
        "pr": (pr_handle, pr_label),
    }


def _get_rhombus(x: float, y: float, left: float, right: float, bottom: float, top: float, **kwargs: Any) -> Polygon:
    """Get rhombus-like shape with vertices at (x, top), (right, y), (x, bottom), and (left, y)."""
    return Polygon(
        np.array([[x, top], [right, y], [x, bottom], [left, y]]),  # type: ignore
        **kwargs,
    )


def _get_bezier_ellipse(
    x: float, y: float, left: float, right: float, bottom: float, top: float, **kwargs: Any
) -> PathPatch:
    """Get ellipse-like shape with radii at (x, top), (right, y), (x, bottom), and (left, y)."""
    # Control points to approximate quarter of ellipse: https://stackoverflow.com/a/16104121
    template = np.array([[0, 1], [0.55, 1], [1, 0.55], [1, 0]])

    xy = np.array([x, y])
    top_right_verts = xy + template * (right - x, top - y)
    bottom_right_verts = xy + template[::-1] * (right - x, bottom - y)
    bottom_left_verts = xy + template * (left - x, bottom - y)
    top_left_verts = xy + template[::-1] * (left - x, top - y)

    codes = [Path.MOVETO] + 12 * [Path.CURVE4]
    verts = np.concatenate([top_right_verts[:-1], bottom_right_verts[:-1], bottom_left_verts[:-1], top_left_verts])
    return PathPatch(Path(verts, codes, closed=True), **kwargs)


def plot_2d_errors(
    x: float,
    y: float,
    xrange: Sequence[float],
    yrange: Sequence[float],
    ax: plt.Axes,
    rhombus: bool = False,
    ellipse: bool = False,
    **kwargs: Any,
) -> tuple[ErrorbarContainer, Optional[Artist]]:
    """Plot points with 2D errorbars and optional shading.

    :param x: Horizontal coordinate of point.
    :param y: Vertical coordinate of point.
    :param xrange: Range of horizontal errorbar (min, max).
    :param yrange: Range of vertical errorbar (min, max).
    :param ax: Axis onto which to plot.
    :param rhombus: Whether to include a rhombus-like shading connecting to the ends of the errorbars.
    :param ellipse: Whether to include an ellipse-like shading connecting to the ends of the errorbars.
    :return: Handles for the errorbar and optional shading.
    """
    default_errorbar_kwargs = dict(capsize=5, fmt="o", mew=1.5, markersize=7)
    default_shading_kwargs = dict(alpha=0.2, lw=0, zorder=-1)

    xlow, xhigh = xrange
    ylow, yhigh = yrange
    errorbar_handle = ax.errorbar(
        x=x, xerr=[[x - xlow], [xhigh - x]], y=y, yerr=[[y - ylow], [yhigh - y]], **default_errorbar_kwargs, **kwargs
    )

    default_shading_kwargs["color"] = errorbar_handle[0].get_color()
    if rhombus:
        shading_handle = ax.add_artist(
            _get_rhombus(x=x, y=y, left=xlow, right=xhigh, bottom=ylow, top=yhigh, **default_shading_kwargs, **kwargs)
        )
    elif ellipse:
        shading_handle = ax.add_artist(
            _get_bezier_ellipse(
                x=x, y=y, left=xlow, right=xhigh, bottom=ylow, top=yhigh, **default_shading_kwargs, **kwargs
            )
        )
    else:
        shading_handle = None
    return errorbar_handle, shading_handle


def plot_bootstrap_calibration(
    data: pd.DataFrame, label_column: str, prob_column: str, ax: Axes, num_bins: int = 5, num_samples: int = 100
) -> None:
    """Plot calibration bars with bootstrapping errorbars for a classifier's outputs."""
    bins = np.linspace(0, 1, num_bins + 1)

    def compute_binned_calibration_stats(sample: pd.DataFrame) -> dict[str, np.ndarray]:
        probs, labels = sample[prob_column], sample[label_column]
        return {
            "proportion": binned_statistic(probs, labels, bins=bins, statistic="mean").statistic,  # type: ignore
            "positives": binned_statistic(probs, labels, bins=bins, statistic="sum").statistic,  # type: ignore
            "total": binned_statistic(probs, labels, bins=bins, statistic="count").statistic,  # type: ignore
        }

    bin_widths = np.diff(bins)
    bin_centres = (bins[:-1] + bins[1:]) / 2

    full_stats = compute_binned_calibration_stats(data)
    quantiles = compute_bootstrap_quantiles(
        data, compute_binned_calibration_stats, quantiles=[0.025, 0.975], num_samples=num_samples
    )

    prop_low, prop_high = quantiles["proportion"]
    full_prop = full_stats["proportion"]

    for i in range(num_bins):
        ax.text(
            bin_centres[i],
            0.01,
            f"{int(full_stats['positives'][i])} / {int(full_stats['total'][i])}",
            ha="center",
            va="bottom",
            color="w",
        )

    ax.bar(
        bins[:-1],
        full_prop,
        yerr=(full_prop - prop_low, prop_high - full_prop),
        width=bin_widths,
        align="edge",
        capsize=8,
    )

    ax.bar(bins[:-1], bin_centres, width=bin_widths, align="edge", fill=None, ls="--")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True proportion")
    ax.set_axisbelow(True)
    ax.grid(color="0.9")


def compute_importance_weights(true_label: np.ndarray, prevalence: Union[float, np.ndarray]) -> np.ndarray:
    """Compute normalised weights for positive and negative instances to match the target prevalence."""
    negatives, positives = np.bincount(true_label)
    pos_weight = prevalence / (1 - prevalence) * negatives / positives
    weights = np.where(true_label, pos_weight, 1.0)
    weights /= weights.sum(-1, keepdims=True)
    return weights


def get_weighted_confusion_matrix(
    true_labels: np.ndarray, pred_labels: np.ndarray, weights: np.ndarray
) -> ConfusionMatrix:
    """Estimate a confusion matrix from samples with arbitrary positive weights."""
    weights = np.squeeze(weights)

    num_total = weights.sum(-1)
    num_positives = (weights * true_labels).sum(-1)
    true_positives = (weights * true_labels * pred_labels).sum(-1)
    false_positives = (weights * (1 - true_labels) * pred_labels).sum(-1)

    return ConfusionMatrix(
        num_total=num_total,
        num_positives=num_positives,  # type: ignore  # Slight abuse here with array instead of int
        true_positives=true_positives,
        false_positives=false_positives,
        validate_args=False,
    )


CytedDecisionFn = Callable[[bool, bool, bool], bool]


def simulate_cyted_workflow(
    decision_fn: CytedDecisionFn,
    he_pred: np.ndarray,
    tff3_pred: np.ndarray,
    true_label: np.ndarray,
    prevalence: Union[float, np.ndarray],
) -> tuple[ConfusionMatrix, np.ndarray, np.ndarray]:
    """Simulate a workflow with a given decision function and compute output statistics.

    :param decision_fn: A Boolean function combining H&E prediction, TFF3 prediction, and pathologist label.
    :param he_pred: Array of H&E model predictions (N,).
    :param tff3_pred: Array of TFF3 model predictions (N,).
    :param true_label: Array of pathologist labels (N,).
    :param prevalence: Barrett's prevalence in the target population.
    :return: The end-to-end confusion matrix; `(N, 3)` array of whether H&E, TFF3, and/or pathologist labels were used
        for each decision; and `(N, 3)` array of prevalences observed by H&E model, TFF3 model, and pathologist.
    """
    decision_tree = DecisionTree(decision_fn)
    decisions = decision_tree.decide(he_pred, tff3_pred, true_label)
    he_used, tff3_used, path_used = decision_tree.were_used(he_pred, tff3_pred, true_label)

    # H&E and TFF3 are always implicitly needed for pathologist review:
    he_used |= path_used
    tff3_used |= path_used

    weights = compute_importance_weights(true_label, prevalence)  # shape: (..., N)
    cm = get_weighted_confusion_matrix(true_label, decisions, weights)
    were_used = np.asarray([he_used, tff3_used, path_used])  # shape: (3, N)

    weighted_were_used = weights[..., None, :] * were_used  # shape: (..., 3, N)
    obs_prevalences = (weighted_were_used @ true_label) / weighted_were_used.sum(-1)

    return cm, weighted_were_used.sum(-1), obs_prevalences


def plot_workflow_sweeps(
    decision_fn: CytedDecisionFn,
    df: pd.DataFrame,
    he_cm: ConfusionMatrixSweep,
    tff3_cm: ConfusionMatrixSweep,
    title: str,
    prevalence: float,
) -> tuple[Figure, Sequence[Axes]]:
    """Simulate workflow and plot statistics for a range of H&E and TFF3 model thresholds.

    Plots end-to-end sensitivity and specificity, and the proportions of cases requiring
    pathologist review and TFF3 staining.

    :param decision_fn: A Boolean function combining H&E prediction, TFF3 prediction, and pathologist label.
    :param df: Dataframe containing H&E and TFF3 model probabilities (`he_prob` and `tff3_prob`)
        and pathologist labels (`true_label`).
    :param he_cm: Confusion matrix sweep for H&E model.
    :param tff3_cm: Confusion matrix sweep for TFF3 model.
    :param title: Overall figure title.
    :param prevalence: Barrett's prevalence in the target population.
    :return: Resulting figure and array of axes, for further customisation.
    """
    # Truncate sensitivity range for better visualisation:
    min_sens = 0.7
    he_cm = he_cm[search_1d(he_cm.sensitivity, min_sens, ascending=True) - 1:]  # noqa: E203
    tff3_cm = tff3_cm[search_1d(tff3_cm.sensitivity, min_sens, ascending=True) - 1:]  # noqa: E203

    he_prob = df["he_prob"].to_numpy()
    tff3_prob = df["tff3_prob"].to_numpy()
    true_label = df["true_label"].to_numpy()

    he_pred_sweep = he_prob >= he_cm.thresholds[:, None, None]
    tff3_pred_sweep = tff3_prob >= tff3_cm.thresholds[None, :, None]

    cm2d, (_, tff3_used, path_used), _ = simulate_cyted_workflow(
        decision_fn, he_pred_sweep, tff3_pred_sweep, true_label, prevalence=prevalence
    )

    fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.01))
    yy, xx = np.meshgrid(tff3_cm.sensitivity, he_cm.sensitivity)

    def plot_heatmap(zz: np.ndarray, ax: Axes, colorbar_kw: dict[str, Any] = {}, **kwargs: Any) -> None:
        handle = ax.pcolormesh(xx, yy, zz, shading="nearest", **kwargs)
        ax.set_aspect(1)
        plt.colorbar(handle, ax=ax, location="top", **colorbar_kw)

    plot_heatmap(cm2d.sensitivity, axs[0], cmap="Reds", vmin=min_sens, vmax=1, colorbar_kw=dict(extend="min"))
    plot_heatmap(cm2d.specificity, axs[1], cmap="Blues", vmin=0.7, vmax=1, colorbar_kw=dict(extend="min"))
    plot_heatmap(path_used, axs[2], cmap="Greens_r", vmin=0, vmax=1)
    plot_heatmap(tff3_used, axs[3], cmap="Oranges_r", vmin=0, vmax=1)

    plt.setp(axs, xlabel="H&E sensitivity")  # type: ignore
    axs[0].set_ylabel("TFF3 sensitivity")
    axs[0].set_title("Final sensitivity", pad=40, fontsize=10)
    axs[1].set_title("Final specificity", pad=40, fontsize=10)
    axs[2].set_title("Fraction requiring pathologist", pad=40, fontsize=10)
    axs[3].set_title("Fraction requiring TFF3", pad=40, fontsize=10)

    fig.suptitle(title)

    return fig, axs  # type: ignore


def compute_all_workflow_stats(
    df: pd.DataFrame,
    decision_fn_dict: dict[str, CytedDecisionFn],
    he_thr: float,
    tff3_thr: float,
    prevalence: float,
) -> dict[tuple[str, str], float]:
    """Simulate and compute statistics for all given workflows.

    :param df: Dataframe containing H&E and TFF3 model probabilities (`he_prob` and `tff3_prob`)
        and pathologist labels (`true_label`).
    :param decision_fn_dict: Dictionary mapping workflow titles to decision functions combining
        H&E prediction, TFF3 prediction, and pathologist label.
    :param he_thr: Threshold for H&E model probabilities.
    :param tff3_thr: Threshold for TFF3 model probabilities.
    :param prevalence: Barrett's prevalence in the target population.
    :return: Dictionary mapping `(title, stat): value`, where `title` is each key in `decision_fn_dict` and `stat` is
        each of `sensitivity`, `specificity`, `tff3_used`, `path_used`, and `path_obs_prev`.
    """
    he_pred = df["he_prob"].to_numpy() >= he_thr
    tff3_pred = df["tff3_prob"].to_numpy() >= tff3_thr
    true_label = df["true_label"].to_numpy()
    stats = {}
    for title, decision_fn in decision_fn_dict.items():
        cm, (_, tff3_used, path_used), (_, _, path_obs_prev) = simulate_cyted_workflow(
            decision_fn, he_pred, tff3_pred, true_label, prevalence=prevalence
        )
        stats[title, "sensitivity"] = float(cm.sensitivity)
        stats[title, "specificity"] = float(cm.specificity)
        stats[title, "tff3_used"] = tff3_used
        stats[title, "path_used"] = path_used
        stats[title, "path_obs_prev"] = path_obs_prev
    return stats


def format_percent_ci(value: float, low: float, high: float, *, precision: int = 0) -> str:
    if high - low < 10 ** -(2 + precision):  # same formatted percentage
        return f"{value:.{precision}%}"
    return f"{value:.{precision}%} [{100 * low:.{precision}f}–{high:.{precision}%}]"


def plot_workflow_operating_points(
    decision_fn_dict: dict[str, CytedDecisionFn],
    workflow_stats_full: dict[tuple[str, str], float],
    workflow_stats_quantiles: dict[tuple[str, str], np.ndarray],
    he_cm: ConfusionMatrixSweep,
    tff3_cm: ConfusionMatrixSweep,
) -> None:
    """Simulate and plot operating points with errorbars for all given workflows.

    :param decision_fn_dict: Dictionary mapping workflow titles to decision functions combining
        H&E prediction, TFF3 prediction, and pathologist label.
    :param workflow_stats_full: Output of `compute_all_workflow_stats()`.
    :param workflow_stats_quantiles: Bootstrap quantiles of `compute_all_workflow_stats()`, as
        returned by `bootstrapping.compute_bootstrap_quantiles()`.
    :param he_cm: Confusion matrix sweep for H&E model, to plot ROC curve.
    :param tff3_cm: Confusion matrix sweep for TFF3 model, to plot ROC curve.
    """
    styles_dict: dict[str, dict[str, Any]] = defaultdict(dict)
    for title in decision_fn_dict:
        if title in ["H&E AND Pathologist", "TFF3 AND Pathologist"]:
            styles_dict[title].update(marker=">", markersize=7)
        if title in ["H&E OR Pathologist", "TFF3 OR Pathologist"]:
            styles_dict[title].update(marker="v", markersize=7)
        if title in ["H&E only", "TFF3 only", "H&E AND TFF3", "H&E OR TFF3"]:
            styles_dict[title].update(mfc="w")
        if "H&E AND TFF3" in title:
            styles_dict[title].update(color="C2")

    plt.figure(figsize=(5, 5))
    for title in decision_fn_dict:
        path_ci = format_percent_ci(
            workflow_stats_full[title, "path_used"], *workflow_stats_quantiles[title, "path_used"]
        )
        tff3_ci = format_percent_ci(
            workflow_stats_full[title, "tff3_used"], *workflow_stats_quantiles[title, "tff3_used"]
        )
        path_prev_ci = format_percent_ci(
            workflow_stats_full[title, "path_obs_prev"], *workflow_stats_quantiles[title, "path_obs_prev"]
        )
        label = f"{title} ({path_ci} path, {tff3_ci} TFF3, {path_prev_ci} path prev)"

        if title == "Pathologist":
            plt.plot(
                workflow_stats_full[title, "specificity"],
                workflow_stats_full[title, "sensitivity"],
                "Xk",
                markersize=10,
                mew=0,
                label=label,
            )
            continue

        errorbar, shading = plot_2d_errors(
            x=workflow_stats_full[title, "specificity"],
            xrange=tuple(workflow_stats_quantiles[title, "specificity"]),
            y=workflow_stats_full[title, "sensitivity"],
            yrange=tuple(workflow_stats_quantiles[title, "sensitivity"]),
            ax=plt.gca(),
            ellipse=True,
            label=label,
        )
        # Ignoring types because setp() supports nested Artist collections, but it's undocumented
        if "H&E" in title and "TFF3" not in title:
            plt.setp((errorbar, shading), color="C0")  # type: ignore
        if "TFF3" in title and "H&E" not in title:
            plt.setp((errorbar, shading), color="C1")  # type: ignore
        if "H&E AND TFF3" in title:
            plt.setp((errorbar, shading), color="C2")  # type: ignore
        if "H&E OR TFF3" in title:
            plt.setp((errorbar, shading), color="C3")  # type: ignore

        if "Path" not in title:
            plt.setp(errorbar[0], mfc="w")  # type: ignore
            plt.setp(errorbar[2], ls=":")  # type: ignore
        if "AND Path" in title:
            plt.setp(errorbar[0], marker="<")  # type: ignore
        if "OR Path" in title:
            plt.setp(errorbar[0], marker="^")  # type: ignore
        if "Consensus" in title:
            plt.setp(errorbar[0], marker="s")  # type: ignore
            plt.setp((errorbar, shading), color="C9")  # type: ignore

    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot(he_cm.specificity, he_cm.sensitivity, "C0-", lw=1, alpha=0.3, label="H&E model ROC curve")
    plt.plot(tff3_cm.specificity, tff3_cm.sensitivity, "C1-", lw=1, alpha=0.3, label="TFF3 model ROC curve")
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.grid(color="0.9")
    plt.gca().invert_xaxis()
    plt.xlabel("Specificity")
    plt.ylabel("Sensitivity")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [handle[0] if isinstance(handle, ErrorbarContainer) else handle for handle in handles]
    plt.figlegend(
        handles, labels, fontsize="small", loc="center left", ncol=1, frameon=False, bbox_to_anchor=(0.9, 0.5)
    )
