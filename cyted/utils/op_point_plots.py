#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter
import pandas as pd
from scipy.stats import binned_statistic


from health_cpath.utils.analysis_plot_utils import format_pr_or_roc_axes
from health_cpath.utils.naming import ResultsKey

from cyted.utils.array_utils import sliced_search_2d
from cyted.utils.conf_matrix import ConfusionMatrix

_GRID_COLOUR = "0.9"


def create_grid_confusion_matrix(num_total: int, num_positives: int, grid_size: int) -> ConfusionMatrix:
    """Generates a 2D confusion matrix sweeping over numbers of true and predicted positives.

    :param num_total: Total number of samples.
    :param num_positives: Number of positive samples.
    :param grid_size: Number of steps to generate along each axis.
    :return: The 2D confusion matrix.
    """
    num_steps = grid_size * 1j  # complex step tells np.mgrid that this is the number of steps, not step size
    tp2d, pp2d = np.mgrid[:num_positives:num_steps, :num_total:num_steps]  # type: ignore
    fp2d = pp2d - tp2d
    cm2d = ConfusionMatrix(
        num_total=num_total,
        num_positives=num_positives,
        true_positives=tp2d,
        false_positives=fp2d,
        validate_args=False,  # full grid will contain some invalid points
    )
    return cm2d


def annotate_levels(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    levels: Sequence[float],
    ax: Axes,
    label_fmt: Any = None,
    frac_x: Optional[float] = None,
    frac_y: Optional[float] = None,
    contour_kwargs: Optional[Dict[str, Any]] = None,
    clabel_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Annotate level contours and labels onto a 2D plot.

    :param xx: Horizontal mesh coordinates, same 2D shape as `yy` and `zz`.
    :param yy: Vertical mesh coordinates, same 2D shape as `xx` and `zz`.
    :param zz: Values, same 2D shape as `xx` and `yy`.
    :param levels: Levels of `zz` to annotate.
    :param ax: Axes to annotate.
    :param label_fmt: Formatter for the level labels, as given to `ax.clabel()`.
    :param frac_x: Horizontal position of the level labels, as a fraction of the width.
        One of `frac_x` or `frac_y` must be specified.
    :param frac_y: Vertical position of the level labels, as a fraction of the height.
        One of `frac_x` or `frac_y` must be specified.
    :param contour_kwargs: Keyword arguments for `ax.contour()`.
    :param clabel_kwargs: Keyword arguments for `ax.clabel()`.
    """
    _contour_kwargs = dict(colors="0.8", linewidths=0.5)
    if contour_kwargs:
        _contour_kwargs.update(contour_kwargs)
    contours = ax.contour(xx, yy, zz, levels=levels, **_contour_kwargs)

    # Get level label coordinates
    label_indices_ij = [sliced_search_2d(zz, level, frac_x, frac_y, ascending=None) for level in levels]
    label_locations_xy = [(xx[i, j], yy[i, j]) for i, j in label_indices_ij]

    _clabel_kwargs = dict(colors="0.8", fontsize="small")
    if clabel_kwargs:
        _clabel_kwargs.update(clabel_kwargs)
    ax.clabel(contours, manual=label_locations_xy, fmt=label_fmt, **_clabel_kwargs)


def add_sens_spec_annotations(
    cm: ConfusionMatrix,
    levels: Sequence[float],
    ax: Axes,
    include_sens: bool = True,
    include_spec: bool = True,
    include_random: bool = True,
    normalised: bool = False,
) -> None:
    """Add sensitivity and specificity annotations to a screening curve plot.

    This also includes hatching for invalid regions of the plane.

    :param cm: Reference confusion matrix from which to retrieve total and positive counts.
    :param levels: Sensitivity and specificity levels to draw, between 0 and 1.
    :param ax: Axes to annotate.
    :param include_sens: Whether to draw sensitivity levels (horizontal).
    :param include_spec: Whether to draw specificity levels (slanted).
    :param include_random: Whether to draw random sampling baseline (diagonal).
    :param normalised: If true, axes are assumed to be normalised to 1, i.e. fraction of true
        positives among positives and of predicted positives over the total. Otherwise, raw counts
        are used.
    """
    cm2d = create_grid_confusion_matrix(cm.num_total, cm.num_positives, grid_size=100)

    xx, yy = cm2d.pred_positives, cm2d.true_positives

    if normalised:
        # Avoid in-place modification
        xx = xx / cm2d.num_total
        yy = yy / cm2d.num_positives

    if include_sens:
        annotate_levels(
            xx, yy, cm2d.sensitivity, levels=levels, frac_x=0.5, label_fmt=lambda x: f"$Sens={100*x:.0f}\\%$", ax=ax
        )

    if include_spec:
        annotate_levels(
            xx, yy, cm2d.specificity, levels=levels, frac_y=0.5, label_fmt=lambda x: f"$Spec={100*x:.0f}\\%$", ax=ax
        )

    if include_random:
        slope = 1 if normalised else cm.prevalence
        ax.axline((0, 0), slope=slope, ls="--", c="grey")
        ax.text(
            0.5,
            0.5,
            " Random",
            ha="center",
            va="center",
            color="grey",
            size=9,
            transform_rotates_text=True,
            rotation=45,
            rotation_mode="anchor",
            bbox=dict(facecolor="w", edgecolor="none", pad=2),
        )

    spc_boundary_contours = ax.contourf(
        xx, yy, cm2d.specificity, hatches=["xxxx", "", "xxxx"], colors="none", levels=[0, 1], extend="both"
    )
    plt.setp(spc_boundary_contours.collections, edgecolor="0.7", linewidth=0)  # type: ignore


def plot_calibration(probs: np.ndarray, labels: np.ndarray, ax: Axes, num_bins: int = 5) -> None:
    """Plot calibration bars for a classifier's outputs.

    :param probs: Predicted probabilities.
    :param labels: True labels.
    :param ax: Axes to plot on.
    :param num_bins: Number of bins to use.
    """
    bins = np.linspace(0, 1, num_bins + 1)

    proportion = binned_statistic(probs, labels, bins=bins).statistic  # type: ignore
    binned_positives = binned_statistic(probs, labels, statistic="sum", bins=bins).statistic  # type: ignore
    binned_total = binned_statistic(probs, labels, statistic="count", bins=bins).statistic  # type: ignore

    bin_widths = np.diff(bins)
    bin_centres = (bins[:-1] + bins[1:]) / 2

    ax.bar(bins[:-1], proportion, width=bin_widths, align="edge")
    ax.bar(bins[:-1], bin_centres, width=bin_widths, align="edge", fill=None, ls="--")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True proportion")
    for i in range(num_bins):
        ax.text(
            bin_centres[i],
            0.02,
            f"{int(binned_positives[i])} / {int(binned_total[i])}",
            ha="center",
            va="bottom",
            color="w",
        )
    ax.set_axisbelow(True)
    ax.grid(color=_GRID_COLOUR)


def format_sens_spec_axes(ax: Axes) -> None:
    """
    Format axes for sensitivity-specificity plots.

    :param ax: Axes to format.
    """
    ax.set(
        xlabel="Specificity",
        ylabel="Sensitivity",
        aspect=1,
        xlim=(1.05, -0.05),
        ylim=(-0.05, 1.05),
    )
    ax.grid(color="0.9")


def plot_roc_and_pr_curves(cms: Sequence[ConfusionMatrix], roc_ax: Axes, pr_ax: Axes, **kwargs: Any) -> None:
    """Plot ROC (sensitivity-specificity) and precision-recall curves.

    :param cms: Confusion matrices whose outputs to visualise, e.g. singleton or cross-validation.
    :param roc_ax: Axes onto which to plot ROC curve.
    :param pr_ax: Axes onto which to plot PR curve.
    :param kwargs: Additional arguments to pass to `plot()` for all curves.
    """
    for cm in cms:
        roc_ax.plot(cm.specificity, cm.sensitivity, **kwargs)
        pr_ax.plot(cm.sensitivity, cm.pos_pred_value, **kwargs)
    mean_prevalence = np.mean([cm.prevalence for cm in cms])

    roc_ax.axline((1, 0), (0, 1), ls="--", c="grey")
    format_sens_spec_axes(roc_ax)
    pr_ax.axhline(mean_prevalence, ls="--", c="grey")
    format_pr_or_roc_axes("pr", pr_ax)


def plot_screening_curve(cms: Sequence[ConfusionMatrix], spec_levels: Sequence[float], ax: Axes) -> None:
    """Plot fraction of retained positives vs manually reviewed cases.

    :param cms: Confusion matrices whose outputs to visualise, e.g. singleton or cross-validation.
    :param spec_levels: Specificity levels to annotate.
    :param ax: Axes onto which to plot.
    """
    add_sens_spec_annotations(cms[0], levels=spec_levels, ax=ax, include_sens=False, normalised=True)
    for cm in cms:
        ax.plot(cm.pred_positives / cm.num_total, cm.sensitivity)

    sens_ticks = ax.get_yticks()
    sens_ticks = np.unique(np.concatenate([sens_ticks, spec_levels]))
    ax.set_yticks(sens_ticks)

    ax.set_xlabel("Fraction of manually reviewed cases")
    ax.set_ylabel("Fraction of retained positives (sensitivity)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(color=_GRID_COLOUR, linewidth=0.5)


def plot_log_histogram(
    values: np.ndarray, num_bins: int, ax: Axes, logx: bool = True, logy: bool = False, **kwargs: Any
) -> None:
    """
    Plot a histogram with logarithmic axis.

    :param values: Values to plot.
    :param num_bins: Number of bins to use.
    :param ax: Axes to plot on.
    :param logx: Whether to use a logarithmic x-axis.
    :param logy: Whether to use a logarithmic y-axis.
    :param kwargs: Additional arguments to pass to `hist()`.
    """
    if logx:
        ax.set_xscale("log")
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), num_bins)
    else:
        bins = np.linspace(values.min(), values.max(), num_bins)

    if logy:
        ax.set_yscale("log")

    ax.hist(values, bins=bins, **kwargs)

    ax.set_axisbelow(True)
    ax.grid(color=_GRID_COLOUR, which="both")


def plot_cdf(values: np.ndarray, ax: Axes, **kwargs: Any) -> None:
    """
    Plot a cumulative distribution function.

    :param values: Values to plot.
    :param ax: Axes to plot on.
    """
    values = np.sort(values[np.isfinite(values)])
    cum_probs = np.linspace(0, 1, len(values))
    ax.plot(values, cum_probs, **kwargs)


def save_plot_roc_and_pr_curves(cm: ConfusionMatrix, add_best2_operating_point: bool, data_dir: Path) -> None:
    _, axs = plt.subplots(1, 2, figsize=(8, 4))
    """
    Saves a plot of ROC and PR curves to a file.

    :param cm: Confusion matrix.
    :param add_best2_operating_point: Whether to add the operating point from pathologist on BEST2.
    :param data_dir: The directory where the plot will be saved.
    """
    plot_roc_and_pr_curves([cm], roc_ax=axs[0], pr_ax=axs[1])
    # Add the operating point from pathologist on BEST2
    if add_best2_operating_point:
        axs[0].errorbar(0.927, 0.817, xerr=0.03, yerr=0.045, marker="o", markersize=5, markerfacecolor="red", capsize=3)
    plt.tight_layout()
    plt.savefig(data_dir / "ensemble_roc_and_pr_curves.png", dpi=300, bbox_inches="tight")


def save_plot_screening_curve(cm: ConfusionMatrix, data_dir: Path) -> None:
    """
    Saves a plot of the screening curve to a file.

    :param cm: Confusion matrix.
    :param data_dir: The directory where the plot will be saved.
    """
    _, _ = plt.subplots(1, 1, figsize=(8, 4))
    plot_screening_curve([cm], spec_levels=[0.8, 0.9, 0.95], ax=plt.gca())
    plt.tight_layout()
    plt.savefig(data_dir / "ensemble_screening_curves.png", dpi=300, bbox_inches="tight")


def save_plot_calibration(
    ensemble_preds: np.ndarray, list_output_dicts: List[Dict[str, pd.Series]], data_dir: Path
) -> None:
    """
    Saves a plot of the calibration curve to a file.

    :param ensemble_preds: Ensemble predictions as a torch tensor.
    :param list_output_dicts: A list of dictionaries containing output of the cross val runs.
    :param data_dir: The directory where the plot will be saved.
    """
    _, _ = plt.subplots(1, 1, figsize=(8, 4))
    plot_calibration(
        probs=np.array(ensemble_preds), labels=np.array(list_output_dicts[0][ResultsKey.TRUE_LABEL]), ax=plt.gca()
    )
    plt.tight_layout()
    plt.savefig(data_dir / "ensemble_calibration.png", dpi=300, bbox_inches="tight")
