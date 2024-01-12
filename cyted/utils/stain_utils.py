#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import skimage.color as skc
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler

from cyted.analysis.analysis_metadata import AnalysisMetadata
from health_cpath.utils.naming import ResultsKey, TileKey

LOG_ADJUST = np.log(1e-6)


def separate_hed_stains(
    image: np.ndarray, bg_colour: Optional[Any] = None, hed_from_rgb: np.ndarray = skc.hed_from_rgb
) -> np.ndarray:
    """
    Convert an RGB image (e.g. H&E or IHC) to HED stain space. Uses the background colour (`bg_colour`)
    of the image if available.

    :param image: RGB image to convert to HED image in HWC format.
    :param bg_color: RGB background of the image as a numpy array.
    :param hed_from_rgb: HED-RGB stain conversion matrix (default=scikit image conversion matrix).
    :return HED image corresponding to the RGB image in HWC format.
    """
    if bg_colour is not None:
        image = image / np.array(bg_colour)
    hed = skc.separate_stains(image, hed_from_rgb)
    return hed


def hed_channels_to_rgb(
    hed_img: np.ndarray, rgb_from_hed: np.ndarray = skc.rgb_from_hed
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct separate RGB images for the given HED optical density image in HWC format.

    :param hed_img: HED channels to convert to RGB images.
    :param rgb_from_hed: RGB-HED stain conversion matrix (default=scikit image conversion matrix).
    :return Tuple of three RGB images in HWC formats, one for each of the three HED channels.
    """
    zeros = np.zeros_like(hed_img[:, :, 0])
    return (
        skc.combine_stains(np.dstack([hed_img[:, :, 0], zeros, zeros]), rgb_from_hed),
        skc.combine_stains(np.dstack([zeros, hed_img[:, :, 1], zeros]), rgb_from_hed),
        skc.combine_stains(np.dstack([zeros, zeros, hed_img[:, :, 2]]), rgb_from_hed),
    )


def to_optical_density(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB pixel intensities in [0, 1] to optical densities.

    :param img: RGB image array.
    :return optical density array of same shape as `img`.
    """
    od_img = np.log(np.maximum(img, 1e-6)) / LOG_ADJUST
    return np.maximum(od_img, 0)


def from_optical_density(od_img: np.ndarray) -> np.ndarray:
    """
    Convert optical densities to RGB pixel intensities in [0, 1].

    :param od_img: Optical density image array.
    :return RGB array of same shape as `od_img`.
    """
    img = np.exp(LOG_ADJUST * od_img)
    return np.clip(img, 0, 1)


def estimate_he_stain_matrix(
    rgb_pixels: np.ndarray, angle_percentile: int = 1, d_from_skimage: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the RGB-HED stain conversion matrix for H&E pixels.

    This function is based on the Macenko method and is only meant to reliably separate
    haematoxylin and eosin stains in H&E images.

    Reference:
    - Macenko et al., "A method for normalizing histology slides for quantitative analysis", ISBI 2009

    Adapted from the StainTools implementation:
    - https://github.com/Peter554/StainTools/blob/master/staintools/stain_extraction/macenko_stain_extractor.py

    :param rgb_pixels: A `(..., 3)`-shaped array of RGB pixel intensities in [0, 1]. Any masking or normalisation
        is expected to be performed before calling this function.
    :param angle_percentile: Bottom (100 - top) percentile for robust angle estimation.
    :param d_from_skimage: Whether to copy the last row of the stain matrix from `skimage`'s default DAB stain.
        If false, use the residual component orthogonal to the estimated H-E plane.
    :return: Tuple containing 3x3 RGB-from-HED and inverse HED-from-RGB stain conversion matrices.
    """
    # Apply PCA to the RGB optical density values of valid pixels
    rgb_od_pixels = to_optical_density(rgb_pixels).reshape(-1, 3)  # shape: (N, 3)
    _, _, principal_components = np.linalg.svd(rgb_od_pixels, full_matrices=False)  # shape: (3, 3)

    # Ensure convenient orientation for basis vectors (positive first coordinate)
    for i in range(3):
        if principal_components[i, 0] < 0:
            principal_components[i] *= -1
    he_plane_basis = principal_components[[0, 1], :]  # shape: (2, 3)

    # Estimate robust min/max planar angles w.r.t. first PC
    he_projections = rgb_od_pixels @ he_plane_basis.T  # shape: (N, 2)
    angles = np.arctan2(he_projections[:, 1], he_projections[:, 0])  # shape: (N,)
    min_angle, max_angle = np.percentile(angles, [angle_percentile, 100 - angle_percentile])

    # Set H-vector and E-vector as min- and max-angle directions in the H-E plane
    h_vector = np.array([np.cos(min_angle), np.sin(min_angle)]) @ he_plane_basis  # shape: (3,)
    e_vector = np.array([np.cos(max_angle), np.sin(max_angle)]) @ he_plane_basis  # shape: (3,)
    if h_vector[0] < e_vector[0]:
        h_vector, e_vector = e_vector, h_vector

    if d_from_skimage:
        # Copy D-vector from skimage's default DAB stain vector
        d_vector = skc.rgb_from_hed[2]  # shape: (3,)
    else:
        # Set D-vector as the principal component perpendicular to the H-E plane (residual)
        d_vector = principal_components[2]  # shape: (3,)

    # Assemble full stain matrix and its inverse to convert to and from RGB
    rgb_from_hed = np.stack([h_vector, e_vector, d_vector])  # shape: (3, 3)
    rgb_from_hed /= np.linalg.norm(rgb_from_hed, axis=1, keepdims=True)
    hed_from_rgb = np.linalg.inv(rgb_from_hed)  # shape: (3, 3)

    return rgb_from_hed, hed_from_rgb


class CorrKey(str, Enum):
    PEARSON = "pearson"
    KENDALL = "kendall"
    SPEARMAN = "spearman"
    MI = "mutual_info"


def find_correlation_stain_attention(
    slide_df: pd.DataFrame,
    stain_df: pd.DataFrame,
    reg_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[CorrKey, Any], pd.DataFrame]:
    """Function to find correlation between stain ratios and attentions for tiles in a slide.

    :param slide_df: Dataframe containing the attention values for each tile in a slide.
    (this can be obtained from `test_output.csv` for a given slide).
    :param stain_df: Dataframe containing the stain ratio values for each tile in a slide.
    :param reg_factor: Factor to convert stain_df coordinates to slide_df coordinates, based on registered dataset.
    :return A tuple containing the following:
    - Normalized attention values in 0-1 range.
    - Normalized stain ratio values in 0-1 range.
    - Unnormalized cumulative attention values (sum to 1).
    - Dictionary of correlation metrics (e.g. Pearson, Kendall, Spearman, MI).
    - A row from `slide_df` containing the true and predicted labels for the slide.
    """

    attentions_slide = []
    stain_ratios_slide = []
    scaler = MinMaxScaler()

    # find the rows in the stain_df that matches the coordinates in the slide_df
    for _, row in stain_df.iterrows():
        slide_df_row = slide_df.loc[
            (slide_df[TileKey.TILE_LEFT] == row.loc[TileKey.TILE_LEFT] * reg_factor)
            & (slide_df[TileKey.TILE_TOP] == row.loc[TileKey.TILE_TOP] * reg_factor)
            & (slide_df[TileKey.TILE_RIGHT] == row.loc[TileKey.TILE_RIGHT] * reg_factor)
            & (slide_df[TileKey.TILE_BOTTOM] == row.loc[TileKey.TILE_BOTTOM] * reg_factor)
        ]
        stain_ratio = row[AnalysisMetadata.BROWN_STAIN_RATIO_TILES] / 100

        # If the row is not empty, append the attention and stain ratio values
        if len(slide_df_row[ResultsKey.BAG_ATTN].values) > 0:
            attention = slide_df_row[ResultsKey.BAG_ATTN].values[0]
            attentions_slide.append(attention)
            stain_ratios_slide.append(stain_ratio)
            row_to_use_for_labels = slide_df_row

    # Normalize values in 0-1 range for the slide
    attentions_slide_norm_unsorted = scaler.fit_transform(np.array(attentions_slide).reshape(-1, 1))
    stain_ratios_slide_norm_unsorted = scaler.fit_transform(np.array(stain_ratios_slide).reshape(-1, 1))

    # Sort stain ratios in descending order and use same indices for attentions
    sorted_indices = np.argsort(stain_ratios_slide_norm_unsorted, axis=0)[::-1].squeeze()
    stain_ratios_slide_norm: np.ndarray = stain_ratios_slide_norm_unsorted[sorted_indices]
    attentions_slide_norm: np.ndarray = attentions_slide_norm_unsorted[sorted_indices]

    # Get cumulative attention values from unnormalized attentions
    attentions_slide_unnorm = np.array(attentions_slide).reshape(-1, 1)[sorted_indices]
    attentions_slide_unnorm_cum = np.cumsum(attentions_slide_unnorm)

    # Calculate correlation metrics
    pcc = np.corrcoef(attentions_slide_norm.flatten(), stain_ratios_slide_norm.flatten())
    tau, _ = kendalltau(attentions_slide_norm.flatten(), stain_ratios_slide_norm.flatten())
    rho, _ = spearmanr(attentions_slide_norm.flatten(), stain_ratios_slide_norm.flatten())
    mi = mutual_info_score(attentions_slide_norm.flatten(), stain_ratios_slide_norm.flatten())

    corr_dict = {CorrKey.PEARSON: pcc[0, 1], CorrKey.KENDALL: tau, CorrKey.SPEARMAN: rho, CorrKey.MI: mi}

    return attentions_slide_norm, stain_ratios_slide_norm, attentions_slide_unnorm_cum, corr_dict, row_to_use_for_labels


def plot_stain_attention_overlap(
    ax: plt.axes, num_tiles: int, attentions: np.ndarray, stain_ratios: np.ndarray
) -> None:
    """
    Function to plot the overlap between stain ratio and attention values for a slide.

    :param ax: Matplotlib axis object.
    :param num_tiles: Number of tiles in the slide.
    :param attentions: Attention values for each tile in the slide.
    :param stain_ratios: Stain ratio values for each tile in the slide (should correspond to `attentions`).
    """
    markerline, stemlines, _ = ax.stem(num_tiles, attentions.squeeze(), basefmt="", use_line_collection=True)
    markerline.set_markersize(0)
    plt.setp(stemlines, linewidth=2, alpha=0.5)

    markerline, stemlines, _ = ax.stem(num_tiles, stain_ratios.squeeze(), basefmt="", use_line_collection=True)
    markerline.set_markersize(0)
    plt.setp(stemlines, linewidth=2, alpha=0.25, color="orange")

    ax.legend(["attention", "stain ratio"], loc="upper right")
    ax.set_xlabel("tile")


def find_entropy_attentions(attentions_slide: np.ndarray) -> float:
    """
    Function to calculate the entropy of attentions for a slide.
    :param attentions_slide: Attention values for each tile in the slide.

    :return entropy_slide_normalized: Normalized entropy value for the attentions in a slide.
    """
    entropy_attn = -sum([p_i * np.log(p_i) for p_i in attentions_slide])
    entropy_slide_normalized = entropy_attn / np.log(len(attentions_slide))
    return entropy_slide_normalized
