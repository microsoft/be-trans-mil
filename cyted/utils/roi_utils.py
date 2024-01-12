#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import numpy as np
import scipy.signal as ss
from scipy.ndimage import distance_transform_cdt
from skimage import filters

from health_ml.utils.box_utils import Box, get_bounding_box


def get_margins(hist: np.ndarray, frac: float) -> tuple[int, int]:
    """Determine indices of the given 1D histogram's `frac` and `1-frac` quantiles."""
    cdf = hist.cumsum() / hist.sum()
    low = int(np.searchsorted(cdf, frac))
    high = int(np.searchsorted(cdf, 1 - frac))
    return low, high


def get_distance_transform(hist: np.ndarray, left_margin: int, right_margin: int) -> np.ndarray:
    """Compute a distance transform for the given 1D histogram.

    :param hist: A 1D array of counts or frequencies.
    :param left_margin: Index below which to ignore.
    :param right_margin: Index above which to ignore.
    :return: An array of the same shape as `hist`, where each element contains
        the distance to the nearest nonzero element of `hist`. All values
        outside the `left_margin:right_margin` range are set to 0.
    """
    background: np.ndarray = hist == 0
    distances = distance_transform_cdt(background[left_margin:right_margin])
    assert isinstance(distances, np.ndarray)

    # Pad distances back to length of full histogram
    padding = (left_margin, len(hist) - right_margin)
    padded_distances = np.pad(distances, (padding,))  # type: ignore
    assert len(padded_distances) == len(hist)

    return padded_distances


def max_split_correlation(x: np.ndarray, index: int) -> float:
    """Compute the maximum normalised cross-correlation between `x[:index]` and `x[index:]`."""
    corr = ss.correlate(x[:index], x[index:], mode="full")
    max_corr = 2 * corr.max() / (x * x).sum()
    return max_corr


def find_best_distance_peak(hist: np.ndarray, distances: np.ndarray, dist_weight: float = 0.01) -> int:
    """Select the distance peak that best splits the histogram in two parts.

    :param hist: A 1D array of counts or frequencies.
    :param distances: Distance transform of `hist`, as returned by
        `get_distance_transform()`.
    :param dist_weight: A coefficient for scaling the relative height of each
        distance peak before adding to the split cross-correlation (see
        `max_split_correlation()`) for disambiguation.
    :return: Best index at which to split the histogram.
    """
    peaks = ss.find_peaks(distances)[0]
    total_distance = distances[peaks].sum()

    # If no peaks (i.e. no vertical gap), fall back to Otsu threshold
    if len(peaks) == 0:
        # Threshold is <= vs >, so we add 1 to use it as a pixel index
        return filters.threshold_otsu(None, hist=hist) + 1

    def score_peak(t: int) -> float:
        max_corr = max_split_correlation(hist, t)
        dist = distances[t] / total_distance
        # Normalised correlations can be very close to each other, especially if
        # there are small isolated tissue fragments. Add the weighted distance
        # to disambiguate.
        score = max_corr + dist_weight * dist
        return score

    best_peak = max(peaks, key=score_peak)
    return best_peak


def get_left_right_roi_bboxes(mask: np.ndarray) -> tuple[Box, Box]:
    """Compute bounding boxes for a mask with two distinct regions side-by-side.

    This is achieved by finding the best vertical split line to separate the
    regions-of-interest, then fitting a bounding box to each one.

    :param mask: A 2D binary array.
    :return: A tuple containing the left and right bounding boxes.
    """
    hist: np.ndarray = mask.mean(0)
    left_margin, right_margin = get_margins(hist, frac=0.2)
    distances = get_distance_transform(hist, left_margin=left_margin, right_margin=right_margin)
    threshold = find_best_distance_peak(hist, distances)

    left_bbox = get_bounding_box(mask[:, :threshold])
    right_offset = (threshold, 0)
    right_bbox = get_bounding_box(mask[:, threshold:]) + right_offset

    return left_bbox, right_bbox
