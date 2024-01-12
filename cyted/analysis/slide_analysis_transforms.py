#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from monai.transforms.transform import MapTransform
from skimage.exposure import histogram
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
from typing import Dict, Optional, Tuple

from cyted.utils.crop_utils import resize_image_array_to_scale_factor
from cyted.analysis.analysis_metadata import AnalysisMetadata
from cyted.utils.stain_utils import hed_channels_to_rgb, separate_hed_stains
from health_cpath.utils.naming import SlideKey, TileKey


class BrownStainRatiod(MapTransform):
    """Estimate brown stain in a WSI using HED stain separation and Otsu thresholding. This is based on the method in
    the paper A deep learning model for molecular label transfer that enables cancer cell identification from
    histopathology images https://www.nature.com/articles/s41698-022-00252-0"""

    def __init__(
        self,
        image_key: str = SlideKey.IMAGE,
        id_key: str = SlideKey.IMAGE_PATH,
        downsample_factor: int = 1,
        background_val: Optional[int] = None,
        area_threshold: int = 64,
        variance_threshold: int = 5000000,
        plot_path: Optional[Path] = None,
        save_binary_mask: bool = False,
    ):
        """
        :param image_key: The key of the WSI image in the data dictionary.
        :param id_key: The key of the WSI id in the data dictionary.
        :param downsample_factor: The downsample factor to use for otsu thresholding. If set to 10, the image is
            downsampled by a factor of 10. Defaults to 1 for no downsampling.
        :param background_val: The background value to use for masking background pixels. If None, the 80th percentile
            of the image is used. Defaults to 255.
        :param area_threshold: The minimum area of brown stain to consider. Defaults to 64.
        :param variance_threshold: The variance threshold to use for considering a stain as brown. Consider all cells
            are normal if variance is too small. Defaults to 5000000.
        :param plot_path: The path to save the plots to. If None, no plots are saved. Defaults to None.
        :param save_binary_mask: Whether to save the binary mask of the brown stain. Defaults to False.
        """
        self.image_key = image_key
        self.id_key = id_key
        self.downsample_factor = downsample_factor
        self.background_val = background_val
        self.area_threshold = area_threshold
        self.variance_threshold = variance_threshold
        self.plot_path = plot_path
        self.save_binary_mask = save_binary_mask

    @property
    def debugging_mode(self) -> bool:
        return self.plot_path is not None

    def set_background_val_sum(self, wsi: np.ndarray) -> int:
        """Set the background value sum for R, G, B channels to use for masking background pixels if not already set.

        :param wsi: The WSI image array in CHW format.
        """
        if self.background_val is None:
            background_val = np.sum(np.quantile(wsi, 0.8, axis=(1, 2)))
        else:
            background_val = self.background_val * 3
        return background_val

    def save_histogram_of_dab_channel(self, dab_channel: np.ndarray) -> None:
        """Save a histogram of the DAB channel after removing outliers.

        :param dab_channel: The DAB channel image array.
        """
        if self.plot_path:
            plot_path = self.plot_path / self.wsi_id
            plot_path.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 1)
            dab_channel_no_outliers = dab_channel[dab_channel < np.quantile(dab_channel, 0.9)]
            axes.hist(dab_channel_no_outliers.flatten(), bins=256, range=(-0.025, 0))
            plt.title(f"{self.wsi_id} DAB histogram label: {self.wsi_label}")
            fig.tight_layout()
            plot_filename = plot_path / "dab_histogram.png"
            plt.savefig(plot_filename)

    def save_intermediate_plot(
        self, wsi_image: np.ndarray, step: str, cmap: Optional[str] = None, add_to_title: str = ""
    ) -> None:
        """Save an intermediate plot if a plot path is set for debugging purposes.

        :param wsi_image: The WSI image to plot in HWC format.
        :param step: The step of the analysis.
        :param cmap: The colormap to use for the image. Defaults to None.
        :param add_to_title: Additional text to add to the title. Defaults to "".
        """
        if self.plot_path:
            plot_path = self.plot_path / self.wsi_id
            plot_path.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 1)
            axes.imshow(wsi_image, cmap=cmap)
            axes.set_title(f"{self.wsi_id} label={self.wsi_label} {add_to_title}")
            fig.tight_layout()
            plot_filename = plot_path / f"{step}.png"
            plt.savefig(plot_filename)

    def get_downsampled_wsi(self, wsi: np.ndarray) -> np.ndarray:
        """Downsample the WSI to the downsample factor.

        :param wsi: The WSI image array in CHW format.:
        :return: The downsampled WSI image array in HWC format.
        """
        scale_factor = max(wsi.shape) / self.downsample_factor
        scale_factor = 1 / scale_factor
        return resize_image_array_to_scale_factor(wsi, scale_factor=scale_factor)

    def get_hem_binary(self, wsi_hed: np.ndarray) -> np.ndarray:
        """Get a binary mask of the HEM channel.

        :param wsi_hed: The HED image array in HWC format.
        :return: The binary mask of the HEM channel.
        """
        hem = -wsi_hed[..., 0]
        hem_thresh = threshold_otsu(hem)
        hem_binary = hem > hem_thresh
        return hem_binary

    def get_masked_dab_values(self, wsi_hed: np.ndarray, hem_binary: np.ndarray) -> np.ndarray:
        """Get the DAB channel values masked by the HEM channel to mask the background pixels.

        :param wsi_hed: The HED image array in HWC format.
        :param hem_binary: The binary mask of the HEM channel to mask the background pixels.
        :return: The DAB channel values masked by the HEM channel.
        """
        dab = -wsi_hed[..., 2]
        dab_masked = np.where(hem_binary == False, dab, 0)  # noqa: E712
        return np.array([i for i in dab_masked.ravel() if i != 0])

    def get_dab_threshold_masked(self, wsi_hed: np.ndarray, hem_binary: np.ndarray) -> Tuple[float, float]:
        """Get the DAB threshold using Otsu thresholding on the masked DAB channel. The mask is created using the HEM
        channel to detect the background.

        :param wsi_hed: The HED image array in HWC format.
        :param hem_binary: The binary mask of the HEM channel to mask the background pixels.
        :return: A tuple of the DAB threshold and the intra-tissue variance of the DAB channel.
        """
        dab_values = self.get_masked_dab_values(wsi_hed, hem_binary)
        # Otsu Thresholding
        hist, bin_centers = histogram(dab_values, 256)
        hist = hist.astype(float)
        # class probabilities for all possible thresholds
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        # class means for all possible thresholds
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
        # Clip ends to align class 1 and class 2 variables:
        # The last value of `weight1`/`mean1` should pair with zero values in
        # `weight2`/`mean2`, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        idx = np.argmax(variance12)
        # Consider all cells are normal if variance is too small
        if variance12[idx] < self.variance_threshold:
            threshold = 0
        else:
            threshold = bin_centers[:-1][idx]
        return threshold, variance12[idx]

    def get_brown_stain_ratio(self, foreground_mask: np.ndarray, dab_binary: np.ndarray) -> Tuple[float, int, int]:
        """Get the brown stain ratio by dividing the number of brown pixels by the number of foreground pixels.

        :param foreground_mask: The binary mask of the foreground.
        :param dab_binary: The binary mask of the DAB channel.
        :return: The brown stain ratio as a float between 0 and 1 and the number of foreground pixels and brown pixels.
        """
        foreground_pixels = foreground_mask.sum()
        brown_pixels = (1 - dab_binary).sum()
        return brown_pixels / foreground_pixels, foreground_pixels, brown_pixels

    def __call__(self, data: Dict) -> Dict:
        # Get the WSI, its id and label, and set the background value
        wsi = data[self.image_key].numpy()
        self.wsi_id = data[self.id_key]
        self.wsi_label = data[SlideKey.LABEL] if SlideKey.LABEL in data else None
        self.background_val_sum = self.set_background_val_sum(wsi)
        self.save_intermediate_plot(wsi.transpose(1, 2, 0), step="original_wsi", cmap="gray")

        # Compute the dab threshold from a downsampled version of the WSI if the downsample ratio is greater than 1
        wsi_small = self.get_downsampled_wsi(wsi) if self.downsample_factor > 1 else wsi.transpose(1, 2, 0)
        wsi_hed_small = separate_hed_stains(wsi_small)
        hem_binary_small = self.get_hem_binary(wsi_hed_small)
        dab_thresh, otsu_var = self.get_dab_threshold_masked(wsi_hed_small, hem_binary_small)

        if self.debugging_mode:
            wsi_hed_rgb_small = hed_channels_to_rgb(wsi_hed_small)
            for i, step in zip([0, 2], ["hem_channel", "dab_channel"]):
                self.save_intermediate_plot(wsi_hed_rgb_small[i], step=step, add_to_title=f"thresh={dab_thresh:.4f}")

        # Get the binary mask of the HEM channel and the DAB channel of the original WSI
        wsi = wsi.transpose(1, 2, 0)
        wsi_hed = separate_hed_stains(wsi) if self.downsample_factor > 1 else wsi_hed_small
        hem_binary = self.get_hem_binary(wsi_hed) if self.downsample_factor > 1 else hem_binary_small
        dab_channel = -wsi_hed[..., 2]
        dab_binary = dab_channel > dab_thresh if dab_thresh != 0 else np.ones_like(dab_channel)

        self.save_histogram_of_dab_channel(dab_channel)
        for binary_mask, step in zip([1 - hem_binary, 1 - dab_binary], ["hem_binary", "dab_binary"]):
            self.save_intermediate_plot(binary_mask, step=step, add_to_title=f"thresh={dab_thresh:.4f}", cmap="gray")

        # Remove small holes in the DAB binary mask and compute the brown stain ratio
        dab_binary_filtered = remove_small_holes(dab_binary, area_threshold=self.area_threshold)
        foreground_mask = wsi.sum(axis=2) < self.background_val_sum
        brown_ratio, foreground_pixels, brown_pixels = self.get_brown_stain_ratio(
            foreground_mask=foreground_mask, dab_binary=dab_binary_filtered
        )
        brown_ratio *= 100

        for binary_mask, step in zip(
            [foreground_mask, 1 - dab_binary_filtered], ["foreground_mask", "dab_binary_filtered"]
        ):
            title = f"area={self.area_threshold} percentage={brown_ratio:.4f}%"
            self.save_intermediate_plot(binary_mask, step=step, add_to_title=title, cmap="gray")

        # Add the results to the data dictionary
        data[AnalysisMetadata.BROWN_STAIN_RATIO] = brown_ratio
        data[AnalysisMetadata.FOREGROUND_PIXELS] = foreground_pixels
        data[AnalysisMetadata.FOREGROUND_HEM_PIXELS] = hem_binary.sum()
        data[AnalysisMetadata.BROWN_PIXELS] = brown_pixels
        data[AnalysisMetadata.DAB_THRESHOLD] = dab_thresh
        data[AnalysisMetadata.OTSU_CLASS_VAR] = otsu_var
        if self.save_binary_mask:
            data[AnalysisMetadata.BROWN_STAIN_BINARY_MASK] = 1 - dab_binary_filtered
            data[AnalysisMetadata.FOREGROUND_MASK] = foreground_mask
        return data


class TilesCountd(MapTransform):
    """Transform to count the number of tiles and add it to the data dictionary."""

    def __init__(self, image_key: str):
        """
        :param image_key: The key of the image in the data dictionary.
        """
        self.image_key = image_key

    def __call__(self, data: Dict) -> Dict:
        data[AnalysisMetadata.TILES_COUNT] = data[self.image_key].shape[0]
        return data


class TilesBrownStainRatiod(MapTransform):
    """Transform to compute the brown stain ratio of a tile and add it to the data dictionary."""

    def __init__(self, image_key: str):
        """
        :param image_key: The key of the image in the data dictionary.
        """
        self.image_key = image_key

    def __call__(self, data: Dict) -> Dict:
        brown_mask = (
            data[AnalysisMetadata.BROWN_STAIN_BINARY_MASK] if AnalysisMetadata.BROWN_STAIN_BINARY_MASK in data else None
        )
        foreground_mask = data[AnalysisMetadata.FOREGROUND_MASK] if AnalysisMetadata.FOREGROUND_MASK in data else None

        tile_count = data[self.image_key].shape[0]
        data[AnalysisMetadata.BROWN_STAIN_RATIO_TILES] = [0] * tile_count
        data[AnalysisMetadata.FOREGROUND_PIXELS_TILES] = [0] * tile_count
        data[AnalysisMetadata.BROWN_PIXELS_TILES] = [0] * tile_count
        if brown_mask is not None and foreground_mask is not None:
            for i in range(tile_count):
                left = data[TileKey.TILE_LEFT][i]
                right = data[TileKey.TILE_RIGHT][i]
                top = data[TileKey.TILE_TOP][i]
                bottom = data[TileKey.TILE_BOTTOM][i]
                brown_mask_tile = brown_mask[top:bottom, left:right]
                foreground_tile = foreground_mask[top:bottom, left:right]
                pixel_count_brown = brown_mask_tile.sum()
                pixel_count_foreground = foreground_tile.sum()
                perc_brown = pixel_count_brown / (pixel_count_foreground + 1e-5) * 100
                data[AnalysisMetadata.BROWN_PIXELS_TILES][i] = pixel_count_brown
                data[AnalysisMetadata.FOREGROUND_PIXELS_TILES][i] = pixel_count_foreground
                data[AnalysisMetadata.BROWN_STAIN_RATIO_TILES][i] = perc_brown
        else:
            logging.warning(
                "Skipping tile analysis because brown stain binary mask or foreground mask is not available."
                "Try setting `self.save_binary_mask` to True."
            )

        return data
