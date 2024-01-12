#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.ticker import EngFormatter, MaxNLocator
from monai.data.wsi_reader import WSIReader
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def format_rounded_length(raw_microns: float) -> Tuple[float, str]:
    """Round and format a length in microns as a "nice" value.

    Will round the given microns length to the nearest multiple of {1, 2, 5} times a power of 10,
    formatted with SI multiples of a metre (µm, mm, etc.). For example, 1.0→"1µm", 123.0→"100µm",
    23,456→"20mm".

    :param raw_microns: A length in micrometres.
    :return: A tuple containing the rounded length and the formatted quantity.
    """
    locator = MaxNLocator(1, steps=[1, 2, 5, 10], integer=True)
    values = locator.tick_values(0, raw_microns)
    rounded_microns = values[1]
    formatter = EngFormatter(unit="m", places=0, sep="")
    label = formatter(rounded_microns * 1e-6)
    return rounded_microns, label


def add_scale_bar(size: float, label: str, ax: plt.Axes) -> None:
    """Add a scale bar to the bottom right of a plot.

    :param size: The size of the bar, in data units.
    :param label: A text label to appear beneath the bar.
    :param ax: Axes onto which to plot the scale bar.
    """
    # Height of scale bar must be given in data coordinates
    transform = ax.transData
    yscale = abs(transform.get_matrix()[1, 1])
    display_height = 2
    height = display_height / yscale

    full_scale_bar = AnchoredSizeBar(
        transform, size, label=label, loc="lower right", frameon=False, size_vertical=height, color="k"
    )

    # Add white outline to scale bar and label text to improve visibility
    bar = full_scale_bar.size_bar.get_children()[0]
    text = full_scale_bar.txt_label.get_children()[0]
    for element in [bar, text]:
        stroke_effect = patheffects.withStroke(linewidth=3, foreground="w")
        element.set_path_effects([stroke_effect])  # type: ignore

    ax.add_artist(full_scale_bar)


def add_auto_scale_bar(mpp: float, ax: plt.Axes, frac_width: float = 0.3) -> None:
    """Add a length scale bar to the bottom right of a plot.

    :param mpp: Spacing, in microns per pixel.
    :param ax: Axes onto which to plot the scale bar.
    :param frac_width: Maximum width of the scale bar, relative to the plot width.
    """
    xlim = ax.get_xlim()
    width_pixels = xlim[1] - xlim[0]
    max_bar_microns = frac_width * width_pixels * mpp
    bar_microns, label = format_rounded_length(max_bar_microns)
    bar_pixels = bar_microns / mpp
    add_scale_bar(size=bar_pixels, label=label, ax=ax)


def load_tile(reader: WSIReader, slide: Any, level: int, xy: Tuple[int, int], tile_size: int) -> np.ndarray:
    """Load a single tile from a slide.

    :param reader: The MONAI WSI reader to use.
    :param slide: The WSI object opened by the reader (e.g. OpenSlide, CuImage).
    :param level: The pyramidal resolution level at which to load the tile.
    :param xy: The coordinates of the centre of the tile, in full-resolution pixels (level 0).
    :param tile_size: The dimensions of the tile to load, in pixels at the chosen level.
    :return: The loaded tile as a channels-first RGB array.
    """
    scale = reader.get_downsample_ratio(slide, level)
    x, y = xy
    left = round(x - scale * (tile_size / 2))
    top = round(y - scale * (tile_size / 2))

    tile, _ = reader.get_data(
        slide,
        location=(top, left),
        size=(tile_size, tile_size),
        level=level,
    )
    return tile


def load_thumbnail(reader: WSIReader, slide: Any) -> np.ndarray:
    """Load a slide at the lowest available resolution.

    :param reader: The MONAI WSI reader to use.
    :param slide: The WSI object opened by the reader (e.g. OpenSlide, CuImage).
    :return: The loaded thumbnail as a channels-first RGB array.
    """
    level_count = reader.get_level_count(slide)
    thumb, _ = reader.get_data(slide, level=level_count - 1)
    return thumb


def mpp_to_power(mpp: float) -> float:
    """Map microns-per-pixel to approx. magnification power (e.g. 1mpp→10x)."""
    exponent = -np.log2(mpp)
    return 10 * np.exp2(np.round(exponent))


def _plot_tile(tile_chw: np.ndarray, ax: plt.Axes) -> AxesImage:
    tile_hwc = tile_chw.transpose(1, 2, 0)
    handle = ax.imshow(tile_hwc, interpolation="nearest")
    ax.set_axis_off()
    return handle


def plot_multi_scale_tiles(
    reader: WSIReader, slide: Any, xy: Tuple[int, int], levels: Sequence[int], tile_size: int = 224
) -> Tuple[Figure, np.ndarray]:
    """Plot tiles at multiple scales to illustrate resolution and field-of-view (FOV).

    The top row of subplots shows the same FOV at decreasing resolutions, whereas the bottom row
    shows fixed-size tiles (e.g. 224x224 pixels) with increasing FOV, following the slide's
    available resolution levels.

    :param reader: The MONAI WSI reader to use.
    :param slide: The WSI object opened by the reader (e.g. OpenSlide, CuImage).
    :param xy: The coordinates of the centre of the tile, in full-resolution pixels (level 0).
    :param levels: The sequence of resolution levels at which to plot.
    :param tile_size: The dimensions of the tiles to load, in pixels.
    :return: A tuple with the Figure and subplot Axes array.
    """
    subplot_size = 3
    n_rows = 2
    fig, axs = plt.subplots(
        n_rows,
        len(levels),
        figsize=(subplot_size * len(levels), n_rows * subplot_size),
        gridspec_kw=dict(hspace=0.02, wspace=0.02),
        squeeze=False,
    )
    assert isinstance(axs, np.ndarray)

    base_mpp, _ = reader.get_mpp(slide, level=0)  # (Y, X) resolution, assumed ~isotropic
    base_power = mpp_to_power(base_mpp)

    for level, (ax0, ax1) in zip(levels, axs.T):
        scale = reader.get_downsample_ratio(slide, level)

        # Plot fixed tile FOVs at decreasing resolutions
        scaled_tile_size = round(tile_size / scale)
        scaled_tile = load_tile(reader, slide, level=level, xy=xy, tile_size=scaled_tile_size)
        _plot_tile(scaled_tile, ax0)

        # Plot fixed-sized tiles with increasing FOV
        tile = load_tile(reader, slide, level=level, xy=xy, tile_size=tile_size)
        _plot_tile(tile, ax1)

        # Add square indicating lowest-level FOV
        if level > 0:
            rect_offset = (tile_size - scaled_tile_size) / 2
            ax1.add_artist(
                Rectangle(
                    xy=(rect_offset, rect_offset),
                    width=scaled_tile_size,
                    height=scaled_tile_size,
                    fill=False,
                    color="b",
                    lw=2,
                    ls="-",
                )
            )

        level_mpp = base_mpp * scale
        add_auto_scale_bar(level_mpp, ax0)
        add_auto_scale_bar(level_mpp, ax1)

        ax0.set_title(f"{base_power / scale:.3g}× ({base_mpp * scale:.2g}mpp)")

    return fig, axs


def plot_rgb_histogram(rgb_img: np.ndarray, ax: plt.Axes, step: int = 8) -> None:
    """Plot a simple stacked RGB intensity histogram."""
    rgb_pixels = rgb_img.reshape(3, -1).T
    ax.hist(
        rgb_pixels,
        histtype="barstacked",
        color=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # type: ignore
        bins=np.arange(-0.5, 256, step=step),
        lw=2,
        density=True,
    )
    ax.set_yticks([])
    ax.set_xlim(-0.5, 255.5)
