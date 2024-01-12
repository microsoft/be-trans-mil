#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
import os
import pickle
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage.color as skc
from health_cpath.utils.naming import SlideKey
from openslide import OpenSlide
from PIL import Image, ImageChops

from cyted.cyted_schema import CytedLabel, CytedSchema
from cyted.preproc.slide_registration import SlideRegistrationPipeline
from cyted.utils.crop_utils import (
    CropAndConvertWSIToTiffd,
    crop_inclusion_boxes,
    extract_inclusion_replace_exclusion_bboxes,
    hardcode_background_from_mask,
    join_images,
    preprocess_bounding_boxes,
    resize_image_array,
    segment_and_post_process,
)
from cyted.utils.stain_utils import hed_channels_to_rgb, separate_hed_stains

MAGNIFICATION_KEY = "magnification"


class RegistrationMetrics(str, Enum):
    MI_BEFORE = "MI_before_registration"  # MI: Mutual information
    MI_AFTER = "MI_after_registration"


def get_background_colour(image: np.ndarray, q: float = 0.5) -> np.ndarray:
    """
    Get the background of an image (median intensity of foreground pixels).

    :param image: Image array to compute the background in HWC format.
    :param q: Quantile to use for background (default=0.5).
    :return The RGB background as a numpy array.
    """
    return np.quantile(image, q=q, axis=(0, 1))  # type: ignore


def add_grid(ax: plt.Axes, step: float) -> None:
    """
    Add gridlines to matplotlib plots.

    :param ax: Axes of the figure to add gridlines to.
    :param step: Step size of the gridlines.
    """
    ax.set(
        xticks=np.arange(0, max(ax.get_xlim()), step),
        xticklabels=[],
        yticks=np.arange(0, max(ax.get_ylim()), step),
        yticklabels=[],
    )
    ax.grid(c="0.5")


def plot_image(
    array_or_image: Union[np.ndarray, sitk.Image], ax: plt.Axes, grid_step: float = 0, title: str = "", **kwargs: Any
) -> None:
    """
    Plot a numpy array or sitk image in matplotlib.

    :param array_or_image: The numpy array or sitk image to plot.
    :param ax: Axes of the figure to plot.
    :param grid_step: Step size of gridlines (default=0).
    :param title: Title of the plot (default="").
    :param kwargs: Arguments of imshow().
    """
    if isinstance(array_or_image, sitk.Image):
        array = sitk.GetArrayViewFromImage(array_or_image)
    else:
        array = array_or_image
    default_kwargs = dict(cmap="turbo")
    default_kwargs.update(kwargs)
    ax.imshow(array, **default_kwargs)
    if grid_step > 0:
        add_grid(ax, step=grid_step)
    ax.set_title(title)


def plot_hed_channels(
    image: np.ndarray,
    bg_colour: Optional[np.ndarray] = None,
    grid_step: float = 100,
    rgb_from_hed: np.ndarray = skc.rgb_from_hed,
) -> None:
    """
    Given a histological image, plot its hematoxylin, eosin and DAB channels.

    :param image: Histological image as numpy array.
    :param bg_color: RGB background of the image as a numpy array.
    :param grid_step: Step size of gridlines (default=100).
    :param rgb_from_hed: RGB-HED stain conversion matrix (default=scikit image conversion matrix).
    """
    hed_from_rgb = np.linalg.inv(rgb_from_hed)

    hed = separate_hed_stains(image, bg_colour, hed_from_rgb=hed_from_rgb)
    hed_rgb = hed_channels_to_rgb(hed, rgb_from_hed=rgb_from_hed)

    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    plot_image(image, axs[0], grid_step=grid_step, title="Original")
    titles = ["Haematoxylin", "Eosin", "DAB"]
    for c in range(3):
        plot_image(hed_rgb[c], axs[c + 1], grid_step=grid_step, title=titles[c])
    fig.tight_layout()


def plot_mask_contour(array_or_image: Union[np.ndarray, sitk.Image], ax: plt.Axes, **kwargs: Any) -> None:
    """
    Plot the contour of a numpy array or sitk image in matplotlib.

    :param array_or_image: The numpy array or sitk image to plot.
    :param ax: Axes of the figure to plot.
    :param kwargs: Additional arguments to give to contour().
    """
    if isinstance(array_or_image, sitk.Image):
        array = sitk.GetArrayViewFromImage(array_or_image)
    else:
        array = array_or_image
    ax.contour(array, levels=[0.5], **kwargs)


def format_euler_params(params: tuple[float, float, float]) -> str:
    """
    Format the radians and degrees parameters.
    """
    return f"({np.rad2deg(params[0]):+.1f}°, {params[1]:+.1f}x, {params[2]:+.1f}y)"


# Declare opt_status as Any and import later to avoid circular imports
def plot_optimiser_trace(opt_status: Any, ax: plt.Axes) -> None:
    """
    Plot the optimizer trace during registration.

    :param opt_status: Optimizer status.
    :param ax: Axes of the figure to plot to.
    """
    from cyted.preproc.slide_registration import OptimiserStatus

    assert isinstance(opt_status, OptimiserStatus)

    (line,) = ax.plot(opt_status.metric_values)
    multires_metric_values = [opt_status.metric_values[index] for index in opt_status.multires_iterations]
    ax.plot(opt_status.multires_iterations, multires_metric_values, "o", c=line.get_color())


def get_cropped_images(
    image_path: Path,
    image_array: np.ndarray,
    image_df: pd.DataFrame,
    bb_df: pd.DataFrame,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Function to get crops for input image.

    :param image_path: Path of the image.
    :param image_array: The image array.
    :param image_df: Image dataframe for the image.
    :param bb_df: Bounding box dataframe for the image..
    :param mask_image_he: Optional mask image for H&E.
    :return A tuple containing crops and background array.
    """
    # Get crops
    in_bboxes, image_array, background = extract_inclusion_replace_exclusion_bboxes(
        image_array=image_array, image_path=image_path, bounding_box_df=bb_df, image_df=image_df
    )
    crops = crop_inclusion_boxes(image_array, in_bboxes)

    bg_color = np.array(background, dtype=float)

    return (crops, bg_color)


def register_crops(
    he_crops: list[np.ndarray],
    tff3_crops: list[np.ndarray],
    he_mask_crops_resized: list[np.ndarray],
    he_bg_color: np.ndarray,
    tff3_bg_color: np.ndarray,
    tff3_image_path: Path,
    input_transforms_path: Optional[Path] = None,
    target_magnification: Optional[float] = None,
) -> tuple[list[np.ndarray], list[sitk.Transform], tuple[int, ...], float, float]:
    """
    Function to register crops of H&E and TFF3 image.

    :param he_crops: A list of H&E crops.
    :param tff3_crops: A list of TFF3 crops.
    :param he_mask_crops_resized: A list of resized H&E mask crops.
    :param he_bg_color: H&E background color.
    :param tff3_bg_color: TFF3 background color.
    :param tff3_image_path: Path of the TFF3 image.
    :param input_transforms_path: Optional path to saved transform.
    :return A tuple containing registered TFF3 crops, SimpleITK transforms, TFF3 background tuple,
    initial metric value and final metric value for registration.
    """
    tff3_crops_registered = []
    transform_crops = []
    total_initial_metric_val = 0.0
    total_final_metric_val = 0.0

    # Load registration transforms if they already exist
    if input_transforms_path:
        if target_magnification is None:
            raise ValueError("Target magnification must be specified if input transforms are provided.")
        input_transforms_files = os.listdir(input_transforms_path)
        input_transform_file = [x for x in input_transforms_files if tff3_image_path.name in x]
        with open(input_transforms_path / input_transform_file[0], "rb") as handle:
            input_transforms_dict = pickle.load(handle)
        total_initial_metric_val = input_transforms_dict[RegistrationMetrics.MI_BEFORE]
        total_final_metric_val = input_transforms_dict[RegistrationMetrics.MI_AFTER]

        # If the transforms were originally computed at a different magnification,
        # we need to rescale the pixel spacing to match the original images.
        original_magnification: float = input_transforms_dict[MAGNIFICATION_KEY]
        rescaling_factor = original_magnification / target_magnification
        rescaled_spacing = (rescaling_factor, rescaling_factor)

    for i in range(len(he_crops)):
        moving_image = sitk.GetImageFromArray(tff3_crops[i].transpose(1, 2, 0), isVector=True)
        fixed_image = sitk.GetImageFromArray(he_crops[i].transpose(1, 2, 0), isVector=True)
        if len(he_mask_crops_resized) > 0:
            fixed_mask = he_mask_crops_resized[i][0].astype(bool)
        else:
            fixed_mask = None
        if input_transforms_path is None:
            register = SlideRegistrationPipeline(
                num_attempts=1,
                num_angles=16,
                num_max_iterations=100,
                fixed_background=he_bg_color,
                moving_background=tff3_bg_color,
            )
        else:
            moving_image.SetSpacing(rescaled_spacing)
            fixed_image.SetSpacing(rescaled_spacing)
            input_transform: sitk.Transform = input_transforms_dict[tff3_image_path.name][i]
            register = SlideRegistrationPipeline(
                fixed_background=he_bg_color, moving_background=tff3_bg_color, input_transform=input_transform
            )
        # H&E mask used for estimating stain matrix if available
        transform, registered_image, initial_metric_crop, final_metric_crop = register(
            moving_image, fixed_image, fixed_mask
        )
        transform_crops.append(transform)
        tff3_crops_registered.append(sitk.GetArrayFromImage(registered_image).transpose(2, 0, 1))
        total_initial_metric_val += initial_metric_crop
        total_final_metric_val += final_metric_crop

    background_tff3 = tuple(tff3_bg_color.astype(int).squeeze())
    return (tff3_crops_registered, transform_crops, background_tff3, total_initial_metric_val, total_final_metric_val)


def join_hardcode_background(
    tff3_crops_registered: list[np.ndarray],
    background_tff3: tuple[int, ...],
    target_magnification: float,
    hardcode_background: Optional[bool] = True,
) -> tuple[np.ndarray, Image.Image]:
    """
    Function to join the crops and hardcode the background (optional), and compute the TFF3 mask image
    to plot difference.

    :param tff3_crops_registered: A list of registered TFF3 crops.
    :param background_tff3: Tuple of TFF3 background color.
    :param target_magnification: Target magnification required.
    :param hardcode_background: Optional flag to hardcode background (default=True).
    :return A tuple containing the TFF3 output image and TFF3 mask image.
    """
    # Join registered TFF3 crops
    joined_image = join_images(tff3_crops_registered, background=background_tff3)

    # TFF3 background hardcoding
    # Resize registered TFF3 crops to 1.25x (since TFF3 masks are computed at this magnification).
    tff3_crops_registered_r = [resize_image_array(crop, target_magnification, 1.25) for crop in tff3_crops_registered]
    tff3_crops_masks = segment_and_post_process(tff3_crops_registered_r, bg_colour=background_tff3)
    # Join mask sections to get single mask image
    tff3_masks_result = resize_image_array(
        join_images(tff3_crops_masks, background=(0, 0, 0)).transpose(2, 0, 1),
        source_magnification=1.25,
        target_magnification=target_magnification,
    ).transpose(1, 2, 0)
    tff3_mask_image = Image.fromarray(tff3_masks_result[:, :, 0], mode="L")
    # Hardcode background to white from TFF3 mask if hardcode_background is true
    if hardcode_background:
        out_image = hardcode_background_from_mask(
            image_array=joined_image.transpose(2, 0, 1), mask_image=tff3_mask_image
        )
        out_image = out_image.transpose(1, 2, 0)
    else:
        out_image = joined_image

    return (out_image, tff3_mask_image)


def save_registration_results(
    tff3_image_path: Path,
    transform_dict: Dict[str, Any],
    tff3_mask_image: Image.Image,
    he_mask_crops_resized: list[np.ndarray],
    output_diff_path: Optional[Path] = None,
    output_transforms_path: Optional[Path] = None,
) -> None:
    """
    Function to save the registration results.

    :param tff3_image_path: Path of the TFF3 image.
    :param tff3_mask_image: TFF3 mask image.
    :param transform_crops: A list of SimpleITK transforms.
    :param he_mask_crops_resized: A list of resized H&E mask crops.
    :param target_magnification: Target magnification required.
    :param output_diff_path: Optional path to store output difference image.
    :param output_transforms_path: Optional path to store output transform.
    """

    # Save registration transforms
    if output_transforms_path:
        transforms_path = output_transforms_path / f"{tff3_image_path.name}_tfm.pickle"
        with open(transforms_path, "wb") as handle:
            pickle.dump(transform_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save difference image
    if output_diff_path and len(he_mask_crops_resized) > 0:
        he_masks_result = join_images(he_mask_crops_resized, background=(0, 0, 0))
        he_mask_image = Image.fromarray(he_masks_result[:, :, 0], mode="L")
        diff_image = ImageChops.difference(tff3_mask_image, he_mask_image)
        diff_path = output_diff_path / f"{tff3_image_path.name}_diff.jpg"
        fig, axes = plt.subplots(1, 1)
        axes.imshow(diff_image, cmap="turbo")
        plt.axis("off")
        fig.tight_layout()
        plt.savefig(diff_path)
        plt.close()


def find_reference_image_and_label_from_csv(
    dataset_csv_path: Path, src_id: str, image_column: str, label_column: str
) -> tuple[str, Any]:
    """
    Function to find the reference image and given label from dataset.csv file.

    :param dataset_csv_path: Path to the dataset CSV file.
    :param src_id: String containing the ID of the image.
    :param image_column: Column of the image path in the dataset CSV.
    :param label_column: Column of label in the dataset CSV.
    :return Tuple containing path of reference image and label.
    """
    if str(dataset_csv_path).endswith(".tsv"):
        dataset_df = pd.read_csv(dataset_csv_path, sep="\t")
    else:
        dataset_df = pd.read_csv(dataset_csv_path)
    dataset_df = dataset_df.reset_index()
    selected_row = dataset_df.loc[dataset_df[CytedSchema.CytedID] == src_id]
    src_he_path = selected_row[image_column].values[0]
    src_label = selected_row[label_column].values[0]
    return (src_he_path, src_label)


def find_reference_he_images_from_folder(fixed_dataset_path: Path, src_id: str) -> List[str]:
    """
    Function to find reference image from a folder, given the image ID.

    :param  fixed_dataset_path: Path of the fixed dataset.
    :param src_id: String containing the ID of the image.
    :return A list of paths matching the image ID from the fixed dataset.
    """
    src_he_paths = os.listdir(fixed_dataset_path)
    src_he_paths_sel = [x for x in src_he_paths if (src_id in x) and (("HE" in x) or ("H&E" in x))]
    return src_he_paths_sel


class RegisterCropConvertWSIToTiffd(CropAndConvertWSIToTiffd):
    """A class to register, crop and convert TFF3 images to H&E images."""

    def __init__(
        self,
        fixed_dataset_path: Path,
        bb_path_fixed: Path,
        label_column: str,
        fixed_masks_path: Optional[Path] = None,
        input_transforms_path: Optional[Path] = None,
        output_diff_path: Optional[Path] = None,
        output_transforms_path: Optional[Path] = None,
        hardcode_background: Optional[bool] = True,
        dataset_csv: Optional[str] = None,
        preprocessed_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param fixed_dataset_path: The path to the fixed (H&E) dataset images.
        :param bb_path_fixed: The path to the bounding box csv file of fixed (H&E) images.
        :param label_column: The label to be used, if dataset csv is provided, only register positively labelled slides.
        :param fixed_masks_path: An optional directory containing H&E binary masks for estimating stain matrix.
        :param input_transforms_path: An optional directory containing registration transforms.
        :param output_diff_path: An optional path to save difference in the registered image and fixed image.
        :param output_transforms_path: An optional directory where output transforms can be stored.
        :param hardcode_background: An optional flag to set/reset background hardcoding (default=True).
        :param dataset_csv: An optional name of the dataset csv file if not using the default name.
        :param preprocessed_path: An optional directory containing preprocessed moving (TFF3) images.
        """
        super().__init__(**kwargs)
        self.fixed_dataset_path = fixed_dataset_path
        self.bb_path_fixed = bb_path_fixed
        self.image_df_he, self.bounding_box_df_he = preprocess_bounding_boxes(self.bb_path_fixed)
        self.image_df_tff3, self.bounding_box_df_tff3 = preprocess_bounding_boxes(self.bounding_box_path)
        self.input_transforms_path = input_transforms_path
        self.output_diff_path = output_diff_path
        self.output_transforms_path = output_transforms_path
        self.masks_path = fixed_masks_path
        self.hardcode_background = hardcode_background
        self.dataset_csv = dataset_csv
        self.preprocessed_moving_path = preprocessed_path
        self.label_column = label_column
        self.initial_metric_val = 0.0
        self.final_metric_val = 0.0

    def get_level_data(self, wsi_obj: OpenSlide, level: int) -> np.ndarray:
        """Gets the level data for the specified level for H&E and TFF3 images.
        When merging cropped and registered bounding boxes, the background is hardcoded to white.

        :param wsi_obj: The whole slide image object of fixed (H&E) image.
        :param level: The level for registering H&E and TFF3 images.
        :return: A registered, cropped and merged TFF3 image array.
        """
        src_tff3_path = Path(wsi_obj._filename)
        src_id = Path(src_tff3_path).name.split(" ")[0]
        src_label = None

        # Find corresponding H&E slide for the given TFF3 slide
        if self.dataset_csv:
            dataset_csv_path = src_tff3_path.parent / self.dataset_csv
            src_he_path, src_label = find_reference_image_and_label_from_csv(
                dataset_csv_path=dataset_csv_path,
                src_id=src_id,
                image_column=CytedSchema.HEImage,
                label_column=self.label_column,
            )
        else:
            src_he_paths_sel = find_reference_he_images_from_folder(
                fixed_dataset_path=self.fixed_dataset_path, src_id=src_id
            )
            if len(src_he_paths_sel) < 1:
                logging.warning(f"H&E reference image not found for TFF3 image {src_tff3_path.name}.")
                return
            if len(src_he_paths_sel) > 1:
                logging.warning(
                    f"Multiple H&E matches {src_he_paths_sel} for TFF3 image {src_tff3_path.name}."
                    "Selecting the first."
                )
            src_he_path = src_he_paths_sel[0]

        image_array_tff3, _ = self.wsi_reader.get_data(wsi_obj, level=level)
        # This is essential so that we can later overwrite the values outside the bounding boxes.
        # If not copied, the image_array will not be the owner of the data and hence can't write.
        image_array_tff3 = image_array_tff3.copy()

        wsi_obj_he = self.wsi_reader.read(self.fixed_dataset_path / src_he_path)
        image_array_he, _ = self.wsi_reader.get_data(wsi_obj_he, level=level)
        image_array_he = image_array_he.copy()

        # If the preprocessed images are available, get these (we will copy negatives to output without registration)
        if self.preprocessed_moving_path:
            wsi_obj_cropped_tff3 = self.wsi_reader.read(
                self.preprocessed_moving_path / src_tff3_path.name.replace("ndpi", "tiff")
            )
            image_array_cropped_tff3, _ = self.wsi_reader.get_data(wsi_obj_cropped_tff3, level=0)
            image_array_cropped_tff3 = image_array_cropped_tff3.copy()

        # Get the mask image for the reference H&E image, if available
        mask_image_he = (self.get_foreground_mask_array(Path(src_he_path)),)

        # Get target magnification from level
        target_magnification: float = self._get_base_objective_power(wsi_obj) / (2**level)

        # Get TFF3 slide label from dataset.csv and register only if slide is positive
        # else copy the TFF3 image from preprocessed TFF3 dataset
        if self.preprocessed_moving_path and src_label == CytedLabel.No:
            logging.info(f"Copying TFF3 negative image {src_tff3_path} without registration.")
            cropped_slide = image_array_cropped_tff3.transpose(1, 2, 0)
        else:
            logging.info(f"Registering TFF3 image {src_tff3_path} to H&E image {src_he_path}.")

            # Get crops for fixed and moving images, and crops of resized masks
            he_crops, he_bg_color = get_cropped_images(
                image_path=Path(src_he_path),
                image_array=image_array_he,
                image_df=self.image_df_he,
                bb_df=self.bounding_box_df_he,
            )

            tff3_crops, tff3_bg_color = get_cropped_images(
                image_path=Path(src_tff3_path),
                image_array=image_array_tff3,
                image_df=self.image_df_tff3,
                bb_df=self.bounding_box_df_tff3,
            )

            if mask_image_he is not None:
                he_mask_array = np.array(mask_image_he[0].convert("RGB")).transpose(2, 0, 1)  # type: ignore
                he_mask_crops, _ = get_cropped_images(
                    image_path=Path(src_he_path),
                    image_array=he_mask_array,
                    image_df=self.image_df_he,
                    bb_df=self.bounding_box_df_he,
                )
            else:
                he_mask_crops = None

            # Resize mask H&E image from 1.25x (since masks are available at 1.25x) to target magnification
            he_mask_crops_resized = []
            if he_mask_crops is not None:
                for i in range(len(he_crops)):
                    he_mask_crop_r = resize_image_array(
                        he_mask_crops[i], source_magnification=1.25, target_magnification=target_magnification
                    )

                    he_mask_crops_resized.append(he_mask_crop_r)

            # Register the crops
            (
                tff3_crops_registered,
                transform_crops,
                background_tff3,
                initial_metric_val,
                final_metric_val,
            ) = register_crops(
                he_crops=he_crops,
                tff3_crops=tff3_crops,
                he_mask_crops_resized=he_mask_crops_resized,
                he_bg_color=he_bg_color,
                tff3_bg_color=tff3_bg_color,
                tff3_image_path=Path(src_tff3_path),
                input_transforms_path=self.input_transforms_path,
                target_magnification=target_magnification,
            )

            # Join crops and hardcode background to white
            cropped_slide, tff3_mask_image = join_hardcode_background(
                tff3_crops_registered=tff3_crops_registered,
                background_tff3=background_tff3,
                target_magnification=target_magnification,
                hardcode_background=self.hardcode_background,
            )

            # Create transforms dictionary
            transform_dict = {
                Path(src_tff3_path).name: transform_crops,
                MAGNIFICATION_KEY: target_magnification,
                RegistrationMetrics.MI_BEFORE: initial_metric_val,
                RegistrationMetrics.MI_AFTER: final_metric_val,
            }

            # Save the resulting difference images, transforms
            save_registration_results(
                tff3_image_path=Path(src_tff3_path),
                transform_dict=transform_dict,
                tff3_mask_image=tff3_mask_image,
                he_mask_crops_resized=he_mask_crops_resized,
                output_diff_path=self.output_diff_path,
                output_transforms_path=self.output_transforms_path,
            )

            # Store original background values and metric values so we can save it later to csv
            self.background_value = background_tff3
            self.initial_metric_val = initial_metric_val
            self.final_metric_val = final_metric_val

        return cropped_slide

    def __call__(self, data: dict) -> dict:
        try:
            super().__call__(data)
            # Only add metric values for converted images
            tiff_path = self.get_tiff_path(Path(data[self.image_key]))
            if tiff_path.exists():
                data[RegistrationMetrics.MI_BEFORE] = self.initial_metric_val
                data[RegistrationMetrics.MI_AFTER] = self.final_metric_val
        except Exception as ex:
            logging.warning(
                f"Error while converting slide {data[SlideKey.IMAGE]} due to {ex}:\n{traceback.format_exc()}"
            )
        return data
