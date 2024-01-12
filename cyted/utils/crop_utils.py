#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import json
import logging
import traceback
from pathlib import Path
from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.morphology as skm
from health_cpath.preprocessing.tiff_conversion import ConvertWSIToTiffd
from health_cpath.utils.naming import SlideKey
from health_ml.utils.box_utils import Box
from openslide import OpenSlide
from PIL import Image

from cyted.cyted_schema import CytedSchema
from cyted.utils.coco_utils import CocoSchema
from cyted.utils.stain_utils import separate_hed_stains

MASK_SUFFIX = "_mask_use"


def preprocess_bounding_boxes(bounding_boxes_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess bounding boxes from COCO format to Pandas DataFrames.

    :param bounding_boxes_path: The path to the bounding boxes JSON file.
    :return: The image and bounding box DataFrames.
    """
    bounding_box_data = json.load(open(bounding_boxes_path))
    image_df = pd.DataFrame(bounding_box_data[CocoSchema.IMAGES])
    bounding_box_df = pd.DataFrame(bounding_box_data[CocoSchema.ANNOTATIONS])
    return image_df, bounding_box_df


def relative_coordinates_to_box(rel_coords: Sequence[float], width: int, height: int) -> Box:
    """Converts relative coordinates to a Box of absolute coordinates.

    :param rel_coords: The relative coordinates.
    :param width: The width of the image.
    :param height: The height of the image.
    :return: A Box of absolute coordinates.
    """
    if len(rel_coords) != 4:
        raise ValueError("Relative coordinates must be a sequence of 4 floats.")
    return Box(
        x=round(width * rel_coords[0]),
        y=round(height * rel_coords[1]),
        w=round(width * rel_coords[2]),
        h=round(height * rel_coords[3]),
    )


def sort_bboxes_left_to_right(bboxes: List[Box]) -> List[Box]:
    """
    Sort a list of boxes in left to right order.

    param bboxes: A list of bounding boxes.
    return: A sorted list of boxes.
    """
    bboxes.sort(key=lambda bb: bb.x)
    return bboxes


def get_exclusion_and_inclusion_bounding_boxes(
    image_path: Path,
    image_df: pd.DataFrame,
    bounding_box_df: pd.DataFrame,
    image_width: int,
    image_height: int,
) -> tuple[list[Box], list[Box]]:
    """Gets the exclusion and inclusion bounding boxes for the given image.

    :param image_path: The path to the image.
    :param image_df: The image DataFrame.
    :param bounding_box_df: The bounding box DataFrame.
    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :return: The exclusion and inclusion bounding boxes.
    """
    # Extract id, width, height from image_df
    image_filename = image_path.name
    image_row = image_df[image_df[CocoSchema.FILENAME].str.contains(image_filename)].squeeze()
    if not isinstance(image_row, pd.Series):
        raise ValueError(f"Image {image_filename} not found in image DataFrame.")

    image_id = image_row[CocoSchema.ID]
    # Find the bounding boxes from bounding_box_df
    image_bounding_boxes = bounding_box_df[bounding_box_df[CocoSchema.IMAGE_ID] == image_id]
    # 1: Inclusion, 2: Exclusion
    exclusion_relative_coordinates = image_bounding_boxes[image_bounding_boxes[CocoSchema.CATEGORY_ID] == 2][
        CocoSchema.BBOX
    ]
    inclusion_relative_coordinates = image_bounding_boxes[image_bounding_boxes[CocoSchema.CATEGORY_ID] == 1][
        CocoSchema.BBOX
    ]

    exclusion_bounding_boxes = get_absolute_bounding_boxes(
        exclusion_relative_coordinates, width=image_width, height=image_height
    )
    inclusion_bounding_boxes = get_absolute_bounding_boxes(
        inclusion_relative_coordinates, width=image_width, height=image_height
    )
    return exclusion_bounding_boxes, inclusion_bounding_boxes


def get_absolute_bounding_boxes(bounding_box_coordinates_list: list[list[float]], width: int, height: int) -> list[Box]:
    """For a list of relative bounding box coordinates, returns a list of absolute bounding boxes."""
    return [relative_coordinates_to_box(coords, width, height) for coords in bounding_box_coordinates_list]


def replace_exclusion_boxes_with_background(
    image_array: np.ndarray, exclusion_bounding_boxes: list[Box]
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Replaces the exclusion boxes with the median background colour.

    :param image_array: The image array.
    :param exclusion_bounding_boxes: The exclusion bounding boxes.
    :return: The image array with the exclusion boxes replaced with the median background colour,
    """
    background: np.ndarray = np.median(image_array, axis=(1, 2), keepdims=True)
    image_array = image_array.copy()
    for exclusion_bounding_box in exclusion_bounding_boxes:
        y_slice, x_slice = exclusion_bounding_box.to_slices()
        image_array[:, y_slice, x_slice] = background

    return image_array, tuple(background.astype(int).squeeze())


def crop_inclusion_boxes(image_array: np.ndarray, inclusion_bounding_boxes: list[Box]) -> list[np.ndarray]:
    """Crops the inclusion boxes from the image array.

    :param image_array: The image array.
    :param inclusion_bounding_boxes: The inclusion bounding boxes.
    :return: The cropped inclusion images.
    """
    inclusion_image_list = []

    for inclusion_bounding_box in inclusion_bounding_boxes:
        y_slice, x_slice = inclusion_bounding_box.to_slices()
        inclusion_image = image_array[:, y_slice, x_slice]
        inclusion_image_list.append(inclusion_image)

    return inclusion_image_list


def join_images(inclusion_image_list: list[np.ndarray], background: tuple[int, ...]) -> np.ndarray:
    """Joins the inclusion images into a single image.

    :param inclusion_image_list: A list of images to joined.
    :param background: The background colours per channel.
    :return: A single image with the inclusion images joined together.
    """
    heights = []
    widths = []
    for image in inclusion_image_list:
        _, h, w = image.shape
        heights.append(h)
        widths.append(w)
    joined_image = Image.new("RGB", (sum(widths), max(heights)), background)
    for i in range(len(inclusion_image_list)):
        im = np.uint8(inclusion_image_list[i]).transpose(1, 2, 0)  # type: ignore
        im2 = Image.fromarray(im)
        joined_image.paste(im=im2, box=(sum(widths[:i]), 0))
    return np.array(joined_image)


def save_whole_slide_plot(image_array: np.ndarray, image_path: Path, plot_path: Path, additional_str: str) -> None:
    """Creates a plot of the cropped and joined slide and saves it to the given path.

    :param image_array: The whole slide image array.
    :param image_path: The path of the original image.
    :param plot_path: The path to save the plot to.
    :param additional_str: An additional string to add to the plot name.
    """
    print_path = plot_path / image_path.name.split(" ")[0]
    fig, axes = plt.subplots(1, 1)
    image_array = image_array.transpose(1, 2, 0)
    axes.imshow(image_array)
    fig.tight_layout()
    output_name = f"{print_path}_{additional_str}.png"
    plt.savefig(output_name)


def resize_and_convert_mask(mask_image: Image, width: int, height: int) -> np.ndarray:
    """Resizes and converts the mask to a boolean array.

    :param mask_image: The binary (e.g. HistoQC) mask image.
    :param width: The width to resize the mask to.
    :param height: The height to resize the mask to.
    :return: A boolean array of the resized mask.
    """
    mask_image_resized = mask_image.resize(size=(width, height), resample=Image.NEAREST)  # w, h
    mask_array = np.array(mask_image_resized).astype(bool)
    return mask_array


def hardcode_background_from_mask(image_array: np.ndarray, mask_image: Image) -> np.ndarray:
    """Hardcodes the background colour from the mask.

    :param image_array: The whole slide image array.
    :param mask_image: The  mask image.
    :return: A whole slide image array with the background hardcoded.
    """
    channels, height, width = image_array.shape  # CHW format
    mask_array = resize_and_convert_mask(mask_image=mask_image, width=width, height=height)
    for i in range(channels):
        image_array[i][np.where(~mask_array)] = 255
    return image_array


def segment_foreground_quantile(x: np.ndarray, q: float = 0.8, r1: int = 5, r2: int = 5) -> np.ndarray:
    """
    Segment an image using a mask from a quantile threshold, followed by morphological closing and opening.

    :param x: Numpy array image to segment in HW format (this can be the Hematoxylin channel of HED image).
    :param q: Quantile to use for threshold (default=0.8).
    :param r1: Radius of the morphological structuring element for closing (default=5).
    :param r2: Radius of the morphological structuring element for opening (default=5).
    :return: Mask boolean array in HW format.
    """
    mask = x > np.quantile(x, q)
    mask = skm.binary_closing(mask, skm.disk(r1))
    mask = skm.binary_opening(mask, skm.disk(r2))
    return mask


def segment_and_post_process(
    crops: List[np.ndarray], bg_colour: tuple[int, ...], se1_size: int = 10, se2_size: int = 10
) -> List[np.ndarray]:
    """
    Segment a list of images using the hematoxyline channel and post-process the segmentation using morphological
    opening and closing operations.

    :param crops: A list of images (crops) as numpy arrays in C, H, W formats.
    :param bg_colour: RGB background of the image as a tuple.
    :param se1_size: Size of the structuring element to use for morphological closing operation (default=10).
    :param se2_size: Size of the structuring element to use for morphological opening operation (default=10).
    :return: A list of segmented mask image arrays in C, H, W format.
    """
    crops_mask = []
    for i in range(len(crops)):
        image_crop = crops[i].transpose(1, 2, 0)
        tff3_hed = separate_hed_stains(image=image_crop, bg_colour=bg_colour)
        mask_crop = segment_foreground_quantile(x=tff3_hed[..., 0], r1=se1_size, r2=se2_size)
        mask_array = mask_crop.astype(np.uint8) * 255
        mask_array_rgb = np.zeros((3, mask_array.shape[0], mask_array.shape[1]))
        mask_array_rgb[0:3] = mask_array
        crops_mask.append(mask_array_rgb)
    return crops_mask


def resize_image_array_to_scale_factor(image_array: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Resize image array given the source and scale factor.

    :param image_array: Numpy array of the image in C, H, W format.
    :param scale_factor: Scale factor to resize the image by.
    :return: Resized image array in C, H, W format.
    """
    resize_size = (int(image_array.shape[2] * scale_factor), int(image_array.shape[1] * scale_factor))
    image_pil = Image.fromarray(image_array.transpose(1, 2, 0))  # HWC format
    image_pil_r = image_pil.resize(size=resize_size)
    return np.asarray(image_pil_r)


def resize_image_array(image_array: np.ndarray, source_magnification: float, target_magnification: float) -> np.ndarray:
    """
    Resize image array given the source and target magnifications.

    :param image_array: Numpy array of the image in C, H, W format.
    :source_magnification: Magnification of the source image array (e.g. 40. for 40x).
    :target_magnification: Magnification of the target image array (e.g. 1.25 for 1.25x).
    :return: Resized image array in C, H, W format.
    """
    factor = target_magnification / source_magnification
    return resize_image_array_to_scale_factor(image_array=image_array, scale_factor=factor).transpose(2, 0, 1)


def extract_inclusion_replace_exclusion_bboxes(
    image_array: np.ndarray,
    image_path: Path,
    bounding_box_df: pd.DataFrame,
    image_df: pd.DataFrame,
    plot_path: Optional[Path] = None,
    input_type: Optional[str] = "",
) -> tuple[List[Box], np.ndarray, tuple[int, ...]]:
    """
    Extract inclusion bounding boces and replace excusion bounding boxes with background.

    :param image_array: The whole slide image array.
    :param image_path: The path to the image.
    :param image_df: The image DataFrame.
    :param bounding_box_df: The bounding box DataFrame.
    :param plot_path: The path to save the plot to (default=None).
    :param input_type: Type of input to use as additional string to name the saved plots, e.g.full/resized (default="").
    :return A tuple containing the list of inclusion bounding boxes, image array after replacing exclusion
    bounding boxes with background, and background value.
    """

    if plot_path:
        save_whole_slide_plot(
            image_array=image_array,
            image_path=image_path,
            plot_path=plot_path,
            additional_str=f"original_slide_{input_type}",
        )

    # Extract exclusion and inclusion bounding boxes and replace exclusion boxes with background
    _, height, width = image_array.shape  # CHW format
    exclusion_bboxes, inclusion_bboxes = get_exclusion_and_inclusion_bounding_boxes(
        image_path,
        image_df,
        bounding_box_df,
        image_width=width,
        image_height=height,
    )

    # Sort inclusion bounding boxes
    inclusion_bboxes = sort_bboxes_left_to_right(inclusion_bboxes)

    # Replace exclusion bounding boxes with background
    image_array, background = replace_exclusion_boxes_with_background(image_array, exclusion_bboxes)
    if plot_path:
        save_whole_slide_plot(image_array, image_path, plot_path, f"after_exclusion_{input_type}")

    return (inclusion_bboxes, image_array, background)


def create_new_slide_from_given_mask(
    image_array: np.ndarray,
    image_path: Path,
    inclusion_bboxes: List[Box],
    mask_image: Image = None,
    plot_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Function to create a slide given the foreground mask image and inclusion bounding boxes.

    :param image_array: The whole slide image array.
    :param mask_image: Mask image of foreground mask to use for background hardcoding.
    :param inclusion_bboxes: The inclusion bounding boxes to use.
    :param plot_path: The path to save the plot to (default=None).
    :return The resulting image array in H, W, C format.
    """
    # Hardcode background using masks
    image_array = hardcode_background_from_mask(image_array=image_array, mask_image=mask_image)
    if plot_path:
        save_whole_slide_plot(image_array, image_path, plot_path, "after_background_hardcoding_slide")

    # Get crops from inclusion bounding boxes on hardcoded image
    inclusion_image_list = crop_inclusion_boxes(image_array, inclusion_bboxes)

    # Join the inclusion hardcoded image crops together in white background
    _background = (255, 255, 255)
    joined_image = join_images(inclusion_image_list, _background)
    if plot_path:
        save_whole_slide_plot(joined_image.transpose(2, 0, 1), image_path, plot_path, "hardcoded_cropped_slide")
    return joined_image


def create_new_slide_by_generating_mask(
    image_array: np.ndarray,
    image_path: Path,
    inclusion_bboxes: List[Box],
    image_df: pd.DataFrame,
    bounding_box_df: pd.DataFrame,
    background: tuple[int, ...],
    plot_path: Optional[Path] = None,
    output_masks_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Function to create a slide and foreground mask image given inclusion bounding boxes.

    :param image_array: The whole slide image array.
    :param inclusion_bboxes: The inclusion bounding boxes to use.
    :param image_df: The image DataFrame.
    :param bounding_box_df: The bounding box DataFrame.
    :param background: A tuple of the original image background values.
    :param plot_path: The path to save the plot to (default=None).
    :param output_masks_path: Path where the output masks will be saved (default=None).
    :return The resulting image array in H, W, C format.
    """
    # Get crops from inclusion bounding boxes on original image
    inclusion_image_list = crop_inclusion_boxes(image_array, inclusion_bboxes)

    # Join the inclusion image crops together using original background
    joined_image = join_images(inclusion_image_list, background)
    if plot_path:
        save_whole_slide_plot(joined_image.transpose(2, 0, 1), image_path, plot_path, "cropped_slide")

    # Resize original image to 1.25x (currently hardcoded for source magnification 10x)
    image_array_r = resize_image_array(image_array=image_array, source_magnification=10.0, target_magnification=1.25)

    # Get inclusion boxes of resized image
    inclusion_bboxes_r, image_array_r, background_r = extract_inclusion_replace_exclusion_bboxes(
        image_array=image_array_r,
        image_path=image_path,
        bounding_box_df=bounding_box_df,
        image_df=image_df,
        plot_path=plot_path,
        input_type="resize",
    )
    inclusion_image_list_r = crop_inclusion_boxes(image_array_r, inclusion_bboxes_r)

    # Get segmentation masks from resized image crops
    crops_masks = segment_and_post_process(crops=inclusion_image_list_r, bg_colour=background_r)

    # Join mask sections to get single mask
    masks_result = join_images(crops_masks, background=(0, 0, 0))
    mask_image = Image.fromarray(masks_result[:, :, 0], mode="L")
    if output_masks_path:
        mask_image.save(f"{output_masks_path}/{image_path.name}{MASK_SUFFIX}.png")
    if plot_path:
        save_whole_slide_plot(masks_result.transpose(2, 0, 1), image_path, plot_path, "cropped_mask")

    # Hardcode background of cropped image using cropped masks
    out_image = hardcode_background_from_mask(image_array=joined_image.transpose(2, 0, 1), mask_image=mask_image)
    if plot_path:
        save_whole_slide_plot(out_image, image_path, plot_path, "after_background_hardcoding_cropped_slide")
    out_image = out_image.transpose(1, 2, 0)

    return out_image


def merge_bounding_boxes_into_new_slide(
    image_array: np.ndarray,
    image_path: Path,
    image_df: pd.DataFrame,
    bounding_box_df: pd.DataFrame,
    mask_image: Image = None,
    output_masks_path: Optional[Path] = None,
    plot_path: Optional[Path] = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Merges the bounding boxes into a new slide.

    :param image_array: The original whole slide image array.
    :param image_path: The path of the original image.
    :param image_df: The image dataframe.
    :param bounding_box_df: The bounding box dataframe.
    :param mask_image: An optional mask image.
    :param masks_path: Directory path of mask image.
    :param output_masks_path: Directory path to save output masks.
    :param plot_path: An optional path to save plots to.
    """

    # Get a list of inclusion boxes, image after replacing exclusion boxes, and original image background
    inclusion_bboxes, image_array, background = extract_inclusion_replace_exclusion_bboxes(
        image_array=image_array,
        image_path=image_path,
        bounding_box_df=bounding_box_df,
        image_df=image_df,
        plot_path=plot_path,
        input_type="full",
    )

    if mask_image is not None:
        # If mask image is provided corresponding to original image, use it for background hardcoding
        out_image = create_new_slide_from_given_mask(
            image_array=image_array,
            image_path=image_path,
            mask_image=mask_image,
            inclusion_bboxes=inclusion_bboxes,
            plot_path=plot_path,
        )
    else:
        # If mask image is not available, create masks on the fly from resized crops
        out_image = create_new_slide_by_generating_mask(
            image_array=image_array,
            image_path=image_path,
            image_df=image_df,
            bounding_box_df=bounding_box_df,
            background=background,
            inclusion_bboxes=inclusion_bboxes,
            plot_path=plot_path,
            output_masks_path=output_masks_path,
        )

    return out_image, background


class CropAndConvertWSIToTiffd(ConvertWSIToTiffd):
    """A class to crop and convert whole slide images to tiff files."""

    def __init__(
        self,
        bounding_box_path: Path,
        masks_path: Optional[Path] = None,
        plot_path: Optional[Path] = None,
        output_masks_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param bounding_box_path: The path to the bounding box csv file.
        :param masks_path: An optional directory containing binary masks for foreground/background hardcoding.
        :param plot_path: An optional path to save intermediate plots for debugging.
        :param output_masks_path: An optional directory where output masks can be stored (if not already provided).
        """
        super().__init__(**kwargs)
        self.bounding_box_path = bounding_box_path
        self.image_df, self.bounding_box_df = preprocess_bounding_boxes(self.bounding_box_path)
        self.plot_path = plot_path
        self.masks_path = masks_path
        self.output_masks_path = output_masks_path
        self.background_value: tuple[int, ...] = (255, 255, 255)

    def get_foreground_mask_array(self, src_path: Path) -> Optional[Image.Image]:
        """Gets the foreground mask array for the specified image. If no mask is found, returns None.

        :param src_path: The src path of the image.
        :return: The foreground mask array or None.
        """
        if self.masks_path is not None:
            ndpi_filename = src_path.name
            mask_path = f"{self.masks_path}/{ndpi_filename}/{ndpi_filename}{MASK_SUFFIX}.png"
            try:
                mask_image = Image.open(mask_path)
            except FileNotFoundError:
                logging.error(
                    f"Foreground mask was not found in the specified location {mask_path}, please check "
                    "`--image_column` and `--masks_path`."
                )
                mask_image = None
        else:
            mask_image = None
        return mask_image

    def get_level_data(self, wsi_obj: OpenSlide, level: int) -> np.ndarray:
        """Gets the level data for the specified level and crops and merges the bounding boxes.
        When cropping the bounding boxes, the background is hardcoded to white if a foreground mask is provided.
        Otherwise, foreground masks will be generated on the fly and background will be hardcoded from the masks.

        :param wsi_obj: The whole slide image object.
        :param level: The level to get the data for.
        :return: A cropped and merged image array.
        """
        image_array, _ = self.wsi_reader.get_data(wsi_obj, level=level)
        # This is essential so that we can later overwrite the values outside the bounding boxes.
        # If not copied, the image_array will not be the owner of the data and hence can't write.
        image_array = image_array.copy()
        src_path = Path(wsi_obj._filename)
        cropped_slide, background_value = merge_bounding_boxes_into_new_slide(
            image_array=image_array,
            image_path=src_path,
            image_df=self.image_df,
            bounding_box_df=self.bounding_box_df,
            plot_path=self.plot_path,
            mask_image=self.get_foreground_mask_array(src_path),
            output_masks_path=self.output_masks_path,
        )
        # Store background values so we can save it later to csv
        self.background_value = background_value
        return cropped_slide

    def __call__(self, data: dict) -> dict:
        try:
            super().__call__(data)
            # Only add background for converted images
            tiff_path = self.get_tiff_path(Path(data[self.image_key]))
            if tiff_path.exists():
                data[CytedSchema.Background] = self.background_value
        except Exception as ex:
            logging.warning(
                f"Error while converting slide {data[SlideKey.IMAGE]} due to {ex}:\n{traceback.format_exc()}"
            )
        return data
