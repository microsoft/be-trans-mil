#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from io import StringIO
import math
import numpy as np
import pandas as pd

from cyted.utils.stain_utils import (
    find_correlation_stain_attention,
    find_entropy_attentions,
    separate_hed_stains,
    hed_channels_to_rgb,
    to_optical_density,
    from_optical_density,
    estimate_he_stain_matrix,
)


def test_separate_hed_stains() -> None:
    rgb_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    hed_image = separate_hed_stains(rgb_image.transpose(1, 2, 0))
    assert hed_image.transpose(2, 0, 1).shape == rgb_image.shape


def test_hed_channels_to_rgb() -> None:
    rgb_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    hed_image = separate_hed_stains(rgb_image.transpose(1, 2, 0))
    hed_channels = hed_channels_to_rgb(hed_image)
    assert len(hed_channels) == 3  # H, E, D channels as RGB images
    assert [(ch_image.shape) == (rgb_image.transpose(1, 2, 0).shape) for ch_image in hed_channels]


def test_optical_density() -> None:
    rgb_image = np.random.rand(3, 100, 100)
    od_image = to_optical_density(rgb_image)
    rgb_image_from_od = from_optical_density(od_image)
    assert rgb_image.all() == rgb_image_from_od.all()
    assert rgb_image_from_od.all() >= 0
    assert rgb_image_from_od.all() <= 1
    assert od_image.all() >= 0
    assert od_image.all() <= 1


def test_estimate_he_stain_matrix() -> None:
    rgb_image = np.random.randint(0, 255, size=(3, 100, 100), dtype=np.uint8)
    image = rgb_image.transpose(1, 2, 0)
    mask = image.mean(2) < 200
    bg_colour = np.median(image[~mask], axis=0)

    assert mask.shape == (rgb_image.shape[1], rgb_image.shape[2])
    assert bg_colour.shape == (3,)

    rgb_pixels = image[mask] / bg_colour
    assert rgb_pixels.shape[-1] == 3  # as required by estimate_he_stain_matrix
    rgb_from_hed, hed_from_rgb = estimate_he_stain_matrix(rgb_pixels=rgb_pixels)

    assert rgb_from_hed.shape == (3, 3)
    assert hed_from_rgb.shape == (3, 3)
    assert np.allclose(hed_from_rgb, np.linalg.inv(rgb_from_hed))
    assert np.allclose(rgb_from_hed, np.linalg.inv(hed_from_rgb))


def test_find_correlation_stain_attention() -> None:
    slide_df_str = "Unnamed: 0,prob_class0,prob_class1,pred_label,true_label,bag_attn,tile_id,top,bottom,right,left\n\
        600622,0.700684,0.299316,0,1,0.000322,y,0,224,18592,18368\n\
            600623,0.700684,0.299316,0,1,0.000237,x,0,224,18816,18592\n"
    slide_df = pd.read_csv(StringIO(slide_df_str))

    stain_df_str = "tile_id,left,right,top,bottom,brown_pixels_tiles,foreground_pixels_tiles,brown_stain_ratio_tiles\n\
        y2,9184,9296,0,112,0,8972,0.0\nx2,9296,9408,0,112,0,8452,0.0"
    stain_df = pd.read_csv(StringIO(stain_df_str))

    (
        attentions_slide_norm,
        stain_ratios_slide_norm,
        attentions_slide_unnorm_cum,
        corr_dict,
        row_to_use_for_labels,
    ) = find_correlation_stain_attention(slide_df=slide_df, stain_df=stain_df, reg_factor=2)

    # check if attentions and stain ratios are normalized and of the right length
    assert (
        len(attentions_slide_norm) == len(stain_ratios_slide_norm) == len(attentions_slide_unnorm_cum) <= len(slide_df)
    )
    assert np.logical_and(attentions_slide_norm >= 0, attentions_slide_norm <= 1).all()
    assert np.logical_and(stain_ratios_slide_norm >= 0, stain_ratios_slide_norm <= 1).all()
    assert np.logical_and(attentions_slide_unnorm_cum >= 0, attentions_slide_unnorm_cum <= 1).all()

    # check if the correlation is between -1 and 1
    assert all(-1 <= value <= 1 for value in corr_dict.values() if not math.isnan(value))

    # check if the row to use for labels is in the slide_df and has same columns as slide_df
    assert row_to_use_for_labels.isin(slide_df).all().all()


def test_find_entropy_attentions() -> None:
    attentions_slide = np.array([0.2, 0.3, 0.1, 0.4])

    expected_entropy = 0.9232196723355078

    actual_entropy = find_entropy_attentions(attentions_slide)

    # check entropy value
    assert np.allclose(actual_entropy, expected_entropy, rtol=1e-05)
    assert actual_entropy >= 0
    assert actual_entropy <= 1
