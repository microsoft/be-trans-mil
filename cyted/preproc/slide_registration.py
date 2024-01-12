#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import SimpleITK as sitk

from cyted.utils.stain_utils import estimate_he_stain_matrix, separate_hed_stains


def rotation_matrix(angle: float) -> np.ndarray:
    """
    Get the rotation matrix given the angle.

    :param angle: Angle to get rotation matrix of.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


class OptimiserStatus:
    """
    Get optimizer status for registration iterations.
    """

    def __init__(self) -> None:
        self.current_iteration_number: int = -1
        self.metric_values: list[float] = []
        self.multires_iterations: list[int] = []

    def on_iteration(self, registration_method: sitk.ImageRegistrationMethod) -> None:
        # Some optimizers report an iteration event for function evaluations and not
        # a complete iteration, we only want to update every iteration.
        if registration_method.GetOptimizerIteration() == self.current_iteration_number:
            return

        self.current_iteration_number = registration_method.GetOptimizerIteration()
        self.metric_values.append(registration_method.GetMetricValue())

    def on_multires_iteration(self) -> None:
        self.multires_iterations.append(len(self.metric_values))


@dataclass
class RegistrationOutput:
    initial_transform: sitk.Transform
    transform: sitk.Transform
    final_metric_value: float
    stopping_condition: str
    opt_status: OptimiserStatus


class SlideRegistrationPipeline:
    """
    Class to perform slide registration between fixed and moving images.
    """

    def __init__(
        self,
        fixed_background: np.ndarray,
        moving_background: np.ndarray,
        num_attempts: int = 1,
        num_angles: int = 1,
        num_max_iterations: int = 100,
        input_transform: Optional[sitk.Transform] = None,
    ) -> None:
        """
        :param fixed_background: An array of the fixed image background values.
        :param moving_background: An array of the moving image background values.
        :param num_attempts: Number of random attempts for registration (default=1).
        :param num_angles: Number of angles to use for registration (default=1).
        :param num_max_iterations: Number of maximum iterations to use for registration (default=100).
        :param input_transform: If this is specified, it will be used for registration (default=None).
        """
        self.random_seed = 12345
        self.num_random_attempts = num_attempts
        self.num_angles = num_angles
        self.num_max_iterations = num_max_iterations

        self.affine_scales = [16, 4]
        self.deformable_scales = [4, 2]
        self.deformable_grid_size = [5, 5]
        self.deformable_grid_order = 3

        self.fixed_background = fixed_background
        self.moving_background = moving_background
        self.input_transform = input_transform

    @staticmethod
    def preprocess(image: sitk.Image, bg_color: np.ndarray, mask: Optional[np.ndarray] = None) -> sitk.Image:
        """Implement separate preprocessing steps appropriate for H&E and IHC. This includes:
        1. [For H&E] Stain matrix estimation (e.g. Macenko), using given mask.
        2. Haematoxylin channel extraction, using estimated H&E stain matrix or default skimage HED matrix for IHC.

        :param image: Simple ITK image for preprocessing.
        :param bg_color: Background color of the image as a numpy array.
        :param mask: Optional mask for H&E image to estimate the H&E stain matrix.
        :return: The preprocessed Simple ITK image containing the hematoxylin channel only.
        """
        array = sitk.GetArrayViewFromImage(image)

        if mask is not None:
            # Macenko method for stain matrix estimation using given mask (for H&E)
            rgb_pixels = array[mask].astype(float) / 255
            _, hed_from_rgb = estimate_he_stain_matrix(rgb_pixels)
            hed = separate_hed_stains(array, bg_colour=bg_color, hed_from_rgb=hed_from_rgb)
        else:
            hed = separate_hed_stains(array, bg_colour=bg_color)
        h_image = sitk.GetImageFromArray(hed[..., 0])
        h_image.CopyInformation(image)
        return sitk.Cast(h_image, sitk.sitkFloat32)

    @staticmethod
    def get_sigma_for_scale(scale: int) -> float:
        return 4.0 * scale  # TODO: Use physical units if necessary

    def __call__(
        self, moving_image: sitk.Image, fixed_image: sitk.Image, fixed_mask: Optional[np.ndarray] = None
    ) -> tuple[sitk.Transform, sitk.Image, float, float]:
        """
        Perform registration between moving image and fixed image.

        :param moving_image: The Simple ITK image to be registered (e.g. IHC image).
        :param fixed_image: The Simple ITK reference image (e.g. H&E image).
        :param fixed_mask: An optional mask for the fixed image.
        :return: A tuple containing the Simple ITK transform, the registered Simple ITK image,
        metric values before and after transform.
        """

        # Apply preprocessing (extract haematoxylin concentrations)
        # For H&E, we use existing mask for estimating stain matrix from Macenko method
        preprocessed_fixed_image = self.preprocess(fixed_image, bg_color=self.fixed_background, mask=fixed_mask)
        preprocessed_moving_image = self.preprocess(moving_image, bg_color=self.moving_background)

        if self.input_transform is None:
            # Run affine image registration
            # Using masks didn't improve results (See registration.ipynb)
            affine_registration_outputs = self.run_affine_registration_n_times(
                fixed_image=preprocessed_fixed_image,
                moving_image=preprocessed_moving_image,
            )

            num_affine_attempts = len(affine_registration_outputs)
            if num_affine_attempts == 0:
                raise RuntimeError(f"Could not register images after {self.num_random_attempts} attempts")

            # Find the best affine transform by picking the smallest metric value
            best_output = min(affine_registration_outputs, key=lambda x: x.final_metric_value)
            best_affine_transform = best_output.transform

            # Apply deformable registration on top of the affine transform
            deformable_registration_output = self.run_deformable_registration(
                fixed_image=preprocessed_fixed_image,
                moving_image=preprocessed_moving_image,
                initial_transform=best_affine_transform,
                seed=self.random_seed + num_affine_attempts,
            )
            final_transform = deformable_registration_output.transform
            # useNearestNeighborExtrapolator prevents black artefacts at the edges
            registered_image = sitk.Resample(
                moving_image, fixed_image, final_transform, useNearestNeighborExtrapolator=True
            )
            initial_metric_value = -affine_registration_outputs[0].opt_status.metric_values[0]  # Losses are negative
            final_metric_value = -deformable_registration_output.final_metric_value
        else:
            # NOTE: Applying a saved transform to an image at a different magnification leads to undesirable shifts.
            final_transform = self.input_transform
            registered_image = sitk.Resample(
                moving_image,
                fixed_image.GetSize(),
                final_transform,
                sitk.sitkLinear,
                fixed_image.GetOrigin(),
                fixed_image.GetSpacing(),
                fixed_image.GetDirection(),
                useNearestNeighborExtrapolator=True,
            )
            initial_metric_value = 0  # these will be copied from saved transform dictionary
            final_metric_value = 0

        return final_transform, registered_image, initial_metric_value, final_metric_value

    def run_affine_registration_n_times(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        fixed_mask: Optional[sitk.Image] = None,
        moving_mask: Optional[sitk.Image] = None,
        rigid: bool = True,
    ) -> list[RegistrationOutput]:
        registration_outputs = []
        # TODO: Include flipping among candidate initialisations (plus all flipped rotations)
        # Flipping could be implemented as affine with [-1, 0, 0, 1] matrix
        # composed with Euler initial transform for rigid, or initialising
        # affine with [-cos, -sin, sin, cos] instead of [cos, -sin, sin, cos].
        angles = np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False)
        for angle in angles:
            for attempt in range(self.num_random_attempts):
                try:
                    output = self.run_affine_registration(
                        fixed_image=fixed_image,
                        moving_image=moving_image,
                        seed=self.random_seed + attempt,
                        angle=angle,
                        moving_mask=moving_mask,
                        fixed_mask=fixed_mask,
                        rigid=rigid,
                    )
                    registration_outputs.append(output)
                except RuntimeError as e:
                    print(e)
                    pass  # retry again with a different random sampling.

        return registration_outputs

    def run_affine_registration(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        seed: int,
        angle: float,
        fixed_mask: Optional[sitk.Image] = None,
        moving_mask: Optional[sitk.Image] = None,
        rigid: bool = True,
    ) -> RegistrationOutput:
        if rigid:
            initial_transform = sitk.Euler2DTransform()
            initial_transform.SetAngle(angle)
        else:
            initial_transform = sitk.AffineTransform(2)  # type: ignore
            initial_transform.SetMatrix(rotation_matrix(angle).flatten())

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, initial_transform, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        return self._run_registration(
            fixed_image=fixed_image,
            moving_image=moving_image,
            seed=seed,
            initial_transform=initial_transform,
            scales=self.affine_scales,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
        )

    def run_deformable_registration(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        seed: int,
        initial_transform: sitk.Transform,
        fixed_mask: Optional[sitk.Image] = None,
        moving_mask: Optional[sitk.Image] = None,
    ) -> RegistrationOutput:
        deformable_transform = sitk.BSplineTransformInitializer(
            fixed_image, self.deformable_grid_size, self.deformable_grid_order
        )

        composite_transform = sitk.CompositeTransform(initial_transform)
        composite_transform.AddTransform(deformable_transform)

        return self._run_registration(
            fixed_image=fixed_image,
            moving_image=moving_image,
            seed=seed,
            initial_transform=composite_transform,
            scales=self.deformable_scales,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
        )

    def _run_registration(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        seed: int,
        initial_transform: sitk.Transform,
        scales: Sequence[int],
        fixed_mask: Optional[sitk.Image] = None,
        moving_mask: Optional[sitk.Image] = None,
    ) -> RegistrationOutput:
        assert fixed_image.GetPixelID() == moving_image.GetPixelID()

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # TODO: Review whether to perform full vs random sampling
        registration_method.SetMetricSamplingStrategy(registration_method.NONE)
        # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        # registration_method.SetMetricSamplingPercentage(0.25, seed=seed)

        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.1,
            numberOfIterations=self.num_max_iterations,
            convergenceMinimumValue=1e-8,
            convergenceWindowSize=20,
        )

        # Scale the step size differently for each parameter, this is critical!!!
        registration_method.SetOptimizerScalesFromPhysicalShift()

        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        sigmas = [self.get_sigma_for_scale(scale) for scale in scales]
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=scales)
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=sigmas)
        # TODO: Uncomment this line if using physical units for sigma (default: pixels)
        # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        if fixed_mask is not None:
            registration_method.SetMetricFixedMask(fixed_mask)
        if moving_mask is not None:
            registration_method.SetMetricMovingMask(moving_mask)

        opt_status = OptimiserStatus()
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: opt_status.on_iteration(registration_method))
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, opt_status.on_multires_iteration)

        final_transform = registration_method.Execute(fixed=fixed_image, moving=moving_image)

        return RegistrationOutput(
            initial_transform=initial_transform,
            transform=final_transform,
            final_metric_value=registration_method.GetMetricValue(),
            stopping_condition=registration_method.GetOptimizerStopConditionDescription(),
            opt_status=opt_status,
        )
