# Copyright (c) 2017, John Skinner
from enum import Enum
import numpy as np
import arvet.util.image_utils as image_utils
from arvet.util.transform import Transform
from arvet.metadata.camera_intrinsics import CameraIntrinsics


class DepthNoiseQuality(Enum):
    NO_NOISE = 0
    GAUSSIAN_NOISE = 1
    KINECT_NOISE = 2


MAXIMUM_QUALITY = DepthNoiseQuality.KINECT_NOISE


def create_noisy_depth_image(left_true_depth: np.ndarray, right_true_depth: np.ndarray,
                             camera_intrinsics: CameraIntrinsics,
                             right_camera_relative_pose: Transform,
                             quality_level: DepthNoiseQuality = MAXIMUM_QUALITY) -> np.ndarray:
    """
    Generate a noisy depth image from a ground truth depth image.
    The image should already be float32 and scaled to meters.
    The output image will be the same size as the ground truth image, with noise introduced.
    :param left_true_depth: A ground-truth depth image captured from the simulator
    :param right_true_depth: A ground-truth depth image captured to the right of the main image.
    :param camera_intrinsics: The intrinsics for both cameras, assumed to be the same
    :param right_camera_relative_pose: The relative pose of the right camera, for projection logic.
    :param quality_level: An integer switch between different noise models, lower is worse. Default best model.
    :return:
    """
    quality_level = DepthNoiseQuality(quality_level)
    if quality_level is DepthNoiseQuality.KINECT_NOISE:
        return kinect_depth_model(left_true_depth, right_true_depth, camera_intrinsics,
                                  right_camera_relative_pose)
    elif quality_level is DepthNoiseQuality.GAUSSIAN_NOISE:
        return naive_gaussian_noise(left_true_depth)
    else:
        return left_true_depth


def naive_gaussian_noise(true_depth: np.ndarray) -> np.ndarray:
    """
    The simplest and least realistic depth noise, we add a gaussian noise to each pixel.
    This is the axial noise component of the kinect depth model, below,
    based on the numbers in:
    Characterizations of noise in Kinect depth images: A review (2014) by
    Tanwi Mallick, Partha Pratim Das, and Arun Kumar Majumdar

    :param true_depth: Ground truth depth image
    :return: A depth image with added noise
    """
    return true_depth + np.random.normal(0, 0.0012 + 0.0019 * np.square(true_depth - 0.4))


def kinect_depth_model(left_true_depth: np.ndarray, right_true_depth: np.ndarray,
                       camera_intrinsics: CameraIntrinsics, baseline: Transform) -> np.ndarray:
    """
    Depth noise based on the original kinect depth sensor
    :param left_true_depth: The left depth image
    :param right_true_depth: The right depth image
    :param camera_intrinsics: The intrinsics of both cameras
    :param baseline: The location of the right camera relative to the left camera
    :return:
    """
    # Coordinate transform baseline into camera coordinates, X right, Y down, Z forward
    # We only actually use the X any Y relative coordinates, assume the Z (forward) relative coordinate
    # is embedded in the right ground truth depth, i.e.: right_gt_depth = B_z + left_gt_depth (if B_x and B_y are zero)
    baseline_x = -1 * baseline.location[1]
    baseline_y = -1 * baseline.location[2]

    # Step 1: Rescale the camera intrisics to the kinect resolution
    fx = 640 * camera_intrinsics.fx / camera_intrinsics.width
    fy = 480 * camera_intrinsics.fy / camera_intrinsics.height
    cx = 640 * camera_intrinsics.cx / camera_intrinsics.width
    cy = 480 * camera_intrinsics.cy / camera_intrinsics.height

    # Step 2: Image resolution - kinect images are 640x480
    if left_true_depth.shape == (480, 640):
        left_depth_points = np.copy(left_true_depth)
    else:
        left_depth_points = image_utils.resize(left_true_depth, (640, 480),
                                               interpolation=image_utils.Interpolation.NEAREST)
    if right_true_depth.shape == (480, 640):
        right_depth_points = np.copy(right_true_depth)
    else:
        right_depth_points = image_utils.resize(right_true_depth, (640, 480),
                                                interpolation=image_utils.Interpolation.NEAREST)
    output_depth = left_depth_points

    # Step 3: Clipping planes - Set to 0 where too close or too far
    shadow_mask = (0.8 < left_depth_points) & (left_depth_points < 4.0)

    # Step 4: Shadows
    # Project the depth points from the right depth image onto the left depth image
    # Places that are not visible from the right image are shadows
    right_points = np.indices((480, 640), dtype=np.float32)
    right_x = right_points[1] - cx
    right_y = right_points[0] - cy

    # Stereo project points in right image into left image
    # x' = fx * (X - B_x) / (Z - B_z) + cx, y' = fy * (Y - B_y) / (Z - B_z) + cy
    # and, as above:
    # X = Z * (x - cx) / fx, Y = Z * (y - cy) / fy
    # Therefore,
    # x' = ((x - cx) * Z - fx * B_x) / (Z - B_z) + cx, similar for y
    right_x = np.multiply(left_depth_points, right_x)  # (x - cx) * Z
    right_y = np.multiply(left_depth_points, right_y)  # (y - cy) * Z
    # x * Z - fx * B_x, y * Z - fy * B_y
    right_x -= baseline_x * fx
    right_y -= baseline_y * fy
    # Divide throughout by Z - B_z, or just the orthographic right depth
    right_x = np.divide(right_x, right_depth_points + 0.00001) + cx
    right_y = np.divide(right_y, right_depth_points + 0.00001) + cy
    shadow_mask &= (right_x >= 0) & (right_y >= 0) & (right_x < 640) & (right_y < 480)
    projected_depth = nearest_sample(right_depth_points, right_x, right_y)
    shadow_mask &= (left_depth_points - projected_depth) < 0.01

    # Step 5: Random dropout of pixels
    shadow_mask &= np.random.choice([False, True], (480, 640), p=(0.2, 0.8))

    # Step 6: Lateral noise - I don't know how to do this quickly

    # Step 7: axial noise
    # Based on the numbers in Characterizations of noise in Kinect depth images: A review (2014) by Mallick et al.
    output_depth += np.random.normal(0, 0.0012 + 0.0019 * np.square(output_depth - 0.4))
    output_depth = np.multiply(shadow_mask, output_depth)

    # Finally, return to an image matching the input size, so that we're aligned with the RGB image
    if left_true_depth.shape != (480, 640):
        output_depth = image_utils.resize(output_depth, (left_true_depth.shape[1],
                                                         left_true_depth.shape[0]),
                                          interpolation=image_utils.Interpolation.NEAREST)
    return output_depth


def nearest_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sample an image from fractional coordinates
    :param image:
    :param x:
    :param y:
    :return:
    """
    x = np.rint(x).astype(np.int)
    y = np.rint(y).astype(np.int)

    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)

    return image[y, x]
