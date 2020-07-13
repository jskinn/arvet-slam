# Copyright (c) 2017, John Skinner
import os.path
import unittest
import numpy as np
import timeit
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.util.transform as tf
import arvet_slam.dataset.ndds.depth_noise as depth_noise


class TestDepthNoise(unittest.TestCase):

    def test_gaussian_noise(self):
        true_depth = get_test_image('left')
        noisy_depth = depth_noise.naive_gaussian_noise(true_depth)

        diff = noisy_depth - true_depth
        self.assertEqual(diff.shape[0] * diff.shape[1], np.count_nonzero(diff))

    def test_kinect_noise_maintains_type(self):
        left_true_depth = 256 * get_test_image('left')
        right_true_depth = 256 * get_test_image('right')

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            left_true_depth.shape[1],
            left_true_depth.shape[0],
            focal_length * left_true_depth.shape[1],
            focal_length * left_true_depth.shape[1],
            0.5 * left_true_depth.shape[1], 0.5 * left_true_depth.shape[0])
        relative_pose = tf.Transform((0, -0.15, 0))
        noisy_depth = depth_noise.kinect_depth_model(left_true_depth,
                                                     right_true_depth, camera_intrinsics, relative_pose)
        self.assertIsNotNone(noisy_depth)
        self.assertNotEqual(np.dtype, noisy_depth.dtype)

    def test_kinect_noise_works_when_not_640_by_480(self):
        left_true_depth = get_test_image('left')[0:64, 0:64]
        right_true_depth = get_test_image('right')[0:64, 0:64]

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            left_true_depth.shape[1],
            left_true_depth.shape[0],
            focal_length * left_true_depth.shape[1],
            focal_length * left_true_depth.shape[1],
            0.5 * left_true_depth.shape[1], 0.5 * left_true_depth.shape[0])
        relative_pose = tf.Transform((0, -0.15, 0))
        noisy_depth = depth_noise.kinect_depth_model(
            left_true_depth, right_true_depth, camera_intrinsics, relative_pose)
        self.assertIsNotNone(noisy_depth)
        self.assertNotEqual(np.dtype, noisy_depth.dtype)

    def test_kinect_noise_produces_reasonable_output(self):
        left_true_depth = get_test_image('left')
        right_true_depth = get_test_image('right')

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            left_true_depth.shape[1],
            left_true_depth.shape[0],
            focal_length * left_true_depth.shape[1],
            focal_length * left_true_depth.shape[1],
            0.5 * left_true_depth.shape[1], 0.5 * left_true_depth.shape[0])
        relative_pose = tf.Transform((0, -0.15, 0))
        noisy_depth = depth_noise.kinect_depth_model(
            left_true_depth, right_true_depth, camera_intrinsics, relative_pose)
        self.assertLessEqual(np.max(noisy_depth), 4.1)  # A little leeway for noise
        self.assertGreaterEqual(np.min(noisy_depth[np.nonzero(noisy_depth)]), 0.7)
        self.assertGreater(np.mean(noisy_depth), 0)  # Assert that something is visible at all
        # image_utils.show_image(noisy_depth / np.max(noisy_depth), 'test depth')

    def test_kinect_noise_is_quick(self):
        left_true_depth = get_test_image('left')
        right_true_depth = get_test_image('right')

        focal_length = 1 / (2 * np.tan(np.pi / 4))
        camera_intrinsics = cam_intr.CameraIntrinsics(
            left_true_depth.shape[1],
            left_true_depth.shape[0],
            focal_length * left_true_depth.shape[1],
            focal_length * left_true_depth.shape[1],
            0.5 * left_true_depth.shape[1], 0.5 * left_true_depth.shape[0])
        relative_pose = tf.Transform((0, -0.15, 0))

        number = 20
        time = timeit.timeit(
            lambda: depth_noise.kinect_depth_model(left_true_depth, right_true_depth,
                                                   camera_intrinsics, relative_pose), number=number)
        # print("Noise time: {0}, total time: {1}".format(time / number, time))
        self.assertLess(time / number, 1)


def get_test_image(suffix: str):
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test-depth-{}.npy'.format(suffix)
    )
    if os.path.isfile(path):
        return np.load(path)
    else:
        raise FileNotFoundError("Could not find test image at {0}".format(path))
