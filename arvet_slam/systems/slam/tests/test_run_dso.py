# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
from pymodm.context_managers import no_auto_dereference

from arvet.util.test_helpers import ExtendedTestCase
from arvet.core.sequence_type import ImageSequenceType
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult
from arvet_slam.systems.slam.direct_sparse_odometry import DSO, RectificationMode
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode


class TestRunDSO(ExtendedTestCase):

    def test_simple_trial_run_rect_none(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.NONE,
            # These should be irrelevant
            rectification_intrinsics=CameraIntrinsics(
                width=320,
                height=240,
                fx=160,
                fy=160,
                cx=160,
                cy=120
            )
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    def test_simple_trial_run_rect_calib(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.CALIB,
            rectification_intrinsics=CameraIntrinsics(
                width=320,
                height=240,
                fx=160,
                fy=160,
                cx=160,
                cy=120
            )
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    def test_simple_trial_run_rect_crop(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        # image_builder.visualise_sequence(max_time, max_time / num_frames)
        # return

        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=480,
                height=480,
                fx=240,
                fy=240,
                cx=240,
                cy=240
            )
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    @unittest.skip("Tends to segfault if running as part of a suite.")
    def test_simple_trial_run_rect_crop_larger_than_source(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=1242, height=376,     # These are the dimensions of the KITTI dataset
            num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=480,
                height=480,     # This is larger than the input height
                fx=240,
                fy=240,
                cx=240,
                cy=240
            )
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    def test_consistency(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.NONE,
            rectification_intrinsics=image_builder.get_camera_intrinsics()
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result1 = subject.finish_trial()

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result2 = subject.finish_trial()

        has_any_estimate = False
        self.assertEqual(len(result1.results), len(result2.results))
        for frame_result_1, frame_result_2 in zip(result1.results, result2.results):
            self.assertEqual(frame_result_1.timestamp, frame_result_2.timestamp)
            self.assertEqual(frame_result_1.tracking_state, frame_result_2.tracking_state)
            if frame_result_1.estimated_motion is None or frame_result_2.estimated_motion is None:
                self.assertEqual(frame_result_1.estimated_motion, frame_result_2.estimated_motion)
            else:
                has_any_estimate = True
                motion1 = frame_result_1.estimated_motion
                motion2 = frame_result_2.estimated_motion

                loc_diff = motion1.location - motion2.location
                self.assertNPClose(loc_diff, np.zeros(3), rtol=0, atol=0.1)     # Absolute tolerance of 10cm is awful
                quat_diff = motion1.rotation_quat(True) - motion2.rotation_quat(True)
                self.assertNPClose(quat_diff, np.zeros(4), rtol=0, atol=0.01)
        self.assertTrue(has_any_estimate)
