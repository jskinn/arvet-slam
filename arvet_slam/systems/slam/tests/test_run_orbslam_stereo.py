# Copyright (c) 2017, John Skinner
import unittest
import os.path
import shutil
from pathlib import Path
from pymodm.context_managers import no_auto_dereference

from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
from arvet.core.sequence_type import ImageSequenceType
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult
from arvet_slam.systems.slam.orbslam2 import OrbSlam2, SensorMode
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode
from arvet_slam.systems.slam.tests.create_vocabulary import create_vocab


class TestRunOrbslamStereo(unittest.TestCase):
    temp_folder = 'temp-test-orbslam2'
    vocab_path = Path(__file__).parent / 'ORBvoc-synth.txt'

    @classmethod
    def setUpClass(cls):
        dbconn.setup_image_manager()
        os.makedirs(cls.temp_folder, exist_ok=True)
        if not cls.vocab_path.exists():  # If there is no vocab file, make one
            print("Creating vocab file, this may take a while...")
            create_vocab(cls.vocab_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder)
        dbconn.tear_down_image_manager()

    def test_simple_trial_run(self):
        # Actually run the system using mocked images
        num_frames = 50
        max_time = 50
        speed = 0.1
        path_manager = PathManager([Path(__file__).parent], self.temp_folder)
        image_builder = DemoImageBuilder(
            mode=ImageMode.STEREO, stereo_offset=0.15,
            width=320, height=240, num_stars=500,
            length=max_time * speed, speed=speed,
            min_size=4, max_size=50
        )
        subject = OrbSlam2(
            vocabulary_file=self.vocab_path,
            mode=SensorMode.STEREO,
            orb_ini_threshold_fast=12,
            orb_min_threshold_fast=7
        )
        subject.resolve_paths(path_manager)
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)
        subject.set_stereo_offset(image_builder.get_stereo_offset())

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
        self.assertTrue(result.has_scale)
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

    def test_run_with_loops(self):
        # Actually run the system using mocked images
        num_frames = [20, 20, 27, 25]
        max_time = 25
        speed = 1
        path_manager = PathManager([Path(__file__).parent], self.temp_folder)
        image_builder = DemoImageBuilder(
            mode=ImageMode.STEREO, stereo_offset=0.15,
            width=320, height=240, num_stars=300,
            length=max_time * speed, speed=speed,
            min_size=4, max_size=50, close_ratio=0.5
        )
        subject = OrbSlam2(
            vocabulary_file=self.vocab_path,
            mode=SensorMode.STEREO,
            orb_ini_threshold_fast=12,
            orb_min_threshold_fast=7
        )
        subject.resolve_paths(path_manager)
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(),
                                      max_time * len(num_frames) / sum(num_frames))
        subject.set_stereo_offset(image_builder.get_stereo_offset())

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for loop_idx, loop_frames in enumerate(num_frames):
            for idx in range(loop_frames):
                time = max_time * idx / loop_frames + max_time * loop_idx
                if loop_idx % 2 == 1:
                    builder_time = max_time * (1 - idx / loop_frames)
                else:
                    builder_time = max_time * idx / loop_frames
                image = image_builder.create_frame(builder_time)
                subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertTrue(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(sum(num_frames), len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    def test_can_run_on_colour(self):
        # Actually run the system using mocked images
        num_frames = 50
        max_time = 50
        speed = 0.1
        path_manager = PathManager([Path(__file__).parent], self.temp_folder)
        image_builder = DemoImageBuilder(
            mode=ImageMode.STEREO, stereo_offset=0.15,
            width=320, height=240, num_stars=500,
            length=max_time * speed, speed=speed,
            min_size=4, max_size=50, colour=True
        )
        subject = OrbSlam2(
            vocabulary_file=self.vocab_path,
            mode=SensorMode.STEREO,
            orb_ini_threshold_fast=12,
            orb_min_threshold_fast=7
        )
        subject.resolve_paths(path_manager)
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)
        subject.set_stereo_offset(image_builder.get_stereo_offset())

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
        self.assertTrue(result.has_scale)
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
