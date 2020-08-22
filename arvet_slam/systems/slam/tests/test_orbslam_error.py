# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import logging
from pathlib import Path
import json
import shutil
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as image_manager
import arvet.util.image_utils as image_utils
import arvet.metadata.image_metadata as imeta
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader

from arvet.util.test_helpers import ExtendedTestCase
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import StereoImage
from arvet.config.path_manager import PathManager

from arvet_slam.systems.slam.orbslam2 import OrbSlam2, SensorMode
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult


def load_dataset_locations():
    conf_json = Path(__file__).parent / 'real_data_locations.json'
    if conf_json.is_file():
        with conf_json.open('r') as fp:
            son = json.load(fp)
            return Path(son['vocab']), Path(son['ndds_location'])
    return None, None


VOCAB_PATH, NDDS_SEQUENCE = load_dataset_locations()


class TestORBSlam2RealData(ExtendedTestCase):
    temp_folder = Path(__file__).parent / 'temp-test-orbslam2'

    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig(level=20)
        dbconn.setup_image_manager()

    @classmethod
    def tearDownClass(cls) -> None:
        dbconn.tear_down_image_manager()
        if cls.temp_folder.exists():
            shutil.rmtree(cls.temp_folder)

    @unittest.skipIf(
        VOCAB_PATH is None or not VOCAB_PATH.exists() or
        NDDS_SEQUENCE is None or not NDDS_SEQUENCE.exists(),
        f"Could not find the NDDS dataset at {NDDS_SEQUENCE}, cannot run integration test"
    )
    def test_simple_trial_run_generated(self):
        # TODO: The state of things:
        # - With the configured sequence, the ORB-SLAM subprocess silently dies about frame 346-347
        # - With my bastardised removal of the subprocess, it seems to work? and not crash?
        # - I have no idea what is different? the pipe?
        sequence_folder, left_path, right_path = ndds_loader.find_files(NDDS_SEQUENCE)
        camera_intrinsics = ndds_loader.read_camera_intrinsics(left_path / '_camera_settings.json')
        max_img_id = ndds_loader.find_max_img_id(lambda idx: left_path / ndds_loader.IMG_TEMPLATE.format(idx))
        with (NDDS_SEQUENCE / 'timestamps.json').open('r') as fp:
            timestamps = json.load(fp)
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        path_manager = PathManager([VOCAB_PATH.parent], self.temp_folder)

        subject = OrbSlam2(
            vocabulary_file=str(VOCAB_PATH.name),
            mode=SensorMode.STEREO,
            vocabulary_branching_factor=5,
            vocabulary_depth=6,
            vocabulary_seed=0,
            depth_threshold=387.0381720715473,
            orb_num_features=598,
            orb_scale_factor=np.power(480 / 104, 1 / 11),  # = (480 / min_height)^(1/num_levels)
            orb_num_levels=11,
            orb_ini_threshold_fast=86,
            orb_min_threshold_fast=48
        )
        subject.resolve_paths(path_manager)
        subject.set_camera_intrinsics(camera_intrinsics, 0.1)

        # Read the first frame data to get the baseline
        left_frame_data = ndds_loader.read_json(left_path / ndds_loader.DATA_TEMPLATE.format(0))
        right_frame_data = ndds_loader.read_json(right_path / ndds_loader.DATA_TEMPLATE.format(0))
        left_camera_pose = ndds_loader.read_camera_pose(left_frame_data)
        right_camera_pose = ndds_loader.read_camera_pose(right_frame_data)
        subject.set_stereo_offset(left_camera_pose.find_relative(right_camera_pose))

        subject.start_trial(ImageSequenceType.SEQUENTIAL, seed=0)
        image_group = 'test'
        with image_manager.get().get_group(image_group, allow_write=True):
            for img_idx in range(max_img_id + 1):
                left_pixels = image_utils.read_colour(left_path / ndds_loader.IMG_TEMPLATE.format(img_idx))
                right_pixels = image_utils.read_colour(right_path / ndds_loader.IMG_TEMPLATE.format(img_idx))

                image = StereoImage(
                    pixels=left_pixels,
                    right_pixels=right_pixels,
                    image_group=image_group,
                    metadata=imeta.ImageMetadata(camera_pose=left_camera_pose),
                    right_metadata=imeta.ImageMetadata(camera_pose=right_camera_pose)
                )
                subject.process_image(image, timestamps[img_idx])
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertEqual(subject, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertEqual(max_img_id + 1, len(result.results))
