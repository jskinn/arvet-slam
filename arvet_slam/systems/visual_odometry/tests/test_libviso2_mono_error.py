# Copyright (c) 2017, John Skinner
import unittest
import bson
from pathlib import Path
import json
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as image_manager
import arvet.util.image_utils as image_utils
import arvet.metadata.image_metadata as imeta
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader

from arvet.util.test_helpers import ExtendedTestCase
from arvet.util.transform import Transform
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import Image

from arvet_slam.systems.visual_odometry.libviso2 import LibVisOMonoSystem, MatcherRefinement
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult


def load_dataset_locations():
    conf_json = Path(__file__).parent / 'real_data_locations.json'
    if conf_json.is_file():
        with conf_json.open('r') as fp:
            son = json.load(fp)
            return Path(son['ndds_location'])
    return None


NDDS_SEQUENCE = load_dataset_locations()


class TestLibVisOMonoRealData(ExtendedTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        dbconn.setup_image_manager()

    @classmethod
    def tearDownClass(cls) -> None:
        dbconn.tear_down_image_manager()

    @unittest.skipIf(
        NDDS_SEQUENCE is None or not NDDS_SEQUENCE.exists(),
        f"Could not find the NDDS dataset at {NDDS_SEQUENCE}, cannot run integration test"
    )
    def test_simple_trial_run_generated(self):
        sequence_folder, left_path, right_path = ndds_loader.find_files(NDDS_SEQUENCE)
        camera_intrinsics = ndds_loader.read_camera_intrinsics(left_path / '_camera_settings.json')
        max_img_id = ndds_loader.find_max_img_id(lambda idx: left_path / ndds_loader.IMG_TEMPLATE.format(idx))
        with (NDDS_SEQUENCE / 'timestamps.json').open('r') as fp:
            timestamps = json.load(fp)

        subject = LibVisOMonoSystem(
            matcher_nms_n=10,
            matcher_nms_tau=66,
            matcher_match_binsize=50,
            matcher_match_radius=245,
            matcher_match_disp_tolerance=2,
            matcher_outlier_disp_tolerance=5,
            matcher_outlier_flow_tolerance=2,
            matcher_multi_stage=False,
            matcher_half_resolution=False,
            matcher_refinement=MatcherRefinement.SUBPIXEL,
            bucketing_max_features=6,
            bucketing_bucket_width=136,
            bucketing_bucket_height=102,
            height=1.0,
            pitch=0.0,
            ransac_iters=439,
            inlier_threshold=4.921875,
            motion_threshold=609.375
        )
        subject.set_camera_intrinsics(camera_intrinsics, 0.1)

        subject.start_trial(ImageSequenceType.SEQUENTIAL, seed=0)
        image_group = 'test'
        with image_manager.get().get_group(image_group, allow_write=True):
            for img_idx in range(max_img_id + 1):
                pixels = image_utils.read_colour(left_path / ndds_loader.IMG_TEMPLATE.format(img_idx))
                image = Image(
                    _id=bson.ObjectId(),
                    pixels=pixels,
                    image_group=image_group,
                    metadata=imeta.ImageMetadata(camera_pose=Transform())
                )
                subject.process_image(image, timestamps[img_idx])
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertEqual(subject, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertEqual(max_img_id + 1, len(result.results))
