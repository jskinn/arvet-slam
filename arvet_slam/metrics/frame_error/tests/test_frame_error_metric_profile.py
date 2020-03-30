import unittest
import unittest.mock as mock
import os
import bson
import numpy as np
import transforms3d as tf3d

import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager

from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.metadata.image_metadata import make_metadata, ImageSourceType
from arvet.core.image import Image
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.metric import MetricResult
from arvet.core.trial_result import TrialResult
import arvet.core.tests.mock_types as mock_types
from arvet.util.transform import Transform, compute_average_pose

from arvet_slam.metrics.frame_error.frame_error_metric import FrameErrorMetric, make_pose_error
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult


#@unittest.skip("Not running profiling")
class TestRunAllDatabaseProfile(unittest.TestCase):
    system = None
    image_source = None
    images = None
    repeats = 10
    num_images = 3009

    @classmethod
    def setUpClass(cls):
        # dbconn.connect_to_test_db()
        # image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        # im_manager.set_image_manager(image_manager)
        #
        # # Ensure we have a clean slate in the database
        # Task._mongometa.collection.drop()
        # mock_types.MockTrialResult._mongometa.collection.drop()
        # Image._mongometa.collection.drop()
        # ImageCollection._mongometa.collection.drop()
        # mock_types.MockSystem._mongometa.collection.drop()

        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()

        image_data = [
            np.random.normal(128, 20, size=(10, 10)).astype(np.uint8)
            for _ in range(cls.num_images)
        ]
        cls.images = [
            Image(
                pixels=pixels,
                metadata=make_metadata(
                    pixels=pixels,
                    source_type=ImageSourceType.SYNTHETIC,
                    camera_pose=Transform(
                        (idx * 15, idx, 0),
                        tf3d.quaternions.axangle2quat((1, 2, 3), 5 * idx * np.pi / (2 * cls.num_images)), w_first=True
                    ),
                    intrinsics=CameraIntrinsics(
                        width=10, height=10,
                        fx=5, fy=5,
                        cx=5, cy=5
                    )
                )
            )
            for idx, pixels in enumerate(image_data)
        ]

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collections for all the models used
        # Image._mongometa.collection.drop()
        # ImageCollection._mongometa.collection.drop()
        # mock_types.MockSystem._mongometa.collection.drop()
        # if os.path.isfile(dbconn.image_file):
        #     os.remove(dbconn.image_file)
        pass

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_profile(self, _):
        import cProfile as profile

        repeats = 3
        random = np.random.RandomState(13)

        # Make some number of trials results to measure
        trial_results = []
        for repeat in range(repeats):
            frame_results = [
                FrameResult(
                    timestamp=idx + random.normal(0, 0.01),
                    image=image,
                    processing_time=random.uniform(0.01, 1),
                    pose=image.camera_pose,
                    estimated_motion=Transform(
                        (14 + random.normal(0, 1), 0.9 + random.normal(0, 0.05), 0.1 + random.normal(0, 0.05)),
                        tf3d.quaternions.axangle2quat(
                            (1, 2, 4), 5 * np.pi / (2 * self.num_images) + random.normal(0, np.pi / 64)),
                        w_first=True
                    ) if idx > 0 else None,
                    tracking_state=TrackingState.OK,
                    num_features=random.randint(10, 1000),
                    num_matches=random.randint(10, 1000)
                )
                for idx, image in enumerate(self.images)
            ]
            trial_result = SLAMTrialResult(
                system=self.system,
                image_source=self.image_source,
                success=True,
                results=frame_results,
                has_scale=False
            )
            trial_results.append(trial_result)

        metric = FrameErrorMetric()

        stats_file = "measure_trials.prof"

        profile.runctx("metric.measure_results(trial_results)",
                       locals=locals(), globals=globals(), filename=stats_file)
