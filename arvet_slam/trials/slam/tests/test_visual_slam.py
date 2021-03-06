# Copyright (c) 2019, John Skinner
import unittest
import unittest.mock as mock
import numpy as np
import transforms3d as tf3d
import pymodm
from pymodm.errors import ValidationError

from arvet.util.transform import Transform
from arvet.util.test_helpers import ExtendedTestCase
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.metadata.image_metadata import make_metadata, ImageSourceType
from arvet.core.image import Image
import arvet.core.tests.mock_types as mock_types
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult


# ------------------------- FRAME RESULT -------------------------


class TestFrameResultMongoModel(pymodm.MongoModel):
    frame_result = pymodm.fields.EmbeddedDocumentField(FrameResult)


class TestPoseErrorDatabase(unittest.TestCase):
    image = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager()

        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        cls.image = Image(
            pixels=pixels, image_group='test',
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        cls.image.save()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        TestFrameResultMongoModel._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()

    def test_stores_and_loads(self):
        frame_result = FrameResult(
            timestamp=10.3,
            image=self.image,
            processing_time=12.44,
            pose=Transform((1, 2, 3), (4, 5, 6, 7)),
            motion=Transform((-2, 1, -3), (-3, 5, 6, -6)),
            estimated_pose=Transform((-1, -2, -3), (-4, -5, -6, 7)),
            estimated_motion=Transform((2, -1, 3), (8, -5, -6, -6)),
            tracking_state=TrackingState.NOT_INITIALIZED,
            loop_edges=[6.223],
            num_features=53,
            num_matches=6
        )

        # Save the model
        model = TestFrameResultMongoModel()
        model.frame_result = frame_result
        model.save()

        # Load all the entities
        all_entities = list(TestFrameResultMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)

        # Something about these fields mean they need to be manually compared first
        self.assertEqual(all_entities[0].frame_result.image, frame_result.image)
        self.assertEqual(all_entities[0].frame_result.tracking_state, frame_result.tracking_state)
        self.assertEqual(all_entities[0].frame_result.pose, frame_result.pose)
        self.assertEqual(all_entities[0].frame_result.motion, frame_result.motion)
        self.assertEqual(all_entities[0].frame_result.estimated_pose, frame_result.estimated_pose)
        self.assertEqual(all_entities[0].frame_result.estimated_motion, frame_result.estimated_motion)
        self.assertEqual(all_entities[0].frame_result, frame_result)
        all_entities[0].delete()

    def test_stores_and_loads_minimal(self):
        frame_result = FrameResult(
            timestamp=10.3,
            image=self.image,
            processing_time=12.44,
            pose=Transform((1, 2, 3), (4, 5, 6, 7)),
            motion=Transform((-2, 1, -3), (-3, 5, 6, -6))
        )

        # Save the model
        model = TestFrameResultMongoModel()
        model.frame_result = frame_result
        model.save()

        # Load all the entities
        all_entities = list(TestFrameResultMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)

        self.assertEqual(all_entities[0].frame_result.image, frame_result.image)
        self.assertEqual(all_entities[0].frame_result.tracking_state, frame_result.tracking_state)
        self.assertEqual(all_entities[0].frame_result.pose, frame_result.pose)
        self.assertEqual(all_entities[0].frame_result.motion, frame_result.motion)
        self.assertEqual(all_entities[0].frame_result.estimated_pose, frame_result.estimated_pose)
        self.assertEqual(all_entities[0].frame_result.estimated_motion, frame_result.estimated_motion)
        self.assertEqual(all_entities[0].frame_result, frame_result)
        all_entities[0].delete()

    def test_stores_and_loads_with_explicitly_null_estimates(self):
        frame_result = FrameResult(
            timestamp=10.3,
            image=self.image,
            processing_time=12.44,
            pose=Transform((1, 2, 3), (4, 5, 6, 7)),
            motion=Transform((-2, 1, -3), (-3, 5, 6, -6)),
            estimated_pose=None,
            estimated_motion=None
        )

        # Save the model
        model = TestFrameResultMongoModel()
        model.frame_result = frame_result
        model.save()

        # Load all the entities
        all_entities = list(TestFrameResultMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)

        self.assertEqual(all_entities[0].frame_result.image, frame_result.image)
        self.assertEqual(all_entities[0].frame_result.tracking_state, frame_result.tracking_state)
        self.assertEqual(all_entities[0].frame_result.pose, frame_result.pose)
        self.assertEqual(all_entities[0].frame_result.motion, frame_result.motion)
        self.assertEqual(all_entities[0].frame_result.estimated_pose, frame_result.estimated_pose)
        self.assertEqual(all_entities[0].frame_result.estimated_motion, frame_result.estimated_motion)
        self.assertEqual(all_entities[0].frame_result, frame_result)
        all_entities[0].delete()

    def test_required_fields_are_required(self):
        model = TestFrameResultMongoModel()

        # no timestamp
        model.frame_result = FrameResult(
            image=self.image,
            processing_time=12.44,
            pose=Transform((1, 2, 3), (4, 5, 6, 7)),
            motion=Transform((-2, 1, -3), (-3, 5, 6, -6))
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no image
        model.frame_result = FrameResult(
            timestamp=10.3,
            processing_time=12.44,
            pose=Transform((1, 2, 3), (4, 5, 6, 7)),
            motion=Transform((-2, 1, -3), (-3, 5, 6, -6))
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no processing time
        model.frame_result = FrameResult(
            timestamp=10.3,
            image=self.image,
            pose=Transform((1, 2, 3), (4, 5, 6, 7)),
            motion=Transform((-2, 1, -3), (-3, 5, 6, -6))
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no pose
        model.frame_result = FrameResult(
            timestamp=10.3,
            image=self.image,
            processing_time=12.44,
            motion=Transform((-2, 1, -3), (-3, 5, 6, -6))
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no motion
        model.frame_result = FrameResult(
            timestamp=10.3,
            image=self.image,
            processing_time=12.44,
            pose=Transform((1, 2, 3), (4, 5, 6, 7))
        )
        with self.assertRaises(ValidationError):
            model.save()


# ------------------------- SLAM TRIAL RESULT -------------------------


class TestSLAMTrialResult(ExtendedTestCase):

    def test_sorts_results_by_timestamp(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)

        timestamps = [(-1 ** idx) * idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 6), w_first=True
                ),
                motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / 6), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), 9 * idx * np.pi / 36), w_first=True
                ),
                estimated_motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 36), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            settings={'key': 'value'},
            results=results,
            has_scale=True
        )
        for idx in range(1, len(results)):
            self.assertGreater(obj.results[idx].timestamp, obj.results[idx - 1].timestamp)

    def test_throws_exception_if_loop_edges_refer_to_missing_timestamp(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)

        timestamps = [(-1 ** idx) * idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 6), w_first=True
                ),
                motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / 6), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), 9 * idx * np.pi / 36), w_first=True
                ),
                estimated_motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 36), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        results[3].loop_edges.append(max(timestamps) + 0.23)
        with self.assertRaises(ValueError):
            SLAMTrialResult(
                system=system,
                image_source=image_source,
                success=True,
                settings={'key': 'value'},
                results=results,
                has_scale=True
            )

    def test_infers_motion_from_pose(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)

        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 36), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=True
        )
        self.assertEqual(Transform(), obj.results[0].motion)
        for idx in range(1, len(results)):
            # The motion should always be the relative pose of the current image relative to the previous one
            self.assertEqual(
                obj.results[idx - 1].pose.find_relative(obj.results[idx].pose), obj.results[idx].motion,
                msg="Did not match at index {0}".format(idx)
            )

    def test_infers_pose_from_motion(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)

        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                motion=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 36), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=True
        )
        self.assertEqual(Transform(), obj.results[0].pose)
        for idx in range(1, len(results)):
            # The motion should always be the relative pose of the current image relative to the previous one
            pose_based_motion = obj.results[idx - 1].pose.find_relative(obj.results[idx].pose)
            stored_motion = obj.results[idx].motion
            self.assertNPClose(pose_based_motion.location, stored_motion.location)
            self.assertNPClose(pose_based_motion.rotation_quat(), stored_motion.rotation_quat())

    def test_infers_estimated_motions_from_estimated_poses(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)

        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 36), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15.1 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), 9 * idx * np.pi / 288), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=True
        )
        self.assertIsNone(obj.results[0].estimated_motion)
        for idx in range(1, len(results)):
            # The motion should always be the relative pose of the current image relative to the previous one
            self.assertEqual(
                obj.results[idx - 1].estimated_pose.find_relative(obj.results[idx].estimated_pose),
                obj.results[idx].estimated_motion,
                msg="Different at index {0}".format(idx)
            )

    def test_infers_estimated_poses_from_estimated_motions(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)

        timestamps = [(-1 ** idx) * idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 36), w_first=True
                ),
                estimated_motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 36), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=True
        )
        # No initial pose, so we can't infer poses from motions
        for idx in range(10):
            self.assertIsNone(obj.results[idx].estimated_pose)

        # Provide a pose estimate, and redo inference
        obj.results[4].estimated_pose = Transform()
        obj.infer_missing_poses_and_motions()

        # Check that inference has filled out all the poses
        for idx in range(len(results)):
            self.assertIsNotNone(obj.results[idx].estimated_pose, msg="pose for frame {0} was None".format(idx))

        # self.assertEqual(Transform(), obj.results[0].estimated_pose)
        for idx in range(1, len(results)):
            # The motion should always be the relative pose of the current image relative to the previous one
            pose_based_motion = obj.results[idx - 1].estimated_pose.find_relative(obj.results[idx].estimated_pose)
            stored_motion = obj.results[idx].estimated_motion
            self.assertNPClose(pose_based_motion.location, stored_motion.location)
            self.assertNPClose(pose_based_motion.rotation_quat(), stored_motion.rotation_quat())

    def test_infers_estimated_poses_from_partial_estimated_motions(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)
        estimate_start = 5

        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (15, 1, 0),
                    (1, 0, 0, 0), w_first=True
                ) if idx > estimate_start else None,    # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        # Provide at least one absolute pose to use as a reference
        results[7].estimated_pose = Transform(
            (15 * 7, 7, 0),
            (1, 0, 0, 0), w_first=True
        )
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=True
        )
        for idx in range(estimate_start):   # We can infer back to the frame before the motion estimates start
            self.assertIsNone(obj.results[idx].estimated_pose)
        for idx in range(estimate_start, 10):
            self.assertNPEqual((15 * idx, idx, 0), obj.results[idx].estimated_pose.location)

    def test_ground_truth_scale_is_average_distance_from_first_estimated_pose(self):
        start = 3
        results = [
            FrameResult(
                timestamp=0.9 * idx,
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform() if idx >= start else None,
                tracking_state=TrackingState.OK
            )
            for idx in range(10)
        ]
        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=True
        )
        self.assertEqual(np.mean([
            np.linalg.norm([15 * (idx - start), (idx - start), 0])
            for idx in range(start + 1, 10)
        ]), obj.ground_truth_scale)

    def test_ground_truth_scale_is_average_movement_of_ground_truth(self):
        start = 3
        results = [
            FrameResult(
                timestamp=0.9 * idx,
                motion=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ) if idx > 0 else None,
                estimated_motion=Transform() if idx >= start else None,
                tracking_state=TrackingState.OK
            )
            for idx in range(10)
        ]
        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=True
        )
        self.assertEqual(np.mean([
            np.linalg.norm([15 * idx, idx, 0])
            for idx in range(start, 10)
        ]), obj.ground_truth_scale)

    def test_estimated_scale_is_average_distance_between_estimated_poses(self):
        estimate_start = 5

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (1.5 * idx, 0.1 * idx, 0),
                    (1, 0, 0, 0), w_first=True
                ) if idx >= estimate_start else None,  # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK
            )
            for idx in range(10)
        ]
        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=False
        )
        self.assertEqual(np.mean([
            np.linalg.norm([1.5 * (idx - estimate_start), 0.1 * (idx - estimate_start), 0])
            for idx in range(estimate_start + 1, 10)
        ]), obj.estimated_scale)

    def test_estimated_scale_is_average_translation_of_estimated_motions(self):
        estimate_start = 5

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (1.5, 0.1, 0),
                    (1, 0, 0, 0), w_first=True
                ) if idx > estimate_start else None,  # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK
            )
            for idx in range(10)
        ]
        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=False
        )
        self.assertEqual(np.sqrt(1.5 * 1.5 + 0.1 * 0.1), obj.estimated_scale)

    def test_get_scaled_motion_returns_None_for_indices_out_of_range(self):
        num_results = 10
        results = [
            FrameResult(
                timestamp=0.9 * idx,
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (1.5, 0.1, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK
            )
            for idx in range(num_results)
        ]
        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=True
        )
        self.assertIsNone(obj.get_scaled_motion(-1))
        self.assertIsNone(obj.get_scaled_motion(-30))
        self.assertIsNone(obj.get_scaled_motion(30 * num_results))

    def test_get_scaled_motion_returns_none_when_no_estimate_is_available(self):
        num_results = 10
        estimate_start = 5

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (1.5, 0.1, 0),
                    (1, 0, 0, 0), w_first=True
                ) if idx > estimate_start else None,  # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK
            )
            for idx in range(num_results)
        ]
        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=True
        )
        for idx in range(num_results):
            if idx <= estimate_start:
                self.assertIsNone(obj.get_scaled_motion(idx))
            else:
                self.assertIsInstance(obj.get_scaled_motion(idx), Transform)

        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=False
        )
        for idx in range(num_results):
            if idx <= estimate_start:
                self.assertIsNone(obj.get_scaled_motion(idx))
            else:
                self.assertIsInstance(obj.get_scaled_motion(idx), Transform)

    def test_get_scaled_motion_returns_the_estimated_motion_if_has_scale(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)
        num_results = 10
        estimate_start = 5

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (1.5, 0.1, 0),
                    tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / (2 * num_results)), w_first=True
                ) if idx > estimate_start else None,  # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx in range(num_results)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=True
        )
        for idx in range(num_results):
            if idx <= estimate_start:
                self.assertIsNone(obj.get_scaled_motion(idx))
            else:
                self.assertEqual(
                    Transform(
                        location=(1.5, 0.1, 0),
                        rotation=tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / (2 * num_results)),
                        w_first=True
                    ),
                    obj.get_scaled_motion(idx)
                )

    def test_get_scaled_motion_rescales_to_the_ground_truth_if_not_has_scale(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)
        num_results = 10
        estimate_start = 4

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (1.5, 0.1, 0),
                    tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / (2 * num_results)), w_first=True
                ) if idx > estimate_start else None,  # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx in range(num_results)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=False
        )
        for idx in range(num_results):
            if idx <= estimate_start:
                self.assertIsNone(obj.get_scaled_motion(idx))
            else:
                motion = obj.get_scaled_motion(idx)
                self.assertNPEqual((15, 1, 0), motion.location)
                self.assertNPEqual(
                    tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / (2 * num_results)),
                    motion.rotation_quat(True)
                )

    def test_get_scaled_motion_matches_scale_of_frames_with_motion_estimates(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)
        num_results = 10
        scale = 0.1
        random = np.random.RandomState(314159)
        motions = [random.normal(0, 10, 3) for _ in range(num_results)]
        mean_motion = np.mean([np.linalg.norm(motion) for motion in motions])

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                motion=Transform(
                    motion,
                    (1, 0, 0, 0), w_first=True
                ) if idx > 0 else None,
                # Only include small motions, which will throw off the scale
                estimated_motion=Transform(
                    scale * motion,
                    tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / (2 * num_results)), w_first=True
                ) if np.linalg.norm(motion) < mean_motion and idx > 0 else None,
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, motion in enumerate(motions)
        ]
        results[0].pose = Transform()
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=False
        )
        for idx in range(num_results):
            if idx <= 0 or np.linalg.norm(motions[idx]) >= mean_motion:
                self.assertIsNone(obj.get_scaled_motion(idx))
            else:
                motion = obj.get_scaled_motion(idx)
                self.assertNPClose(
                    motions[idx], motion.location,
                    rtol=0, atol=1e-14
                )
                self.assertNPEqual(
                    tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / (2 * num_results)),
                    motion.rotation_quat(True)
                )

    def test_get_scaled_pose_returns_None_for_indices_out_of_range(self):
        num_results = 10
        results = [
            FrameResult(
                timestamp=0.9 * idx,
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (1.5, 0.1, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK
            )
            for idx in range(num_results)
        ]
        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=True
        )
        self.assertIsNone(obj.get_scaled_pose(-1))
        self.assertIsNone(obj.get_scaled_pose(-30))
        self.assertIsNone(obj.get_scaled_pose(30 * num_results))

    def test_get_scaled_pose_returns_none_when_no_estimate_is_available(self):
        num_results = 10
        estimate_start = 5

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (1.5 * idx, 0.1 * idx, 0),
                    (1, 0, 0, 0), w_first=True
                ) if idx > estimate_start else None,  # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK
            )
            for idx in range(num_results)
        ]
        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=True
        )
        for idx in range(num_results):
            if idx <= estimate_start:
                self.assertIsNone(obj.get_scaled_pose(idx))
            else:
                self.assertIsInstance(obj.get_scaled_pose(idx), Transform)

        obj = SLAMTrialResult(
            success=True,
            results=results,
            has_scale=False
        )
        for idx in range(num_results):
            if idx <= estimate_start:
                self.assertIsNone(obj.get_scaled_pose(idx))
            else:
                self.assertIsInstance(obj.get_scaled_pose(idx), Transform)

    def test_get_scaled_pose_returns_the_estimated_pose_if_has_scale(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)
        num_results = 10
        estimate_start = 5

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (1.5 * idx, 0.1 * idx, 0),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / (2 * num_results)), w_first=True
                ) if idx > estimate_start else None,  # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx in range(num_results)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=True
        )
        for idx in range(num_results):
            if idx <= estimate_start:
                self.assertIsNone(obj.get_scaled_pose(idx))
            else:
                self.assertEqual(
                    Transform(
                        location=(1.5 * idx, 0.1 * idx, 0),
                        rotation=tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / (2 * num_results)),
                        w_first=True
                    ),
                    obj.get_scaled_pose(idx)
                )

    def test_get_scaled_pose_rescales_to_the_ground_truth_if_not_has_scale(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        mock_image = mock.create_autospec(Image)
        num_results = 10
        estimate_start = 4

        results = [
            FrameResult(
                timestamp=0.9 * idx,
                image=mock_image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (15 * idx, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (1.5 * idx, 0.1 * idx, 0),
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / (2 * num_results)), w_first=True
                ) if idx >= estimate_start else None,  # Motions start the frame after 'estimate_start'
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx in range(num_results)
        ]
        obj = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=results,
            has_scale=False
        )
        for idx in range(num_results):
            if idx < estimate_start:
                self.assertIsNone(obj.get_scaled_pose(idx))
            else:
                pose = obj.get_scaled_pose(idx)
                self.assertNPClose((15 * idx, idx, 0), pose.location, rtol=0, atol=1e15)
                self.assertNPEqual(
                    tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / (2 * num_results)),
                    pose.rotation_quat(True)
                )


class TestSLAMTrialResultDatabase(unittest.TestCase):
    system = None
    image_source = None
    image = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager()

        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.image = mock_types.make_image(1)
        cls.image.save()
        cls.system.save()
        cls.image_source.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        SLAMTrialResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        SLAMTrialResult._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()

    def test_stores_and_loads_motion_only(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ) if idx > 0 else None,
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        results[0].estimated_pose = Transform()
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            settings={'key': 'value'},
            results=results,
            run_time=10.4,
            has_scale=True
        )
        obj.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_motion_only_minimal(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ) if idx > 0 else None,
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results,
        )
        obj.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_motion_only_partial(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ) if idx >= 5 else None,
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results,
        )
        obj.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_pose_only(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            settings={'key': 'value'},
            run_time=10.4,
            results=results,
            has_scale=True
        )
        obj.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_pose_only_partial(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ) if idx >= 5 else None,
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )
        obj.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_pose_only_minimal(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )
        obj.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_no_estimate(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )
        obj.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_when_images_cannot_be_written(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ) if idx > 0 else None,
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        results[0].estimated_pose = Transform()
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            settings={'key': 'value'},
            results=results,
            run_time=10.4,
            has_scale=True
        )
        # Open the image source group read-only
        with im_manager.get().get_group(self.image_source.get_image_group(), allow_write=False):
            obj.save(cascade=True)

            # Load all the entities
            all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_required_fields_are_required(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]

        # No system
        obj = SLAMTrialResult(
            image_source=self.image_source,
            success=True,
            results=results
        )
        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('system', err_context.exception.message)

        # No image_source
        obj = SLAMTrialResult(
            system=self.system,
            success=True,
            results=results
        )
        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('image_source', err_context.exception.message)

        # No success
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            results=results
        )
        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('success', err_context.exception.message)

        # No results
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True
        )
        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('results', err_context.exception.message)

        # empty results
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=[]
        )
        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('results', err_context.exception.message)

    def test_is_invalid_when_timestamp_doesnt_increase(self):
        results = [
            FrameResult(
                timestamp=3,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx in range(10)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )
        with self.assertRaises(ValidationError):
            obj.save()

    def test_is_invalid_when_initial_motion_is_not_zero(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                motion=Transform(
                    (15 + np.random.normal(0, 1), np.random.normal(0, 0.1), 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )
        # Force the motion to be non-zero, without inference
        obj.results[0].motion = Transform((15, 1, 0))
        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('motion', err_context.exception.message)

    def test_is_invalid_when_pose_and_motion_dont_match(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                motion=Transform(   # This is definitely not the correct motion
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )
        with self.assertRaises(ValidationError):
            obj.save()

    def test_is_invalid_when_estimated_pose_and_estimated_motion_dont_match(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(   # This is definitely not the correct motion
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )
        with self.assertRaises(ValidationError):
            obj.save()

    def test_is_invalid_when_motion_is_none_and_pose_is_not(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )

        # Clear the true motion on a single result
        obj.results[5].motion = None

        with self.assertRaises(ValidationError):
            obj.save()

    def test_is_invalid_when_pose_can_be_inferred(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )

        # Clear the true pose on a single result
        obj.results[5].pose = None

        with self.assertRaises(ValidationError):
            obj.save()

    def test_is_invalid_when_first_motion_is_not_none(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=Transform(
                    (15, 1, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )

        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('estimated motion', err_context.exception.message)

    def test_is_invalid_when_estimated_pose_is_none_estimated_motion_is_not(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )

        # Clear the estimated pose after it is used to infer the motion
        del obj.results[5].estimated_pose

        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('5', err_context.exception.message)  # Make sure the frame with the error is mentioned

    def test_is_invalid_when_estimated_motion_is_none_estimated_pose_is_not(self):
        timestamps = [idx + np.random.normal(0, 0.01) for idx in range(10)]
        results = [
            FrameResult(
                timestamp=timestamp,
                image=self.image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0), w_first=True
                ),
                tracking_state=TrackingState.OK,
                loop_edges=[np.random.choice(timestamps)],
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, timestamp in enumerate(timestamps)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=results
        )

        # Clear the estimated motion after it is inferred
        del obj.results[5].estimated_motion

        with self.assertRaises(ValidationError) as err_context:
            obj.save()
        self.assertIn('5', err_context.exception.message)  # Make sure the frame with the error is mentioned
