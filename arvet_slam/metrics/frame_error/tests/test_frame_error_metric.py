import unittest
import unittest.mock as mock
import os
import numpy as np

import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager

from arvet.core.image import Image
from arvet.core.metric import MetricResult
from arvet.core.trial_result import TrialResult
import arvet.core.tests.mock_types as mock_types
from arvet.util.transform import Transform, compute_average_pose

from arvet_slam.metrics.frame_error.frame_error_metric import FrameErrorMetric, make_pose_error
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult


class TestFrameErrorMetricDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        FrameErrorMetric._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        FrameErrorMetric._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads(self):
        obj = FrameErrorMetric()
        obj.save()

        # Load all the entities
        all_entities = list(FrameErrorMetric.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_get_instance_returns_the_same_instance(self):
        metric1 = FrameErrorMetric.get_instance()
        metric2 = FrameErrorMetric.get_instance()
        self.assertIsNone(metric1.identifier)
        self.assertIsNone(metric2.identifier)

        metric1.save()
        metric3 = FrameErrorMetric.get_instance()
        self.assertIsNotNone(metric1.identifier)
        self.assertIsNotNone(metric3.identifier)
        self.assertEqual(metric1.identifier, metric3.identifier)

    def test_measure_results_returns_failed_result_that_can_be_saved(self):
        system1 = mock_types.MockSystem()
        system2 = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        system1.save()
        system2.save()
        image_source.save()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source, system=system1, success=True)
            for _ in range(3)
        ]
        group2 = [
            mock_types.MockTrialResult(image_source=image_source, system=system2, success=True)
            for _ in range(3)
        ]
        trial_results = group1 + group2
        for trial_result in trial_results:
            trial_result.save()

        metric = FrameErrorMetric.get_instance()
        metric.save()

        result = metric.measure_results(trial_results)
        result.full_clean()
        self.assertTrue(result.is_valid())
        result.save()

        all_entities = list(MetricResult.objects.all())
        self.assertEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], result)
        all_entities[0].delete()

    def test_measure_results_returns_successful_result_that_can_be_saved(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        system.save()
        image_source.save()

        images = [
            mock_types.make_image(idx)
            for idx in range(10)
        ]
        for image in images:
            image.save()

        frame_results = [
            FrameResult(
                timestamp=idx + np.random.normal(0, 0.01),
                image=image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, image in enumerate(images)
        ]
        trial_result = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=frame_results,
            has_scale=True
        )
        trial_result.save()

        metric = FrameErrorMetric.get_instance()
        metric.save()

        result = metric.measure_results([trial_result])
        self.assertTrue(result.is_valid())
        result.save()

        all_entities = list(MetricResult.objects.all())
        self.assertEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], result)
        all_entities[0].delete()


class TestFrameErrorMetric(unittest.TestCase):

    def test_returns_failed_metric_if_any_trials_have_failed(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source, system=system, success=True)
            for _ in range(3)
        ]
        group2 = [
            mock_types.MockTrialResult(image_source=image_source, system=system, success=False)
            for _ in range(1)
        ]

        metric = FrameErrorMetric()
        result = metric.measure_results(group1 + group2)

        self.assertFalse(result.success)
        self.assertGreater(len(result.message), 1)
        self.assertIn('failed', result.message)

    def test_returns_failed_metric_if_the_trials_are_from_different_systems(self):
        system1 = mock_types.MockSystem()
        system2 = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source, system=system1, success=True)
            for _ in range(3)
        ]
        group2 = [
            mock_types.MockTrialResult(image_source=image_source, system=system2, success=True)
            for _ in range(3)
        ]

        metric = FrameErrorMetric()
        result = metric.measure_results(group1 + group2)

        self.assertFalse(result.success)
        self.assertGreater(len(result.message), 1)
        self.assertIn('system', result.message)

    def test_returns_failed_metric_if_the_trials_are_from_different_image_sources(self):
        system = mock_types.MockSystem()
        image_source1 = mock_types.MockImageSource()
        image_source2 = mock_types.MockImageSource()

        group1 = [
            mock_types.MockTrialResult(image_source=image_source1, system=system, success=True)
            for _ in range(3)
        ]
        group2 = [
            mock_types.MockTrialResult(image_source=image_source2, system=system, success=True)
            for _ in range(3)
        ]

        metric = FrameErrorMetric()
        result = metric.measure_results(group1 + group2)

        self.assertFalse(result.success)
        self.assertGreater(len(result.message), 1)
        self.assertIn('image source', result.message)

    def test_stores_superset_of_all_image_properties_in_result(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        images = [
            mock.create_autospec(Image)  # Even though autospecs are slow, if it isn't an Image we can't store it
            for _ in range(10)
        ]
        for image in images:
            image.get_columns.return_value = set()
        images[0].get_columns.return_value = {'my_column_1'}
        images[1].get_columns.return_value = {'my_column_2'}

        frame_results = [
            FrameResult(
                timestamp=idx + np.random.normal(0, 0.01),
                image=image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, image in enumerate(images)
        ]
        trial_result = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=frame_results,
            has_scale=True
        )

        metric = FrameErrorMetric()
        result = metric.measure_results([trial_result])

        self.assertEqual({'my_column_1', 'my_column_2'}, set(result.image_columns))

    def test_measure_single_trial_with_all_properties(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        images = [
            mock.create_autospec(Image)  # Even though autospecs are slow, if it isn't an Image we can't store it
            for _ in range(10)
        ]
        for image in images:
            image.get_columns.return_value = set()

        frame_results = [
            FrameResult(
                timestamp=idx + np.random.normal(0, 0.01),
                image=image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, image in enumerate(images)
        ]
        trial_result = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=frame_results,
            has_scale=True
        )

        metric = FrameErrorMetric()
        result = metric.measure_results([trial_result])

        self.assertTrue(result.success)
        self.assertEqual(len(frame_results), len(result.errors))
        for idx, frame_error in enumerate(result.errors):
            frame_result = trial_result.results[idx]
            self.assertEqual(0, frame_error.repeat)
            self.assertEqual(frame_result.timestamp, frame_error.timestamp)
            self.assertEqual(frame_result.image, frame_error.image)
            self.assertEqual(TrackingState.OK, frame_error.tracking)
            self.assertEqual(frame_result.num_features, frame_error.num_features)
            self.assertEqual(frame_result.num_matches, frame_error.num_matches)

            self.assertErrorsEqual(make_pose_error(frame_result.estimated_pose, frame_result.pose),
                                   frame_error.absolute_error)
            self.assertIsNone(frame_error.noise)
            if idx <= 0:
                # No relative error on the first frame, because there is no motion
                self.assertIsNone(frame_error.relative_error)
            else:
                self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, frame_result.motion),
                                       frame_error.relative_error)

    def test_measure_single_trial_lost_part_way_through(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        images = [
            mock.create_autospec(Image)
            for _ in range(10)
        ]
        for image in images:
            image.get_columns.return_value = set()

        lost_start = 3
        lost_end = 7
        frame_results = [
            FrameResult(
                timestamp=idx + np.random.normal(0, 0.01),
                image=image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ) if not lost_start <= idx < lost_end else None,
                tracking_state=TrackingState.OK if not lost_start <= idx < lost_end else TrackingState.LOST,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, image in enumerate(images)
        ]
        trial_result = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            results=frame_results,
            has_scale=True
        )

        metric = FrameErrorMetric()
        result = metric.measure_results([trial_result])

        self.assertTrue(result.success)
        self.assertEqual(len(frame_results), len(result.errors))
        for idx, frame_error in enumerate(result.errors):
            frame_result = frame_results[idx]
            self.assertEqual(0, frame_error.repeat)
            self.assertEqual(frame_result.timestamp, frame_error.timestamp)
            self.assertEqual(frame_result.image, frame_error.image)
            self.assertEqual(frame_result.tracking_state, frame_error.tracking)
            self.assertIsNone(frame_error.noise)
            self.assertEqual(frame_result.num_features, frame_error.num_features)
            self.assertEqual(frame_result.num_matches, frame_error.num_matches)

            if idx == 0 or idx == lost_end:
                # the first frame has estimated pose, but no estimated motion
                self.assertEqual(make_pose_error(frame_result.estimated_pose, frame_result.pose),
                                 frame_error.absolute_error)
                self.assertIsNone(frame_error.relative_error)
            elif lost_start <= idx < lost_end:
                # Lost, expect errors to be None
                self.assertIsNone(frame_error.absolute_error)
                self.assertIsNone(frame_error.relative_error)
            else:
                self.assertErrorsEqual(make_pose_error(frame_result.estimated_pose, frame_result.pose),
                                       frame_error.absolute_error)
                self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, frame_result.motion),
                                       frame_error.relative_error)

    def test_measure_multiple_trials_with_all_properties(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        repeats = 3

        images = [
            mock.create_autospec(Image)  # Even though autospecs are slow, if it isn't an Image we can't store it
            for _ in range(10)
        ]
        for image in images:
            image.get_columns.return_value = set()

        estimated_motions = [
            [
                Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ) for _ in range(repeats)
            ] for _ in range(len(images))
        ]
        average_motions = [compute_average_pose(image_motions) for image_motions in estimated_motions]

        trial_results = []
        for repeat in range(repeats):
            frame_results = [
                FrameResult(
                    timestamp=idx + np.random.normal(0, 0.01),
                    image=image,
                    processing_time=np.random.uniform(0.01, 1),
                    pose=Transform(
                        (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                        (1, 0, 0, 0)
                    ),
                    estimated_motion=estimated_motions[idx][repeat],
                    tracking_state=TrackingState.OK,
                    num_features=np.random.randint(10, 1000),
                    num_matches=np.random.randint(10, 1000)
                )
                for idx, image in enumerate(images)
            ]
            frame_results[0].estimated_pose = Transform()
            trial_result = SLAMTrialResult(
                system=system,
                image_source=image_source,
                success=True,
                results=frame_results,
                has_scale=True
            )
            trial_results.append(trial_result)

        metric = FrameErrorMetric()
        result = metric.measure_results(trial_results)

        self.assertTrue(result.success)
        self.assertEqual(repeats * len(images), len(result.errors))
        for repeat in range(repeats):
            for idx in range(len(images)):
                frame_error = result.errors[repeat * len(images) + idx]
                frame_result = trial_results[repeat].results[idx]
                self.assertEqual(repeat, frame_error.repeat)
                self.assertEqual(frame_result.timestamp, frame_error.timestamp)
                self.assertEqual(frame_result.image, frame_error.image)
                self.assertEqual(TrackingState.OK, frame_error.tracking)
                self.assertEqual(frame_result.num_features, frame_error.num_features)
                self.assertEqual(frame_result.num_matches, frame_error.num_matches)
                self.assertErrorsEqual(make_pose_error(frame_result.estimated_pose, frame_result.pose),
                                       frame_error.absolute_error)
                self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, frame_result.motion),
                                       frame_error.relative_error)
                self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, average_motions[idx]),
                                       frame_error.noise)

    def assertErrorsEqual(self, pose_error_1, pose_error_2):
        self.assertEqual(pose_error_1.x, pose_error_2.x)
        self.assertEqual(pose_error_1.y, pose_error_2.y)
        self.assertEqual(pose_error_1.z, pose_error_2.z)
        self.assertEqual(pose_error_1.length, pose_error_2.length)
        self.assertEqual(pose_error_1.rot, pose_error_2.rot)
        # Special handling for direction, which might be NaN
        if np.isnan(pose_error_1.direction):
            self.assertTrue(np.isnan(pose_error_2.direction))
        else:
            self.assertEqual(pose_error_1.direction, pose_error_2.direction)

    def assertErrorsClose(self, pose_error_1, pose_error_2):
        self.assertAlmostEqual(pose_error_1.x, pose_error_2.x)
        self.assertAlmostEqual(pose_error_1.y, pose_error_2.y)
        self.assertAlmostEqual(pose_error_1.z, pose_error_2.z)
        self.assertAlmostEqual(pose_error_1.length, pose_error_2.length)
        self.assertAlmostEqual(pose_error_1.rot, pose_error_2.rot)
        # Special handling for direction, which might be NaN
        if np.isnan(pose_error_1.direction):
            self.assertTrue(np.isnan(pose_error_2.direction))
        else:
            self.assertAlmostEqual(pose_error_1.direction, pose_error_2.direction)
