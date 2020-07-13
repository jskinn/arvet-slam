import unittest
import unittest.mock as mock
import bson
import itertools
import numpy as np
import transforms3d as tf3d

import arvet.database.tests.database_connection as dbconn

from arvet.core.image import Image
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.metric import MetricResult
from arvet.core.trial_result import TrialResult
import arvet.core.tests.mock_types as mock_types
from arvet.util.transform import Transform, compute_average_pose
from arvet.util.test_helpers import ExtendedTestCase

from arvet_slam.metrics.frame_error.frame_error_metric import FrameErrorMetric, make_pose_error, \
    align_trajectory_to_ground_truth
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult


class TestFrameErrorMetricDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager()

    def setUp(self):
        # Clear temporary data from the test database before each test, to prevent cross-contamination
        FrameErrorMetric.objects.all().delete()
        MetricResult.objects.all().delete()

    def tearDown(self) -> None:
        # Clear temporary data from the test database before each test, to prevent cross-contamination
        FrameErrorMetric.objects.all().delete()
        MetricResult.objects.all().delete()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        FrameErrorMetric._mongometa.collection.drop()
        MetricResult._mongometa.collection.drop()
        TrialResult._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()

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

        # Clean up
        metric.delete()
        system1.delete()
        system2.delete()
        image_source.delete()

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
        # self.assertTrue(result.is_valid())
        result.save(cascade=True)

        all_entities = list(MetricResult.objects.all())
        self.assertEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], result)
        all_entities[0].delete()

        # Clean up
        metric.delete()
        system.delete()
        image_source.delete()


class TestFrameErrorMetric(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        dbconn.setup_image_manager()

    @classmethod
    def tearDownClass(cls) -> None:
        dbconn.tear_down_image_manager()

    def test_only_slam_trial_results_are_appropriate(self):
        metric = FrameErrorMetric()
        self.assertTrue(metric.is_trial_appropriate(mock.create_autospec(SLAMTrialResult)))
        self.assertFalse(metric.is_trial_appropriate(mock_types.MockTrialResult()))

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_failed_metric_if_given_no_trials(self, _):
        metric = FrameErrorMetric()
        result = metric.measure_results([])

        self.assertFalse(result.success)
        self.assertGreater(len(result.message), 1)
        self.assertIn('zero', result.message)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_failed_metric_if_any_trials_have_failed(self, _):
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

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_failed_metric_if_the_trials_are_from_different_systems(self, _):
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

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_failed_metric_if_the_trials_are_from_different_image_sources(self, _):
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

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_failed_metric_if_the_trials_have_different_numbers_of_results(self, _):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()

        group1 = [
            SLAMTrialResult(image_source=image_source, system=system, success=True,
                            results=[FrameResult(timestamp=frame_idx) for frame_idx in range(5)])
            for _ in range(3)
        ]
        group2 = [
            SLAMTrialResult(image_source=image_source, system=system, success=True,
                            results=[FrameResult(timestamp=frame_idx) for frame_idx in range(6)])
            for _ in range(3)
        ]

        metric = FrameErrorMetric()
        result = metric.measure_results(group1 + group2)

        self.assertFalse(result.success)
        self.assertGreater(len(result.message), 1)
        self.assertIn('number of frames', result.message)
        self.assertIn('5', result.message)
        self.assertIn('6', result.message)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_autoloads_image_sources_and_systems(self, mock_autoload):
        system = mock_types.MockSystem(_id=bson.ObjectId())
        image_source = mock_types.MockImageSource(_id=bson.ObjectId())

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
        metric.measure_results([trial_result])

        self.assertTrue(mock_autoload.called)
        # Unfortunately, we can't force a state where the references are ids and not objects here
        self.assertIn(mock.call(VisionSystem, []), mock_autoload.call_args_list)
        self.assertIn(mock.call(ImageSource, []), mock_autoload.call_args_list)


class TestFrameErrorMetricOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        dbconn.setup_image_manager()

    @classmethod
    def tearDownClass(cls) -> None:
        dbconn.tear_down_image_manager()

    def setUp(self) -> None:
        self.system = mock_types.MockSystem()
        self.image_source = mock_types.MockImageSource()

        self.images = [
            mock.create_autospec(Image)  # Even though autospecs are slow, if it isn't an Image we can't store it
            for _ in range(10)
        ]
        for image in self.images:
            image.get_columns.return_value = set()

    def make_frame_results_from_poses(self, estimated_poses):
        frame_results = [
            FrameResult(
                timestamp=idx + np.random.normal(0, 0.01),
                image=image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_pose=estimated_poses[idx],
                tracking_state=TrackingState.OK if estimated_poses[idx] is not None else TrackingState.LOST,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000),
                loop_edges=[]
            )
            for idx, image in enumerate(self.images)
        ]
        # Infer loop closures when an estimate returns from being lost (but not when first initialised)
        last_found_timestamp = None
        was_lost_prev = False
        for idx in range(len(frame_results)):
            if estimated_poses[idx] is not None:
                if was_lost_prev and last_found_timestamp is not None:
                    frame_results[idx].loop_edges.append(last_found_timestamp)
                last_found_timestamp = frame_results[idx].timestamp
                was_lost_prev = False
            else:
                was_lost_prev = True
        return frame_results

    def make_frame_results_from_motions(self, estimated_motions, set_initial_pose=False):
        results = [
            FrameResult(
                timestamp=idx + np.random.normal(0, 0.01),
                image=image,
                processing_time=np.random.uniform(0.01, 1),
                pose=Transform(
                    (idx * 15, idx, 0),
                    (1, 0, 0, 0), w_first=True
                ),
                estimated_motion=estimated_motions[idx] if idx > 0 else None,
                tracking_state=TrackingState.OK if estimated_motions[idx] is not None else TrackingState.LOST,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx, image in enumerate(self.images)
        ]
        # Optionally set an origin for the first pose
        if isinstance(set_initial_pose, Transform):
            results[0].estimated_pose = set_initial_pose
        elif bool(set_initial_pose):
            results[0].estimated_pose = Transform()

        # Infer loop closures when the motion stops being none
        last_found_timestamp = None
        was_lost_prev = False
        for idx in range(len(results)):
            if results[idx].estimated_motion is not None:
                if was_lost_prev and last_found_timestamp is not None:
                    results[idx].loop_edges.append(last_found_timestamp)
                last_found_timestamp = results[idx].timestamp
                was_lost_prev = False
            else:
                was_lost_prev = True
        return results

    def make_trial(self, frame_results, has_scale=True):
        return SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            results=frame_results,
            has_scale=bool(has_scale)
        )

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_measure_single_trial(self, _):
        frame_results = self.make_frame_results_from_poses([
            Transform(location=(idx * 15.1, 1.01 * idx, 0))
            for idx in range(len(self.images))
        ])
        trial_result = self.make_trial(frame_results)

        metric = FrameErrorMetric()
        result = metric.measure_results([trial_result])

        self.assertTrue(result.success)
        self.assertEqual(1, len(result.errors))

        # Trial errors, mostly from tracking information
        trial_error = result.errors[0]
        self.assertEqual([], trial_error.frames_lost)
        self.assertEqual([len(self.images)], trial_error.frames_found)
        self.assertEqual([], trial_error.times_lost)
        self.assertEqual([
            frame_results[-1].timestamp - frame_results[0].timestamp
        ], trial_error.times_found)
        self.assertEqual([], trial_error.distances_lost)
        self.assertEqual([
            sum(np.linalg.norm(frame_result.motion.location) for frame_result in frame_results)
        ], trial_error.distances_found)

        # Per-frame errors
        self.assertEqual(len(frame_results), len(trial_error.frame_errors))
        for idx, frame_error in enumerate(trial_error.frame_errors):
            frame_result = trial_result.results[idx]
            self.assertEqual(0, frame_error.repeat)
            self.assertEqual(frame_result.timestamp, frame_error.timestamp)
            self.assertEqual(frame_result.image, frame_error.image)
            self.assertEqual(frame_result.tracking_state, frame_error.tracking)
            self.assertEqual(frame_result.num_features, frame_error.num_features)
            self.assertEqual(frame_result.num_matches, frame_error.num_matches)
            self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
            self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))
            self.assertIsNone(frame_error.noise)
            self.assertIsNone(frame_error.systemic_error)

            self.assertAlmostEqual(0.1 * idx, frame_error.absolute_error.x, places=13)
            self.assertAlmostEqual(0.01 * idx, frame_error.absolute_error.y, places=13)
            self.assertEqual(0, frame_error.absolute_error.z)
            self.assertAlmostEqual(idx * np.sqrt(0.1 * 0.1 + 0.01 * 0.01), frame_error.absolute_error.length, places=13)
            self.assertEqual(0, frame_error.absolute_error.rot)

            if idx == 0:
                self.assertIsNone(frame_error.relative_error)
            else:

                self.assertAlmostEqual(0.1, frame_error.relative_error.x, places=13)
                self.assertAlmostEqual(0.01, frame_error.relative_error.y, places=13)
                self.assertEqual(0, frame_error.relative_error.z)
                self.assertAlmostEqual(np.sqrt(0.1 * 0.1 + 0.01 * 0.01), frame_error.relative_error.length, places=13)
                self.assertEqual(0, frame_error.relative_error.rot)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_measure_single_trial_lost_at_the_beginning(self, _):
        start = 3
        frame_results = self.make_frame_results_from_poses([
            Transform(location=(idx * 15.1, 1.01 * idx, 0)) if idx >= start else None
            for idx in range(len(self.images))
        ])
        trial_result = self.make_trial(frame_results)

        metric = FrameErrorMetric()
        result = metric.measure_results([trial_result])

        self.assertTrue(result.success)
        self.assertEqual(1, len(result.errors))

        # Tracking information
        trial_error = result.errors[0]
        self.assertEqual([start], trial_error.frames_lost)
        self.assertEqual([len(self.images) - start], trial_error.frames_found)
        self.assertEqual([
            frame_results[start - 1].timestamp - frame_results[0].timestamp
        ], trial_error.times_lost)
        self.assertEqual([
            frame_results[-1].timestamp - frame_results[start - 1].timestamp
        ], trial_error.times_found)
        self.assertEqual([
            sum(np.linalg.norm(frame_result.motion.location) for frame_result in frame_results[:start])
        ], trial_error.distances_lost)
        self.assertEqual([
            sum(np.linalg.norm(frame_result.motion.location) for frame_result in frame_results[start:])
        ], trial_error.distances_found)

        # Sanity check the tracking information, the totals should be right
        self.assertEqual(len(self.images), sum(itertools.chain(trial_error.frames_lost, trial_error.frames_found)))
        self.assertAlmostEqual(frame_results[-1].timestamp - frame_results[0].timestamp,
                               sum(itertools.chain(trial_error.times_lost, trial_error.times_found)), places=14)
        self.assertEqual(np.linalg.norm(frame_results[-1].pose.location - frame_results[0].pose.location),
                         sum(itertools.chain(trial_error.distances_lost, trial_error.distances_found)))

        # Frame errors
        self.assertEqual(len(frame_results), len(trial_error.frame_errors))
        for idx, frame_error in enumerate(trial_error.frame_errors):
            frame_result = trial_result.results[idx]
            self.assertEqual(0, frame_error.repeat)
            self.assertEqual(frame_result.timestamp, frame_error.timestamp)
            self.assertEqual(frame_result.image, frame_error.image)
            self.assertEqual(frame_result.tracking_state, frame_error.tracking)
            self.assertEqual(frame_result.num_features, frame_error.num_features)
            self.assertEqual(frame_result.num_matches, frame_error.num_matches)
            self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
            self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))
            self.assertIsNone(frame_error.noise)
            self.assertIsNone(frame_error.systemic_error)

            # Absolute error should be 0 when both estimates appear, and grow after that
            if idx < start:
                self.assertIsNone(frame_error.absolute_error)
            else:
                self.assertAlmostEqual(0.1 * (idx - start), frame_error.absolute_error.x)
                self.assertAlmostEqual(0.01 * (idx - start), frame_error.absolute_error.y)
                self.assertEqual(0, frame_error.absolute_error.z)
                self.assertAlmostEqual((idx - start) * np.sqrt(0.1 * 0.1 + 0.01 * 0.01),
                                       frame_error.absolute_error.length, places=13)
                self.assertEqual(0, frame_error.absolute_error.rot)

            # Relative error should be undefined at and before the start frame, and constant after
            if idx <= start:
                self.assertIsNone(frame_error.relative_error)
            else:
                self.assertAlmostEqual(0.1, frame_error.relative_error.x, places=13)
                self.assertAlmostEqual(0.01, frame_error.relative_error.y, places=13)
                self.assertEqual(0, frame_error.relative_error.z)
                self.assertAlmostEqual(np.sqrt(0.1 * 0.1 + 0.01 * 0.01), frame_error.relative_error.length, places=13)
                self.assertEqual(0, frame_error.relative_error.rot)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_measure_single_trial_lost_part_way_through(self, _):
        lost_start = 3
        lost_end = 7
        frame_results = self.make_frame_results_from_poses([
            Transform(
                location=(idx * 15.1, 1.1 * idx, 0)
            ) if not lost_start <= idx < lost_end else None
            for idx in range(len(self.images))
        ])
        trial_result = self.make_trial(frame_results)

        metric = FrameErrorMetric()
        result = metric.measure_results([trial_result])

        self.assertTrue(result.success)
        self.assertEqual(1, len(result.errors))

        # Tracking information
        trial_error = result.errors[0]
        self.assertEqual([lost_end - lost_start], trial_error.frames_lost)
        self.assertEqual([lost_start, len(self.images) - lost_end], trial_error.frames_found)
        self.assertEqual([
            frame_results[lost_end - 1].timestamp - frame_results[lost_start - 1].timestamp
        ], trial_error.times_lost)
        self.assertEqual([
            frame_results[lost_start - 1].timestamp - frame_results[0].timestamp,
            frame_results[-1].timestamp - frame_results[lost_end - 1].timestamp
        ], trial_error.times_found)
        self.assertEqual([
            sum(np.linalg.norm(frame_result.motion.location) for frame_result in frame_results[lost_start:lost_end])
        ], trial_error.distances_lost)
        self.assertEqual([
            sum(np.linalg.norm(frame_result.motion.location) for frame_result in frame_results[:lost_start]),
            sum(np.linalg.norm(frame_result.motion.location) for frame_result in frame_results[lost_end:])
        ], trial_error.distances_found)

        # Sanity check the tracking information, the totals should be right
        self.assertEqual(len(self.images), sum(itertools.chain(trial_error.frames_lost, trial_error.frames_found)))
        self.assertAlmostEqual(frame_results[-1].timestamp - frame_results[0].timestamp,
                               sum(itertools.chain(trial_error.times_lost, trial_error.times_found)), places=14)
        self.assertEqual(np.linalg.norm(frame_results[-1].pose.location - frame_results[0].pose.location),
                         sum(itertools.chain(trial_error.distances_lost, trial_error.distances_found)))

        self.assertEqual(len(frame_results), len(trial_error.frame_errors))
        for idx, frame_error in enumerate(trial_error.frame_errors):
            frame_result = frame_results[idx]
            self.assertEqual(0, frame_error.repeat)
            self.assertEqual(frame_result.timestamp, frame_error.timestamp)
            self.assertEqual(frame_result.image, frame_error.image)
            self.assertEqual(frame_result.tracking_state, frame_error.tracking)
            self.assertIsNone(frame_error.noise)
            self.assertIsNone(frame_error.systemic_error)
            self.assertEqual(frame_result.num_features, frame_error.num_features)
            self.assertEqual(frame_result.num_matches, frame_error.num_matches)

            if idx == 0 or idx == lost_end:
                # the first frame has estimated pose, but no estimated motion
                self.assertErrorsEqual(make_pose_error(frame_result.estimated_pose, frame_result.pose),
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

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_measure_multiple_trials(self, _):
        repeats = 3

        estimated_motions = [
            [
                Transform(
                    location=(np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1))
                ) for _ in range(repeats)
            ] for _ in range(len(self.images))
        ]
        average_motions = [compute_average_pose(image_motions) for image_motions in estimated_motions]

        trial_results = []
        expected_times_found = [[] for _ in range(repeats)]
        expected_distance_found = [[] for _ in range(repeats)]
        for repeat in range(repeats):
            frame_results = self.make_frame_results_from_motions([
                estimated_motions[idx][repeat]
                for idx in range(len(self.images))
            ], set_initial_pose=True)
            trial_results.append(self.make_trial(frame_results))
            expected_times_found[repeat] = [frame_results[-1].timestamp - frame_results[0].timestamp]
            expected_distance_found[repeat] = [
                sum(np.linalg.norm(frame_result.motion.location) for frame_result in frame_results)]

        metric = FrameErrorMetric()
        result = metric.measure_results(trial_results)

        self.assertTrue(result.success)
        self.assertEqual(repeats, len(result.errors))
        for repeat in range(repeats):
            trial_errors = result.errors[repeat]

            # Tracking information
            self.assertEqual([], trial_errors.frames_lost)
            self.assertEqual([len(self.images)], trial_errors.frames_found)
            self.assertEqual([], trial_errors.times_lost)
            self.assertEqual(expected_times_found[repeat], trial_errors.times_found)
            self.assertEqual([], trial_errors.distances_lost)
            self.assertEqual(expected_distance_found[repeat], trial_errors.distances_found)

            self.assertEqual(len(self.images), len(trial_errors.frame_errors))
            for idx in range(len(self.images)):
                frame_error = trial_errors.frame_errors[idx]
                frame_result = trial_results[repeat].results[idx]
                self.assertEqual(repeat, frame_error.repeat)
                self.assertEqual(frame_result.timestamp, frame_error.timestamp)
                self.assertEqual(frame_result.image, frame_error.image)
                self.assertEqual(TrackingState.OK, frame_error.tracking)
                self.assertEqual(frame_result.num_features, frame_error.num_features)
                self.assertEqual(frame_result.num_matches, frame_error.num_matches)
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))
                self.assertErrorsEqual(make_pose_error(frame_result.estimated_pose, frame_result.pose),
                                       frame_error.absolute_error)
                if idx == 0:
                    self.assertIsNone(frame_error.relative_error)
                    self.assertIsNone(frame_error.noise)
                    self.assertIsNone(frame_error.systemic_error)
                else:
                    self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, frame_result.motion),
                                           frame_error.relative_error)
                    self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, average_motions[idx]),
                                           frame_error.noise)
                    self.assertErrorsEqual(make_pose_error(average_motions[idx], frame_result.motion),
                                           frame_error.systemic_error)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_measure_multiple_trials_lost_at_different_times(self, _):
        lost_start = 3
        lost_end = 6
        repeats = 3

        estimated_motions = [
            [
                Transform(
                    location=(np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1))
                ) if not lost_start + repeat <= idx < lost_end + repeat else None
                for repeat in range(repeats)
            ] for idx in range(len(self.images))
        ]
        average_motions = [
            compute_average_pose(motion for motion in image_motions if motion is not None)
            if len(image_motions) - image_motions.count(None) > 1 else None
            for image_motions in estimated_motions
        ]

        trial_results = []
        for repeat in range(repeats):
            frame_results = self.make_frame_results_from_motions([
                estimated_motions[idx][repeat]
                for idx in range(len(self.images))
            ], set_initial_pose=True)
            trial_results.append(self.make_trial(frame_results))

        metric = FrameErrorMetric()
        result = metric.measure_results(trial_results)

        self.assertTrue(result.success)
        self.assertEqual(repeats, len(result.errors))
        for repeat in range(repeats):
            trial_errors = result.errors[repeat]
            frame_results = trial_results[repeat].results
            self.assertEqual([lost_end - lost_start], trial_errors.frames_lost)
            self.assertEqual([lost_start + repeat, len(self.images) - lost_end - repeat], trial_errors.frames_found)
            self.assertEqual([
                frame_results[lost_end + repeat - 1].timestamp - frame_results[lost_start + repeat - 1].timestamp
            ], trial_errors.times_lost)
            self.assertEqual([
                frame_results[lost_start + repeat - 1].timestamp - frame_results[0].timestamp,
                frame_results[-1].timestamp - frame_results[lost_end + repeat - 1].timestamp
            ], trial_errors.times_found)
            self.assertEqual([sum(
                np.linalg.norm(frame_result.motion.location)
                for frame_result in frame_results[lost_start + repeat:lost_end + repeat]
            )], trial_errors.distances_lost)
            self.assertEqual([
                sum(np.linalg.norm(frame_result.motion.location) for frame_result in
                    frame_results[:lost_start + repeat]),
                sum(np.linalg.norm(frame_result.motion.location) for frame_result in frame_results[lost_end + repeat:])
            ], trial_errors.distances_found)

            # Sanity check the tracking information, the totals should be right
            self.assertEqual(
                len(self.images), sum(itertools.chain(trial_errors.frames_lost, trial_errors.frames_found)))
            self.assertAlmostEqual(
                frame_results[-1].timestamp - frame_results[0].timestamp,
                sum(itertools.chain(trial_errors.times_lost, trial_errors.times_found)),
                places=14
            )
            self.assertEqual(np.linalg.norm(frame_results[-1].pose.location - frame_results[0].pose.location),
                             sum(itertools.chain(trial_errors.distances_lost, trial_errors.distances_found)))

            self.assertEqual(len(self.images), len(trial_errors.frame_errors))
            for idx in range(len(self.images)):
                frame_error = trial_errors.frame_errors[idx]
                frame_result = trial_results[repeat].results[idx]
                self.assertEqual(repeat, frame_error.repeat)
                self.assertEqual(frame_result.timestamp, frame_error.timestamp)
                self.assertEqual(frame_result.image, frame_error.image)
                self.assertEqual(frame_result.tracking_state, frame_error.tracking)
                self.assertEqual(frame_result.num_features, frame_error.num_features)
                self.assertEqual(frame_result.num_matches, frame_error.num_matches)
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))

                # Absolute error should be none after it becomes lost, since we don't have a reference to bring it back
                if lost_start + repeat <= idx:
                    self.assertIsNone(frame_error.absolute_error)
                else:
                    self.assertErrorsEqual(make_pose_error(frame_result.estimated_pose, frame_result.pose),
                                           frame_error.absolute_error)
                # relative error should be none for the lost frames
                if idx == 0 or lost_start + repeat <= idx < lost_end + repeat:
                    self.assertIsNone(frame_error.relative_error)
                else:
                    self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, frame_result.motion),
                                           frame_error.relative_error)
                # Noise should be none when we are lost or when there is less than two estimates
                if idx == 0 or lost_start + repeat <= idx < lost_end + repeat or \
                        sum(not lost_start + ridx <= idx < lost_end + ridx for ridx in range(repeats)) <= 1:
                    self.assertIsNone(frame_error.noise)
                else:
                    self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, average_motions[idx]),
                                           frame_error.noise)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_zero_error_or_noise_for_misaligned_trajectories(self, _):
        repeats = 3

        # The mis-aligned origin of the ground truth poses. Shouldn't affect the error
        gt_offset = Transform(
            (18, 66, -33),
            tf3d.quaternions.axangle2quat((5, -3, -5), 8 * np.pi / 17)
        )
        estimate_offsets = [
            Transform(
                np.random.uniform(-10, 10, size=3),
                tf3d.quaternions.axangle2quat(np.random.uniform(-10, 10, size=3),  np.random.uniform(-np.pi, np.pi))
            )
            for _ in range(repeats)
        ]

        trial_results = []
        for repeat in range(repeats):
            frame_results = [
                FrameResult(
                    timestamp=idx,
                    image=image,
                    processing_time=np.random.uniform(0.01, 1),
                    pose=gt_offset.find_independent(Transform(
                        (idx * 15, idx, 0),
                        tf3d.quaternions.axangle2quat((-2, 6, -1), idx * np.pi / 27), w_first=True
                    )),
                    estimated_pose=estimate_offsets[repeat].find_independent(Transform(
                        (idx * 15, idx, 0),
                        tf3d.quaternions.axangle2quat((-2, 6, -1), idx * np.pi / 27), w_first=True
                    )),
                    tracking_state=TrackingState.OK
                )
                for idx, image in enumerate(self.images)
            ]
            trial_results.append(self.make_trial(frame_results))

        metric = FrameErrorMetric()
        result = metric.measure_results(trial_results)

        self.assertTrue(result.success)
        self.assertEqual(repeats, len(result.errors))
        for repeat in range(repeats):
            self.assertEqual(len(self.images), len(result.errors[repeat].frame_errors))
            for idx in range(len(self.images)):
                frame_error = result.errors[repeat].frame_errors[idx]
                frame_result = trial_results[repeat].results[idx]
                self.assertEqual(repeat, frame_error.repeat)
                self.assertEqual(frame_result.timestamp, frame_error.timestamp)
                self.assertEqual(frame_result.image, frame_error.image)
                self.assertEqual(TrackingState.OK, frame_error.tracking)
                self.assertEqual(frame_result.num_features, frame_error.num_features)
                self.assertEqual(frame_result.num_matches, frame_error.num_matches)
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))

                # Errors should all be close to zero.
                # We need more leeway on the rotation error, because it is fed through an arccos,
                # So a tiny change away from 1 makes it large quickly
                self.assertErrorIsAlmostZero(frame_error.absolute_error, places=12, rot_places=7)
                if idx <= 0:
                    # No relative error or noise on the first frame, because there is no motion
                    self.assertIsNone(frame_error.relative_error)
                    self.assertIsNone(frame_error.noise)
                    self.assertIsNone(frame_error.systemic_error)
                else:
                    self.assertErrorIsAlmostZero(frame_error.relative_error, places=12, rot_places=7)
                    self.assertErrorIsAlmostZero(frame_error.noise, rot_places=7)
                    self.assertErrorIsAlmostZero(frame_error.systemic_error, rot_places=7)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_zero_error_or_noise_for_misaligned_partial_trajectories(self, _):
        repeats = 3

        # The mis-aligned origin of the ground truth poses. Shouldn't matter
        gt_offset = Transform(
            (18, 66, -33),
            tf3d.quaternions.axangle2quat((5, -3, -5), 8 * np.pi / 17)
        )
        estimate_offsets = [
            Transform(
                np.random.uniform(-10, 10, size=3),
                tf3d.quaternions.axangle2quat(np.random.uniform(-10, 10, size=3),  np.random.uniform(-np.pi, np.pi))
            )
            for _ in range(repeats)
        ]

        trial_results = []
        for repeat in range(repeats):
            frame_results = [
                FrameResult(
                    timestamp=idx,
                    image=image,
                    processing_time=np.random.uniform(0.01, 1),
                    pose=gt_offset.find_relative(Transform(
                        (idx * 15, idx, 0),
                        tf3d.quaternions.axangle2quat((-2, 6, -1), idx * np.pi / 27), w_first=True
                    )),
                    estimated_pose=estimate_offsets[repeat].find_relative(Transform(
                        (idx * 15, idx, 0),
                        tf3d.quaternions.axangle2quat((-2, 6, -1), idx * np.pi / 27), w_first=True
                    )) if idx % 4 != 3 else None,   # 2 holes, at 4 and 8 making 3 different segments
                    tracking_state=TrackingState.OK,
                    num_features=np.random.randint(10, 1000),
                    num_matches=np.random.randint(10, 1000)
                )
                for idx, image in enumerate(self.images)
            ]
            trial_results.append(self.make_trial(frame_results))

        metric = FrameErrorMetric()
        result = metric.measure_results(trial_results)

        self.assertTrue(result.success)
        self.assertEqual(repeats, len(result.errors))
        for repeat in range(repeats):
            self.assertEqual(len(self.images), len(result.errors[repeat].frame_errors))
            for idx in range(len(self.images)):
                frame_error = result.errors[repeat].frame_errors[idx]
                frame_result = trial_results[repeat].results[idx]
                self.assertEqual(repeat, frame_error.repeat)
                self.assertEqual(frame_result.timestamp, frame_error.timestamp)
                self.assertEqual(frame_result.image, frame_error.image)
                self.assertEqual(TrackingState.OK, frame_error.tracking)
                self.assertEqual(frame_result.num_features, frame_error.num_features)
                self.assertEqual(frame_result.num_matches, frame_error.num_matches)
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))

                # Errors should all be zero, when
                if idx % 4 == 3:
                    self.assertIsNone(frame_error.absolute_error)
                else:
                    self.assertErrorIsAlmostZero(frame_error.absolute_error, places=12, rot_places=6)

                if idx <= 0 or idx % 4 == 3 or idx % 4 == 0:
                    # No relative error on the first frame, because there is no motion
                    self.assertIsNone(frame_error.relative_error)
                    self.assertIsNone(frame_error.noise)
                    self.assertIsNone(frame_error.systemic_error)
                else:
                    self.assertErrorIsAlmostZero(frame_error.relative_error, places=12, rot_places=6)
                    self.assertErrorIsAlmostZero(frame_error.noise, places=12, rot_places=6)
                    self.assertErrorIsAlmostZero(frame_error.systemic_error, places=12, rot_places=6)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_zero_error_or_noise_for_misscaled_trajectory_that_doesnt_have_scale(self, _):
        repeats = 3

        # Make 3 trials, each with a different scale, they should be rescaled to correct
        trial_results = []
        for repeat in range(repeats):
            scale = 5 * (repeat + 1) / 8
            frame_results = self.make_frame_results_from_poses([
                Transform(location=(scale * idx * 15, scale * idx, 0))
                for idx in range(len(self.images))
            ])
            trial_result = self.make_trial(frame_results, has_scale=False)
            trial_results.append(trial_result)

        metric = FrameErrorMetric()
        result = metric.measure_results(trial_results)

        self.assertTrue(result.success)
        self.assertEqual(repeats, len(result.errors))
        for repeat in range(repeats):
            self.assertEqual(len(self.images), len(result.errors[repeat].frame_errors))
            for idx in range(len(self.images)):
                frame_error = result.errors[repeat].frame_errors[idx]
                frame_result = trial_results[repeat].results[idx]
                self.assertEqual(repeat, frame_error.repeat)
                self.assertEqual(frame_result.timestamp, frame_error.timestamp)
                self.assertEqual(frame_result.image, frame_error.image)
                self.assertEqual(TrackingState.OK, frame_error.tracking)
                self.assertEqual(frame_result.num_features, frame_error.num_features)
                self.assertEqual(frame_result.num_matches, frame_error.num_matches)
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))

                # Errors should all be zero. We can be stricter on the rotation error in this test because
                # the scale only affects the translation.
                self.assertErrorIsAlmostZero(frame_error.absolute_error, places=12)

                if idx <= 0:
                    # No relative error or noise on the first frame, because there is no motion
                    self.assertIsNone(frame_error.relative_error)
                    self.assertIsNone(frame_error.noise)
                    self.assertIsNone(frame_error.systemic_error)
                else:
                    self.assertErrorIsAlmostZero(frame_error.relative_error)
                    self.assertErrorIsAlmostZero(frame_error.noise)
                    self.assertErrorIsAlmostZero(frame_error.systemic_error)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_returns_nonzero_error_and_noise_for_misscaled_trajectory_that_has_scale(self, _):
        repeats = 3

        # Make 3 trials, each with a different scale, they should be rescaled to correct
        trial_results = []
        for repeat in range(repeats):
            scale = 5 * (repeat + 1) / 8
            frame_results = self.make_frame_results_from_poses([
                Transform(location=(scale * idx * 15, scale * idx, 0))
                for idx in range(len(self.images))
            ])
            trial_result = self.make_trial(frame_results, has_scale=True)
            trial_results.append(trial_result)
        avg_scale = sum(5 * (repeat + 1) / 8 for repeat in range(repeats)) / 3

        metric = FrameErrorMetric()
        result = metric.measure_results(trial_results)

        self.assertTrue(result.success)
        self.assertEqual(repeats, len(result.errors))
        for repeat in range(repeats):
            self.assertEqual(len(self.images), len(result.errors[repeat].frame_errors))
            for idx in range(len(self.images)):
                frame_error = result.errors[repeat].frame_errors[idx]
                frame_result = trial_results[repeat].results[idx]
                self.assertEqual(repeat, frame_error.repeat)
                self.assertEqual(frame_result.timestamp, frame_error.timestamp)
                self.assertEqual(frame_result.image, frame_error.image)
                self.assertEqual(TrackingState.OK, frame_error.tracking)
                self.assertEqual(frame_result.num_features, frame_error.num_features)
                self.assertEqual(frame_result.num_matches, frame_error.num_matches)
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))

                # Errors should all be non-zero. We can be stricter on the rotation error in this test because
                # the scale only affects the translation.
                scale = 5 * (repeat + 1) / 8

                self.assertEqual(15 * (scale * idx - idx), frame_error.absolute_error.x)
                self.assertEqual(scale * idx - idx, frame_error.absolute_error.y)
                self.assertEqual(0, frame_error.absolute_error.z)
                self.assertEqual(
                    np.linalg.norm([15 * (scale * idx - idx), scale * idx - idx, 0]),
                    frame_error.absolute_error.length
                )
                self.assertEqual(0, frame_error.absolute_error.rot)

                if idx <= 0:
                    # No relative error or noise on the first frame, because there is no motion
                    self.assertIsNone(frame_error.relative_error)
                    self.assertIsNone(frame_error.noise)
                    self.assertIsNone(frame_error.systemic_error)
                else:
                    self.assertEqual(15 * scale - 15, frame_error.relative_error.x)
                    self.assertEqual(scale - 1, frame_error.relative_error.y)
                    self.assertEqual(0, frame_error.relative_error.z)
                    self.assertEqual(
                        np.linalg.norm([15 * scale - 15, scale - 1, 0]),
                        frame_error.relative_error.length
                    )
                    self.assertEqual(0, frame_error.relative_error.rot)

                    self.assertEqual(15 * (scale - avg_scale), frame_error.noise.x)
                    self.assertEqual(scale - avg_scale, frame_error.noise.y)
                    self.assertEqual(0, frame_error.noise.z)
                    self.assertEqual(
                        np.linalg.norm([15 * (scale - avg_scale), scale - avg_scale, 0]),
                        frame_error.noise.length
                    )
                    self.assertEqual(0, frame_error.noise.rot)

                    self.assertEqual((15 * avg_scale - 15), frame_error.systemic_error.x)
                    self.assertEqual(avg_scale - 1, frame_error.systemic_error.y)
                    self.assertEqual(0, frame_error.systemic_error.z)
                    self.assertEqual(
                        np.linalg.norm([15 * avg_scale - 15, avg_scale - 1, 0]),
                        frame_error.systemic_error.length
                    )
                    self.assertEqual(0, frame_error.systemic_error.rot)

    @mock.patch('arvet_slam.metrics.frame_error.frame_error_metric.autoload_modules')
    def test_measure_multiple_many_random_trials(self, _):
        repeats = 5

        estimated_motions = [
            [
                Transform(
                    np.random.uniform(-10, 10, size=3),
                    tf3d.quaternions.axangle2quat(np.random.uniform(-10, 10, size=3),  np.random.uniform(-np.pi, np.pi))
                ) if idx > 0 and np.random.uniform(0, 10) > 1 else None
                for _ in range(repeats)
            ] for idx in range(len(self.images))
        ]
        average_motions = [
            compute_average_pose(motion for motion in image_motions if motion is not None)
            if len(image_motions) - image_motions.count(None) > 1 else None
            for image_motions in estimated_motions
        ]

        trial_results = []
        for repeat in range(repeats):
            frame_results = self.make_frame_results_from_motions([
                estimated_motions[idx][repeat]
                for idx in range(len(self.images))
            ], set_initial_pose=True)
            trial_results.append(self.make_trial(frame_results))

        metric = FrameErrorMetric()
        result = metric.measure_results(trial_results)

        self.assertTrue(result.success)
        self.assertEqual(repeats, len(result.errors))
        for repeat in range(repeats):
            has_been_lost = False
            self.assertEqual(len(self.images), len(result.errors[repeat].frame_errors))
            for idx in range(len(self.images)):
                frame_error = result.errors[repeat].frame_errors[idx]
                frame_result = trial_results[repeat].results[idx]
                self.assertEqual(repeat, frame_error.repeat)
                self.assertEqual(frame_result.timestamp, frame_error.timestamp)
                self.assertEqual(frame_result.image, frame_error.image)
                self.assertEqual(frame_result.tracking_state, frame_error.tracking)
                self.assertEqual(frame_result.num_features, frame_error.num_features)
                self.assertEqual(frame_result.num_matches, frame_error.num_matches)
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_distances))
                self.assertEqual(len(frame_result.loop_edges), len(frame_error.loop_angles))

                # Absolute error should be none after it becomes lost, since we don't have a reference to bring it back
                if idx > 0 and (estimated_motions[idx][repeat] is None or has_been_lost):
                    self.assertIsNone(frame_error.absolute_error)
                    has_been_lost = True
                else:
                    self.assertErrorsEqual(make_pose_error(frame_result.estimated_pose, frame_result.pose),
                                           frame_error.absolute_error)
                # relative error should be none for the lost frames
                if estimated_motions[idx][repeat] is None:
                    self.assertIsNone(frame_error.relative_error)
                else:
                    self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, frame_result.motion),
                                           frame_error.relative_error)
                # Noise should be none when we are lost or when there is less than two estimates
                if estimated_motions[idx][repeat] is None or \
                        len(estimated_motions[idx]) - estimated_motions[idx].count(None) <= 1:
                    self.assertIsNone(frame_error.noise)
                else:
                    self.assertErrorsEqual(make_pose_error(frame_result.estimated_motion, average_motions[idx]),
                                           frame_error.noise)

                # Systemic errror should be none iff the average motion is None
                if average_motions[idx] is None:
                    self.assertIsNone(frame_error.systemic_error)
                else:
                    self.assertErrorsEqual(make_pose_error(average_motions[idx], frame_result.motion),
                                           frame_error.systemic_error)

    def assertErrorsEqual(self, pose_error_1, pose_error_2, rot_places=13):
        self.assertEqual(pose_error_1.x, pose_error_2.x)
        self.assertEqual(pose_error_1.y, pose_error_2.y)
        self.assertEqual(pose_error_1.z, pose_error_2.z)
        self.assertEqual(pose_error_1.length, pose_error_2.length)
        self.assertAlmostEqual(pose_error_1.rot, pose_error_2.rot, places=rot_places)
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

    def assertErrorIsZero(self, pose_error, rot_places=13):
        self.assertEqual(0, pose_error.x)
        self.assertEqual(0, pose_error.y)
        self.assertEqual(0, pose_error.z)
        self.assertEqual(0, pose_error.length)
        self.assertTrue(np.isnan(pose_error.direction))
        self.assertAlmostEqual(0, pose_error.rot, places=rot_places)

    def assertErrorIsAlmostZero(self, pose_error, places=13, rot_places=13):
        self.assertAlmostEqual(0, pose_error.x, places=places)
        self.assertAlmostEqual(0, pose_error.y, places=places)
        self.assertAlmostEqual(0, pose_error.z, places=places)
        self.assertAlmostEqual(0, pose_error.length, places=places)
        # self.assertTrue(np.isnan(pose_error.direction))
        self.assertAlmostEqual(0, pose_error.rot, places=rot_places)


class TestAlignTrajectoryToGroundTruth(ExtendedTestCase):

    def assertQuatClose(self, q1, q2, msg='', rtol=1e-05, atol=1e-08):
        if np.dot(q1, q2) < 1:
            q2 = -1 * np.asarray(q2)
        self.assertNPClose(q1, q2, msg=msg, rtol=rtol, atol=atol)

    def test_align_simple_offset(self):
        offset = np.array([15, 22, -1])
        points = [np.array([15 * idx - 3 * idx * idx, -22 * idx, 200 + 55 * idx - 5 * idx]) for idx in range(30)]
        gt_trajectory = [Transform(location=point) for point in points]
        estimated_trajectory = [Transform(location=point + offset) for point in points]
        shift, scale = align_trajectory_to_ground_truth(estimated_trajectory, gt_trajectory, compute_scale=False)
        self.assertNPEqual(offset, shift.location)
        self.assertNPEqual([1, 0, 0, 0], shift.rotation_quat(True))
        self.assertEqual(1, scale)
        for idx in range(len(points)):
            self.assertEqual(gt_trajectory[idx], shift.find_relative(estimated_trajectory[idx]))

    def test_align_simple_rotation(self):
        rotation = tf3d.quaternions.axangle2quat((0, -1, 0), 3 * np.pi / 4)
        points = [np.array([15 * idx - 3 * idx * idx, -22 * idx, 200 + 50 * idx]) for idx in range(30)]
        gt_trajectory = [Transform(location=point) for point in points]
        estimated_trajectory = [
            Transform(location=tf3d.quaternions.rotate_vector(point, rotation), rotation=rotation, w_first=True)
            for point in points
        ]
        shift, scale = align_trajectory_to_ground_truth(estimated_trajectory, gt_trajectory, compute_scale=False)
        self.assertNPClose([0, 0, 0], shift.location, rtol=0, atol=1e-12)
        self.assertQuatClose(rotation, shift.rotation_quat(True), rtol=0, atol=1e-12)
        self.assertEqual(1, scale)
        for idx in range(len(points)):
            shifted_pose = shift.find_relative(estimated_trajectory[idx])
            self.assertNPClose(gt_trajectory[idx].location, shifted_pose.location, rtol=0, atol=1e-12)
            self.assertQuatClose(gt_trajectory[idx].rotation_quat(True), shifted_pose.rotation_quat(True),
                                 rtol=0, atol=1e-12)

    def test_combined_rotation_and_translation(self):
        offset = np.array([-17, -43, 12])
        rotation = tf3d.quaternions.axangle2quat((-3, 1, 2), 7 * np.pi / 9)
        points = [np.array([
            (idx - 34) * (55 - idx),
            122 - 16 * idx,
            30 + 35 * idx - 18 * idx * idx
        ]) for idx in range(100)]
        gt_trajectory = [Transform(location=point) for point in points]
        estimated_trajectory = [
            Transform(location=tf3d.quaternions.rotate_vector(point, rotation) + offset,
                      rotation=rotation, w_first=True)
            for point in points
        ]
        shift, scale = align_trajectory_to_ground_truth(estimated_trajectory, gt_trajectory, compute_scale=False)
        self.assertNPClose(offset, shift.location, rtol=0, atol=1e-9)
        self.assertNPClose(rotation, shift.rotation_quat(True), rtol=0, atol=1e-10)
        self.assertEqual(1, scale)
        for idx in range(len(points)):
            shifted_pose = shift.find_relative(estimated_trajectory[idx])
            self.assertNPClose(gt_trajectory[idx].location, shifted_pose.location, rtol=0, atol=1e-8)   # This is bad
            self.assertQuatClose(gt_trajectory[idx].rotation_quat(True), shifted_pose.rotation_quat(True),
                                 rtol=0, atol=1e-10)

    def test_combined_rotation_translation_and_symmetric_scale(self):
        offset = np.array([-17, -43, 12])
        scale = np.pi
        rotation = tf3d.quaternions.axangle2quat((-3, 1, 2), 7 * np.pi / 9)
        points = [np.array([
            (idx - 34) * (55 - idx),
            122 - 16 * idx,
            30 + 35 * idx - 18 * idx * idx
        ]) for idx in range(100)]
        gt_trajectory = [Transform(location=point) for point in points]
        estimated_trajectory = [
            Transform(location=tf3d.quaternions.rotate_vector(point, rotation) * scale + offset,
                      rotation=rotation, w_first=True)
            for point in points
        ]
        shift, est_scale = align_trajectory_to_ground_truth(estimated_trajectory, gt_trajectory,
                                                            compute_scale=True, use_symmetric_scale=True)
        self.assertNPClose(offset, shift.location, rtol=0, atol=1e-8)
        self.assertNPClose(rotation, shift.rotation_quat(True), rtol=0, atol=1e-10)
        self.assertAlmostEqual(1 / scale, est_scale, places=10)
        for idx in range(len(points)):
            shifted_pose = shift.find_relative(estimated_trajectory[idx])
            self.assertNPClose(gt_trajectory[idx].location, est_scale * shifted_pose.location, rtol=0, atol=1e-8)
            self.assertQuatClose(gt_trajectory[idx].rotation_quat(True), shifted_pose.rotation_quat(True),
                                 rtol=0, atol=1e-10)

    def test_combined_rotation_translation_and_asymmetric_scale(self):
        offset = np.array([-17, -43, 12])
        scale = np.pi
        rotation = tf3d.quaternions.axangle2quat((-3, 1, 2), 7 * np.pi / 9)
        points = [np.array([
            (idx - 34) * (55 - idx),
            122 - 16 * idx,
            30 + 35 * idx - 18 * idx * idx
        ]) for idx in range(100)]
        gt_trajectory = [Transform(location=point) for point in points]
        estimated_trajectory = [
            Transform(location=tf3d.quaternions.rotate_vector(point, rotation) * scale + offset,
                      rotation=rotation, w_first=True)
            for point in points
        ]
        shift, est_scale = align_trajectory_to_ground_truth(estimated_trajectory, gt_trajectory,
                                                            compute_scale=True, use_symmetric_scale=False)
        self.assertNPClose(offset, shift.location, rtol=0, atol=1e-8)
        self.assertNPClose(rotation, shift.rotation_quat(True), rtol=0, atol=1e-10)
        self.assertAlmostEqual(1 / scale, est_scale, places=10)
        for idx in range(len(points)):
            shifted_pose = shift.find_relative(estimated_trajectory[idx])
            self.assertNPClose(gt_trajectory[idx].location, est_scale * shifted_pose.location, rtol=0, atol=1e-8)
            self.assertQuatClose(gt_trajectory[idx].rotation_quat(True), shifted_pose.rotation_quat(True),
                                 rtol=0, atol=1e-10)
