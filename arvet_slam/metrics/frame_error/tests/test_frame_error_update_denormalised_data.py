import unittest
from pathlib import Path
from itertools import chain
import numpy as np
import transforms3d as t3
import pymodm.fields as fields

import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.util.transform import Transform
from arvet.core.image import Image
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image_collection import ImageCollection
from arvet.core.system import VisionSystem
from arvet.core.trial_result import TrialResult
from arvet.core.tests.mock_types import make_image, MockSystem, MockMetric
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult
from arvet_slam.metrics.frame_error.frame_error_metric import FrameErrorMetric
from arvet_slam.metrics.frame_error.frame_error_result import make_frame_error, FrameError,\
    TrialErrors, make_frame_error_result, FrameErrorResult, json_value
import arvet_slam.metrics.frame_error.frame_error_update_denormalised_data as update_data


class SystemWithProperties(MockSystem):
    properties = fields.DictField(blank=True)

    def get_columns(self):
        return set(self.properties)

    def get_properties(self, columns=None, settings=None):
        properties = super(SystemWithProperties, self).get_properties(columns, settings)
        properties.update(self.properties)
        properties.update(settings)
        return properties


class MetricWithProperties(MockMetric):
    properties = fields.DictField(blank=True)

    def get_columns(self):
        return set(self.properties)

    def get_properties(self, columns=None):
        properties = super(MetricWithProperties, self).get_properties(columns)
        properties.update(self.properties)
        return properties


class TestUpdateFrameError(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model

        FrameError.objects.all().delete()
        FrameErrorResult.objects.all().delete()
        TrialResult.objects.all().delete()
        VisionSystem.objects.all().delete()
        ImageCollection.objects.all().delete()
        FrameErrorMetric.objects.all().delete()
        Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()

    def test_update_system_information(self):
        random = np.random.RandomState(13)
        system = SystemWithProperties(properties={'my_system_property': random.randint(10, 420)})
        system.save()
        metric = MockMetric()
        metric.save()
        image_collection = make_image_collection(10)
        trial_results = make_trials(system, image_collection, 3, random)

        trial_errors = []
        all_frame_errors = []
        for repeat_idx, trial_result in enumerate(trial_results):
            frame_errors = []
            for frame_result in trial_result.results:
                frame_error = FrameError(
                    trial_result=trial_result,
                    image=frame_result.image,
                    repeat=repeat_idx,
                    timestamp=frame_result.timestamp,
                    motion=frame_result.motion,
                    processing_time=frame_result.processing_time,
                    num_features=frame_result.num_features,
                    num_matches=frame_result.num_matches,
                    tracking=frame_result.tracking_state,
                    absolute_error=None,
                    relative_error=None,
                    noise=None
                )
                frame_error.save()
                frame_errors.append(frame_error)
                all_frame_errors.append(frame_error)
            trial_errors.append(TrialErrors(
                frame_errors=frame_errors
            ))
        # Trial results are found based on the metric results, so we need to make that
        metric_result = make_frame_error_result(
            metric=metric,
            trial_results=trial_results,
            errors=trial_errors
        )
        metric_result.save()

        update_data.update_frame_errors_system_properties()

        # Check that the image properties have been updated
        for frame_error in all_frame_errors:
            frame_error.refresh_from_db()
            system_properties = frame_error.trial_result.system.get_properties(
                None, frame_error.trial_result.settings)
            # Convert the properties to a JSON serialisable value
            system_properties = {str(k): json_value(v) for k, v in system_properties.items()}
            self.assertEqual(system_properties, frame_error.system_properties)

    def test_update_system_information_only_missing(self):
        random = np.random.RandomState(13)
        system = SystemWithProperties(properties={'my_system_property': random.randint(10, 420)})
        system.save()
        metric = MockMetric()
        metric.save()
        image_collection = make_image_collection(10)
        trial_results = make_trials(system, image_collection, 3, random)

        trial_errors = []
        all_frame_errors = []
        for repeat_idx, trial_result in enumerate(trial_results):
            frame_errors = []
            for frame_result in trial_result.results:
                frame_error = FrameError(
                    trial_result=trial_result,
                    image=frame_result.image,
                    repeat=repeat_idx,
                    timestamp=frame_result.timestamp,
                    motion=frame_result.motion,
                    processing_time=frame_result.processing_time,
                    num_features=frame_result.num_features,
                    num_matches=frame_result.num_matches,
                    tracking=frame_result.tracking_state,
                    absolute_error=None,
                    relative_error=None,
                    noise=None
                )
                frame_error.save()
                frame_errors.append(frame_error)
                all_frame_errors.append(frame_error)
            trial_errors.append(TrialErrors(
                frame_errors=frame_errors
            ))
        # Trial results are found based on the metric results, so we need to make that
        metric_result = make_frame_error_result(
            metric=metric,
            trial_results=trial_results,
            errors=trial_errors
        )
        metric_result.save()

        update_data.update_frame_errors_system_properties()

        # Check that the image properties have been updated
        for frame_error in all_frame_errors:
            frame_error.refresh_from_db()
            system_properties = frame_error.trial_result.system.get_properties(
                None, frame_error.trial_result.settings)
            # Convert the properties to a JSON serialisable value
            system_properties = {str(k): json_value(v) for k, v in system_properties.items()}
            self.assertEqual(system_properties, frame_error.system_properties)

    def test_update_image_information(self):
        random = np.random.RandomState(13)
        system = MockSystem()
        system.save()
        metric = MockMetric()
        metric.save()
        image_collection = make_image_collection(10)
        trial_results = make_trials(system, image_collection, 3, random)

        frame_errors = []
        for repeat_idx, trial_result in enumerate(trial_results):
            for frame_result in trial_result.results:
                frame_error = FrameError(
                    trial_result=trial_result,
                    image=frame_result.image,
                    repeat=repeat_idx,
                    timestamp=frame_result.timestamp,
                    motion=frame_result.motion,
                    processing_time=frame_result.processing_time,
                    num_features=frame_result.num_features,
                    num_matches=frame_result.num_matches,
                    tracking=frame_result.tracking_state,
                    absolute_error=None,
                    relative_error=None,
                    noise=None
                )
                frame_error.save()
                frame_errors.append(frame_error)

        update_data.update_frame_error_image_information()

        # Check that the image properties have been updated
        for frame_error in frame_errors:
            frame_error.refresh_from_db()
            image_properties = frame_error.image.get_properties()
            # Convert the properties to a JSON serialisable value
            image_properties = {str(k): json_value(v) for k, v in image_properties.items()}
            self.assertEqual(image_properties, frame_error.image_properties)

    def test_update_image_information_only_missing(self):
        random = np.random.RandomState(13)
        system = MockSystem()
        system.save()
        metric = MockMetric()
        metric.save()
        image_collection = make_image_collection(10)
        trial_results = make_trials(system, image_collection, 3, random)

        non_updated_ids = set()
        frame_errors = []
        for repeat_idx, trial_result in enumerate(trial_results):
            for frame_idx, frame_result in enumerate(trial_result.results):
                frame_error = FrameError(
                    trial_result=trial_result,
                    image=frame_result.image,
                    repeat=repeat_idx,
                    timestamp=frame_result.timestamp,
                    motion=frame_result.motion,
                    processing_time=frame_result.processing_time,
                    num_features=frame_result.num_features,
                    num_matches=frame_result.num_matches,
                    tracking=frame_result.tracking_state,
                    absolute_error=None,
                    relative_error=None,
                    noise=None
                )
                if int(frame_idx) % 4 == 3:
                    # These properties should remain
                    frame_error.image_properties = {
                        'default_property_1': 1.2, 'default_property_2': str(frame_result.image.pk)
                    }
                    non_updated_ids.add(frame_result.image.pk)
                elif int(frame_error.repeat + frame_error.timestamp) % 4 == 0:
                    # These ones should be overriden because there will be at least one other
                    # frame error that is missing these properties, and we update them all at once
                    frame_error.image_properties = {
                        'overwritten_property_1': 3.2, 'overwritten_property_2': str(frame_result.image.pk)
                    }
                frame_error.save()
                frame_errors.append(frame_error)

        update_data.update_frame_error_image_information(only_missing=True)

        # Check that the image properties have been updated
        for frame_error in frame_errors:
            frame_error.refresh_from_db()
            image_properties = frame_error.image.get_properties()
            # Convert the properties to a JSON serialisable value
            image_properties = {str(k): json_value(v) for k, v in image_properties.items()}
            if frame_error.image.pk in non_updated_ids:
                self.assertEqual({
                    'default_property_1': 1.2, 'default_property_2': str(frame_error.image.pk)
                }, frame_error.image_properties)
            else:
                self.assertEqual(image_properties, frame_error.image_properties)


class TestUpdateFrameErrorResult(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model

        FrameError.objects.all().delete()
        FrameErrorResult.objects.all().delete()
        TrialResult.objects.all().delete()
        VisionSystem.objects.all().delete()
        ImageCollection.objects.all().delete()
        FrameErrorMetric.objects.all().delete()
        Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()

    def test_update_image_source_properties(self):
        random = np.random.RandomState(13)
        system = MockSystem()
        system.save()
        metric = MockMetric()
        metric.save()
        image_collection = make_image_collection(10, dataset='my_dataset', sequence_name='test_sequence_01')
        trial_results = make_trials(system, image_collection, 3, random)

        trial_errors = []
        for repeat_idx, trial_result in enumerate(trial_results):
            frame_errors = []
            for frame_result in trial_result.results:
                frame_error = make_frame_error(
                    trial_result=trial_result,
                    frame_result=frame_result,
                    image=None,
                    system=None,
                    repeat_index=repeat_idx,
                    loop_distances=[],
                    loop_angles=[],
                    absolute_error=None,
                    relative_error=None,
                    noise=None
                )
                frame_error.save()
                frame_errors.append(frame_error)
            trial_errors.append(TrialErrors(
                frame_errors=frame_errors
            ))
        metric_result = FrameErrorResult(
            metric=metric,
            trial_results=trial_results,
            system=system,
            image_source=image_collection,
            success=True,
            errors=trial_errors
            # Metric result is missing the image source properties
        )
        metric_result.save()

        update_data.update_frame_error_result_image_source_properties()
        metric_result.refresh_from_db()

        image_source_properties = image_collection.get_properties()
        image_source_properties = {str(k): json_value(v) for k, v in image_source_properties.items()}
        self.assertEqual(image_source_properties, metric_result.image_source_properties)

    def test_update_metric_properties(self):
        random = np.random.RandomState(13)
        system = MockSystem()
        system.save()
        metric = MetricWithProperties(properties={'metric_property_1': 'linear', 'metric_property_2': 10})
        metric.save()
        image_collection = make_image_collection(10)
        trial_results = make_trials(system, image_collection, 3, random)

        trial_errors = []
        for repeat_idx, trial_result in enumerate(trial_results):
            frame_errors = []
            for frame_result in trial_result.results:
                frame_error = make_frame_error(
                    trial_result=trial_result,
                    frame_result=frame_result,
                    image=None,
                    system=None,
                    repeat_index=repeat_idx,
                    loop_distances=[],
                    loop_angles=[],
                    absolute_error=None,
                    relative_error=None,
                    noise=None
                )
                frame_error.save()
                frame_errors.append(frame_error)
            trial_errors.append(TrialErrors(
                frame_errors=frame_errors
            ))
        metric_result = FrameErrorResult(
            metric=metric,
            trial_results=trial_results,
            system=system,
            image_source=image_collection,
            success=True,
            errors=trial_errors
            # Metric result is missing the metric properties
        )
        metric_result.save()

        update_data.update_frame_error_result_metric_properties()
        metric_result.refresh_from_db()

        metric_properties = metric.get_properties()
        metric_properties = {str(k): json_value(v) for k, v in metric_properties.items()}
        self.assertEqual(metric_properties, metric_result.metric_properties)

    def test_update_frame_error_columns(self):
        random = np.random.RandomState(13)
        system = MockSystem()
        system.save()
        metric = MetricWithProperties(properties={'metric_property_1': 'linear', 'metric_property_2': 10})
        metric.save()
        image_collection = make_image_collection(10)
        trial_results = make_trials(system, image_collection, 3, random)

        trial_errors = []
        all_frame_errors = []
        for repeat_idx, trial_result in enumerate(trial_results):
            frame_errors = []
            for frame_idx, frame_result in enumerate(trial_result.results):
                frame_error = make_frame_error(
                    trial_result=trial_result,
                    frame_result=frame_result,
                    image=None,
                    system=None,
                    repeat_index=repeat_idx,
                    loop_distances=[],
                    loop_angles=[],
                    absolute_error=None,
                    relative_error=None,
                    noise=None
                )
                # Add a different extra property to each frame error
                frame_error.system_properties[f"secret_property_{frame_idx}"] = frame_idx * frame_result.timestamp
                frame_error.save()
                frame_errors.append(frame_error)
                all_frame_errors.append(frame_error)
            trial_errors.append(TrialErrors(
                frame_errors=frame_errors
            ))
        metric_result = FrameErrorResult(
            metric=metric,
            trial_results=trial_results,
            system=system,
            image_source=image_collection,
            success=True,
            errors=trial_errors
            # Metric result is missing the frame error columns list
        )
        metric_result.save()

        update_data.update_frame_error_result_columns()
        metric_result.refresh_from_db()

        frame_error_columns = list(set(chain.from_iterable(
            frame_error.get_columns()
            for frame_error in all_frame_errors
        )))
        self.assertEqual(frame_error_columns, metric_result.frame_columns)


def make_image_collection(length, **kwargs):
    images = [make_image(idx) for idx in range(length)]
    timestamps = [idx * 0.9 for idx in range(len(images))]
    for image in images:
        image.save()
    image_collection = ImageCollection(
        images=images,
        timestamps=timestamps,
        sequence_type=ImageSequenceType.SEQUENTIAL,
        **kwargs
    )
    image_collection.save()
    return image_collection


def make_trials(system: VisionSystem, image_collection: ImageCollection, repeats: int, random: np.random.RandomState):
    # Get the true motions, for making trials
    true_motions = [
        image_collection.images[frame_idx - 1].camera_pose.find_relative(
            image_collection.images[frame_idx].camera_pose)
        if frame_idx > 0 else None
        for frame_idx in range(len(image_collection))
    ]

    # Make some plausible trial results
    trial_results = []
    for repeat in range(repeats):
        start_idx = random.randint(0, len(image_collection) - 2)
        frame_results = [FrameResult(
            timestamp=timestamp,
            image=image,
            pose=image.camera_pose,
            processing_time=random.uniform(0.001, 1.0),
            estimated_motion=true_motions[frame_idx].find_independent(Transform(
                location=random.normal(0, 1, 3),
                rotation=t3.quaternions.axangle2quat(random.uniform(-1, 1, 3), random.normal(0, np.pi / 2)),
                w_first=True
            )) if frame_idx > start_idx else None,
            tracking_state=TrackingState.OK if frame_idx > start_idx else TrackingState.NOT_INITIALIZED,
            num_matches=random.randint(10, 100)
        ) for frame_idx, (timestamp, image) in enumerate(image_collection)]
        frame_results[start_idx].estimated_pose = Transform()
        trial_settings = {
            'random': random.randint(0, 10),
            'repeat': repeat
        }
        trial_result = SLAMTrialResult(
            system=system,
            image_source=image_collection,
            success=True,
            results=frame_results,
            has_scale=False,
            settings=trial_settings
        )
        trial_result.save()
        trial_results.append(trial_result)
    return trial_results
