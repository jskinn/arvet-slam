import unittest
import unittest.mock as mock
import os
import bson
import pymodm
from pymodm.errors import ValidationError
import numpy as np
import transforms3d as tf3d

import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.util.transform import Transform
from arvet.metadata.image_metadata import make_metadata, ImageSourceType
import arvet.core.tests.mock_types as mock_types
from arvet.core.image import Image
from arvet.core.system import VisionSystem
from arvet.core.trial_result import TrialResult
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.metrics.frame_error.frame_error_metric import make_pose_error, PoseError, FrameError, TrialErrors, \
    FrameErrorResult


class TestMakePoseError(unittest.TestCase):

    def test_pose_error_includes_trans_error(self):
        ref = Transform((10, 12, 14))
        pose = Transform((11, 15, 19))
        pose_error = make_pose_error(pose, ref)
        self.assertEqual(1, pose_error.x)
        self.assertEqual(3, pose_error.y)
        self.assertEqual(5, pose_error.z)
        self.assertEqual(np.sqrt(1 + 9 + 25), pose_error.length)

    def test_trans_error_can_be_zero(self):
        ref = Transform((1, 2, 3))
        pose = Transform((1, 2, 3))
        pose_error = make_pose_error(pose, ref)
        self.assertEqual(0, pose_error.x)
        self.assertEqual(0, pose_error.y)
        self.assertEqual(0, pose_error.z)
        self.assertEqual(0, pose_error.length)

    def test_pose_error_finds_direction_of_trans_error(self):
        ref = Transform((1, 0, 0))
        pose = Transform((2, 1, 0))
        pose_error = make_pose_error(pose, ref)
        self.assertAlmostEqual(np.pi / 4, pose_error.direction)

    def test_pose_error_direction_is_nan_when_reference_motion_is_zero(self):
        ref = Transform((0, 0, 0))
        pose = Transform((1, 1, 0))
        pose_error = make_pose_error(pose, ref)
        self.assertTrue(np.isnan(pose_error.direction))

    def test_direction_is_nan_when_error_is_zero(self):
        ref = Transform((1, 2, 3))
        pose = Transform((1, 2, 3))
        pose_error = make_pose_error(pose, ref)
        self.assertTrue(np.isnan(pose_error.direction))

    def test_includes_rotation_error(self):
        pose = Transform(rotation=tf3d.quaternions.axangle2quat((1, 2, 3), np.pi / 6), w_first=True)
        ref = Transform((1, 1, 0))
        pose_error = make_pose_error(pose, ref)
        self.assertAlmostEqual(np.pi / 6, pose_error.rot)


# ------------------------- POSE ERROR -------------------------


class TestPoseErrorMongoModel(pymodm.MongoModel):
    pose_error = pymodm.fields.EmbeddedDocumentField(PoseError)


class TestPoseErrorDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        TestPoseErrorMongoModel._mongometa.collection.drop()

    def test_stores_and_loads(self):
        pose_error = PoseError(
            x=10,
            y=11,
            z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 36,
            rot=np.pi/5
        )

        # Save the model
        model = TestPoseErrorMongoModel()
        model.pose_error = pose_error
        model.save()

        # Load all the entities
        all_entities = list(TestPoseErrorMongoModel.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].pose_error, pose_error)
        all_entities[0].delete()

    def test_required_fields_are_required(self):
        model = TestPoseErrorMongoModel()

        # no x
        model.pose_error = PoseError(
            y=11,
            z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 36,
            rot=np.pi/5
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no y
        model.pose_error = PoseError(
            x=10,
            z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 36,
            rot=np.pi/5
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no z
        model.pose_error = PoseError(
            x=10,
            y=11,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 36,
            rot=np.pi/5
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no length
        model.pose_error = PoseError(
            x=10,
            y=11,
            z=12,
            direction=np.pi / 36,
            rot=np.pi/5
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no direction
        model.pose_error = PoseError(
            x=10,
            y=11,
            z=12,
            length=np.sqrt(100 + 121 + 144),
            rot=np.pi/5
        )
        with self.assertRaises(ValidationError):
            model.save()

        # no rot
        model.pose_error = PoseError(
            x=10,
            y=11,
            z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 36,
        )
        with self.assertRaises(ValidationError):
            model.save()


# ------------------------- FRAME ERROR -------------------------


def mock_system_get_properties(columns=None, settings=None):
    if columns is None:
        columns = {'system_column', 'runtime_column'}
    if settings is None:
        settings = {}
    properties = {}
    if 'system_column' in columns:
        properties['system_column'] = 823.3
    properties.update({k: v for k, v in settings.items() if k in columns})
    return properties


class TestFrameErrorGetProperties(unittest.TestCase):

    def setUp(self) -> None:
        self.system = mock.Mock(spec=VisionSystem)
        self.system.get_columns.return_value = {'system_column', 'runtime_column'}
        self.system.get_properties.side_effect = mock_system_get_properties
        self.trial = mock.Mock(spec=TrialResult)
        self.trial.system = self.system
        self.trial.settings = {'runtime_column': 42}

        self.pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.image = Image(
            pixels=self.pixels,
            metadata=make_metadata(self.pixels)
        )
        self.motion = Transform(
            (1.33, -0.233, -0.0343),
            (-0.5, 0.5, 0.5, -0.5)
        )

        self.repeat = 1
        self.timestamp = 1.3
        self.tracking = TrackingState.OK
        self.processing_time = 0.568
        self.num_features = 423
        self.num_matches = 238
        self.absolute_error = PoseError(
            x=10,
            y=11,
            z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 7,
            rot=np.pi / 5
        )
        self.relative_error = PoseError(
            x=13,
            y=14,
            z=15,
            length=np.sqrt(169 + 196 + 225),
            direction=np.pi / 36,
            rot=np.pi / 2
        )
        self.noise = PoseError(
            x=16,
            y=17,
            z=18,
            length=np.sqrt(256 + 289 + 324),
            direction=np.pi / 8,
            rot=np.pi / 16
        )

        self.frame_error = FrameError(
            repeat=self.repeat,
            timestamp=self.timestamp,
            image=self.image,
            trial_result=self.trial,
            processing_time=self.processing_time,
            tracking=self.tracking,
            motion=self.motion,
            num_features=self.num_features,
            num_matches=self.num_matches,
            absolute_error=self.absolute_error,
            relative_error=self.relative_error,
            noise=self.noise
        )

    def test_get_properties_returns_all_properties_by_default(self):
        expected_properties = dict(self.image.get_properties())
        expected_properties.update(self.system.get_properties(settings=self.trial.settings))
        expected_properties.update({
            'repeat': self.repeat,
            'timestamp': self.timestamp,
            'tracking': self.tracking == TrackingState.OK,
            'processing_time': self.processing_time,
            'motion_x': self.motion.x,
            'motion_y': self.motion.y,
            'motion_z': self.motion.z,
            'motion_roll': self.motion.euler[0],
            'motion_pitch': self.motion.euler[1],
            'motion_yaw': self.motion.euler[2],
            'num_features': self.num_features,
            'num_matches': self.num_matches,

            'abs_error_x': self.absolute_error.x,
            'abs_error_y': self.absolute_error.y,
            'abs_error_z': self.absolute_error.z,
            'abs_error_length': self.absolute_error.length,
            'abs_error_direction': self.absolute_error.direction,
            'abs_rot_error': self.absolute_error.rot,

            'trans_error_x': self.relative_error.x,
            'trans_error_y': self.relative_error.y,
            'trans_error_z': self.relative_error.z,
            'trans_error_length': self.relative_error.length,
            'trans_error_direction': self.relative_error.direction,
            'rot_error': self.relative_error.rot,

            'trans_noise_x': self.noise.x,
            'trans_noise_y': self.noise.y,
            'trans_noise_z': self.noise.z,
            'trans_noise_length': self.noise.length,
            'trans_noise_direction': self.noise.direction,
            'rot_noise': self.noise.rot
        })
        self.assertEqual(expected_properties, self.frame_error.get_properties())

    def test_get_properties_returns_all_image_columns(self):
        properties = self.frame_error.get_properties()
        for column_name in self.image.get_columns():
            self.assertIn(column_name, properties)

    def test_get_properties_returns_all_system_columns(self):
        properties = self.frame_error.get_properties()
        for column_name in self.system.get_columns():
            self.assertIn(column_name, properties)

    def test_get_properties_returns_only_the_requested_properties(self):
        expected_properties = dict(self.image.get_properties())
        expected_properties.update(self.system.get_properties(settings=self.trial.settings))
        expected_properties.update({
            'repeat': self.repeat,
            'timestamp': self.timestamp,
            'tracking': self.tracking == TrackingState.OK,
            'processing_time': self.processing_time,
            'motion_x': self.motion.x,
            'motion_y': self.motion.y,
            'motion_z': self.motion.z,
            'motion_roll': self.motion.euler[0],
            'motion_pitch': self.motion.euler[1],
            'motion_yaw': self.motion.euler[2],
            'num_features': self.num_features,
            'num_matches': self.num_matches,

            'abs_error_x': self.absolute_error.x,
            'abs_error_y': self.absolute_error.y,
            'abs_error_z': self.absolute_error.z,
            'abs_error_length': self.absolute_error.length,
            'abs_error_direction': self.absolute_error.direction,
            'abs_rot_error': self.absolute_error.rot,

            'trans_error_x': self.relative_error.x,
            'trans_error_y': self.relative_error.y,
            'trans_error_z': self.relative_error.z,
            'trans_error_length': self.relative_error.length,
            'trans_error_direction': self.relative_error.direction,
            'rot_error': self.relative_error.rot,

            'trans_noise_x': self.noise.x,
            'trans_noise_y': self.noise.y,
            'trans_noise_z': self.noise.z,
            'trans_noise_length': self.noise.length,
            'trans_noise_direction': self.noise.direction,
            'rot_noise': self.noise.rot
        })
        columns = list(expected_properties.keys())
        np.random.shuffle(columns)
        colums_1 = {column for idx, column in enumerate(columns) if idx % 2 == 0}
        colums_2 = set(columns) - colums_1

        properties = self.frame_error.get_properties(colums_1)
        for column in colums_1:
            self.assertEqual(expected_properties[column], properties[column])
        for column in colums_2:
            self.assertNotIn(column, properties)

        properties = self.frame_error.get_properties(colums_2)
        for column in colums_1:
            self.assertNotIn(column, properties)
        for column in colums_2:
            self.assertEqual(expected_properties[column], properties[column])

    def test_get_properties_allows_extra_properties(self):
        extra_properties = {'foo': 'bar', 'baz': 1.33}
        image_columns = list(self.image.get_columns())
        image_columns = image_columns[:len(image_columns) // 2]
        image_properties = self.image.get_properties(image_columns)

        expected_properties = {
            **extra_properties,
            'repeat': self.repeat,
            'tracking': self.tracking == TrackingState.OK,

            'abs_error_x': self.absolute_error.x,
            'abs_error_z': self.absolute_error.z,
            'abs_error_length': self.absolute_error.length,
            'trans_error_y': self.relative_error.y,

            'trans_noise_length': self.noise.length,
            'trans_noise_direction': self.noise.direction,
            'rot_noise': self.noise.rot,

            'num_features': self.num_features,
            'num_matches': self.num_matches,
            **image_properties
        }
        self.assertEqual(expected_properties, self.frame_error.get_properties(
            [*expected_properties.keys(), *image_columns],
            extra_properties
        ))

    def test_get_properties_returns_properties_with_minimal_parameters(self):
        frame_error = FrameError(
            repeat=self.repeat,
            timestamp=self.timestamp,
            trial_result=self.trial,
            image=self.image,
            motion=self.motion
        )
        expected_properties = dict(self.image.get_properties())
        expected_properties.update(self.system.get_properties(self.trial.settings))
        expected_properties.update({
            'repeat': self.repeat,
            'timestamp': self.timestamp,
            'tracking': True,
            'processing_time': np.nan,
            'motion_x': self.motion.x,
            'motion_y': self.motion.y,
            'motion_z': self.motion.z,
            'motion_roll': self.motion.euler[0],
            'motion_pitch': self.motion.euler[1],
            'motion_yaw': self.motion.euler[2],
            'num_features': 0,
            'num_matches': 0,

            'abs_error_x': np.nan,
            'abs_error_y': np.nan,
            'abs_error_z': np.nan,
            'abs_error_length': np.nan,
            'abs_error_direction': np.nan,
            'abs_rot_error': np.nan,

            'trans_error_x': np.nan,
            'trans_error_y': np.nan,
            'trans_error_z': np.nan,
            'trans_error_length': np.nan,
            'trans_error_direction': np.nan,
            'rot_error': np.nan,

            'trans_noise_x': np.nan,
            'trans_noise_y': np.nan,
            'trans_noise_z': np.nan,
            'trans_noise_length': np.nan,
            'trans_noise_direction': np.nan,
            'rot_noise': np.nan,

            'system_column': 823.3,
            'runtime_column': 42
        })
        self.assertEqual(expected_properties, frame_error.get_properties())


class TestFrameErrorDatabase(unittest.TestCase):
    system = None
    image_source = None
    trial_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        cls.system = mock_types.MockSystem()
        cls.system.save()

        cls.image_source = mock_types.MockImageSource()
        cls.image_source.save()

        cls.trial_result = mock_types.MockTrialResult(system=cls.system, image_source=cls.image_source, success=True)
        cls.trial_result.save()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        mock_types.MockSystem.objects.all().delete()
        mock_types.MockImageSource.objects.all().delete()
        mock_types.MockMetric.objects.all().delete()
        mock_types.MockTrialResult.objects.all().delete()
        FrameError.objects.all().delete()
        Image._mongometa.collection.drop()
        im_manager.set_image_manager(None)
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = Image(
            pixels=pixels,
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        image.save()
        frame_error = FrameError(
            trial_result=self.trial_result,
            repeat=1,
            timestamp=1.3,
            image=image,
            tracking=TrackingState.OK,
            motion=Transform((1.2, 0.1, -0.03), (0, 1, 0, 0)),
            processing_time=0.223,
            num_features=423,
            num_matches=238,
            absolute_error=PoseError(
                x=10,
                y=11,
                z=12,
                length=np.sqrt(100 + 121 + 144),
                direction=np.pi / 7,
                rot=np.pi / 5
            ),
            relative_error=PoseError(
                x=13,
                y=14,
                z=15,
                length=np.sqrt(169 + 196 + 225),
                direction=np.pi / 36,
                rot=np.pi / 2
            ),
            noise=PoseError(
                x=16,
                y=17,
                z=18,
                length=np.sqrt(256 + 289 + 324),
                direction=np.pi / 8,
                rot=np.pi / 16
            )
        )

        # Clear the models
        FrameError._mongometa.collection.drop()

        # Save the model
        frame_error.save()

        # Load all the entities
        all_entities = list(FrameError.objects.all())
        self.assertEqual(len(all_entities), 1)

        # For some reason, we need to force the load of these properties for the objects to compare equal
        loaded_frame_error = all_entities[0]
        self.assertEqual(frame_error.image, image)
        self.assertEqual(frame_error.tracking, loaded_frame_error.tracking)
        self.assertEqual(frame_error.absolute_error, loaded_frame_error.absolute_error)
        self.assertEqual(frame_error.relative_error, loaded_frame_error.relative_error)
        self.assertEqual(frame_error.noise, loaded_frame_error.noise)

        self.assertEqual(frame_error, loaded_frame_error)
        loaded_frame_error.delete()

    def test_required_fields_are_required(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = Image(
            pixels=pixels,
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        image.save()

        # No repeat
        frame_error = FrameError(
            timestamp=1.3,
            image=image,
            motion=Transform((1.2, 0.1, -0.03), (0, 1, 0, 0))
        )
        with self.assertRaises(ValidationError):
            frame_error.save()

        # No timestamp
        frame_error = FrameError(
            repeat=1,
            image=image,
            motion=Transform((1.2, 0.1, -0.03), (0, 1, 0, 0)),
        )
        with self.assertRaises(ValidationError):
            frame_error.save()

        # No image
        frame_error = FrameError(
            repeat=1,
            timestamp=1.3,
            motion=Transform((1.2, 0.1, -0.03), (0, 1, 0, 0))
        )
        with self.assertRaises(ValidationError):
            frame_error.save()

        # No motion
        frame_error = FrameError(
            repeat=1,
            timestamp=1.3,
            image=image
        )
        with self.assertRaises(ValidationError):
            frame_error.save()

    def test_stores_and_loads_without_saving_images(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = Image(
            pixels=pixels,
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        image.save()

        prev_image_manager = im_manager.get()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=False)
        im_manager.set_image_manager(image_manager)

        frame_error = FrameError(
            trial_result=self.trial_result,
            repeat=1,
            timestamp=1.3,
            image=image,
            tracking=TrackingState.OK,
            motion=Transform((1.2, 0.1, -0.03), (0, 1, 0, 0)),
            processing_time=0.223,
            num_features=423,
            num_matches=238,
            absolute_error=PoseError(
                x=10,
                y=11,
                z=12,
                length=np.sqrt(100 + 121 + 144),
                direction=np.pi / 7,
                rot=np.pi / 5
            ),
            relative_error=PoseError(
                x=13,
                y=14,
                z=15,
                length=np.sqrt(169 + 196 + 225),
                direction=np.pi / 36,
                rot=np.pi / 2
            ),
            noise=PoseError(
                x=16,
                y=17,
                z=18,
                length=np.sqrt(256 + 289 + 324),
                direction=np.pi / 8,
                rot=np.pi / 16
            )
        )

        # Clear the models
        FrameError._mongometa.collection.drop()

        # Save the model
        frame_error.save(cascade=True)

        # Load all the entities
        all_entities = list(FrameError.objects.all())
        self.assertEqual(len(all_entities), 1)

        # For some reason, we need to force the load of these properties for the objects to compare equal
        loaded_frame_error = all_entities[0]
        self.assertEqual(frame_error.image, image)
        self.assertEqual(frame_error.tracking, loaded_frame_error.tracking)
        self.assertEqual(frame_error.absolute_error, loaded_frame_error.absolute_error)
        self.assertEqual(frame_error.relative_error, loaded_frame_error.relative_error)
        self.assertEqual(frame_error.noise, loaded_frame_error.noise)

        self.assertEqual(frame_error, loaded_frame_error)
        loaded_frame_error.delete()

        # restore the image manager
        im_manager.set_image_manager(prev_image_manager)


# ------------------------- FRAME ERROR RESULT -------------------------


class TestFrameErrorResult(unittest.TestCase):

    @staticmethod
    def make_trial_errors(mock_trial, mock_images):
        return [
            TrialErrors(
                frame_errors=[
                    FrameError(
                        repeat=repeat,
                        timestamp=1.3 * idx,
                        image=mock_image,
                        trial_result=mock_trial,
                        tracking=TrackingState.OK,
                        motion=Transform((1.2, 0.1, -0.03), (-0.5, 0.5, -0.5, -0.5)),
                        num_features=423,
                        num_matches=238,
                        absolute_error=make_pose_error(Transform((1.1 * idx, 0, 0)), Transform((idx, 0, 0))),
                        relative_error=make_pose_error(Transform((1.1, 0, 0)), Transform((1, 0, 0))),
                        noise=make_pose_error(Transform((1.05, 0, 0)), Transform((1, 0, 0)))
                    ) for idx, mock_image in enumerate(mock_images)
                ]
            ) for repeat in range(2)
        ]

    def test_get_columns_includes_system_columns(self):
        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_columns.return_value = {'mock_column_1', 'mock_column_2'}
        mock_image_source.get_columns.return_value = set()
        mock_metric.get_columns.return_value = set()

        result = FrameErrorResult(system=mock_system, image_source=mock_image_source, metric=mock_metric)
        columns = result.get_columns()
        self.assertIn('mock_column_1', columns)
        self.assertIn('mock_column_2', columns)

    def test_get_columns_includes_image_source_columns(self):
        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_columns.return_value = set()
        mock_image_source.get_columns.return_value = {'mock_column_1', 'mock_column_2'}
        mock_metric.get_columns.return_value = set()

        result = FrameErrorResult(system=mock_system, image_source=mock_image_source, metric=mock_metric)
        columns = result.get_columns()
        self.assertIn('mock_column_1', columns)
        self.assertIn('mock_column_2', columns)

    def test_get_columns_includes_metric_columns(self):
        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_columns.return_value = set()
        mock_image_source.get_columns.return_value = set()
        mock_metric.get_columns.return_value = {'mock_column_1', 'mock_column_2'}

        result = FrameErrorResult(system=mock_system, image_source=mock_image_source, metric=mock_metric)
        columns = result.get_columns()
        self.assertIn('mock_column_1', columns)
        self.assertIn('mock_column_2', columns)

    def test_get_columns_includes_frame_error_columns(self):
        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_columns.return_value = set()
        mock_image_source.get_columns.return_value = set()
        mock_metric.get_columns.return_value = set()

        result = FrameErrorResult(system=mock_system, image_source=mock_image_source, metric=mock_metric)
        columns = result.get_columns()
        for column in {
            'repeat',
            'timestamp',
            'tracking',
            'processing_time',
            'motion_x',
            'motion_y',
            'motion_z',
            'motion_roll',
            'motion_pitch',
            'motion_yaw',
            'num_features',
            'num_matches',

            'abs_error_x',
            'abs_error_y',
            'abs_error_z',
            'abs_error_length',
            'abs_error_direction',
            'abs_rot_error',

            'trans_error_x',
            'trans_error_y',
            'trans_error_z',
            'trans_error_length',
            'trans_error_direction',
            'rot_error',

            'trans_noise_x',
            'trans_noise_y',
            'trans_noise_z',
            'trans_noise_length',
            'trans_noise_direction',
            'rot_noise',

            'num_features',
            'num_matches',
        }:
            self.assertIn(column, columns)

    def test_get_columns_includes_image_columns(self):
        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_columns.return_value = set()
        mock_image_source.get_columns.return_value = set()
        mock_metric.get_columns.return_value = set()

        result = FrameErrorResult(system=mock_system, image_source=mock_image_source, metric=mock_metric,
                                  image_columns=['mock_column_1', 'mock_column_2'])
        columns = result.get_columns()
        self.assertIn('mock_column_1', columns)
        self.assertIn('mock_column_2', columns)

    def test_get_columns_includes_values_computable_from_the_tracking_state(self):
        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_columns.return_value = set()
        mock_image_source.get_columns.return_value = set()
        mock_metric.get_columns.return_value = set()

        result = FrameErrorResult(system=mock_system, image_source=mock_image_source, metric=mock_metric)
        columns = result.get_columns()
        for column in {
            'min_frames_lost',
            'max_frames_lost',
            'mean_frames_lost',
            'median_frames_lost',
            'std_frames_lost',
            'min_frames_found',
            'max_frames_found',
            'mean_frames_found',
            'median_frames_found',
            'std_frames_found',
            'min_times_lost',
            'max_times_lost',
            'mean_times_lost',
            'median_times_lost',
            'std_times_lost',
            'min_times_found',
            'max_times_found',
            'mean_times_found',
            'median_times_found',
            'std_times_found',
            'min_distance_lost',
            'max_distance_lost',
            'mean_distance_lost',
            'median_distance_lost',
            'std_distance_lost',
            'min_distance_found',
            'max_distance_found',
            'mean_distance_found',
            'median_distance_found',
            'std_distance_found'
        }:
            self.assertIn(column, columns)

    def test_get_results_returns_a_result_for_each_frame_error(self):
        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_trial = mock.create_autospec(mock_types.MockTrialResult)

        mock_system.get_columns.return_value = set()
        mock_system.get_columns.return_value = set()
        mock_system.get_properties.return_value = {}
        mock_image_source.get_columns.return_value = set()
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_columns.return_value = set()
        mock_metric.get_properties.return_value = {}
        mock_trial.system = mock_system
        mock_trial.settings = {}

        error_columns = {
            'repeat',
            'timestamp',
            'tracking',
            'processing_time',
            'motion_x',
            'motion_y',
            'motion_z',
            'motion_roll',
            'motion_pitch',
            'motion_yaw',
            'num_features',
            'num_matches',

            'abs_error_x',
            'abs_error_y',
            'abs_error_z',
            'abs_error_length',
            'abs_error_direction',
            'abs_rot_error',

            'trans_error_x',
            'trans_error_y',
            'trans_error_z',
            'trans_error_length',
            'trans_error_direction',
            'rot_error',

            'trans_noise_x',
            'trans_noise_y',
            'trans_noise_z',
            'trans_noise_length',
            'trans_noise_direction',
            'rot_noise'
        }

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_columns.return_value = set()
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        results = subject.get_results()

        self.assertEqual(len(trial_errors) * len(mock_images), len(results))
        for result in results:
            for column in error_columns:
                self.assertIn(column, result)

    def test_get_results_returns_only_requested_properties(self):
        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = {}
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_properties.return_value = {}
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source)

        error_columns = {
            'repeat',
            'timestamp',
            'abs_error_x',
            'abs_error_length',
            'trans_error_y',
            'trans_noise_x',
            'trans_noise_z',
            'trans_noise_length',
            'rot_noise',
            'num_matches'
        }

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        results = subject.get_results(error_columns)

        self.assertEqual(len(trial_errors) * len(mock_images), len(results))
        for result in results:
            self.assertEqual(error_columns, set(result.keys()))

    def test_get_results_returns_system_properties_on_all_results(self):
        system_properties = {'mock_column_1': 'foo', 'mock_column_2': 12.33}

        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = system_properties
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_properties.return_value = {}
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source)

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.pk = bson.ObjectId()
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        results = subject.get_results()
        for result in results:
            for column_name, value in system_properties.items():
                self.assertIn(column_name, result)
                self.assertEqual(value, result[column_name])

    def test_get_results_passes_column_list_and_settings_through_to_system_get_properties(self):
        columns = {'trans_error_x', 'mock_column_1'}
        settings = {'in_fx': 320, 'in_fy': 319.2}

        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = {'mock_column_1': 'foo', 'mock_column_2': 12.33}
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_properties.return_value = {}
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source, settings=settings)

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        subject.get_results(columns)
        self.assertEqual(mock.call(columns, settings), mock_system.get_properties.call_args)

    def test_get_results_returns_image_source_properties_on_all_results(self):
        image_source_properties = {'mock_column_1': 'foo', 'mock_column_2': 12.33}

        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = {}
        mock_image_source.get_properties.return_value = image_source_properties
        mock_metric.get_properties.return_value = {}
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source)

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        results = subject.get_results()
        for result in results:
            for column_name, value in image_source_properties.items():
                self.assertIn(column_name, result)
                self.assertEqual(value, result[column_name])

    def test_get_results_passes_column_list_through_to_image_source_get_properties(self):
        columns = {'trans_error_x', 'mock_column_1'}

        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = {}
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_properties.return_value = {}
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source)

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        subject.get_results(columns)
        self.assertEqual(mock.call(columns), mock_image_source.get_properties.call_args)

    def test_get_results_returns_metric_properties_on_all_results(self):
        metric_properties = {'mock_column_1': 'foo', 'mock_column_2': 12.33}

        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = {}
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_properties.return_value = metric_properties
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source)

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        results = subject.get_results()
        for result in results:
            for column_name, value in metric_properties.items():
                self.assertIn(column_name, result)
                self.assertEqual(value, result[column_name])

    def test_get_results_passes_column_list_through_to_metric_get_properties(self):
        columns = {'trans_error_x', 'mock_column_1'}

        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = {}
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_properties.return_value = {}
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source)

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        subject.get_results(columns)
        self.assertEqual(mock.call(columns), mock_metric.get_properties.call_args)

    def test_get_results_returns_image_properties_on_all_results(self):
        image_properties = {'mock_column_1': 'foo', 'mock_column_2': 12.33}

        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = {}
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_properties.return_value = {}
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source)

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for idx, mock_image in enumerate(mock_images):
            mock_image.get_properties.return_value = image_properties
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        results = subject.get_results()
        for result in results:
            for column_name, value in image_properties.items():
                self.assertIn(column_name, result)
                self.assertEqual(value, result[column_name])

    def test_get_results_passes_column_list_through_to_image_get_properties(self):
        columns = {'trans_error_x', 'mock_column_1'}

        mock_system = mock.create_autospec(mock_types.MockSystem)
        mock_image_source = mock.create_autospec(mock_types.MockImageSource)
        mock_metric = mock.create_autospec(mock_types.MockMetric)
        mock_system.get_properties.return_value = {}
        mock_image_source.get_properties.return_value = {}
        mock_metric.get_properties.return_value = {}
        mock_trial = mock_types.MockTrialResult(system=mock_system, image_source=mock_image_source)

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = FrameErrorResult(system=mock_system, image_source=mock_image_source,
                                   metric=mock_metric, errors=trial_errors)
        subject.get_results(columns)
        for mock_image in mock_images:
            self.assertEqual(mock.call(columns), mock_image.get_properties.call_args)


class TestFrameErrorResultDatabase(unittest.TestCase):
    system = None
    image_source = None
    metric = None
    trial_result = None
    images = []
    trial_errors = []
    frame_errors = []

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        cls.system = mock_types.MockSystem()
        cls.system.save()

        cls.image_source = mock_types.MockImageSource()
        cls.image_source.save()

        cls.metric = mock_types.MockMetric()
        cls.metric.save()

        cls.trial_result = mock_types.MockTrialResult(system=cls.system, image_source=cls.image_source, success=True)
        cls.trial_result.save()

        cls.images = []
        for idx in range(3):
            pixels = np.random.uniform(0, 255, (5, 5))
            image = Image(
                pixels=pixels,
                metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
            )
            image.save()
            cls.images.append(image)

        for repeat in range(3):
            frame_errors = []
            for idx, image in enumerate(cls.images):
                true_pose = Transform(
                    location=(3.5 * idx, 0.7 * idx, 0),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 12),
                    w_first=True
                )
                est_pose = Transform(
                    location=(3.7 * idx, 0.6 * idx, -0.01 * idx),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), 1.1 * idx * np.pi / 12),
                    w_first=True
                )
                true_motion = Transform(
                    location=(3.5 * (idx - 1), 0.7 * (idx - 1), 0),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), (idx - 1) * np.pi / 12),
                    w_first=True
                ).find_relative(true_pose)
                est_motion = Transform(
                    location=(3.7 * (idx - 1), 0.6 * (idx - 1), -0.01 * (idx - 1)),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), 1.1 * (idx - 1) * np.pi / 12),
                    w_first=True
                ).find_relative(est_pose)
                avg_pose = Transform(
                    location=(3.6 * idx, 0.65 * idx, -0.015 * idx),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), 1.1 * idx * np.pi / 12),
                    w_first=True
                )
                frame_error = FrameError(
                    trial_result=cls.trial_result,
                    repeat=repeat,
                    timestamp=1.3 * idx,
                    image=image,
                    motion=true_motion,
                    tracking=TrackingState.OK,
                    num_features=423,
                    num_matches=238,
                    absolute_error=make_pose_error(est_pose, true_pose),
                    relative_error=make_pose_error(est_motion, true_motion),
                    noise=make_pose_error(est_pose, avg_pose)
                )
                frame_error.save()
                cls.frame_errors.append(frame_error)
                frame_errors.append(frame_error)
            cls.trial_errors.append(TrialErrors(
                frame_errors=frame_errors,
                frames_lost=[],
                frames_found=[len(cls.images)],
                times_lost=[],
                times_found=[1.3 * len(cls.images)],
                distances_lost=[],
                distances_found=[float(np.linalg.norm((3.5 * len(cls.images), 0.7 * len(cls.images), 0)))]
            ))

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        mock_types.MockSystem.objects.all().delete()
        mock_types.MockImageSource.objects.all().delete()
        mock_types.MockMetric.objects.all().delete()
        mock_types.MockTrialResult.objects.all().delete()
        FrameError._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads(self):
        # Save the model
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True,
            system=self.system,
            image_source=self.image_source,
            errors=self.trial_errors
        )
        result.save(cascade=True)

        # Load all the entities
        all_entities = list(FrameErrorResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], result)
        all_entities[0].delete()

    def test_stores_and_loads_failed(self):
        # Save the model
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=False,
            system=self.system,
            image_source=self.image_source,
            errors=[]
        )
        result.save()

        # Load all the entities
        all_entities = list(FrameErrorResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], result)
        all_entities[0].delete()

    def test_required_fields_are_required(self):
        # no metric
        result = FrameErrorResult(
            trial_results=[self.trial_result],
            success=True,
            system=self.system,
            image_source=self.image_source,
            errors=self.trial_errors
        )
        with self.assertRaises(ValidationError):
            result.save()

        # no trial results
        result = FrameErrorResult(
            metric=self.metric,
            success=True,
            system=self.system,
            image_source=self.image_source,
            errors=self.trial_errors
        )
        with self.assertRaises(ValidationError):
            result.save()

        # no empty trial results
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[],
            success=True,
            system=self.system,
            image_source=self.image_source,
            errors=self.trial_errors
        )
        with self.assertRaises(ValidationError):
            result.save()

        # no success
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            system=self.system,
            image_source=self.image_source,
            errors=self.trial_errors
        )
        with self.assertRaises(ValidationError):
            result.save()

        # no system
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True,
            image_source=self.image_source,
            errors=self.frame_errors
        )
        with self.assertRaises(ValidationError):
            result.save()

        # no image source
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True,
            system=self.system,
            errors=self.trial_errors
        )
        with self.assertRaises(ValidationError):
            result.save()

        # no errors
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True,
            system=self.system,
            image_source=self.image_source
        )
        with self.assertRaises(ValidationError):
            result.save()

    def test_delete_removes_frame_errors(self):
        trial_errors = []
        all_frame_errors = []
        for repeat in range(3):
            frame_errors = []
            for idx, image in enumerate(self.images):
                true_pose = Transform(
                    location=(3.5 * idx, 0.7 * idx, 0),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 12),
                    w_first=True
                )
                est_pose = Transform(
                    location=(3.7 * idx, 0.6 * idx, -0.01 * idx),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), 1.1 * idx * np.pi / 12),
                    w_first=True
                )
                true_motion = Transform(
                    location=(3.5 * (idx - 1), 0.7 * (idx - 1), 0),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), (idx - 1) * np.pi / 12),
                    w_first=True
                ).find_relative(true_pose)
                est_motion = Transform(
                    location=(3.7 * (idx - 1), 0.6 * (idx - 1), -0.01 * (idx - 1)),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), 1.1 * (idx - 1) * np.pi / 12),
                    w_first=True
                ).find_relative(est_pose)
                avg_pose = Transform(
                    location=(3.6 * idx, 0.65 * idx, -0.015 * idx),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), 1.1 * idx * np.pi / 12),
                    w_first=True
                )
                frame_error = FrameError(
                    trial_result=self.trial_result,
                    repeat=repeat,
                    timestamp=1.3 * idx,
                    image=image,
                    motion=true_motion,
                    tracking=TrackingState.OK,
                    num_features=423,
                    num_matches=238,
                    absolute_error=make_pose_error(est_pose, true_pose),
                    relative_error=make_pose_error(est_motion, true_motion),
                    noise=make_pose_error(est_pose, avg_pose)
                )
                frame_error.save()
                frame_errors.append(frame_error)
            trial_errors.append(TrialErrors(
                frame_errors=frame_errors
            ))
            all_frame_errors.extend(frame_errors)
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True,
            system=self.system,
            image_source=self.image_source,
            errors=trial_errors
        )
        result.save()

        frame_error_ids = [err.pk for err in all_frame_errors]
        result_id = result.pk

        self.assertEqual(len(all_frame_errors), FrameError.objects.raw({'_id': {'$in': frame_error_ids}}).count())
        self.assertEqual(1, FrameErrorResult.objects.raw({'_id': result_id}).count())

        result.delete()

        self.assertEqual(0, FrameError.objects.raw({'_id': {'$in': frame_error_ids}}).count())
        self.assertEqual(0, FrameErrorResult.objects.raw({'_id': result_id}).count())
