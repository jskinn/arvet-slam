import unittest
import unittest.mock as mock
import bson
import pymodm
from pymodm.errors import ValidationError
import numpy as np
import transforms3d as tf3d

import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as image_manager
from arvet.util.transform import Transform
from arvet.metadata.image_metadata import make_metadata, ImageSourceType
import arvet.core.tests.mock_types as mock_types
from arvet.core.image import Image
from arvet.core.system import VisionSystem
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult
from arvet_slam.metrics.frame_error.frame_error_result import make_pose_error, PoseError, make_frame_error, FrameError,\
    TrialErrors, make_frame_error_result, FrameErrorResult


class CountedImageSource(mock_types.MockImageSource):
    instance_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CountedImageSource.instance_count += 1


class CountedImage(Image):
    instance_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CountedImage.instance_count += 1


class CountedSystem(mock_types.MockSystem):
    instance_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CountedSystem.instance_count += 1


class CountedTrialResult(SLAMTrialResult):
    instance_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CountedTrialResult.instance_count += 1


class CountedMetric(mock_types.MockMetric):
    instance_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CountedMetric.instance_count += 1


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


class TestFrameError(unittest.TestCase):

    def setUp(self) -> None:
        self.system = mock.Mock(spec=VisionSystem)
        self.system.pk = bson.ObjectId()
        self.system.get_columns.return_value = {'system_column', 'runtime_column'}
        self.system.get_properties.side_effect = mock_system_get_properties
        self.trial = mock_types.MockTrialResult(system=self.system, settings={'runtime_column': 42})

        self.pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.image = Image(
            _id=bson.ObjectId(),
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
        self.loop_distances = [0.223, 1.93]
        self.loop_angles = [np.pi / 22, np.pi / 180]
        self.num_features = 423
        self.num_matches = 238

        frame_result = FrameResult(
            timestamp=self.timestamp,
            image=self.image,
            processing_time=self.processing_time,
            pose=self.image.camera_pose,
            motion=self.motion,
            estimated_pose=Transform(),
            estimated_motion=Transform(),
            tracking_state=self.tracking,
            num_features=self.num_features,
            num_matches=self.num_matches
        )

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
        self.systemic_error = PoseError(
            x=19,
            y=20,
            z=21,
            length=np.sqrt(19 * 19 + 400 + 21 * 21),
            direction=np.pi / 9,
            rot=np.pi / 27
        )

        self.frame_error = make_frame_error(
            repeat_index=self.repeat,
            image=self.image,
            trial_result=self.trial,
            system=self.system,
            frame_result=frame_result,
            loop_distances=self.loop_distances,
            loop_angles=self.loop_angles,
            absolute_error=self.absolute_error,
            relative_error=self.relative_error,
            noise=self.noise,
            systemic_error=self.systemic_error
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

            'is_loop_closure': len(self.loop_distances) > 0,
            'num_loop_closures': len(self.loop_distances),
            'max_loop_closure_distance': max(self.loop_distances),
            'min_loop_closure_distance': min(self.loop_distances),
            'mean_loop_closure_distance': np.mean(self.loop_distances),
            'max_loop_closure_angle': max(self.loop_angles),
            'min_loop_closure_angle': min(self.loop_angles),
            'mean_loop_closure_angle': np.mean(self.loop_angles),

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
            'rot_noise': self.noise.rot,

            'systemic_x': self.systemic_error.x,
            'systemic_y': self.systemic_error.y,
            'systemic_z': self.systemic_error.z,
            'systemic_length': self.systemic_error.length,
            'systemic_direction': self.systemic_error.direction,
            'systemic_rot': self.systemic_error.rot
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
            'rot_noise': self.noise.rot,

            'systemic_x': self.systemic_error.x,
            'systemic_y': self.systemic_error.y,
            'systemic_z': self.systemic_error.z,
            'systemic_length': self.systemic_error.length,
            'systemic_direction': self.systemic_error.direction,
            'systemic_rot': self.systemic_error.rot
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

    def test_make_frame_error_uses_image_from_frame_result_if_None(self):
        frame_result = FrameResult(
            timestamp=self.timestamp,
            image=self.image,
            processing_time=self.processing_time,
            pose=self.image.camera_pose,
            motion=self.motion,
            estimated_pose=Transform(),
            estimated_motion=Transform(),
            tracking_state=self.tracking,
            num_features=self.num_features,
            num_matches=self.num_matches
        )
        pose_error = PoseError(
            x=10, y=11, z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 7, rot=np.pi / 5
        )
        frame_error = make_frame_error(
            trial_result=self.trial,
            frame_result=frame_result,
            image=None,
            system=None,
            repeat_index=self.repeat,
            loop_distances=self.loop_distances,
            loop_angles=self.loop_angles,
            absolute_error=pose_error,
            relative_error=pose_error,
            noise=pose_error,
            systemic_error=pose_error
        )
        self.assertEqual(self.image, frame_error.image)

    def test_make_frame_error_uses_image_from_frame_result_if_doesnt_match(self):
        image2 = Image(
            _id=bson.ObjectId(), pixels=self.pixels, metadata=make_metadata(self.pixels)
        )
        frame_result = FrameResult(
            timestamp=self.timestamp,
            image=self.image,
            processing_time=self.processing_time,
            pose=self.image.camera_pose,
            motion=self.motion,
            estimated_pose=Transform(),
            estimated_motion=Transform(),
            tracking_state=self.tracking,
            num_features=self.num_features,
            num_matches=self.num_matches
        )
        pose_error = PoseError(
            x=10, y=11, z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 7, rot=np.pi / 5
        )
        frame_error = make_frame_error(
            trial_result=self.trial,
            frame_result=frame_result,
            image=image2,
            system=None,
            repeat_index=self.repeat,
            loop_distances=self.loop_distances,
            loop_angles=self.loop_angles,
            absolute_error=pose_error,
            relative_error=pose_error,
            noise=pose_error,
            systemic_error=pose_error
        )
        self.assertEqual(self.image, frame_error.image)

    def test_make_frame_error_fills_out_necessary_data_for_properties(self):
        frame_result = FrameResult(
            timestamp=self.timestamp,
            image=self.image,
            processing_time=self.processing_time,
            pose=self.image.camera_pose,
            motion=self.motion,
            estimated_pose=Transform(),
            estimated_motion=Transform(),
            tracking_state=self.tracking,
            num_features=self.num_features,
            num_matches=self.num_matches
        )
        absolute_error = PoseError(
            x=10, y=11, z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 7, rot=np.pi / 5
        )
        relative_error = PoseError(
            x=13,
            y=14,
            z=15,
            length=np.sqrt(169 + 196 + 225),
            direction=np.pi / 36,
            rot=np.pi / 2
        )
        noise = PoseError(
            x=16,
            y=17,
            z=18,
            length=np.sqrt(256 + 289 + 324),
            direction=np.pi / 8,
            rot=np.pi / 16
        )
        systemic_error = PoseError(
            x=19,
            y=20,
            z=21,
            length=np.sqrt(19 * 19 + 400 + 21 * 21),
            direction=np.pi / 9,
            rot=np.pi / 27
        )
        frame_error = make_frame_error(
            trial_result=self.trial,
            frame_result=frame_result,
            image=None,
            system=None,
            repeat_index=self.repeat,
            loop_distances=self.loop_distances,
            loop_angles=self.loop_angles,
            absolute_error=absolute_error,
            relative_error=relative_error,
            noise=noise,
            systemic_error=systemic_error
        )
        expected_properties = dict(self.image.get_properties())
        expected_properties.update(self.system.get_properties(self.trial.settings))
        expected_properties.update({
            'repeat': self.repeat,
            'timestamp': self.timestamp,
            'tracking': True,
            'processing_time': self.processing_time,
            'motion_x': self.motion.x,
            'motion_y': self.motion.y,
            'motion_z': self.motion.z,
            'motion_roll': self.motion.euler[0],
            'motion_pitch': self.motion.euler[1],
            'motion_yaw': self.motion.euler[2],
            'num_features': self.num_features,
            'num_matches': self.num_matches,

            'is_loop_closure': len(self.loop_distances) > 0,
            'num_loop_closures': len(self.loop_distances),
            'max_loop_closure_distance': max(self.loop_distances),
            'min_loop_closure_distance': min(self.loop_distances),
            'mean_loop_closure_distance': np.mean(self.loop_distances),
            'max_loop_closure_angle': max(self.loop_angles),
            'min_loop_closure_angle': min(self.loop_angles),
            'mean_loop_closure_angle': np.mean(self.loop_angles),

            'abs_error_x': absolute_error.x,
            'abs_error_y': absolute_error.y,
            'abs_error_z': absolute_error.z,
            'abs_error_length': absolute_error.length,
            'abs_error_direction': absolute_error.direction,
            'abs_rot_error': absolute_error.rot,

            'trans_error_x': relative_error.x,
            'trans_error_y': relative_error.y,
            'trans_error_z': relative_error.z,
            'trans_error_length': relative_error.length,
            'trans_error_direction': relative_error.direction,
            'rot_error': relative_error.rot,

            'trans_noise_x': noise.x,
            'trans_noise_y': noise.y,
            'trans_noise_z': noise.z,
            'trans_noise_length': noise.length,
            'trans_noise_direction': noise.direction,
            'rot_noise': noise.rot,

            'systemic_x': systemic_error.x,
            'systemic_y': systemic_error.y,
            'systemic_z': systemic_error.z,
            'systemic_length': systemic_error.length,
            'systemic_direction': systemic_error.direction,
            'systemic_rot': systemic_error.rot,

            'system_column': 823.3,
            'runtime_column': 42
        })
        self.assertEqual(expected_properties, frame_error.get_properties())

    def test_make_frame_error_uses_settings_from_system_from_trial_result_if_None(self):
        frame_result = FrameResult(
            timestamp=self.timestamp,
            image=self.image,
            processing_time=self.processing_time,
            pose=self.image.camera_pose,
            motion=self.motion,
            estimated_pose=Transform(),
            estimated_motion=Transform(),
            tracking_state=self.tracking,
            num_features=self.num_features,
            num_matches=self.num_matches
        )
        pose_error = PoseError(
            x=10, y=11, z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 7, rot=np.pi / 5
        )
        frame_error = make_frame_error(
            trial_result=self.trial,
            frame_result=frame_result,
            image=None,
            system=None,
            loop_distances=self.loop_distances,
            loop_angles=self.loop_angles,
            repeat_index=self.repeat,
            absolute_error=pose_error,
            relative_error=pose_error,
            noise=pose_error,
            systemic_error=pose_error
        )
        expected_properties = self.system.get_properties(None, self.trial.settings)
        self.assertEqual(expected_properties, frame_error.system_properties)

    def test_make_frame_error_uses_settings_from_system_from_trial_result_if_doesnt_match(self):
        alt_system = mock.Mock(spec=VisionSystem)
        alt_system.pk = bson.ObjectId()
        alt_system.get_properties.return_value = {'foo': 'bar', 'baz': 1066}
        frame_result = FrameResult(
            timestamp=self.timestamp,
            image=self.image,
            processing_time=self.processing_time,
            pose=self.image.camera_pose,
            motion=self.motion,
            estimated_pose=Transform(),
            estimated_motion=Transform(),
            tracking_state=self.tracking,
            num_features=self.num_features,
            num_matches=self.num_matches
        )
        pose_error = PoseError(
            x=10, y=11, z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 7, rot=np.pi / 5
        )
        frame_error = make_frame_error(
            trial_result=self.trial,
            frame_result=frame_result,
            image=None,
            system=alt_system,
            repeat_index=self.repeat,
            loop_distances=self.loop_distances,
            loop_angles=self.loop_angles,
            absolute_error=pose_error,
            relative_error=pose_error,
            noise=pose_error,
            systemic_error=pose_error
        )
        expected_properties = self.system.get_properties(None, self.trial.settings)
        self.assertEqual(expected_properties, frame_error.system_properties)

    def test_make_frame_error_raises_exception_if_loop_distances_does_not_match_loop_angles(self):
        frame_result = FrameResult(
            timestamp=self.timestamp,
            image=self.image,
            processing_time=self.processing_time,
            pose=self.image.camera_pose,
            motion=self.motion,
            estimated_pose=Transform(),
            estimated_motion=Transform(),
            tracking_state=self.tracking,
            num_features=self.num_features,
            num_matches=self.num_matches
        )
        with self.assertRaises(ValueError):
            make_frame_error(
                repeat_index=self.repeat,
                image=self.image,
                trial_result=self.trial,
                system=self.system,
                frame_result=frame_result,
                loop_distances=self.loop_distances,
                loop_angles=self.loop_angles + [np.pi / 47],
                absolute_error=self.absolute_error,
                relative_error=self.relative_error,
                noise=self.noise,
                systemic_error=self.systemic_error
            )


class TestFrameErrorDatabase(unittest.TestCase):
    system = None
    image_source = None
    trial_result = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager()

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
        dbconn.tear_down_image_manager()

    def test_stores_and_loads(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = Image(
            pixels=pixels, image_group='group',
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
            pixels=pixels, image_group='test',
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
            image_group='test',
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
        with image_manager.get().get_group(self.image_source.get_image_group(), allow_write=False):
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

    def test_make_frame_error_output_stores_and_loads(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = Image(
            pixels=pixels, image_group='test',
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        image.save()
        frame_result = FrameResult(
            timestamp=1.3,
            image=image,
            processing_time=0.223,
            pose=Transform(),
            motion=Transform((1.2, 0.1, -0.03), (0, 1, 0, 0)),
            estimated_pose=Transform(),
            estimated_motion=Transform(),
            tracking_state=TrackingState.OK,
            num_features=423,
            num_matches=238
        )

        frame_error = make_frame_error(
            trial_result=self.trial_result,
            frame_result=frame_result,
            image=image,
            system=self.trial_result.system,
            repeat_index=1,
            loop_distances=[22.3, 0.166],
            loop_angles=[np.pi / 38, 16 * np.pi / 237],
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
            ),
            systemic_error=PoseError(
                x=19,
                y=20,
                z=21,
                length=np.sqrt(19 * 19 + 400 + 21 * 21),
                direction=np.pi / 27,
                rot=np.pi / 19
            )
        )

        # Clear the models
        FrameError.objects.all().delete()

        # Save the model
        frame_error.save()

        # Load all the entities
        all_entities = list(FrameError.objects.all())
        self.assertEqual(len(all_entities), 1)

        loaded_frame_error = all_entities[0]
        self.assertEqual(frame_error.image, loaded_frame_error.image)
        self.assertEqual(frame_error.trial_result, loaded_frame_error.trial_result)
        self.assertEqual(frame_error.repeat, loaded_frame_error.repeat)
        self.assertEqual(frame_error.timestamp, loaded_frame_error.timestamp)
        self.assertEqual(frame_error.motion, loaded_frame_error.motion)
        self.assertEqual(frame_error.processing_time, loaded_frame_error.processing_time)
        self.assertEqual(frame_error.num_features, loaded_frame_error.num_features)
        self.assertEqual(frame_error.num_matches, loaded_frame_error.num_matches)
        self.assertEqual(frame_error.tracking, loaded_frame_error.tracking)
        self.assertEqual(frame_error.loop_distances, loaded_frame_error.loop_distances)
        self.assertEqual(frame_error.loop_angles, loaded_frame_error.loop_angles)
        self.assertEqual(frame_error.absolute_error, loaded_frame_error.absolute_error)
        self.assertEqual(frame_error.relative_error, loaded_frame_error.relative_error)
        self.assertEqual(frame_error.noise, loaded_frame_error.noise)
        self.assertEqual(frame_error.systemic_error, loaded_frame_error.systemic_error)
        self.assertEqual(frame_error.system_properties, loaded_frame_error.system_properties)
        self.assertEqual(frame_error.image_properties, loaded_frame_error.image_properties)
        loaded_frame_error.delete()

    def test_make_frame_error_doesnt_dereference_frame_result_image_if_valid(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = CountedImage(
            pixels=pixels, image_group='test',
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        image.save()

        system = mock_types.MockSystem()
        system.save()

        image_source = mock_types.MockImageSource()
        image_source.save()

        trial_result = SLAMTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            settings={'foo': 'bar'},
            has_scale=True,
            results=[FrameResult(
                timestamp=0.1,
                image=image,
                processing_time=0.023,
                pose=Transform(),
                motion=Transform(),
                estimated_pose=Transform(),
                estimated_motion=None,
                tracking_state=TrackingState.OK,
                num_features=23,
                num_matches=12
            )],
        )
        trial_result.save()
        trial_result_id = trial_result.pk
        del trial_result    # Remove both the SLAM trial result and the frame result from memory, so it must be loaded

        CountedImage.instance_count = 0
        trial_result = SLAMTrialResult.objects.get({'_id': trial_result_id})
        self.assertEqual(0, CountedImage.instance_count)

        make_frame_error(
            trial_result=trial_result,
            frame_result=trial_result.results[0],
            image=image,
            system=system,
            repeat_index=1,
            loop_distances=[],
            loop_angles=[],
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
            ),
            systemic_error=PoseError(
                x=19,
                y=20,
                z=21,
                length=np.sqrt(19 * 19 + 400 + 21 * 21),
                direction=np.pi / 27,
                rot=np.pi / 19
            )
        )
        self.assertEqual(0, CountedImage.instance_count)

        trial_result.delete()
        system.delete()
        image_source.delete()
        image.delete()

    def test_make_frame_error_doesnt_dereference_system_if_valid(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = Image(
            pixels=pixels, image_group='test',
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        image.save()

        system = CountedSystem()
        system.save()

        image_source = mock_types.MockImageSource()
        image_source.save()

        # This needs to be the most recently defined subclass for the no_auto_dereference to work
        trial_result = CountedTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            settings={'foo': 'bar'},
            has_scale=True,
            results=[FrameResult(
                timestamp=0.1,
                image=image,
                processing_time=0.023,
                pose=Transform(),
                motion=Transform(),
                estimated_pose=Transform(),
                estimated_motion=None,
                tracking_state=TrackingState.OK,
                num_features=23,
                num_matches=12
            )],
        )
        trial_result.save()
        trial_result_id = trial_result.pk
        del trial_result    # Remove both the SLAM trial result and the frame result from memory, so it must be loaded

        CountedSystem.instance_count = 0
        trial_result = CountedTrialResult.objects.get({'_id': trial_result_id})
        self.assertEqual(0, CountedSystem.instance_count)

        make_frame_error(
            trial_result=trial_result,
            frame_result=trial_result.results[0],
            image=image,
            system=system,
            repeat_index=1,
            loop_distances=[],
            loop_angles=[],
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
            ),
            systemic_error=PoseError(
                x=19,
                y=20,
                z=21,
                length=np.sqrt(19 * 19 + 400 + 21 * 21),
                direction=np.pi / 27,
                rot=np.pi / 19
            )
        )
        self.assertEqual(0, CountedSystem.instance_count)

        trial_result.delete()
        system.delete()
        image_source.delete()

    def test_get_properties_doesnt_dereference_image_or_trial_result(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = CountedImage(
            pixels=pixels, image_group='test',
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        image.save()

        system = mock_types.MockSystem()
        system.save()

        image_source = mock_types.MockImageSource()
        image_source.save()

        frame_result = FrameResult(
            timestamp=0.1,
            image=image,
            processing_time=0.023,
            pose=Transform(),
            motion=Transform(),
            estimated_pose=Transform(),
            estimated_motion=None,
            tracking_state=TrackingState.OK,
            num_features=23,
            num_matches=12
        )

        trial_result = CountedTrialResult(system=system, image_source=image_source,
                                          success=True, results=[frame_result])
        trial_result.save()

        frame_error = make_frame_error(
            trial_result=trial_result,
            frame_result=frame_result,
            image=image,
            system=system,
            repeat_index=1,
            loop_distances=[],
            loop_angles=[],
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
            ),
            systemic_error=PoseError(
                x=19,
                y=20,
                z=21,
                length=np.sqrt(19 * 19 + 400 + 21 * 21),
                direction=np.pi / 27,
                rot=np.pi / 19
            )
        )
        frame_error.save()
        frame_error_id = frame_error.pk
        del frame_error    # Delete the frame_error to clear it's references

        CountedImage.instance_count = 0
        CountedTrialResult.instance_count = 0
        frame_error = FrameError.objects.get({'_id': frame_error_id})
        self.assertEqual(0, CountedImage.instance_count)
        self.assertEqual(0, CountedTrialResult.instance_count)
        frame_error.get_properties()    # Get all the properties
        self.assertEqual(0, CountedImage.instance_count)
        self.assertEqual(0, CountedTrialResult.instance_count)

        frame_error.delete()
        trial_result.delete()
        system.delete()
        image_source.delete()
        image.delete()


# ------------------------- FRAME ERROR RESULT -------------------------


class TestFrameErrorResult(unittest.TestCase):

    def test_get_columns_includes_image_source_properties_keys(self):
        result = FrameErrorResult(image_source_properties={'mock_column_1': 'foo', 'mock_column_2': 28})
        columns = result.get_columns()
        self.assertIn('mock_column_1', columns)
        self.assertIn('mock_column_2', columns)

    def test_get_columns_includes_metric_properties_keys(self):
        result = FrameErrorResult(metric_properties={'mock_column_1': 'foo', 'mock_column_2': 28})
        columns = result.get_columns()
        self.assertIn('mock_column_1', columns)
        self.assertIn('mock_column_2', columns)

    def test_get_columns_includes_frame_error_columns(self):
        result = FrameErrorResult(frame_columns=['mock_column_1', 'mock_column_2'])
        columns = result.get_columns()
        self.assertIn('mock_column_1', columns)
        self.assertIn('mock_column_2', columns)

    def test_get_columns_includes_values_computable_from_the_tracking_state(self):
        result = FrameErrorResult()
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


class TestMakeFrameErrorResult(unittest.TestCase):

    def test_make_frame_error_result_links_to_system_from_trial_result(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        metric = mock_types.MockMetric()
        trial_result = mock_types.MockTrialResult(system=system, image_source=image_source)

        result = make_frame_error_result(metric=metric, trial_results=[trial_result], errors=[])
        self.assertEqual(system, result.system)

    def test_make_frame_error_result_copies_image_source_properties(self):
        system = mock_types.MockSystem()
        image_source = mock.create_autospec(mock_types.MockImageSource)
        metric = mock_types.MockMetric()
        trial_result = mock_types.MockTrialResult(system=system, image_source=image_source)

        image_source_properties = {'foo': 'bar', 'baz': 22}
        image_source.get_properties.return_value = image_source_properties

        result = make_frame_error_result(metric=metric, trial_results=[trial_result], errors=[])
        self.assertEqual(image_source_properties, result.image_source_properties)
        columns = result.get_columns()
        self.assertIn('foo', columns)
        self.assertIn('baz', columns)

    def test_make_frame_error_result_copies_metric_properties(self):
        system = mock_types.MockSystem()
        image_source = mock_types.MockImageSource()
        metric = mock.create_autospec(mock_types.MockMetric)
        trial_result = mock_types.MockTrialResult(system=system, image_source=image_source)

        metric_properties = {'metric_property_1': 'bar', 'metric_property_2': 22}
        metric.get_properties.return_value = metric_properties

        result = make_frame_error_result(metric=metric, trial_results=[trial_result], errors=[])
        self.assertEqual(metric_properties, result.metric_properties)
        columns = result.get_columns()
        self.assertIn('metric_property_1', columns)
        self.assertIn('metric_property_2', columns)

    def test_make_frame_error_result_caches_frame_error_columns(self):
        image = mock.create_autospec(Image)
        system = mock.create_autospec(mock_types.MockSystem)
        image_source = mock_types.MockImageSource()
        metric = mock_types.MockMetric()
        trial_result = mock_types.MockTrialResult(system=system, image_source=image_source)

        frame_result = FrameResult(
            timestamp=0.1,
            image=image,
            processing_time=0.023,
            pose=Transform(),
            motion=Transform(),
            estimated_pose=Transform(),
            estimated_motion=None,
            tracking_state=TrackingState.OK,
            num_features=23,
            num_matches=12
        )
        pose_error = PoseError(
            x=10, y=11, z=12,
            length=np.sqrt(100 + 121 + 144),
            direction=np.pi / 7, rot=np.pi / 5
        )

        # Make two frame errors with different set of columns
        system.get_properties.return_value = {'system_property_1': 'foobar'}
        image.get_properties.return_value = {'image_property_A': 3.14159}
        frame_error_1 = make_frame_error(
            trial_result=trial_result,
            frame_result=frame_result,
            image=image,
            system=system,
            repeat_index=1,
            loop_distances=[],
            loop_angles=[],
            absolute_error=pose_error,
            relative_error=pose_error,
            noise=pose_error,
            systemic_error=pose_error
        )

        system.get_properties.return_value = {'system_property_2': 65535}
        image.get_properties.return_value = {'image_property_charlie': "they're in the trees"}
        frame_error_2 = make_frame_error(
            trial_result=trial_result,
            frame_result=frame_result,
            image=image,
            system=system,
            repeat_index=1,
            loop_distances=[],
            loop_angles=[],
            absolute_error=pose_error,
            relative_error=pose_error,
            noise=pose_error,
            systemic_error=pose_error
        )
        self.assertNotEqual(frame_error_1.get_columns(), frame_error_2.get_columns())

        result = make_frame_error_result(metric=metric, trial_results=[trial_result],
                                         errors=[TrialErrors(frame_errors=[frame_error_1, frame_error_2])])
        columns = result.get_columns()
        # All the columns from all the frame errors should show up in get_columns. It promises to be quite a list.
        for column in frame_error_1.get_columns():
            self.assertIn(column, columns)
        for column in frame_error_2.get_columns():
            self.assertIn(column, columns)


class TestFrameErrorResultGetResults(unittest.TestCase):

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
                        noise=make_pose_error(Transform((1.05, 0, 0)), Transform((1, 0, 0))),
                        system_properties={'system_property': 'BAZ', f"setting_{repeat}": f"BLUE_{repeat * 8}"},
                        image_properties={'image_property': idx * 3.5}
                    ) for idx, mock_image in enumerate(mock_images)
                ]
            ) for repeat in range(2)
        ]

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
        mock_trial.image_source = mock_image_source
        mock_trial.settings = {}

        # Make images and errors
        mock_images = [mock.create_autospec(Image) for _ in range(3)]
        for mock_image in mock_images:
            mock_image.get_columns.return_value = set()
            mock_image.get_properties.return_value = {}
        trial_errors = self.make_trial_errors(mock_trial, mock_images)

        subject = make_frame_error_result(
            metric=mock_metric,
            trial_results=[mock_trial],
            errors=trial_errors
        )
        results = subject.get_results()

        self.assertEqual(len(trial_errors) * len(mock_images), len(results))
        for errors in trial_errors:
            for frame_error in errors.frame_errors:
                properties = frame_error.get_properties()
                found = False
                for result in results:
                    if all(
                            column in result and
                            (result[column] == properties[column] or (
                                    np.isnan(result[column]) and np.isnan(properties[column])
                            ))
                            for column in properties.keys()
                    ):
                        found = True
                        break
                self.assertTrue(found, f"Could not find frame result {properties} in {results}")

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

        subject = make_frame_error_result(
            metric=mock_metric,
            trial_results=[mock_trial],
            errors=trial_errors
        )
        results = subject.get_results(error_columns)

        self.assertEqual(len(trial_errors) * len(mock_images), len(results))
        for result in results:
            self.assertEqual(error_columns, set(result.keys()))


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
        dbconn.setup_image_manager()

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
                pixels=pixels, image_group='test',
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
        dbconn.tear_down_image_manager()

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

        # Re-load the entity
        loaded_result = FrameErrorResult.objects.get({'_id': result.pk})
        self.assertEqual(loaded_result, result)
        result.delete()

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

        # Re-load the entity
        loaded_result = FrameErrorResult.objects.get({'_id': result.pk})
        self.assertEqual(loaded_result, result)
        result.delete()

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

    def test_stores_and_loads_frame_errors_if_cascade(self):
        # make some unsaved frame and trial errors
        all_trial_errors = []
        for repeat in range(3):
            frame_errors = []
            tracking_start = repeat
            for idx, image in enumerate(self.images):
                true_pose = Transform(
                    location=(3.5 * idx, 0.7 * idx, 0),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), idx * np.pi / 12),
                    w_first=True
                )
                est_pose = Transform(
                    location=((3.6 + 0.1 * repeat) * idx, (0.8 - 0.15 * repeat) * idx, -0.01 * idx),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), 1.1 * idx * np.pi / 12),
                    w_first=True
                )
                true_motion = Transform(
                    location=(3.5 * (idx - 1), 0.7 * (idx - 1), 0),
                    rotation=tf3d.quaternions.axangle2quat((1, 2, 3), (idx - 1) * np.pi / 12),
                    w_first=True
                ).find_relative(true_pose)
                est_motion = Transform(
                    location=((3.6 + 0.1 * repeat) * (idx - 1), 0.6 * (idx - 1), -0.01 * (idx - 1)),
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
                    tracking=TrackingState.OK if idx > tracking_start else TrackingState.LOST,
                    num_features=423 - 6 * (repeat * idx),
                    num_matches=int(638 * (idx + 1) / (repeat + 3)),
                    absolute_error=make_pose_error(est_pose, true_pose) if idx > tracking_start else None,
                    relative_error=make_pose_error(est_motion, true_motion) if idx > tracking_start else None,
                    noise=make_pose_error(est_pose, avg_pose) if idx > tracking_start else None
                )
                frame_errors.append(frame_error)
            all_trial_errors.append(TrialErrors(
                frame_errors=frame_errors,
                frames_lost=[tracking_start],
                frames_found=[len(self.images) - tracking_start],
                times_lost=[1.3 * tracking_start],
                times_found=[1.3 * (len(self.images) - tracking_start)],
                distances_lost=[float(np.linalg.norm((
                    3.5 * tracking_start,
                    0.7 * tracking_start,
                    0
                )))],
                distances_found=[float(np.linalg.norm((
                    3.5 * (len(self.images) - tracking_start),
                    0.7 * (len(self.images) - tracking_start),
                    0
                )))]
            ))

        # Save the model, which should also save the trial and frame errors.
        result = FrameErrorResult(
            metric=self.metric,
            trial_results=[self.trial_result],
            success=True,
            system=self.system,
            image_source=self.image_source,
            errors=all_trial_errors
        )
        result.save(cascade=True)

        # Re-load the entity
        loaded_result = FrameErrorResult.objects.get({'_id': result.pk})
        self.assertEqual(loaded_result, result)

        # Check the referenced trial and fame results are the same
        self.assertEqual(len(all_trial_errors), len(loaded_result.errors))
        for trial_errors, loaded_trial_errors in zip(all_trial_errors, loaded_result.errors):
            self.assertEqual(trial_errors.frames_lost, loaded_trial_errors.frames_lost)
            self.assertEqual(trial_errors.frames_found, loaded_trial_errors.frames_found)
            self.assertEqual(trial_errors.times_lost, loaded_trial_errors.times_lost)
            self.assertEqual(trial_errors.times_found, loaded_trial_errors.times_found)
            self.assertEqual(trial_errors.distances_lost, loaded_trial_errors.distances_lost)
            self.assertEqual(trial_errors.distances_found, loaded_trial_errors.distances_found)
            self.assertEqual(len(trial_errors.frame_errors), len(loaded_trial_errors.frame_errors))

            for frame_error, loaded_frame_error in zip(trial_errors.frame_errors, loaded_trial_errors.frame_errors):
                self.assertEqual(frame_error.pk, loaded_frame_error.pk)
                self.assertEqual(frame_error, loaded_frame_error)

        # Clean up
        result.delete()

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

    def test_make_frame_error_result_returns_a_result_that_can_be_saved(self):
        result = make_frame_error_result(
            metric=self.metric,
            trial_results=[self.trial_result],
            errors=self.trial_errors
        )
        result.save(cascade=True)

        # Re-load the entity
        loaded_result = FrameErrorResult.objects.get({'_id': result.pk})
        self.assertEqual(loaded_result, result)
        result.delete()

    def test_get_results_does_not_dereference_referenced_objects(self):
        pixels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        image = CountedImage(
            pixels=pixels, image_group='test',
            metadata=make_metadata(pixels, source_type=ImageSourceType.SYNTHETIC)
        )
        image.save()

        image_source = CountedImageSource()
        image_source.save()

        system = CountedSystem()
        system.save()

        trial_result = CountedTrialResult(
            system=system,
            image_source=image_source,
            success=True,
            settings={'foo': 'bar'},
            has_scale=True,
            results=[FrameResult(
                timestamp=0.1,
                image=image,
                processing_time=0.023,
                pose=Transform(),
                motion=Transform(),
                estimated_pose=Transform(),
                estimated_motion=None,
                tracking_state=TrackingState.OK,
                num_features=23,
                num_matches=12
            )],
        )
        trial_result.save()

        metric = CountedMetric()
        metric.save()

        result = make_frame_error_result(
            metric=metric,
            trial_results=[trial_result],
            errors=[TrialErrors(
                frame_errors=[make_frame_error(
                    trial_result=trial_result,
                    frame_result=trial_result.results[0],
                    image=image,
                    system=system,
                    repeat_index=1,
                    loop_distances=[],
                    loop_angles=[],
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
                    ),
                    systemic_error=PoseError(
                        x=19,
                        y=20,
                        z=21,
                        length=np.sqrt(19 * 19 + 400 + 21 * 21),
                        direction=np.pi / 27,
                        rot=np.pi / 19
                    )
                )]
            )]
        )
        result.save(cascade=True)
        result_id = result.pk
        del result  # Clear the object and all its references. This will force them to be re-loaded from the database

        CountedSystem.instance_count = 0
        CountedImageSource.instance_count = 0
        CountedImage.instance_count = 0
        CountedTrialResult.instance_count = 0
        CountedMetric.instance_count = 0
        result = FrameErrorResult.objects.get({'_id': result_id})
        result.get_results()
        self.assertEqual(0, CountedSystem.instance_count)
        self.assertEqual(0, CountedImageSource.instance_count)
        self.assertEqual(0, CountedImage.instance_count)
        self.assertEqual(0, CountedTrialResult.instance_count)
        self.assertEqual(0, CountedMetric.instance_count)
