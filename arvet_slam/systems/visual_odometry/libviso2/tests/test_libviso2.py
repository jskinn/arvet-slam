# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import transforms3d as tf3d
import arvet.util.transform as tf
import arvet.util.dict_utils as du
import arvet.core.sequence_type
import arvet.core.image
import arvet.metadata.image_metadata as imeta
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.database.tests.test_entity
import arvet_slam.systems.visual_odometry.libviso2.libviso2 as viso
import arvet_slam.trials.slam.visual_slam as slam_trial


class TestLibVisO(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def get_class(self):
        return viso.LibVisOSystem

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'matcher_nms_n': np.random.randint(0, 5),
            'matcher_nms_tau': np.random.randint(0, 100),
            'matcher_match_binsize': np.random.randint(0, 100),
            'matcher_match_radius': np.random.randint(0, 500),
            'matcher_match_disp_tolerance': np.random.randint(0, 10),
            'matcher_outlier_disp_tolerance': np.random.randint(0, 10),
            'matcher_outlier_flow_tolerance': np.random.randint(0, 10),
            'matcher_multi_stage': np.random.choice([True, False]),
            'matcher_half_resolution': np.random.choice([True, False]),
            'matcher_refinement': np.random.randint(0, 3),
            'bucketing_max_features': np.random.randint(0, 10),
            'bucketing_bucket_width': np.random.randint(0, 100),
            'bucketing_bucket_height': np.random.randint(0, 100),
            'ransac_iters': np.random.randint(0, 100),
            'inlier_threshold': np.random.uniform(0.0, 3.0),
            'reweighting': np.random.choice([True, False])
        })
        return viso.LibVisOSystem(*args, **kwargs)

    def assert_models_equal(self, system1, system2):
        """
        Helper to assert that two viso systems are equal
        Libviso2 systems don't have any persistent configuration, so they're all equal with the same id.
        :param system1:
        :param system2:
        :return:
        """
        if (not isinstance(system1, viso.LibVisOSystem) or
                not isinstance(system2, viso.LibVisOSystem)):
            self.fail('object was not a LibVisOSystem')
        self.assertEqual(system1.identifier, system2.identifier)

    def test_can_start_and_stop_trial(self):
        subject = viso.LibVisOSystem()
        subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        result = subject.finish_trial()
        self.assertIsInstance(result, slam_trial.SLAMTrialResult)

    def test_simple_trial_run(self):
        # Actually run the system using mocked images
        subject = viso.LibVisOSystem()
        subject.set_camera_intrinsics(cam_intr.CameraIntrinsics(
            width=320,
            height=240,
            fx=160,
            fy=160,
            cx=160,
            cy=120
        ))
        subject.set_stereo_baseline(50)
        subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        num_frames = 50
        for time in range(num_frames):
            image = create_frame(time / num_frames)
            subject.process_image(image, 4 * time / num_frames)
        result = subject.finish_trial()
        self.assertIsInstance(result, slam_trial.SLAMTrialResult)


class TestMakeRelativePose(unittest.TestCase):

    def test_returns_transform_object(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = viso.make_relative_pose(frame_delta)
        self.assertIsInstance(pose, tf.Transform)

    def test_rearranges_location_coordinates(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPEqual((13.2, -10, 22.4), pose.location)

    def test_changes_rotation_each_axis(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        # Roll, rotation around z-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 0, 1), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around x-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((1, 0, 0), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((0, -np.pi / 6, 0), pose.euler)

        # Yaw, rotation around negative y-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 1, 0), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((0, 0, -np.pi / 6), pose.euler)

    def test_combined(self):
        frame_delta = np.identity(4)
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            frame_delta[0:3, 3] = -loc[1], -loc[2], loc[0]
            frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((-rot_axis[1], -rot_axis[2], rot_axis[0]),
                                                              rot_angle, False)
            pose = viso.make_relative_pose(frame_delta)
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


def create_frame(time):
    frame = np.zeros((320, 240), dtype=np.uint8)
    right_frame = np.zeros((320, 240), dtype=np.uint8)
    speed = 200
    f = frame.shape[1] / 2
    cx = frame.shape[1] / 2
    cy = frame.shape[0] / 2
    stars = [{
        'pos': (
            (127 * idx + 34 * idx * idx) % 400 - 200,
            (320 - 17 * idx + 7 * idx * idx) % 400 - 200,
            (183 * idx - speed * time) % 500 + 0.01),
        'width': idx % 31 + 1,
        'height': idx % 27 + 1,
        'colour': idx * 7 % 256
    } for idx in range(300)]
    stars.sort(key=lambda s: s['pos'][2], reverse=True)

    for star in stars:
        x, y, z = star['pos']

        left = int(np.round(f * ((x - star['width'] / 2) / z) + cx))
        right = int(np.round(f * ((x + star['width'] / 2) / z) + cx))

        top = int(np.round(f * ((y - star['height'] / 2) / z) + cy))
        bottom = int(np.round(f * ((y + star['height'] / 2) / z) + cy))

        left = max(0, min(frame.shape[1], left))
        right = max(0, min(frame.shape[1], right))
        top = max(0, min(frame.shape[0], top))
        bottom = max(0, min(frame.shape[0], bottom))

        frame[top:bottom, left:right] = star['colour']

        left = int(np.round(f * ((x + 50 - star['width'] / 2) / z) + cx))
        right = int(np.round(f * ((x + 50 + star['width'] / 2) / z) + cx))

        left = max(0, min(frame.shape[1], left))
        right = max(0, min(frame.shape[1], right))

        right_frame[top:bottom, left:right] = star['colour']

    return arvet.core.image.StereoImage(
        left_data=frame,
        right_data=right_frame,
        metadata=imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            hash_=0x0000000,
            camera_pose=tf.Transform(
                location=np.array((1, 2, 3)) + time * np.array((4, 5, 6)),
                rotation=tf3d.quaternions.axangle2quat((1, 2, 34), np.pi / 36 + time * np.pi / 15)
            )
        )
    )
