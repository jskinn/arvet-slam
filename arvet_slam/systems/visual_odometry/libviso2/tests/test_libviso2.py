# Copyright (c) 2017, John Skinner
import unittest
import os
import numpy as np
import transforms3d as tf3d
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager

import arvet.util.transform as tf
import arvet.metadata.image_metadata as imeta
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.system import VisionSystem
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import StereoImage
from arvet.core.image_collection import ImageCollection

import arvet_slam.systems.visual_odometry.libviso2.libviso2 as viso
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult


class TestLibVisODatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        VisionSystem._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        VisionSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads(self):
        kwargs = {
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
        }
        obj = viso.LibVisOSystem(**kwargs)
        obj.save()

        # Load all the entities
        all_entities = list(VisionSystem.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_result_saves(self):
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        # Make an image collection with some number of images
        images = []
        num_images = 10
        for time in range(num_images):
            image = create_frame(time / num_images)
            image.save()
            images.append(image)
        image_collection = ImageCollection(
            images=images,
            timestamps=list(range(len(images))),
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        image_collection.save()

        subject = viso.LibVisOSystem()
        subject.save()

        # Actually run the system using mocked images
        subject.set_camera_intrinsics(CameraIntrinsics(
            width=320,
            height=240,
            fx=120,
            fy=120,
            cx=160,
            cy=120
        ))
        subject.set_stereo_offset(tf.Transform([0, -25, 0]))
        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for time, image in enumerate(images):
            subject.process_image(image, time)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)
        result.image_source = image_collection
        result.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], result)
        all_entities[0].delete()

        SLAMTrialResult._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        StereoImage._mongometa.collection.drop()


class TestLibVisO(unittest.TestCase):

    def test_can_start_and_stop_trial(self):
        subject = viso.LibVisOSystem()
        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)

    def test_simple_trial_run(self):
        # Actually run the system using mocked images
        subject = viso.LibVisOSystem()
        subject.set_camera_intrinsics(CameraIntrinsics(
            width=320,
            height=240,
            fx=120,
            fy=120,
            cx=160,
            cy=120
        ))
        subject.set_stereo_offset(tf.Transform([0, -25, 0]))
        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        num_frames = 50
        for time in range(num_frames):
            image = create_frame(time / num_frames)
            subject.process_image(image, 4 * time / num_frames)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)


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
    frame = np.zeros((240, 320), dtype=np.uint8)
    right_frame = np.zeros((240, 320), dtype=np.uint8)
    speed = 200
    baseline = 25
    f = frame.shape[1] / 2
    cx = frame.shape[1] / 2
    cy = frame.shape[0] / 2
    num_stars = 300
    z_values = [600 - idx * 2 - speed * time for idx in range(num_stars)]
    stars = [{
        'pos': (
            (127 * idx + 34 * idx * idx) % 400 - 200,
            (320 - 17 * idx + 7 * idx * idx) % 400 - 200,
            z_value
        ),
        'width': idx % 31 + 1,
        'height': idx % 27 + 1,
        'colour': 50 + (idx * 7 % 206)
    } for idx, z_value in enumerate(z_values) if z_value > 0]

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

        left = int(np.round(f * ((x + baseline - star['width'] / 2) / z) + cx))
        right = int(np.round(f * ((x + baseline + star['width'] / 2) / z) + cx))

        left = max(0, min(frame.shape[1], left))
        right = max(0, min(frame.shape[1], right))

        right_frame[top:bottom, left:right] = star['colour']

    left_metadata = imeta.make_metadata(
        pixels=frame,
        source_type=imeta.ImageSourceType.SYNTHETIC,
        camera_pose=tf.Transform(
            location=[time * speed, 0, 0],
            rotation=[0, 0, 0, 1]
        ),
        intrinsics=CameraIntrinsics(
            width=frame.shape[1],
            height=frame.shape[0],
            fx=f,
            fy=f,
            cx=cx,
            cy=cy
        )
    )
    right_metadata = imeta.make_right_metadata(
        pixels=right_frame,
        left_metadata=left_metadata,
        camera_pose=tf.Transform(
            location=[time * speed, -baseline, 0],
            rotation=[0, 0, 0, 1]
        ),
        intrinsics=CameraIntrinsics(
            width=frame.shape[1],
            height=frame.shape[0],
            fx=f,
            fy=f,
            cx=cx,
            cy=cy
        )
    )
    return StereoImage(
        pixels=frame,
        right_pixels=right_frame,
        metadata=left_metadata,
        right_metadata=right_metadata
    )
