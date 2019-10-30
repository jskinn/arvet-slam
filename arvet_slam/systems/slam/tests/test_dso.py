# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import logging
import os.path
from pathlib import Path
import shutil
import numpy as np
import transforms3d as tf3d
from bson import ObjectId
from dso import UndistortPinhole, UndistortRadTan

import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.util.transform import Transform
from arvet.util.test_helpers import ExtendedTestCase
from arvet.config.path_manager import PathManager
import arvet.metadata.image_metadata as imeta
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image_source import ImageSource
from arvet.core.image_collection import ImageCollection
from arvet.core.image import Image
from arvet.core.system import VisionSystem

from arvet_slam.trials.slam.visual_slam import SLAMTrialResult
from arvet_slam.systems.slam.direct_sparse_odometry import DSO, RectificationMode, \
    make_undistort_from_mode, make_undistort_from_out_intrinsics, make_pose


class TestDSODatabase(unittest.TestCase):
    temp_folder = 'temp-test-dso'
    path_manager = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        os.makedirs(cls.temp_folder, exist_ok=True)
        logging.disable(logging.CRITICAL)
        cls.path_manager = PathManager([Path(__file__).parent], cls.temp_folder)

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        VisionSystem.objects.all().delete()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        VisionSystem._mongometa.collection.drop()
        SLAMTrialResult._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        logging.disable(logging.NOTSET)
        shutil.rmtree(cls.temp_folder)

    def test_stores_and_loads(self):
        obj = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=320,
                fy=320,
                cx=320,
                cy=240
            )
        )
        obj.save()

        # Load all the entities
        all_entities = list(VisionSystem.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].rectification_intrinsics, obj.rectification_intrinsics)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_minimal_args(self):
        obj = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=320,
                fy=320,
                cx=320,
                cy=240
            )
        )
        obj.save()

        # Load all the entities
        all_entities = list(VisionSystem.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0].rectification_intrinsics, obj.rectification_intrinsics)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_result_saves(self):
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)

        # Make an image collection with some number of images
        images = []
        num_images = 10
        for time in range(num_images):
            image = make_image()
            image.metadata.camera_pose = Transform((0.25 * (14 - time), -1.1 * time, 0.11 * time))
            image.save()
            images.append(image)
        image_collection = ImageCollection(
            images=images,
            timestamps=list(range(len(images))),
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        image_collection.save()

        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=320,
                fy=320,
                cx=320,
                cy=240
            )
        )
        subject.save()
        subject.set_camera_intrinsics(image_collection.camera_intrinsics, image_collection.average_timestep)
        subject.resolve_paths(self.path_manager)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for timestamp, image in image_collection:
            subject.process_image(image, timestamp)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)
        for frame_result in result.results:
            self.assertIsNotNone(frame_result.image)
        result.image_source = image_collection
        result.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], result)
        self.assertEqual(all_entities[0].system, subject)
        self.assertEqual(all_entities[0].image_source, image_collection)
        for idx, (timestamp, image) in enumerate(image_collection):
            self.assertEqual(all_entities[0].results[idx].image, image)
            self.assertEqual(all_entities[0].results[idx].timestamp, timestamp)
        all_entities[0].delete()

        SLAMTrialResult.objects.all().delete()
        VisionSystem.objects.all().delete()
        ImageCollection.objects.all().delete()


class TestDSO(unittest.TestCase):
    temp_folder = 'temp-test-dso'
    path_manager = None

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        os.makedirs(cls.temp_folder, exist_ok=True)
        cls.path_manager = PathManager([Path(__file__).parent], cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        shutil.rmtree(cls.temp_folder)

    def test_is_image_source_appropriate_returns_true_for_sequential_image_sources(self):
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=320,
                fy=320,
                cx=320,
                cy=240
            )
        )
        mock_image_source = mock.create_autospec(ImageSource)

        mock_image_source.sequence_type = ImageSequenceType.SEQUENTIAL
        self.assertTrue(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.sequence_type = ImageSequenceType.NON_SEQUENTIAL
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.sequence_type = ImageSequenceType.INTERACTIVE
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

    def test_start_trial_raises_exception_for_non_sequential_image_sources(self):
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=500,
                height=500,
                fx=250,
                fy=250,
                cx=250,
                cy=250
            )
        )
        subject.set_camera_intrinsics(CameraIntrinsics(
            width=860,
            height=500,
            fx=400,
            fy=400,
            cx=450,
            cy=335
        ), 0.0333333)
        with self.assertRaises(RuntimeError):
            subject.start_trial(ImageSequenceType.NON_SEQUENTIAL)
        with self.assertRaises(RuntimeError):
            subject.start_trial(ImageSequenceType.INTERACTIVE)

    def test_start_trial_raises_exception_if_no_intrinsics_are_set(self):
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=500,
                height=500,
                fx=250,
                fy=250,
                cx=250,
                cy=250
            )
        )
        with self.assertRaises(RuntimeError):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)

    def test_start_trial_doesnt_crash(self):
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=500,
                height=500,
                fx=250,
                fy=250,
                cx=250,
                cy=250
            )
        )
        subject.set_camera_intrinsics(CameraIntrinsics(
            width=860,
            height=500,
            fx=400,
            fy=400,
            cx=450,
            cy=335
        ), 0.0333333)
        subject.start_trial(ImageSequenceType.SEQUENTIAL)

    def test_process_image_raises_exception_if_unstarted(self):
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=500,
                height=500,
                fx=250,
                fy=250,
                cx=250,
                cy=250
            )
        )
        image = make_image()
        with self.assertRaises(RuntimeError):
            subject.process_image(image, 0.0)

    def test_finish_trial_raises_exception_if_unstarted(self):
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=500,
                height=500,
                fx=250,
                fy=250,
                cx=250,
                cy=250
            )
        )
        with self.assertRaises(RuntimeError):
            subject.finish_trial()


class TestMakeUndistortFromMode(unittest.TestCase):

    def test_returns_undistort_pinhole_with_crop_but_no_distortion(self):
        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=860,
                height=500,
                fx=400,
                fy=400,
                cx=450,
                cy=335
            ),
            rectification_mode=RectificationMode.CROP,
            out_width=500,
            out_height=500
        )
        self.assertIsInstance(undistorter, UndistortPinhole)

    def test_returns_undistort_pinhole_with_no_rectification(self):
        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50
            ),
            rectification_mode=RectificationMode.NONE,
            out_width=100,
            out_height=100
        )
        self.assertIsInstance(undistorter, UndistortPinhole)

    def test_returns_undistort_radtan_with_distortion_and_rectification(self):
        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=860,
                height=500,
                fx=400,
                fy=400,
                cx=450,
                cy=335,
                k1=0.01
            ),
            rectification_mode=RectificationMode.CROP,
            out_width=500,
            out_height=500
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=860,
                height=500,
                fx=400,
                fy=400,
                cx=450,
                cy=335,
                k2=0.01
            ),
            rectification_mode=RectificationMode.CROP,
            out_width=500,
            out_height=500
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=860,
                height=500,
                fx=400,
                fy=400,
                cx=450,
                cy=335,
                p1=0.01
            ),
            rectification_mode=RectificationMode.CROP,
            out_width=500,
            out_height=500
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=860,
                height=500,
                fx=400,
                fy=400,
                cx=450,
                cy=335,
                p1=0.01
            ),
            rectification_mode=RectificationMode.CROP,
            out_width=500,
            out_height=500
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

    def test_returns_undistort_radtan_with_distortion_but_no_rectification(self):
        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
                k1=0.01
            ),
            rectification_mode=RectificationMode.NONE,
            out_width=100,
            out_height=100
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
                k2=0.01
            ),
            rectification_mode=RectificationMode.NONE,
            out_width=100,
            out_height=100
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
                p1=0.01
            ),
            rectification_mode=RectificationMode.NONE,
            out_width=100,
            out_height=100
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_mode(
            intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
                p2=0.01
            ),
            rectification_mode=RectificationMode.NONE,
            out_width=100,
            out_height=100
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

    def test_throws_exception_if_given_calib_rectification(self):
        with self.assertRaises(ValueError):
            make_undistort_from_mode(
                intrinsics=CameraIntrinsics(
                    width=860,
                    height=500,
                    fx=400,
                    fy=400,
                    cx=450,
                    cy=335,
                    p1=0.01
                ),
                rectification_mode=RectificationMode.CALIB,
                out_width=500,
                out_height=500
            )

    def test_throws_exception_if_out_dimensions_do_not_match_in_but_rectification_is_none(self):
        with self.assertRaises(RuntimeError):
            make_undistort_from_mode(
                intrinsics=CameraIntrinsics(
                    width=860,
                    height=500,
                    fx=400,
                    fy=400,
                    cx=450,
                    cy=335,
                    p1=0.01
                ),
                rectification_mode=RectificationMode.NONE,
                out_width=500,
                out_height=500
            )


class TestMakeUndistortFromOutIntrinsics(unittest.TestCase):

    def test_returns_pinhole_if_there_is_no_distortion(self):
        undistorter = make_undistort_from_out_intrinsics(
            intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=700,
                fy=700,
                cx=320,
                cy=240
            ),
            out_intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50
            ),
        )
        self.assertIsInstance(undistorter, UndistortPinhole)

    def test_returns_radtan_if_there_is_distortion(self):
        undistorter = make_undistort_from_out_intrinsics(
            intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=700,
                fy=700,
                cx=320,
                cy=240,
                k1=0.01
            ),
            out_intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
            ),
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_out_intrinsics(
            intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=700,
                fy=700,
                cx=320,
                cy=240,
                k2=0.01
            ),
            out_intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
            ),
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_out_intrinsics(
            intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=700,
                fy=700,
                cx=320,
                cy=240,
                p1=0.01
            ),
            out_intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
            ),
        )
        self.assertIsInstance(undistorter, UndistortRadTan)

        undistorter = make_undistort_from_out_intrinsics(
            intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=700,
                fy=700,
                cx=320,
                cy=240,
                p2=0.01
            ),
            out_intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100,
                fy=100,
                cx=50,
                cy=50,
            ),
        )
        self.assertIsInstance(undistorter, UndistortRadTan)


class TestMakePose(ExtendedTestCase):

    def test_returns_transform_object(self):
        location = (100, 200, 300)
        rotation = (0.1, 0.2, 0.3, np.sqrt(1 - 0.3 * 0.3 - 0.2 * 0.2 - 0.1 * 0.1))
        pose = make_pose(location, rotation)
        self.assertIsInstance(pose, Transform)

    def test_rearranges_location_coordinates(self):
        location = (100, 200, 300)
        rotation = (0.1, 0.2, 0.3, np.sqrt(1 - 0.3 * 0.3 - 0.2 * 0.2 - 0.1 * 0.1))
        pose = make_pose(location, rotation)
        self.assertNPEqual((300, -100, -200), pose.location)

    def test_changes_rotation_each_axis(self):
        location = (100, 200, 300)
        rotation = tf3d.quaternions.axangle2quat((0, 0, 1), np.pi / 6)
        rotation = (rotation[1], rotation[2], rotation[3], rotation[0])  # Rearrange to w last
        # Roll, rotation around z-axis for DSO
        pose = make_pose(location, rotation)
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around x-axis for DSO
        rotation = tf3d.quaternions.axangle2quat((1, 0, 0), np.pi / 6, True)
        rotation = (rotation[1], rotation[2], rotation[3], rotation[0])  # Rearrange to w last
        pose = make_pose(location, rotation)
        self.assertNPClose((0, -np.pi / 6, 0), pose.euler)

        # Yaw, rotation around negative y-axis for libviso2
        rotation = tf3d.quaternions.axangle2quat((0, 1, 0), np.pi / 6, True)
        rotation = (rotation[1], rotation[2], rotation[3], rotation[0])  # Rearrange to w last
        pose = make_pose(location, rotation)
        self.assertNPClose((0, 0, -np.pi / 6), pose.euler)

    def test_combined(self):
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            rot = tf3d.quaternions.axangle2quat((-rot_axis[1], -rot_axis[2], rot_axis[0]),
                                                rot_angle, False)
            rot = (rot[1], rot[2], rot[3], rot[0])  # Rearrange to w last
            pose = make_pose((-loc[1], -loc[2], loc[0]), rot)
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))


def make_image():
    pixels = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
    metadata = imeta.make_metadata(
        pixels=pixels,
        source_type=imeta.ImageSourceType.SYNTHETIC,
        camera_pose=Transform(location=[13.8, 2.3, -9.8]),
        intrinsics=CameraIntrinsics(
            width=pixels.shape[1],
            height=pixels.shape[0],
            fx=16, fy=16, cx=16, cy=16
        )
    )
    return Image(
        _id=ObjectId(),
        pixels=pixels,
        metadata=metadata
    )
