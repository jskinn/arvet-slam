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
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
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

    def test_get_instance_throws_exception_without_rectification_mode(self):
        with self.assertRaises(ValueError):
            DSO.get_instance(rectification_intrinsics=CameraIntrinsics(
                width=640,
                height=480,
                fx=320,
                fy=320,
                cx=320,
                cy=240
            ))

    def test_get_instance_throws_exception_without_intrinsics(self):
        for rectification_mode in {RectificationMode.CROP, RectificationMode.NONE, RectificationMode.CALIB}:
            with self.assertRaises(ValueError):
                DSO.get_instance(rectification_mode=rectification_mode)

    def test_get_instance_can_create_an_instance(self):
        rectification_mode = np.random.choice([RectificationMode.CALIB, RectificationMode.NONE, RectificationMode.CROP])
        intrinsics = CameraIntrinsics(
            width=int(np.random.randint(100, 500)),
            height=int(np.random.randint(100, 500)),
            fx=int(np.random.randint(100, 500)),
            fy=int(np.random.randint(100, 500)),
            cx=int(np.random.randint(100, 500)),
            cy=int(np.random.randint(100, 500))
        )
        obj = DSO.get_instance(
            rectification_mode=rectification_mode,
            rectification_intrinsics=intrinsics
        )
        self.assertIsInstance(obj, DSO)
        self.assertEqual(rectification_mode, obj.rectification_mode)
        self.assertEqual(intrinsics, obj.rectification_intrinsics)

        # Check the object can be saved
        obj.save()

    def test_get_instance_returns_an_existing_instance(self):
        intrinsics = CameraIntrinsics(
            width=int(np.random.randint(100, 500)),
            height=int(np.random.randint(100, 500)),
            fx=int(np.random.randint(100, 500)),
            fy=int(np.random.randint(100, 500)),
            cx=int(np.random.randint(100, 500)),
            cy=int(np.random.randint(100, 500))
        )
        for rectification_mode in {RectificationMode.CROP, RectificationMode.NONE, RectificationMode.CALIB}:
            obj = DSO(rectification_mode=rectification_mode, rectification_intrinsics=intrinsics)
            obj.save()

            result = DSO.get_instance(rectification_mode=rectification_mode, rectification_intrinsics=intrinsics)
            self.assertIsInstance(result, DSO)
            self.assertEqual(obj.pk, result.pk)
            self.assertEqual(obj, result)

    def test_get_instance_only_checks_width_height_for_crop_and_none_rectification(self):
        intrinsics = CameraIntrinsics(
            width=int(np.random.randint(100, 500)),
            height=int(np.random.randint(100, 500)),
            fx=int(np.random.randint(100, 500)),
            fy=int(np.random.randint(100, 500)),
            cx=int(np.random.randint(100, 500)),
            cy=int(np.random.randint(100, 500))
        )
        for rectification_mode in {RectificationMode.CROP, RectificationMode.NONE}:
            obj = DSO(rectification_mode=rectification_mode, rectification_intrinsics=intrinsics)
            obj.save()

            alt_intrinsics = CameraIntrinsics(
                width=intrinsics.width,
                height=intrinsics.height,
                fx=intrinsics.fx + 10,  # these are different, but it shouldn't matter
                fy=intrinsics.fy + 10,
                cx=intrinsics.cx + 10,
                cy=intrinsics.cy + 10
            )
            result = DSO.get_instance(rectification_mode=rectification_mode, rectification_intrinsics=alt_intrinsics)
            self.assertIsInstance(result, DSO)
            self.assertEqual(obj.pk, result.pk)
            self.assertEqual(obj, result)

    def test_get_instance_ignores_distortion_for_calib_rectification(self):
        intrinsics = CameraIntrinsics(
            width=int(np.random.randint(100, 500)),
            height=int(np.random.randint(100, 500)),
            fx=int(np.random.randint(100, 500)),
            fy=int(np.random.randint(100, 500)),
            cx=int(np.random.randint(100, 500)),
            cy=int(np.random.randint(100, 500)),
            k1=float(np.random.uniform(-0.1, 0.1)),
            k2=float(np.random.uniform(-0.1, 0.1)),
            k3=float(np.random.uniform(-0.1, 0.1)),
            p1=float(np.random.uniform(-0.1, 0.1)),
            p2=float(np.random.uniform(-0.1, 0.1))
        )
        obj = DSO(rectification_mode=RectificationMode.CALIB, rectification_intrinsics=intrinsics)
        obj.save()

        alt_intrinsics = CameraIntrinsics(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,  # these are different, but it shouldn't matter
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            k1=intrinsics.k1 * -3.2,
            k2=intrinsics.k2 + 0.31,
            k3=intrinsics.k3 + 0.31,
            p1=intrinsics.p1 + 0.31,
            p2=intrinsics.p2 + 0.31
        )
        result = DSO.get_instance(rectification_mode=RectificationMode.CALIB, rectification_intrinsics=alt_intrinsics)
        self.assertIsInstance(result, DSO)
        self.assertEqual(obj.pk, result.pk)
        self.assertEqual(obj, result)


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
        mock_image_source.camera_intrinsics = mock.Mock()
        mock_image_source.camera_intrinsics.width = 640
        mock_image_source.camera_intrinsics.height = 480

        mock_image_source.sequence_type = ImageSequenceType.SEQUENTIAL
        self.assertTrue(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.sequence_type = ImageSequenceType.NON_SEQUENTIAL
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.sequence_type = ImageSequenceType.INTERACTIVE
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

    def test_is_image_source_appropriate_returns_true_for_image_resolutions_that_are_valid(self):
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
        mock_image_source.camera_intrinsics = mock.Mock()

        mock_image_source.camera_intrinsics.height = 480

        # DSO requires at least a 3 layer pyramid from resolution halving
        # Valid dimensions can be halved at least 2 times, with resolutions greater than 5000 pixels
        # - 16 is too small, it can be halved, but isn't enough pixels
        # - 202 cannot be halved twice
        # - 480 is valid
        mock_image_source.camera_intrinsics.height = 640
        for width in [16, 202, 480]:
            mock_image_source.camera_intrinsics.width = width
            self.assertEqual(width % 4 == 0 and (width / 4) * (640 / 4) > 5000,
                             subject.is_image_source_appropriate(mock_image_source),
                             f"Failed with resolution {width}x640")

        mock_image_source.camera_intrinsics.width = 640
        for height in [16, 202, 480]:
            mock_image_source.camera_intrinsics.height = height
            self.assertEqual(height % 4 == 0 and (height / 4) * (640 / 4) > 5000,
                             subject.is_image_source_appropriate(mock_image_source),
                             f"Failed with resolution 640x{height}")

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

    def test_make_settings_includes_intrinsics(self):
        intrinsics = CameraIntrinsics(
            width=860,
            height=500,
            fx=400,
            fy=400,
            cx=450,
            cy=335
        )
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
        subject.set_camera_intrinsics(intrinsics, 0.0333333)
        settings = subject.make_settings()
        self.assertEqual('Pinhole', settings['undistort_mode'])
        self.assertEqual(intrinsics.width, settings['in_width'])
        self.assertEqual(intrinsics.height, settings['in_height'])
        self.assertEqual(intrinsics.fx, settings['in_fx'])
        self.assertEqual(intrinsics.fy, settings['in_fy'])
        self.assertEqual(intrinsics.cx, settings['in_cx'])
        self.assertEqual(intrinsics.cy, settings['in_cy'])
        self.assertEqual(intrinsics.p1, settings['in_p1'])
        self.assertEqual(intrinsics.p2, settings['in_p2'])
        self.assertEqual(intrinsics.k1, settings['in_k1'])
        self.assertEqual(intrinsics.k2, settings['in_k2'])

    def test_make_settings_includes_distortion_and_changes_undistort_mode(self):
        intrinsics = CameraIntrinsics(
            width=860,
            height=500,
            fx=400,
            fy=400,
            cx=450,
            cy=335,
            k1=0.3312,
            k2=0.13543,
            p1=-0.3170,
            p2=-0.9989
        )
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
        subject.set_camera_intrinsics(intrinsics, 0.0333333)
        settings = subject.make_settings()
        self.assertEqual('RadTan', settings['undistort_mode'])
        self.assertEqual(intrinsics.p1, settings['in_p1'])
        self.assertEqual(intrinsics.p2, settings['in_p2'])
        self.assertEqual(intrinsics.k1, settings['in_k1'])
        self.assertEqual(intrinsics.k2, settings['in_k2'])

    def test_make_settings_uses_input_width_height_for_rect_none(self):
        intrinsics = CameraIntrinsics(
            width=860,
            height=500,
            fx=400,
            fy=400,
            cx=450,
            cy=335
        )
        subject = DSO(
            rectification_mode=RectificationMode.NONE,
            rectification_intrinsics=CameraIntrinsics(
                width=400,
                height=400,
                fx=200,
                fy=200,
                cx=200,
                cy=200
            )
        )
        subject.set_camera_intrinsics(intrinsics, 0.0333333)
        settings = subject.make_settings()
        self.assertEqual(intrinsics.width, settings['out_width'])
        self.assertEqual(intrinsics.height, settings['out_height'])
        self.assertNotIn('out_fx', settings)
        self.assertNotIn('out_fy', settings)
        self.assertNotIn('out_cx', settings)
        self.assertNotIn('out_cy', settings)

    def test_make_settings_uses_output_width_height_for_rect_crop(self):
        intrinsics = CameraIntrinsics(
            width=860,
            height=500,
            fx=400,
            fy=400,
            cx=450,
            cy=335
        )
        out_intrinsics = CameraIntrinsics(
            width=400,
            height=300,
            fx=200,
            fy=200,
            cx=200,
            cy=150
        )
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=out_intrinsics
        )
        subject.set_camera_intrinsics(intrinsics, 0.0333333)
        settings = subject.make_settings()
        self.assertEqual(out_intrinsics.width, settings['out_width'])
        self.assertEqual(out_intrinsics.height, settings['out_height'])
        self.assertNotIn('out_fx', settings)
        self.assertNotIn('out_fy', settings)
        self.assertNotIn('out_cx', settings)
        self.assertNotIn('out_cy', settings)

    def test_make_settings_provides_output_calib_for_rect_calib(self):
        intrinsics = CameraIntrinsics(
            width=860,
            height=500,
            fx=400,
            fy=400,
            cx=450,
            cy=335
        )
        out_intrinsics = CameraIntrinsics(
            width=640,
            height=480,
            fx=388.2,
            fy=389.9,
            cx=315.5,
            cy=265.3
        )
        subject = DSO(
            rectification_mode=RectificationMode.CALIB,
            rectification_intrinsics=out_intrinsics
        )
        subject.set_camera_intrinsics(intrinsics, 0.0333333)
        settings = subject.make_settings()
        self.assertEqual(out_intrinsics.width, settings['out_width'])
        self.assertEqual(out_intrinsics.height, settings['out_height'])
        self.assertEqual(out_intrinsics.fx, settings['out_fx'])
        self.assertEqual(out_intrinsics.fy, settings['out_fy'])
        self.assertEqual(out_intrinsics.cx, settings['out_cx'])
        self.assertEqual(out_intrinsics.cy, settings['out_cy'])

    def test_get_properties_reads_intrinsics_from_settings(self):
        out_intrinsics = CameraIntrinsics(
            width=600,
            height=400,
            fx=388.2,
            fy=389.9,
            cx=315.5,
            cy=265.3
        )
        settings = {
            'rectification_mode': 'CALIB',
            'undistort_mode': 'RadTan',
            'in_width': 860,
            'in_height': 500,
            'in_fx': 401.3,
            'in_fy': 399.8,
            'in_cx': 450.1,
            'in_cy': 335.2,
            'in_p1': -0.3151,
            'in_p2': 0.8715,
            'in_k1': 0.11123,
            'in_k2': -0.00123,
            'out_width': 640,
            'out_height': 460,
            'out_fx': 388.2,
            'out_fy': 389.9,
            'out_cx': 315.5,
            'out_cy': 265.3
        }
        subject = DSO(
            rectification_mode=RectificationMode.CALIB,
            rectification_intrinsics=out_intrinsics
        )
        properties = subject.get_properties(settings=settings)
        self.assertEqual(subject.rectification_mode, properties['rectification_mode'])
        for column in set(settings.keys()) - {'rectification_mode'}:
            self.assertEqual(settings[column], properties[column])

    def test_get_properties_prioritises_system_rectification_mode(self):
        out_intrinsics = CameraIntrinsics(
            width=600,
            height=400,
            fx=378.2,
            fy=289.9,
            cx=325.5,
            cy=268.9
        )
        for sys_rect_mode in [
            RectificationMode.NONE, RectificationMode.CROP, RectificationMode.CALIB
        ]:
            subject = DSO(
                rectification_mode=sys_rect_mode,
                rectification_intrinsics=out_intrinsics
            )
            for settings_rect_mode in {
                RectificationMode.NONE, RectificationMode.CROP, RectificationMode.CALIB
            } - {sys_rect_mode}:
                settings = {
                    'rectification_mode': str(settings_rect_mode.name),
                    'out_fx': 388.2,
                    'out_fy': 389.9,
                    'out_cx': 315.5,
                    'out_cy': 265.3
                }
                properties = subject.get_properties(settings=settings)
                self.assertEqual(sys_rect_mode, properties['rectification_mode'])
                if sys_rect_mode is RectificationMode.CALIB:
                    self.assertEqual(settings['out_fx'], properties['out_fx'])
                    self.assertEqual(settings['out_fy'], properties['out_fy'])
                    self.assertEqual(settings['out_cx'], properties['out_cx'])
                    self.assertEqual(settings['out_cy'], properties['out_cy'])
                else:
                    self.assertTrue(np.isnan(properties['out_fx']))
                    self.assertTrue(np.isnan(properties['out_fx']))
                    self.assertTrue(np.isnan(properties['out_cx']))
                    self.assertTrue(np.isnan(properties['out_cy']))

    def test_get_properties_prioritises_settings_over_stored_intrinsics(self):
        out_intrinsics = CameraIntrinsics(
            width=600,
            height=400,
            fx=378.2,
            fy=289.9,
            cx=325.5,
            cy=268.9
        )
        settings = {
            'rectification_mode': 'CALIB',
            'undistort_mode': 'RadTan',
            'in_width': 860,
            'in_height': 500,
            'in_fx': 401.3,
            'in_fy': 399.8,
            'in_cx': 450.1,
            'in_cy': 335.2,
            'in_p1': -0.3151,
            'in_p2': 0.8715,
            'in_k1': 0.11123,
            'in_k2': -0.00123,
            'out_width': 640,
            'out_height': 460,
            'out_fx': 388.2,
            'out_fy': 389.9,
            'out_cx': 315.5,
            'out_cy': 265.3
        }
        subject = DSO(
            rectification_mode=RectificationMode.CALIB,
            rectification_intrinsics=out_intrinsics
        )
        properties = subject.get_properties(settings=settings)
        for column in set(settings.keys()) - {'rectification_mode'}:
            self.assertEqual(settings[column], properties[column])

    def test_get_properties_returns_nan_for_out_intrinsics_if_mode_is_not_CALIB(self):
        out_intrinsics = CameraIntrinsics(
            width=640,
            height=400,
            fx=388.2,
            fy=389.9,
            cx=315.5,
            cy=265.3
        )
        for rect_mode in [RectificationMode.CROP, RectificationMode.NONE]:
            subject = DSO(
                rectification_mode=rect_mode,
                rectification_intrinsics=out_intrinsics
            )

            settings = {
                'rectification_mode': str(rect_mode.name),
                'undistort_mode': 'RadTan',
                'in_width': 860,
                'in_height': 500,
                'in_fx': 401.3,
                'in_fy': 399.8,
                'in_cx': 450.1,
                'in_cy': 335.2,
                'in_p1': -0.3151,
                'in_p2': 0.8715,
                'in_k1': 0.11123,
                'in_k2': -0.00123,
                'out_width': 640,
                'out_height': 460
            }
            properties = subject.get_properties(settings=settings)
            self.assertTrue(np.isnan(properties['out_fx']))
            self.assertTrue(np.isnan(properties['out_fx']))
            self.assertTrue(np.isnan(properties['out_cx']))
            self.assertTrue(np.isnan(properties['out_cy']))

            settings = {
                'rectification_mode': str(rect_mode.name),
                'undistort_mode': 'RadTan',
                'in_width': 860,
                'in_height': 500,
                'in_fx': 401.3,
                'in_fy': 399.8,
                'in_cx': 450.1,
                'in_cy': 335.2,
                'in_p1': -0.3151,
                'in_p2': 0.8715,
                'in_k1': 0.11123,
                'in_k2': -0.00123,
                'out_width': 640,
                'out_height': 460,
                # even if they are included in settings
                'out_fx': 388.2,
                'out_fy': 389.9,
                'out_cx': 315.5,
                'out_cy': 265.3
            }
            properties = subject.get_properties(settings=settings)
            self.assertTrue(np.isnan(properties['out_fx']))
            self.assertTrue(np.isnan(properties['out_fx']))
            self.assertTrue(np.isnan(properties['out_cx']))
            self.assertTrue(np.isnan(properties['out_cy']))

    def test_get_properties_returns_nan_for_runtime_values_missing_from_settings_with_rect_CROP(self):
        out_intrinsics = CameraIntrinsics(
            width=640,
            height=400,
            fx=388.2,
            fy=389.9,
            cx=315.5,
            cy=265.3
        )
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=out_intrinsics
        )
        properties = subject.get_properties()
        self.assertEqual(RectificationMode.CROP, properties['rectification_mode'])
        self.assertEqual(out_intrinsics.width, properties['out_width'])
        self.assertEqual(out_intrinsics.height, properties['out_height'])
        for prop in [
            'undistort_mode',
            'in_width',
            'in_height',
            'in_fx',
            'in_fy',
            'in_cx',
            'in_p1',
            'in_p2',
            'in_k1',
            'in_k2',
            'out_fx',
            'out_fy',
            'out_cx',
            'out_cy'
        ]:
            self.assertTrue(np.isnan(properties[prop]))

    def test_get_properties_returns_nan_for_runtime_values_missing_from_settings_with_rect_CALIB(self):
        out_intrinsics = CameraIntrinsics(
            width=640,
            height=400,
            fx=388.2,
            fy=389.9,
            cx=315.5,
            cy=265.3
        )
        subject = DSO(
            rectification_mode=RectificationMode.CALIB,
            rectification_intrinsics=out_intrinsics
        )
        properties = subject.get_properties()
        self.assertEqual(RectificationMode.CALIB, properties['rectification_mode'])
        self.assertEqual(out_intrinsics.width, properties['out_width'])
        self.assertEqual(out_intrinsics.height, properties['out_height'])
        self.assertEqual(out_intrinsics.fx, properties['out_fx'])
        self.assertEqual(out_intrinsics.fy, properties['out_fy'])
        self.assertEqual(out_intrinsics.cx, properties['out_cx'])
        self.assertEqual(out_intrinsics.cy, properties['out_cy'])
        for prop in [
            'undistort_mode',
            'in_width',
            'in_height',
            'in_fx',
            'in_fy',
            'in_cx',
            'in_p1',
            'in_p2',
            'in_k1',
            'in_k2'
        ]:
            self.assertTrue(np.isnan(properties[prop]))

    def test_get_properties_only_returns_specified_columns_CROP(self):
        out_intrinsics = CameraIntrinsics(
            width=640,
            height=400,
            fx=388.2,
            fy=389.9,
            cx=315.5,
            cy=265.3
        )
        settings = {
            'rectification_mode': 'CROP',
            'undistort_mode': 'RadTan',
            'in_width': 860,
            'in_height': 500,
            'in_fx': 401.3,
            'in_fy': 399.8,
            'in_cx': 450.1,
            'in_cy': 335.2,
            'in_p1': -0.3151,
            'in_p2': 0.8715,
            'in_k1': 0.11123,
            'in_k2': -0.00123,
            'out_width': out_intrinsics.width,
            'out_height': out_intrinsics.height
        }
        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=out_intrinsics
        )
        columns = list(subject.get_columns() - {'rectification_mode'})
        np.random.shuffle(columns)
        columns1 = [col for idx, col in enumerate(columns) if idx % 2 == 0]
        columns2 = list(set(columns) - set(columns1))

        properties = subject.get_properties(columns1, settings=settings)
        for column in columns1:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns2:
            self.assertNotIn(column, properties)

        properties = subject.get_properties(['rectification_mode'] + columns2, settings=settings)
        self.assertIn('rectification_mode', properties)
        self.assertEqual(subject.rectification_mode, properties['rectification_mode'])
        for column in columns2:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns1:
            self.assertNotIn(column, properties)

    def test_get_properties_only_returns_specified_columns_CALIB(self):
        out_intrinsics = CameraIntrinsics(
            width=640,
            height=400,
            fx=388.2,
            fy=389.9,
            cx=315.5,
            cy=265.3
        )
        settings = {
            'rectification_mode': 'CALIB',
            'undistort_mode': 'RadTan',
            'in_width': 860,
            'in_height': 500,
            'in_fx': 401.3,
            'in_fy': 399.8,
            'in_cx': 450.1,
            'in_cy': 335.2,
            'in_p1': -0.3151,
            'in_p2': 0.8715,
            'in_k1': 0.11123,
            'in_k2': -0.00123,
            'out_width': out_intrinsics.width,
            'out_height': out_intrinsics.height,
            'out_fx': out_intrinsics.fx,
            'out_fy': out_intrinsics.fy,
            'out_cx': out_intrinsics.cx,
            'out_cy': out_intrinsics.cy
        }
        subject = DSO(
            rectification_mode=RectificationMode.CALIB,
            rectification_intrinsics=out_intrinsics
        )
        columns = list(subject.get_columns() - {'rectification_mode'})
        np.random.shuffle(columns)
        columns1 = [col for idx, col in enumerate(columns) if idx % 2 == 0]
        columns2 = list(set(columns) - set(columns1))

        properties = subject.get_properties(columns1, settings=settings)
        for column in columns1:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns2:
            self.assertNotIn(column, properties)

        properties = subject.get_properties(['rectification_mode'] + columns2, settings=settings)
        self.assertIn('rectification_mode', properties)
        self.assertEqual(subject.rectification_mode, properties['rectification_mode'])
        for column in columns2:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns1:
            self.assertNotIn(column, properties)


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
