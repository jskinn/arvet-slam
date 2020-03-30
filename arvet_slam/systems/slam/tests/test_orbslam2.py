# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import logging
import os.path
from pathlib import Path
from io import StringIO
import string
from queue import Empty as QueueEmpty
import shutil
import tempfile
import multiprocessing
import multiprocessing.queues
import numpy as np
import transforms3d as tf3d
from bson import ObjectId
from pymodm.context_managers import no_auto_dereference

import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.util.transform import Transform
from arvet.util.test_helpers import ExtendedTestCase
from arvet.util.image_utils import convert_to_grey
from arvet.config.path_manager import PathManager
import arvet.metadata.image_metadata as imeta
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image_source import ImageSource
from arvet.core.image_collection import ImageCollection
from arvet.core.image import Image, StereoImage
from arvet.core.system import VisionSystem

from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult
from arvet_slam.systems.slam.orbslam2 import OrbSlam2, SensorMode, dump_config, nested_to_dotted, \
    make_relative_pose, run_orbslam
from arvet_slam.systems.slam.tests.create_vocabulary import create_vocab


class TestOrbSlam2Database(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

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

    def test_stores_and_loads(self):
        obj = OrbSlam2(
            vocabulary_file='im-a-file-{0}'.format(np.random.randint(0, 100000)),
            mode=np.random.choice([SensorMode.MONOCULAR, SensorMode.STEREO, SensorMode.RGBD]),
            vocabulary_branching_factor=np.random.randint(0, 100),
            vocabulary_depth=np.random.randint(0, 100),
            vocabulary_seed=np.random.randint(0, 2**31),
            depth_threshold=np.random.uniform(10, 100),
            orb_num_features=np.random.randint(10, 10000),
            orb_scale_factor=np.random.uniform(0.5, 2),
            orb_num_levels=np.random.randint(3, 20),
            orb_ini_threshold_fast=np.random.randint(10, 20),
            orb_min_threshold_fast=np.random.randint(3, 10)
        )
        obj.save()

        # Load all the entities
        all_entities = list(VisionSystem.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_stores_and_loads_minimal_args(self):
        obj = OrbSlam2(
            mode=np.random.choice([SensorMode.MONOCULAR, SensorMode.STEREO, SensorMode.RGBD])
        )
        obj.save()

        # Load all the entities
        all_entities = list(VisionSystem.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_get_instance_throws_exception_without_sensor_mode(self):
        with self.assertRaises(ValueError):
            OrbSlam2.get_instance()

    def test_get_instance_can_create_an_instance(self):
        voc_file = 'im-a-file-{0}'.format(np.random.randint(0, 100000))
        sensor_mode = SensorMode.RGBD
        vocabulary_branching_factor = np.random.randint(2, 256)
        vocabulary_depth = np.random.randint(4, 16)
        vocabulary_seed = np.random.randint(0, 2**31)
        depth_threshold = np.random.uniform(10, 100.0)
        orb_num_features = np.random.randint(100, 2000)
        orb_scale_factor = np.random.uniform(0.5, 2.0)
        orb_num_levels = np.random.randint(5, 10)
        orb_ini_threshold_fast = np.random.randint(11, 20)
        orb_min_threshold_fast = np.random.randint(5, 10)
        obj = OrbSlam2.get_instance(
            vocabulary_file=voc_file,
            mode=sensor_mode,
            vocabulary_branching_factor=vocabulary_branching_factor,
            vocabulary_depth=vocabulary_depth,
            vocabulary_seed=vocabulary_seed,
            depth_threshold=depth_threshold,
            orb_num_features=orb_num_features,
            orb_scale_factor=orb_scale_factor,
            orb_num_levels=orb_num_levels,
            orb_ini_threshold_fast=orb_ini_threshold_fast,
            orb_min_threshold_fast=orb_min_threshold_fast
        )
        self.assertIsInstance(obj, OrbSlam2)
        self.assertEqual(voc_file, obj.vocabulary_file)
        self.assertEqual(sensor_mode, obj.mode)
        self.assertEqual(vocabulary_branching_factor, obj.vocabulary_branching_factor)
        self.assertEqual(vocabulary_depth, obj.vocabulary_depth)
        self.assertEqual(vocabulary_seed, obj.vocabulary_seed)
        self.assertEqual(depth_threshold, obj.depth_threshold)
        self.assertEqual(orb_num_features, obj.orb_num_features)
        self.assertEqual(orb_scale_factor, obj.orb_scale_factor)
        self.assertEqual(orb_num_levels, obj.orb_num_levels)
        self.assertEqual(orb_ini_threshold_fast, obj.orb_ini_threshold_fast)
        self.assertEqual(orb_min_threshold_fast, obj.orb_min_threshold_fast)

        # Check the object can be saved
        obj.save()

    def test_creates_an_instance_with_defaults_by_default(self):
        for sensor_mode in {SensorMode.MONOCULAR, SensorMode.STEREO, SensorMode.RGBD}:
            obj = OrbSlam2(mode=sensor_mode)
            result = OrbSlam2.get_instance(mode=sensor_mode)

            self.assertEqual(obj.depth_threshold, result.depth_threshold)
            self.assertEqual(obj.orb_num_features, result.orb_num_features)
            self.assertEqual(obj.orb_scale_factor, result.orb_scale_factor)
            self.assertEqual(obj.orb_num_levels, result.orb_num_levels)
            self.assertEqual(obj.orb_ini_threshold_fast, result.orb_ini_threshold_fast)
            self.assertEqual(obj.orb_min_threshold_fast, result.orb_min_threshold_fast)

    def test_get_instance_returns_an_existing_instance_defaults(self):
        for sensor_mode in {SensorMode.MONOCULAR, SensorMode.STEREO, SensorMode.RGBD}:
            obj = OrbSlam2(mode=sensor_mode)
            obj.save()

            result = OrbSlam2.get_instance(mode=sensor_mode)
            self.assertIsInstance(result, OrbSlam2)
            self.assertEqual(obj.pk, result.pk)
            self.assertEqual(obj, result)

    def test_get_instance_returns_existing_instance_complex(self):
        voc_file = 'im-a-file-{0}'.format(np.random.randint(0, 100000))
        sensor_mode = SensorMode.RGBD
        vocabulary_branching_factor = np.random.randint(2, 256)
        vocabulary_depth = np.random.randint(4, 16)
        vocabulary_seed = np.random.randint(0, 2 ** 31)
        depth_threshold = np.random.uniform(10, 100.0)
        orb_num_features = np.random.randint(100, 2000)
        orb_scale_factor = np.random.uniform(0.5, 2.0)
        orb_num_levels = np.random.randint(5, 10)
        orb_ini_threshold_fast = np.random.randint(11, 20)
        orb_min_threshold_fast = np.random.randint(5, 10)

        obj = OrbSlam2(
            vocabulary_file=voc_file,
            mode=sensor_mode,
            vocabulary_branching_factor=vocabulary_branching_factor,
            vocabulary_depth=vocabulary_depth,
            vocabulary_seed=vocabulary_seed,
            depth_threshold=depth_threshold,
            orb_num_features=orb_num_features,
            orb_scale_factor=orb_scale_factor,
            orb_num_levels=orb_num_levels,
            orb_ini_threshold_fast=orb_ini_threshold_fast,
            orb_min_threshold_fast=orb_min_threshold_fast
        )
        obj.save()

        result = OrbSlam2.get_instance(
            vocabulary_file=voc_file,
            mode=sensor_mode,
            vocabulary_branching_factor=vocabulary_branching_factor,
            vocabulary_depth=vocabulary_depth,
            vocabulary_seed=vocabulary_seed,
            depth_threshold=depth_threshold,
            orb_num_features=orb_num_features,
            orb_scale_factor=orb_scale_factor,
            orb_num_levels=orb_num_levels,
            orb_ini_threshold_fast=orb_ini_threshold_fast,
            orb_min_threshold_fast=orb_min_threshold_fast
        )
        self.assertEqual(obj.pk, result.pk)
        self.assertEqual(obj, result)

    def test_get_instance_returns_existing_instance_without_vocab_file(self):
        sensor_mode = SensorMode.RGBD
        vocabulary_branching_factor = np.random.randint(2, 256)
        vocabulary_depth = np.random.randint(4, 16)
        vocabulary_seed = np.random.randint(0, 2 ** 31)
        depth_threshold = np.random.uniform(10, 100.0)
        orb_num_features = np.random.randint(100, 2000)
        orb_scale_factor = np.random.uniform(0.5, 2.0)
        orb_num_levels = np.random.randint(5, 10)
        orb_ini_threshold_fast = np.random.randint(11, 20)
        orb_min_threshold_fast = np.random.randint(5, 10)

        obj = OrbSlam2(
            vocabulary_file='im-a-file',
            mode=sensor_mode,
            vocabulary_branching_factor=vocabulary_branching_factor,
            vocabulary_depth=vocabulary_depth,
            vocabulary_seed=vocabulary_seed,
            depth_threshold=depth_threshold,
            orb_num_features=orb_num_features,
            orb_scale_factor=orb_scale_factor,
            orb_num_levels=orb_num_levels,
            orb_ini_threshold_fast=orb_ini_threshold_fast,
            orb_min_threshold_fast=orb_min_threshold_fast
        )
        obj.save()

        result = OrbSlam2.get_instance(
            mode=sensor_mode,
            vocabulary_branching_factor=vocabulary_branching_factor,
            vocabulary_depth=vocabulary_depth,
            vocabulary_seed=vocabulary_seed,
            depth_threshold=depth_threshold,
            orb_num_features=orb_num_features,
            orb_scale_factor=orb_scale_factor,
            orb_num_levels=orb_num_levels,
            orb_ini_threshold_fast=orb_ini_threshold_fast,
            orb_min_threshold_fast=orb_min_threshold_fast
        )
        self.assertEqual(obj.pk, result.pk)
        self.assertEqual(obj, result)

    def test_get_instance_ignores_vocab_settings_if_file_is_specified(self):
        voc_file = 'im-a-file-{0}'.format(np.random.randint(0, 100000))
        sensor_mode = SensorMode.RGBD
        vocabulary_branching_factor = np.random.randint(2, 256)
        vocabulary_depth = np.random.randint(4, 16)
        vocabulary_seed = np.random.randint(4, 2 ** 31)
        depth_threshold = np.random.uniform(10, 100.0)
        orb_num_features = np.random.randint(100, 2000)
        orb_scale_factor = np.random.uniform(0.5, 2.0)
        orb_num_levels = np.random.randint(5, 10)
        orb_ini_threshold_fast = np.random.randint(11, 20)
        orb_min_threshold_fast = np.random.randint(5, 10)

        obj = OrbSlam2(
            vocabulary_file=voc_file,
            mode=sensor_mode,
            vocabulary_branching_factor=vocabulary_branching_factor,
            vocabulary_depth=vocabulary_depth,
            vocabulary_seed=vocabulary_seed,
            depth_threshold=depth_threshold,
            orb_num_features=orb_num_features,
            orb_scale_factor=orb_scale_factor,
            orb_num_levels=orb_num_levels,
            orb_ini_threshold_fast=orb_ini_threshold_fast,
            orb_min_threshold_fast=orb_min_threshold_fast
        )
        obj.save()

        result = OrbSlam2.get_instance(
            vocabulary_file=voc_file,
            mode=sensor_mode,
            vocabulary_branching_factor=vocabulary_branching_factor + 1,
            vocabulary_depth=vocabulary_depth - 1,
            vocabulary_seed=vocabulary_seed // 2,
            depth_threshold=depth_threshold,
            orb_num_features=orb_num_features,
            orb_scale_factor=orb_scale_factor,
            orb_num_levels=orb_num_levels,
            orb_ini_threshold_fast=orb_ini_threshold_fast,
            orb_min_threshold_fast=orb_min_threshold_fast
        )
        self.assertEqual(obj.pk, result.pk)
        self.assertEqual(obj, result)


class TestOrbSlam2ResultDatabase(unittest.TestCase):
    temp_folder = Path(__file__).parent / 'temp-test-orbslam2'
    vocab_file = 'ORBvoc-synth.txt'
    vocab_path = Path(__file__).parent / vocab_file
    path_manager = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        os.makedirs(cls.temp_folder, exist_ok=True)
        logging.disable(logging.CRITICAL)
        cls.path_manager = PathManager([Path(__file__).parent], cls.temp_folder)

        if not cls.vocab_path.exists():  # If there is no vocab file, make one
            print("Creating vocab file, this may take a while...")
            create_vocab(cls.vocab_path)

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

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_result_saves_mono(self, mock_multiprocessing):
        # Make an image collection with some number of images
        images = []
        num_images = 10
        for time in range(num_images):
            image = make_image(SensorMode.MONOCULAR)
            image.metadata.camera_pose = Transform((0.25 * (14 - time), -1.1 * time, 0.11 * time))
            image.save()
            images.append(image)
        image_collection = ImageCollection(
            images=images,
            timestamps=list(range(len(images))),
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        image_collection.save()

        # Mock the subprocess to control the orbslam output, we don't want to actually run it.
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.qsize.return_value = 0
        mock_queue.get.side_effect = [
            'ORBSLAM Ready!',
            {
                idx: [
                    0.122 + 0.09 * idx,  # Processing Time
                    15 + idx,  # Number of features
                    6 + idx,  # Number of matches
                    TrackingState.OK,  # Tracking state
                    [  # Estimated pose
                        [1, 0, 0, idx],
                        [0, 1, 0, -0.1 * idx],
                        [0, 0, 1, 0.22 * (14 - idx)]
                    ]
                ]
                for idx in range(10)
            }
        ]
        mock_multiprocessing.Queue.return_value = mock_queue

        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocab_file)
        subject.save()
        subject.set_camera_intrinsics(image_collection.camera_intrinsics, image_collection.average_timestep)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for timestamp, image in image_collection:
            subject.process_image(image, timestamp)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)
        self.assertEqual(len(image_collection), len(result.results))
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

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_result_saves_stereo(self, mock_multiprocessing):
        # Make an image collection with some number of images
        images = []
        num_images = 10
        stereo_offset = Transform([0, 0.12, 0])
        for time in range(num_images):
            image = make_image(SensorMode.STEREO)
            img_pose = Transform((0.25 * (14 - time), -1.1 * time, 0.11 * time))
            image.metadata.camera_pose = img_pose
            image.right_metadata.camera_pose = img_pose.find_independent(stereo_offset)
            image.save()
            images.append(image)
        image_collection = ImageCollection(
            images=images,
            timestamps=list(range(len(images))),
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        image_collection.save()

        # Mock the subprocess to control the orbslam output, we don't want to actually run it.
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.qsize.return_value = 0
        mock_queue.get.side_effect = [
            'ORBSLAM Ready!',
            {
                idx: [
                    0.122 + 0.09 * idx,  # Processing Time
                    15 + idx,  # Number of features
                    6 + idx,  # Number of matches
                    TrackingState.OK,  # Tracking state
                    [  # Estimated pose
                        [1, 0, 0, idx],
                        [0, 1, 0, -0.1 * idx],
                        [0, 0, 1, 0.22 * (14 - idx)]
                    ]
                ]
                for idx in range(10)
            }
        ]
        mock_multiprocessing.Queue.return_value = mock_queue

        subject = OrbSlam2(mode=SensorMode.STEREO, vocabulary_file=self.vocab_file)
        subject.save()
        subject.set_camera_intrinsics(image_collection.camera_intrinsics, image_collection.average_timestep)
        subject.resolve_paths(self.path_manager)
        subject.set_stereo_offset(image_collection.stereo_offset)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for timestamp, image in image_collection:
            subject.process_image(image, timestamp)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)
        self.assertEqual(len(image_collection), len(result.results))
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

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_result_saves_rgbd(self, mock_multiprocessing):
        # Make an image collection with some number of images
        images = []
        num_images = 10
        for time in range(num_images):
            image = make_image(SensorMode.RGBD)
            img_pose = Transform((0.25 * (14 - time), -1.1 * time, 0.11 * time))
            image.metadata.camera_pose = img_pose
            image.save()
            images.append(image)
        image_collection = ImageCollection(
            images=images,
            timestamps=list(range(len(images))),
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        image_collection.save()

        # Mock the subprocess to control the orbslam output, we don't want to actually run it.
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.qsize.return_value = 0
        mock_queue.get.side_effect = [
            'ORBSLAM Ready!',
            {
                idx: [
                    0.122 + 0.09 * idx,  # Processing Time
                    15 + idx,  # Number of features
                    6 + idx,  # Number of matches
                    TrackingState.OK,  # Tracking state
                    [  # Estimated pose
                        [1, 0, 0, idx],
                        [0, 1, 0, -0.1 * idx],
                        [0, 0, 1, 0.22 * (14 - idx)]
                    ]
                ]
                for idx in range(10)
            }
        ]
        mock_multiprocessing.Queue.return_value = mock_queue

        subject = OrbSlam2(mode=SensorMode.RGBD, vocabulary_file=self.vocab_file)
        subject.save()
        subject.set_camera_intrinsics(image_collection.camera_intrinsics, image_collection.average_timestep)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for timestamp, image in image_collection:
            subject.process_image(image, timestamp)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)
        self.assertEqual(len(image_collection), len(result.results))
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


class TestOrbSlam2(unittest.TestCase):
    temp_folder = Path(__file__).parent / 'temp-test-orbslam2'
    vocabulary_file = 'ORBvoc-temp.txt'
    path_manager = None

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        os.makedirs(cls.temp_folder, exist_ok=True)
        path = Path(__file__).parent
        (path / cls.vocabulary_file).touch()    # This file just has to exist where the path manager can find it
        cls.path_manager = PathManager([path], cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
        (Path(__file__).parent / cls.vocabulary_file).unlink()
        shutil.rmtree(cls.temp_folder)

    def test_get_properties_is_overridden_by_settings(self):
        settings = {
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
            'in_k3': 0.01443,
            'base': 15.223,
            'vocabulary_branching_factor': 12,
            'vocabulary_depth': 5,
            'vocabulary_seed': 378627802,
            'vocabulary_file': 'my_vocab_file',
            'mode': str(SensorMode.RGBD.name),
            'depth_threshold': 123,
            'orb_num_features': 4082,
            'orb_scale_factor': 1.1,
            'orb_num_levels': 3,
            'orb_ini_threshold_fast': 6,
            'orb_min_threshold_fast': 3
        }
        subject = OrbSlam2(
            mode=SensorMode.MONOCULAR,
            depth_threshold=22,
            orb_num_features=332,
            orb_scale_factor=1.02,
            orb_num_levels=8,
            orb_ini_threshold_fast=22,
            orb_min_threshold_fast=16
        )
        properties = subject.get_properties(settings=settings)
        self.assertEqual(SensorMode.RGBD, properties['mode'])
        for column in set(settings.keys()) - {'mode'}:
            self.assertEqual(settings[column], properties[column])

    def test_get_properties_reads_from_object_or_is_nan_when_not_in_settings(self):
        subject = OrbSlam2(
            mode=SensorMode.MONOCULAR,
            vocabulary_file='my_vocab_file',
            vocabulary_branching_factor=17,
            vocabulary_depth=5,
            vocabulary_seed=273635835,
            depth_threshold=22,
            orb_num_features=332,
            orb_scale_factor=1.02,
            orb_num_levels=8,
            orb_ini_threshold_fast=22,
            orb_min_threshold_fast=16
        )
        properties = subject.get_properties()
        self.assertEqual(SensorMode.MONOCULAR, properties['mode'])
        self.assertEqual('my_vocab_file', properties['vocabulary_file'])
        self.assertEqual(17, properties['vocabulary_branching_factor'])
        self.assertEqual(5, properties['vocabulary_depth'])
        self.assertEqual(273635835, properties['vocabulary_seed'])
        self.assertEqual(22, properties['depth_threshold'])
        self.assertEqual(332, properties['orb_num_features'])
        self.assertEqual(1.02, properties['orb_scale_factor'])
        self.assertEqual(8, properties['orb_num_levels'])
        self.assertEqual(22, properties['orb_ini_threshold_fast'])
        self.assertEqual(16, properties['orb_min_threshold_fast'])
        for column in [
            'in_width',
            'in_height',
            'in_fx',
            'in_fy',
            'in_cx',
            'in_cy',
            'in_p1',
            'in_p2',
            'in_k1',
            'in_k2',
            'in_k3',
            'base'
        ]:
            self.assertTrue(np.isnan(properties[column]))

    def test_get_properties_only_returns_the_requested_properties(self):
        settings = {
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
            'in_k3': 0.01443,
            'base': 15.223,
            'vocabulary_branching_factor': 3,
            'vocabulary_depth': 21,
            'vocabulary_seed': 163463436,
            'vocabulary_file': 'my_vocab_file',
            'mode': str(SensorMode.RGBD.name),
            'depth_threshold': 123,
            'orb_num_features': 4082,
            'orb_scale_factor': 1.1,
            'orb_num_levels': 3,
            'orb_ini_threshold_fast': 6,
            'orb_min_threshold_fast': 3
        }
        subject = OrbSlam2(
            mode=SensorMode.MONOCULAR,
            vocabulary_file='my_vocab_file',
            vocabulary_branching_factor=12,
            vocabulary_depth=5,
            vocabulary_seed=378627802,
            depth_threshold=22,
            orb_num_features=332,
            orb_scale_factor=1.02,
            orb_num_levels=8,
            orb_ini_threshold_fast=22,
            orb_min_threshold_fast=16
        )
        columns = list(subject.get_columns())
        np.random.shuffle(columns)
        columns1 = {column for idx, column in enumerate(columns) if idx % 2 == 0 and column != 'mode'}
        columns2 = set(columns) - columns1

        properties = subject.get_properties(columns1, settings=settings)
        for column in columns1:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns2:
            self.assertNotIn(column, properties)

        properties = subject.get_properties(columns2, settings=settings)
        for column in columns2:
            self.assertIn(column, properties)
            if column == 'mode':
                self.assertEqual(SensorMode.RGBD, properties[column])
            elif column in settings:
                self.assertEqual(settings[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns1:
            self.assertNotIn(column, properties)

    def test_is_image_source_appropriate_returns_true_for_monocular_systems_and_sequential_image_sources(self):
        subject = OrbSlam2(mode=SensorMode.MONOCULAR)
        mock_image_source = mock.create_autospec(ImageSource)
        mock_image_source.camera_intrinsics = CameraIntrinsics(width=640, height=480)

        mock_image_source.sequence_type = ImageSequenceType.SEQUENTIAL
        self.assertTrue(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.sequence_type = ImageSequenceType.NON_SEQUENTIAL
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.sequence_type = ImageSequenceType.INTERACTIVE
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

    def test_is_image_source_appropriate_returns_true_for_stereo_systems_if_stereo_is_available(self):
        subject = OrbSlam2(mode=SensorMode.STEREO)
        mock_image_source = mock.create_autospec(ImageSource)
        mock_image_source.camera_intrinsics = CameraIntrinsics(width=640, height=480)

        mock_image_source.sequence_type = ImageSequenceType.SEQUENTIAL
        mock_image_source.is_stereo_available = True
        self.assertTrue(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.is_stereo_available = False
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.sequence_type = ImageSequenceType.NON_SEQUENTIAL
        mock_image_source.is_stereo_available = True
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source = mock.create_autospec(ImageSource)
        mock_image_source.sequence_type = ImageSequenceType.INTERACTIVE
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

    def test_is_image_source_appropriate_returns_true_for_rgbd_systems_if_depth_is_available(self):
        subject = OrbSlam2(mode=SensorMode.RGBD)
        mock_image_source = mock.create_autospec(ImageSource)
        mock_image_source.camera_intrinsics = CameraIntrinsics(width=640, height=480)

        mock_image_source.sequence_type = ImageSequenceType.SEQUENTIAL
        mock_image_source.is_depth_available = True
        self.assertTrue(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.is_depth_available = False
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source.sequence_type = ImageSequenceType.NON_SEQUENTIAL
        mock_image_source.is_depth_available = True
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

        mock_image_source = mock.create_autospec(ImageSource)
        mock_image_source.sequence_type = ImageSequenceType.INTERACTIVE
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

    def test_is_image_source_appropriate_returns_false_if_the_feature_pyramid_scales_a_dimension_to_zero(self):
        num_levels = 4
        for scale_factor in [1.2, 2.0]:     # Try for different scale factors, to make sure it's not a constant
            subject = OrbSlam2(mode=SensorMode.RGBD, orb_scale_factor=scale_factor, orb_num_levels=num_levels)
            mock_image_source = mock.create_autospec(ImageSource)
            mock_image_source.sequence_type = ImageSequenceType.SEQUENTIAL

            # The feature pyramid will reduce this to 0 exactly on the lowest dimension, but 1 more is still 1
            zero_dim = np.floor(32.5 * (scale_factor ** (num_levels - 1)))
            # The feature pyramid will reduce this to a negative number (roughly -15)
            negative_dim = np.floor(17 * scale_factor ** (num_levels - 1))

            mock_image_source.camera_intrinsics = CameraIntrinsics(width=zero_dim, height=zero_dim + 1)
            self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

            mock_image_source.camera_intrinsics = CameraIntrinsics(width=negative_dim, height=zero_dim + 1)
            self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

            mock_image_source.camera_intrinsics = CameraIntrinsics(width=zero_dim + 1, height=zero_dim)
            self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

            mock_image_source.camera_intrinsics = CameraIntrinsics(width=zero_dim + 1, height=negative_dim)
            self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

            mock_image_source.camera_intrinsics = CameraIntrinsics(width=zero_dim + 1, height=zero_dim + 1)
            self.assertTrue(subject.is_image_source_appropriate(mock_image_source))

    def test_is_image_source_appropriate_returns_false_if_the_aspect_ratios_are_wrong(self):
        subject = OrbSlam2(mode=SensorMode.RGBD, orb_scale_factor=2, orb_num_levels=2)
        mock_image_source = mock.create_autospec(ImageSource)
        mock_image_source.sequence_type = ImageSequenceType.SEQUENTIAL

        # This image source has width/height < 0.5, which will round to 0, ruining nIni
        mock_image_source.camera_intrinsics = CameraIntrinsics(width=256, height=1024)
        self.assertFalse(subject.is_image_source_appropriate(mock_image_source))

        # This is fine.
        mock_image_source.camera_intrinsics = CameraIntrinsics(width=1024, height=256)
        self.assertTrue(subject.is_image_source_appropriate(mock_image_source))

    def test_save_settings_raises_error_without_paths_configured(self):
        intrinsics = CameraIntrinsics(
            width=640, height=480, fx=320, fy=320, cx=320, cy=240
        )
        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(intrinsics, 1 / 30)

        with self.assertRaises(RuntimeError):
            subject.save_settings()

    def test_save_settings_raises_error_without_camera_intrinsics(self):
        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.resolve_paths(self.path_manager)

        with self.assertRaises(RuntimeError):
            subject.save_settings()

    def test_save_settings_raises_error_without_camera_baseline_if_stereo(self):
        intrinsics = CameraIntrinsics(width=640, height=480, fx=320, fy=320, cx=320, cy=240)
        subject = OrbSlam2(mode=SensorMode.STEREO, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(intrinsics, 1 / 30)
        subject.resolve_paths(self.path_manager)

        with self.assertRaises(RuntimeError):
            subject.save_settings()

    @mock.patch('arvet_slam.systems.slam.orbslam2.tempfile', autospec=tempfile)
    def test_save_settings_monocular_saves_to_a_temporary_file(self, mock_tempfile):
        mock_tempfile.mkstemp.return_value = (12, 'my_temp_file.yml')
        mock_open = mock.mock_open()
        mock_open.return_value = StringIO()

        intrinsics = CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240)
        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(intrinsics, 1 / 30)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.save_settings()

        self.assertTrue(mock_open.called)
        self.assertEqual(Path('my_temp_file.yml'), mock_open.call_args[0][0])

    @mock.patch('arvet_slam.systems.slam.orbslam2.tempfile', autospec=tempfile)
    def test_save_settings_monocular_writes_camera_configuration(self, mock_tempfile):
        mock_tempfile.mkstemp.return_value = (12, 'my_temp_file.yml')
        mock_file = InspectableStringIO()
        mock_open = mock.mock_open()
        mock_open.return_value = mock_file

        intrinsics = CameraIntrinsics(
            width=640, height=480, fx=320, fy=321, cx=322, cy=240,
            k1=1, k2=2, k3=3, p1=4, p2=5
        )
        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(intrinsics, 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.save_settings()

        contents = mock_file.getvalue()
        lines = contents.split('\n')
        self.assertGreater(len(lines), 0)

        # Camera Configuration
        self.assertIn('Camera.width: 640', lines)
        self.assertIn('Camera.height: 480', lines)
        self.assertIn('Camera.fx: 320.0', lines)
        self.assertIn('Camera.fy: 321.0', lines)
        self.assertIn('Camera.cx: 322.0', lines)
        self.assertIn('Camera.cy: 240.0', lines)
        self.assertIn('Camera.k1: 1.0', lines)
        self.assertIn('Camera.k2: 2.0', lines)
        self.assertIn('Camera.k3: 3.0', lines)
        self.assertIn('Camera.p1: 4.0', lines)
        self.assertIn('Camera.p2: 5.0', lines)
        self.assertIn('Camera.fps: 29.0', lines)
        self.assertIn('Camera.RGB: 1', lines)

    @mock.patch('arvet_slam.systems.slam.orbslam2.tempfile', autospec=tempfile)
    def test_save_settings_stereo_writes_stereo_baseline(self, mock_tempfile):
        mock_tempfile.mkstemp.return_value = (12, 'my_temp_file.yml')
        mock_file = InspectableStringIO()
        mock_open = mock.mock_open()
        mock_open.return_value = mock_file

        intrinsics = CameraIntrinsics(
            width=640, height=480, fx=320, fy=321, cx=322, cy=240,
            k1=1, k2=2, k3=3, p1=4, p2=5
        )
        subject = OrbSlam2(mode=SensorMode.STEREO, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(intrinsics, 1 / 29)
        subject.set_stereo_offset(Transform([0.012, -0.142, 0.09]))
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.save_settings()

        contents = mock_file.getvalue()
        lines = contents.split('\n')
        self.assertGreater(len(lines), 0)

        # Camera baseline.
        # Should be fx times the right-offset of the camera (-1 * the y component)
        self.assertIn('Camera.bf: {0}'.format(0.142 * 320), lines)

    @mock.patch('arvet_slam.systems.slam.orbslam2.tempfile', autospec=tempfile)
    def test_save_settings_writes_system_configuration(self, mock_tempfile):
        mock_tempfile.mkstemp.return_value = (12, 'my_temp_file.yml')
        mock_file = InspectableStringIO()
        mock_open = mock.mock_open()
        mock_open.return_value = mock_file

        intrinsics = CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240)
        subject = OrbSlam2(
            mode=SensorMode.MONOCULAR,
            vocabulary_file=self.vocabulary_file,
            depth_threshold=58.2,
            orb_num_features=2337,
            orb_scale_factor=1.32,
            orb_num_levels=16,
            orb_ini_threshold_fast=25,
            orb_min_threshold_fast=14
        )
        subject.set_camera_intrinsics(intrinsics, 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.save_settings()

        contents = mock_file.getvalue()
        lines = contents.split('\n')
        self.assertGreater(len(lines), 0)

        # Camera Configuration
        self.assertIn('ThDepth: 58.2', lines)
        self.assertIn('DepthMapFactor: 1.0', lines)
        self.assertIn('ORBextractor.nFeatures: 2337', lines)
        self.assertIn('ORBextractor.scaleFactor: 1.32', lines)
        self.assertIn('ORBextractor.nLevels: 16', lines)
        self.assertIn('ORBextractor.iniThFAST: 25', lines)
        self.assertIn('ORBextractor.minThFAST: 14', lines)

    @mock.patch('arvet_slam.systems.slam.orbslam2.tempfile', autospec=tempfile)
    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_saves_settings_file(self, _, mock_tempfile):
        random = np.random.RandomState(22)
        width = random.randint(300, 800)
        height = random.randint(300, 800)
        fx = random.uniform(0.9, 1.1) * width
        fy = random.uniform(0.9, 1.1) * height
        cx = random.uniform(0, 1) * width
        cy = random.uniform(0, 1) * height
        k1 = random.uniform(0, 1)
        k2 = random.uniform(0, 1)
        k3 = random.uniform(0, 1)
        p1 = random.uniform(0, 1)
        p2 = random.uniform(0, 1)
        framerate = float(random.randint(200, 600) / 64)
        stereo_offset = Transform(random.uniform(-1, 1, size=3))

        mock_tempfile.mkstemp.return_value = (12, 'my_temp_file.yml')
        mock_file = InspectableStringIO()
        mock_open = mock.mock_open()
        mock_open.return_value = mock_file

        intrinsics = CameraIntrinsics(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2
        )
        subject = OrbSlam2(
            mode=SensorMode.STEREO,
            vocabulary_file=self.vocabulary_file,
            depth_threshold=random.uniform(0, 255),
            orb_num_features=random.randint(0, 8000),
            orb_scale_factor=random.uniform(0, 2),
            orb_num_levels=random.randint(1, 10),
            orb_ini_threshold_fast=random.randint(15, 100),
            orb_min_threshold_fast=random.randint(0, 15)
        )
        subject.resolve_paths(self.path_manager)
        subject.set_camera_intrinsics(intrinsics, 1 / framerate)
        subject.set_stereo_offset(stereo_offset)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)

        contents = mock_file.getvalue()
        lines = contents.split('\n')
        self.assertGreater(len(lines), 0)

        self.assertEqual('%YAML:1.0', lines[0])
        self.assertIn('Camera.fx: {0}'.format(fx), lines)
        self.assertIn('Camera.fy: {0}'.format(fy), lines)
        self.assertIn('Camera.cx: {0}'.format(cx), lines)
        self.assertIn('Camera.cy: {0}'.format(cy), lines)
        self.assertIn('Camera.k1: {0}'.format(k1), lines)
        self.assertIn('Camera.k2: {0}'.format(k2), lines)
        self.assertIn('Camera.k3: {0}'.format(k3), lines)
        self.assertIn('Camera.p1: {0}'.format(p1), lines)
        self.assertIn('Camera.p2: {0}'.format(p2), lines)
        self.assertIn('Camera.width: {0}'.format(width), lines)
        self.assertIn('Camera.height: {0}'.format(height), lines)
        self.assertIn('Camera.fps: {0}'.format(framerate), lines)
        self.assertIn('Camera.bf: {0}'.format(-1 * stereo_offset.location[1] * fx), lines)
        self.assertIn('ThDepth: {0}'.format(subject.depth_threshold), lines)
        self.assertIn('ORBextractor.nFeatures: {0}'.format(subject.orb_num_features), lines)
        self.assertIn('ORBextractor.scaleFactor: {0}'.format(subject.orb_scale_factor), lines)
        self.assertIn('ORBextractor.nLevels: {0}'.format(subject.orb_num_levels), lines)
        self.assertIn('ORBextractor.iniThFAST: {0}'.format(subject.orb_ini_threshold_fast), lines)
        self.assertIn('ORBextractor.minThFAST: {0}'.format(subject.orb_min_threshold_fast), lines)

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_uses_id_in_settings_file_to_avoid_collisions(self, _):
        sys_id = ObjectId()
        mock_open = mock.mock_open()

        subject = OrbSlam2(_id=sys_id, mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_open.called)
        self.assertIn(str(sys_id), str(mock_open.call_args[0][0]))

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_finds_available_file(self, _):
        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        self.assertIsNone(subject._settings_file)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        self.assertIsNotNone(subject._settings_file)
        self.assertTrue(os.path.isfile(subject._settings_file))

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_does_nothing_for_non_sequential_input(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        mock_open = mock.mock_open()
        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(ImageSequenceType.NON_SEQUENTIAL)
        self.assertFalse(mock_multiprocessing.Process.called)
        self.assertFalse(mock_process.start.called)
        self.assertFalse(mock_open.called)

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_starts_a_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process

        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertEqual(run_orbslam, mock_multiprocessing.Process.call_args[1]['target'])
        self.assertTrue(mock_process.start.called)

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_waits_for_a_response_from_a_subprocess(self, mock_multiprocessing):
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.get.return_value = 'ORBSLAM Ready!'
        mock_multiprocessing.Queue.return_value = mock_queue

        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_queue.get.called)

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_kills_subprocess_and_raises_exception_if_it_gets_no_response(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process

        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.get.side_effect = QueueEmpty
        mock_multiprocessing.Queue.return_value = mock_queue

        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        subject.save_settings()
        settings_path = subject._settings_file
        with self.assertRaises(RuntimeError):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_process.start.called)
        self.assertTrue(mock_process.terminate.called)
        self.assertTrue(mock_process.join.called)
        self.assertTrue(mock_process.kill.called)
        self.assertFalse(settings_path.exists())

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_process_image_mono_sends_image_to_subprocess(self, mock_multiprocessing):
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)     # Have to be specific to get the class
        mock_queue.qsize.return_value = 0
        mock_multiprocessing.Queue.return_value = mock_queue
        image = make_image(SensorMode.MONOCULAR)
        greyscale_pixels = convert_to_grey(image.pixels)

        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertIn(mock_queue, mock_multiprocessing.Process.call_args[1]['args'])

        subject.process_image(image, 12)
        self.assertTrue(mock_queue.put.called)
        self.assertIn(12, [elem for elem in mock_queue.put.call_args[0][0] if isinstance(elem, int)])
        self.assertTrue(any(np.array_equal(greyscale_pixels, elem) for elem in mock_queue.put.call_args[0][0]))

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_process_image_rgbd_sends_image_and_depth_to_subprocess(self, mock_multiprocessing):
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)     # Have to be specific to get the class
        mock_queue.qsize.return_value = 0
        mock_multiprocessing.Queue.return_value = mock_queue
        image = make_image(SensorMode.RGBD)
        greyscale_pixels = convert_to_grey(image.pixels)
        float32_depth = image.depth.astype(np.float32)

        subject = OrbSlam2(mode=SensorMode.RGBD, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertIn(mock_queue, mock_multiprocessing.Process.call_args[1]['args'])

        subject.process_image(image, 12)
        self.assertTrue(mock_queue.put.called)
        self.assertIn(12, [elem for elem in mock_queue.put.call_args[0][0] if isinstance(elem, int)])
        self.assertTrue(any(np.array_equal(greyscale_pixels, elem) for elem in mock_queue.put.call_args[0][0]))
        self.assertTrue(any(np.array_equal(float32_depth, elem) for elem in mock_queue.put.call_args[0][0]))

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_process_image_stereo_sends_left_and_right_image_to_subprocess(self, mock_multiprocessing):
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)     # Have to be specific to get the class
        mock_queue.qsize.return_value = 0
        mock_multiprocessing.Queue.return_value = mock_queue
        image = make_image(SensorMode.STEREO)
        greyscale_left = convert_to_grey(image.left_pixels)
        greyscale_right = convert_to_grey(image.right_pixels)

        subject = OrbSlam2(mode=SensorMode.STEREO, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)
        subject.set_stereo_offset(Transform(location=(0.2, -0.6, 0.01)))

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertIn(mock_queue, mock_multiprocessing.Process.call_args[1]['args'])

        subject.process_image(image, 12)
        self.assertTrue(mock_queue.put.called)
        self.assertIn(12, [elem for elem in mock_queue.put.call_args[0][0] if isinstance(elem, int)])
        self.assertTrue(np.any([np.array_equal(greyscale_left, elem) for elem in mock_queue.put.call_args[0][0]]))
        self.assertTrue(np.any([np.array_equal(greyscale_right, elem) for elem in mock_queue.put.call_args[0][0]]))

    def test_finish_trial_raises_exception_if_unstarted(self):
        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)
        with self.assertRaises(RuntimeError):
            subject.finish_trial()

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_finish_trial_joins_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.qsize.return_value = 0
        mock_queue.get.side_effect = [
            'ORBSLAM Ready!',
            {
                1.3 * idx: [
                    0.122, 15, 6, TrackingState.OK,
                    [
                        1, 0, 0, idx,
                        0, 1, 0, -0.1 * idx,
                        0, 0, 1, 0.22 * (14 - idx)
                    ]
                ]
                for idx in range(10)
            }
        ]
        mock_multiprocessing.Queue.return_value = mock_queue

        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)

        subject.finish_trial()
        self.assertTrue(mock_queue.put.called)
        self.assertIsNone(mock_queue.put.call_args[0][0])
        self.assertTrue(mock_process.join.called)

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_finish_trial_returns_result_with_data_from_subprocess(self, mock_multiprocessing):
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.qsize.return_value = 0
        mock_queue.get.side_effect = [
            'ORBSLAM Ready!',
            {
                1.3 * idx: [
                    0.122 + 0.09 * idx,     # Processing Time
                    15 + idx,               # Number of features
                    6 + idx,                # Number of matches
                    TrackingState.OK,       # Tracking state
                    [   # Estimated pose
                        [1, 0, 0, idx],
                        [0, 1, 0, -0.1 * idx],
                        [0, 0, 1, 0.22 * (14 - idx)]
                    ]
                ]
                for idx in range(10)
            }
        ]
        mock_multiprocessing.Queue.return_value = mock_queue
        image_ids = []
        intrinsics = CameraIntrinsics(
            width=640, height=480, fx=320, fy=321, cx=322, cy=240,
            k1=0.11, k2=-.33, k3=0.077, p1=1.3, p2=-0.44
        )

        subject = OrbSlam2(
            mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file,
            depth_threshold=42.0,
            orb_num_features=1337,
            orb_scale_factor=1.03,
            orb_num_levels=22,
            orb_ini_threshold_fast=7,
            orb_min_threshold_fast=4
        )
        subject.set_camera_intrinsics(intrinsics, 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(10):
            image = make_image(SensorMode.MONOCULAR)
            image.metadata.camera_pose = Transform((0.25 * (14 - idx), -1.1 * idx, 0.11 * idx))
            image_ids.append(image.pk)
            subject.process_image(image, 1.3 * idx)
        trial_result = subject.finish_trial()
        self.assertIsInstance(trial_result, SLAMTrialResult)
        self.assertTrue(trial_result.success)
        self.assertGreater(trial_result.run_time, 0)
        for key, value in {
            'in_fx': intrinsics.fx,
            'in_fy': intrinsics.fy,
            'in_cx': intrinsics.cx,
            'in_cy': intrinsics.cy,
            'in_k1': intrinsics.k1,
            'in_k2': intrinsics.k2,
            'in_p1': intrinsics.p1,
            'in_p2': intrinsics.p2,
            'in_k3': intrinsics.k3,
            'in_width': intrinsics.width,
            'in_height': intrinsics.height,
            'base': float('nan'),
            'vocabulary_file': str(subject.vocabulary_file),
            'mode': str(subject.mode.name),
            'depth_threshold': subject.depth_threshold,
            'orb_num_features': subject.orb_num_features,
            'orb_scale_factor': subject.orb_scale_factor,
            'orb_num_levels': subject.orb_num_levels,
            'orb_ini_threshold_fast': subject.orb_ini_threshold_fast,
            'orb_min_threshold_fast': subject.orb_min_threshold_fast
        }.items():
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(trial_result.settings[key]))
            else:
                self.assertEqual(value, trial_result.settings[key])
        self.assertEqual(10, len(trial_result.results))
        for idx in range(10):
            frame_result = trial_result.results[idx]
            with no_auto_dereference(FrameResult):
                self.assertEqual(image_ids[idx], frame_result.image)
            self.assertEqual(Transform((0.25 * (14 - idx), -1.1 * idx, 0.11 * idx)), frame_result.pose)
            self.assertEqual(1.3 * idx, frame_result.timestamp)
            self.assertEqual(15 + idx, frame_result.num_features)
            self.assertEqual(6 + idx, frame_result.num_matches)
            # Coordinates of the estimated pose should be rearranged
            self.assertEqual(Transform([0.22 * (14 - idx), -1 * idx, 0.1 * idx]), frame_result.estimated_pose)

    @mock.patch('arvet_slam.systems.slam.orbslam2.logging', autospec=logging)
    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_logs_timestamps_returned_by_subprocess_without_matching_frame(self, mock_multiprocessing, mock_logging):
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.qsize.return_value = 0
        mock_queue.get.side_effect = [
            'ORBSLAM Ready!',
            {
                1.3 * idx: [
                    0.122 + 0.09 * idx,     # Processing Time
                    15 + idx,               # Number of features
                    6 + idx,                # Number of matches
                    TrackingState.OK,       # Tracking state
                    [  # Estimated pose
                        1, 0, 0, idx,
                        0, 1, 0, -0.1 * idx,
                        0, 0, 1, 0.22 * (14 - idx)
                    ]
                ]
                for idx in range(10)
            }
        ]
        mock_multiprocessing.Queue.return_value = mock_queue

        mock_logger = mock.MagicMock()
        mock_logging.getLogger.return_value = mock_logger

        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(ImageSequenceType.SEQUENTIAL)
        # Finish without giving it any frames
        trial_result = subject.finish_trial()

        self.assertIsInstance(trial_result, SLAMTrialResult)
        self.assertFalse(trial_result.success)
        self.assertEqual(0, len(trial_result.results))
        self.assertTrue(mock_logger.warning.called)
        for idx in range(10):
            # Look for the missing timestamps in the log messages
            self.assertTrue(any(str(1.3 * idx) in call_args[0][0] for call_args in mock_logger.warning.call_args_list))

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_finish_trial_cleans_up_and_raises_exception_if_cannot_get_data_from_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_queue.qsize.return_value = 0
        mock_queue.get.side_effect = ['ORBSLAM ready!', QueueEmpty()]
        mock_multiprocessing.Queue.return_value = mock_queue

        subject = OrbSlam2(mode=SensorMode.MONOCULAR, vocabulary_file=self.vocabulary_file)
        subject.set_camera_intrinsics(CameraIntrinsics(width=640, height=480, fx=320, fy=321, cx=322, cy=240), 1 / 29)
        subject.resolve_paths(self.path_manager)

        subject.save_settings()
        settings_file = subject._settings_file
        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        with self.assertRaises(RuntimeError):
            subject.finish_trial()
        self.assertTrue(mock_process.join.called)
        self.assertFalse(settings_file.exists())


class TestDumpConfig(unittest.TestCase):
    config = {
        'Camera': {
            'fx': 320,
            'fy': 240,
            'cx': 320,
            'cy': 240,
            'k1': 0,
            'k2': 0,
            'p1': 0,
            'p2': 0,
            'k3': 0,
            'width': 640,
            'height': 480,
            'fps': 30.0,
            'RGB': 1
        },
        'ThDepth': 70,
        'DepthMapFactor': 1.0,
        'ORBextractor': {
            'nFeatures': 2000,
            'scaleFactor': 1.2,
            'nLevels': 8,
            'iniThFAST': 12,
            'minThFAST': 7
        },
        'Viewer': {
            'KeyFrameSize': 0.05,
            'KeyFrameLineWidth': 1,
            'GraphLineWidth': 0.9,
            'PointSize': 2,
            'CameraSize': 0.08,
            'CameraLineWidth': 3,
            'ViewpointX': 0,
            'ViewpointY': -0.7,
            'ViewpointZ': -1.8,
            'ViewpointF': 500
        }
    }

    def test_opens_and_writes_to_specified_file(self):
        mock_file = InspectableStringIO()
        mock_open = mock.mock_open()
        mock_open.return_value = mock_file

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            dump_config('test_conf.yml', self.config)
        self.assertTrue(mock_open.called)
        self.assertEqual('test_conf.yml', mock_open.call_args[0][0])
        self.assertGreater(len(mock_file.getvalue()), 0)

    def test_dump_config_writes_yaml_header(self):
        mock_file = InspectableStringIO()
        mock_open = mock.mock_open()
        mock_open.return_value = mock_file

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            dump_config('test_conf.yml', self.config)

        contents = mock_file.getvalue()
        lines = contents.split('\n')
        self.assertGreater(len(lines), 0)
        self.assertEqual('%YAML:1.0', lines[0])

    def test_nested_to_dotted_converts_arbitrary_dicts(self):
        key_chars = list(string.ascii_letters + string.digits)
        expected_key = 'foobar'
        value = 12
        nested_dict = {expected_key: value}
        self.assertEqual(nested_to_dotted(nested_dict), {expected_key: value})
        for _ in range(10):
            new_key = ''.join(np.random.choice(key_chars) for _ in range(10))
            nested_dict = {new_key: nested_dict}
            expected_key = new_key + '.' + expected_key
            self.assertEqual(nested_to_dotted(nested_dict), {expected_key: value})

    def test_converts_nested_keys_to_dots(self):
        mock_file = InspectableStringIO()
        mock_open = mock.mock_open()
        mock_open.return_value = mock_file

        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            dump_config('test_conf.yml', self.config)

        contents = mock_file.getvalue()
        lines = contents.split('\n')

        expected_conf = nested_to_dotted(self.config)
        for key, value in expected_conf.items():
            self.assertIn('{key}: {value}'.format(key=key, value=value), lines)


class TestMakeRelativePose(ExtendedTestCase):

    def test_returns_transform_object(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = make_relative_pose(frame_delta)
        self.assertIsInstance(pose, Transform)

    def test_rearranges_location_coordinates(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = make_relative_pose(frame_delta)
        self.assertNPEqual((13.2, -10, 22.4), pose.location)

    def test_changes_rotation_each_axis(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        # Roll, rotation around z-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 0, 1), np.pi / 6, True)
        pose = make_relative_pose(frame_delta)
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around x-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((1, 0, 0), np.pi / 6, True)
        pose = make_relative_pose(frame_delta)
        self.assertNPClose((0, -np.pi / 6, 0), pose.euler)

        # Yaw, rotation around negative y-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 1, 0), np.pi / 6, True)
        pose = make_relative_pose(frame_delta)
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
            pose = make_relative_pose(frame_delta)
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))


class InspectableStringIO(StringIO):
    """
    A tiny modification on StringIO to preserve the value
    This can be returned from a mocked open() call to act like a file,
    and then allow inspection of the file contents.
    Does not have mos
    """

    def __init__(self):
        super(InspectableStringIO, self).__init__()
        self.final_value = ''

    def close(self) -> None:
        self.final_value = self.getvalue()
        super(InspectableStringIO, self).close()

    def getvalue(self) -> str:
        if self.closed:
            return self.final_value
        return super(InspectableStringIO, self).getvalue()


def make_image(img_type: SensorMode):
    pixels = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
    depth = None
    if img_type == SensorMode.RGBD:
        depth = np.random.normal(1.0, 0.01, size=(32, 32)).astype(np.float16)
    metadata = imeta.make_metadata(
        pixels=pixels,
        depth=depth,
        source_type=imeta.ImageSourceType.SYNTHETIC,
        camera_pose=Transform(location=[13.8, 2.3, -9.8]),
        intrinsics=CameraIntrinsics(
            width=pixels.shape[1],
            height=pixels.shape[0],
            fx=16, fy=16, cx=16, cy=16
        )
    )
    if img_type == SensorMode.STEREO:
        right_pixels = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
        right_metadata = imeta.make_right_metadata(
            right_pixels, metadata,
            intrinsics=CameraIntrinsics(
                width=right_pixels.shape[1],
                height=right_pixels.shape[0],
                fx=16, fy=16, cx=16, cy=16
            )
        )
        return StereoImage(
            _id=ObjectId(),
            pixels=pixels,
            metadata=metadata,
            right_pixels=right_pixels,
            right_metadata=right_metadata
        )
    elif img_type == SensorMode.RGBD:
        return Image(
            _id=ObjectId(),
            pixels=pixels,
            depth=depth,
            metadata=metadata
        )
    return Image(
        _id=ObjectId(),
        pixels=pixels,
        metadata=metadata
    )
