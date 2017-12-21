# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import os.path
import shutil
import numpy as np
import multiprocessing
import multiprocessing.queues
import bson.objectid as oid
import arvet.database.tests.test_entity
import arvet.util.dict_utils as du
import arvet.metadata.image_metadata as imeta
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.core.sequence_type
import arvet.core.image
import arvet_slam.systems.slam.orbslam2
import orbslam2

_temp_folder = 'temp-test-orbslam2'


class TestORBSLAM2(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs(_temp_folder, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_temp_folder)

    def get_class(self):
        return arvet_slam.systems.slam.orbslam2.ORBSLAM2

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'vocabulary_file': 'imafile-{0}'.format(np.random.randint(10, 20)),
            'settings': {
                'Camera': {
                    'fx': np.random.uniform(10, 1000),
                    'fy': np.random.uniform(10, 1000),
                    'cx': np.random.uniform(10, 1000),
                    'cy': np.random.uniform(10, 1000),

                    'k1': np.random.uniform(-1, 1),
                    'k2': np.random.uniform(-1, 2),
                    'p1': np.random.uniform(0, 1),
                    'p2': np.random.uniform(0, 1),
                    'k3': np.random.uniform(-10, 10),
                    'fps': np.random.uniform(10, 100),
                    'RGB': np.random.randint(0, 1),
                },
                'ORBextractor': {
                    'nFeatures': np.random.randint(0, 8000),
                    'scaleFactor': np.random.uniform(0, 2),
                    'nLevels': np.random.randint(1, 10),
                    'iniThFAST': np.random.randint(0, 100),
                    'minThFAST': np.random.randint(0, 20)
                },
                'Viewer': {
                    'KeyFrameSize': np.random.uniform(0, 1),
                    'KeyFrameLineWidth': np.random.uniform(0, 3),
                    'GraphLineWidth': np.random.uniform(0, 3),
                    'PointSize': np.random.uniform(0, 3),
                    'CameraSize': np.random.uniform(0, 1),
                    'CameraLineWidth': np.random.uniform(0, 3),
                    'ViewpointX': np.random.uniform(0, 10),
                    'ViewpointY': np.random.uniform(0, 10),
                    'ViewpointZ': np.random.uniform(0, 10),
                    'ViewpointF': np.random.uniform(0, 10)
                }
            },
            'mode': arvet_slam.systems.slam.orbslam2.SensorMode(np.random.randint(0, 3)),
            'temp_folder': _temp_folder
        })
        return arvet_slam.systems.slam.orbslam2.ORBSLAM2(*args, **kwargs)

    def assert_models_equal(self, trial_result1, trial_result2):
        """
        Helper to assert that two SLAM trial results models are equal
        :param trial_result1:
        :param trial_result2:
        :return:
        """
        if (not isinstance(trial_result1, arvet_slam.systems.slam.orbslam2.ORBSLAM2) or
                not isinstance(trial_result2, arvet_slam.systems.slam.orbslam2.ORBSLAM2)):
            self.fail('object was not a ORBSLAM2')
        self.assertEqual(trial_result1.identifier, trial_result2.identifier)
        self.assertEqual(trial_result1._vocabulary_file, trial_result2._vocabulary_file)
        self.assertEqual(trial_result1._orbslam_settings, trial_result2._orbslam_settings)
        self.assertEqual(trial_result1.get_settings(), trial_result2.get_settings())

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_starts_a_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        subject = self.make_instance()
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertEqual(arvet_slam.systems.slam.orbslam2.run_orbslam,
                         mock_multiprocessing.Process.call_args[1]['target'])
        self.assertTrue(mock_process.start.called)

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_saves_settings_file(self, _):
        width = np.random.randint(300, 800)
        height = np.random.randint(300, 800)
        fx = np.random.uniform(0.9, 1.1) * width
        fy = np.random.uniform(0.9, 1.1) * height
        cx = np.random.uniform(0, 1) * width
        cy = np.random.uniform(0, 1) * height
        k1 = np.random.uniform(0, 1)
        k2 = np.random.uniform(0, 1)
        k3 = np.random.uniform(0, 1)
        p1 = np.random.uniform(0, 1)
        p2 = np.random.uniform(0, 1)
        baseline = np.random.uniform(0, 1)

        settings = {
            'Camera': {
                'fps': np.random.uniform(10, 100),
                'RGB': np.random.randint(0, 1)
            },
            'ThDepth': np.random.randint(0, 255),
            'DepthMapFactor': np.random.randint(0, 255),
            'ORBextractor': {
                'nFeatures': np.random.randint(0, 8000),
                'scaleFactor': np.random.uniform(0, 2),
                'nLevels': np.random.randint(1, 10),
                'iniThFAST': np.random.randint(0, 100),
                'minThFAST': np.random.randint(0, 20)
            }
        }

        mock_open = mock.mock_open()
        subject = self.make_instance(settings=settings)
        subject.set_camera_intrinsics(cam_intr.CameraIntrinsics(
            width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2
        ))
        subject.set_stereo_baseline(baseline)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_open.called)
        filename = mock_open.call_args[0][0]
        self.assertTrue(os.path.isfile(filename))

        mock_file = mock_open()
        self.assertTrue(mock_file.write.called)

        # Join together all the file contents
        file_contents = ""
        for args in mock_file.write.call_args_list:
            file_contents += args[0][0]

        self.assertTrue(file_contents.startswith('%YAML:1.0\n'))
        self.assertIn('Camera.fx: {0}'.format(fx), file_contents)
        self.assertIn('Camera.fy: {0}'.format(fy), file_contents)
        self.assertIn('Camera.cx: {0}'.format(cx), file_contents)
        self.assertIn('Camera.cy: {0}'.format(cy), file_contents)
        self.assertIn('Camera.k1: {0}'.format(k1), file_contents)
        self.assertIn('Camera.k2: {0}'.format(k2), file_contents)
        self.assertIn('Camera.k3: {0}'.format(k3), file_contents)
        self.assertIn('Camera.p1: {0}'.format(p1), file_contents)
        self.assertIn('Camera.p2: {0}'.format(p2), file_contents)
        self.assertIn('Camera.width: {0}'.format(width), file_contents)
        self.assertIn('Camera.height: {0}'.format(height), file_contents)
        self.assertIn('Camera.fps: {0}'.format(settings['Camera']['fps']), file_contents)
        self.assertIn('Camera.bf: {0}'.format(baseline * fx), file_contents)
        self.assertIn('Camera.RGB: {0}'.format(settings['Camera']['RGB']), file_contents)
        self.assertIn('ThDepth: {0}'.format(settings['ThDepth']), file_contents)
        self.assertIn('DepthMapFactor: {0}'.format(settings['DepthMapFactor']), file_contents)
        self.assertIn('ORBextractor.nFeatures: {0}'.format(settings['ORBextractor']['nFeatures']), file_contents)
        self.assertIn('ORBextractor.scaleFactor: {0}'.format(settings['ORBextractor']['scaleFactor']), file_contents)
        self.assertIn('ORBextractor.nLevels: {0}'.format(settings['ORBextractor']['nLevels']), file_contents)
        self.assertIn('ORBextractor.iniThFAST: {0}'.format(settings['ORBextractor']['iniThFAST']), file_contents)
        self.assertIn('ORBextractor.minThFAST: {0}'.format(settings['ORBextractor']['minThFAST']), file_contents)

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_uses_id_in_settings_file(self, _):
        mock_open = mock.mock_open()
        sys_id = oid.ObjectId()
        temp_folder = _temp_folder
        subject = self.make_instance(temp_folder=temp_folder, id_=sys_id)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_open.called)
        self.assertIn(str(sys_id), mock_open.call_args[0][0])

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_finds_available_file(self, _):
        mock_open = mock.mock_open()
        subject = self.make_instance()
        self.assertIsNone(subject._settings_file)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertIsNotNone(subject._settings_file)
        self.assertTrue(os.path.isfile(subject._settings_file))

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_start_trial_does_nothing_for_non_sequential_input(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        mock_open = mock.mock_open()
        subject = self.make_instance()
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
        self.assertFalse(mock_multiprocessing.Process.called)
        self.assertFalse(mock_process.start.called)
        self.assertFalse(mock_open.called)

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_process_image_mono_sends_image_to_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)     # Have to be specific to get the class
        mock_queue.qsize.return_value = 0
        mock_multiprocessing.Process.return_value = mock_process
        mock_multiprocessing.Queue.return_value = mock_queue

        mock_image_data = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
        image = arvet.core.image.Image(data=mock_image_data, metadata=imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            hash_=b'\x00\x00\x00\x00\x00\x00\x00\x01'
        ))

        subject = self.make_instance(mode=arvet_slam.systems.slam.orbslam2.SensorMode.MONOCULAR)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertIn(mock_queue, mock_multiprocessing.Process.call_args[1]['args'])

        subject.process_image(image, 12)
        self.assertTrue(mock_queue.put.called)
        self.assertIn(12, [elem for elem in mock_queue.put.call_args[0][0] if isinstance(elem, int)])
        self.assertTrue(np.any([np.array_equal(mock_image_data, elem) for elem in mock_queue.put.call_args[0][0]]))

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_process_image_rgbd_sends_image_and_depth_to_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)     # Have to be specific to get the class
        mock_queue.qsize.return_value = 0
        mock_multiprocessing.Process.return_value = mock_process
        mock_multiprocessing.Queue.return_value = mock_queue

        mock_image_data = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
        mock_depth_data = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
        image = arvet.core.image.Image(data=mock_image_data, depth_data=mock_depth_data, metadata=imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            hash_=b'\x00\x00\x00\x00\x00\x00\x00\x01'
        ))

        subject = self.make_instance(mode=arvet_slam.systems.slam.orbslam2.SensorMode.RGBD)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertIn(mock_queue, mock_multiprocessing.Process.call_args[1]['args'])

        subject.process_image(image, 12)
        self.assertTrue(mock_queue.put.called)
        self.assertIn(12, [elem for elem in mock_queue.put.call_args[0][0] if isinstance(elem, int)])
        self.assertTrue(np.any([np.array_equal(mock_image_data, elem) for elem in mock_queue.put.call_args[0][0]]))
        self.assertTrue(np.any([np.array_equal(mock_depth_data, elem) for elem in mock_queue.put.call_args[0][0]]))

    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_process_image_stereo_sends_left_and_right_image_to_subprocess(self, mock_multiprocessing):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_queue = mock.create_autospec(multiprocessing.queues.Queue)     # Have to be specific to get the class
        mock_queue.qsize.return_value = 0
        mock_multiprocessing.Process.return_value = mock_process
        mock_multiprocessing.Queue.return_value = mock_queue

        mock_left_data = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
        mock_right_data = np.random.randint(0, 255, (32, 32, 3), dtype='uint8')
        image = arvet.core.image.StereoImage(left_data=mock_left_data, right_data=mock_right_data,
                                             metadata=imeta.ImageMetadata(
                                                 source_type=imeta.ImageSourceType.SYNTHETIC,
                                                 hash_=b'\x00\x00\x00\x00\x00\x00\x00\x01'
                                             ))

        subject = self.make_instance(mode=arvet_slam.systems.slam.orbslam2.SensorMode.STEREO)
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock.mock_open(), create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        self.assertTrue(mock_multiprocessing.Process.called)
        self.assertIn(mock_queue, mock_multiprocessing.Process.call_args[1]['args'])

        subject.process_image(image, 12)
        self.assertTrue(mock_queue.put.called)
        self.assertIn(12, [elem for elem in mock_queue.put.call_args[0][0] if isinstance(elem, int)])
        self.assertTrue(np.any([np.array_equal(mock_left_data, elem) for elem in mock_queue.put.call_args[0][0]]))
        self.assertTrue(np.any([np.array_equal(mock_right_data, elem) for elem in mock_queue.put.call_args[0][0]]))

    @mock.patch('arvet_slam.systems.slam.orbslam2.os', autospec=os)
    @mock.patch('arvet_slam.systems.slam.orbslam2.multiprocessing', autospec=multiprocessing)
    def test_finish_trial_waits_for_output(self, mock_multiprocessing, mock_os):
        mock_process = mock.create_autospec(multiprocessing.Process)
        mock_multiprocessing.Process.return_value = mock_process
        mock_open = mock.mock_open()
        subject = self.make_instance()
        with mock.patch('arvet_slam.systems.slam.orbslam2.open', mock_open, create=True):
            subject.start_trial(arvet.core.sequence_type.ImageSequenceType.NON_SEQUENTIAL)
        # TODO: Finish testing finish_trial

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


class TestRunOrbslam(unittest.TestCase):

    def test_calls_initialize_and_shutdown(self):
        mock_input_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_input_queue.get.side_effect = [None, None]
        mock_output_queue = mock.create_autospec(multiprocessing.queues.Queue)
        mock_orbslam = mock.create_autospec(orbslam2)
        mock_system = mock.create_autospec(orbslam2.System)
        mock_orbslam.System.return_value = mock_system
        with mock.patch.dict('sys.modules', orbslam2=mock_orbslam):
            arvet_slam.systems.slam.orbslam2.run_orbslam(mock_output_queue, mock_input_queue, '', '',
                                                         arvet_slam.systems.slam.orbslam2.SensorMode.RGBD)
        self.assertTrue(mock_system.initialize.called)
        self.assertTrue(mock_system.shutdown.called)
