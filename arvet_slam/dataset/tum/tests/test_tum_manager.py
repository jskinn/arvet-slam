# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import os.path
from shutil import rmtree
from arvet_slam.dataset.tum.tum_manager import TUMManager, dataset_names


class TestTUMManager(unittest.TestCase):

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_find_roots_finds_folders_with_the_names_of_datasets(self, mock_tum_loader):
        mock_dataset_root = os.path.join(os.path.dirname(__file__), 'mock_tum_dataset')
        teddy_root = os.path.join(mock_dataset_root, 'rgbd_dataset_freiburg1_teddy')
        os.makedirs(teddy_root, exist_ok=True)

        marker = "/foobar"
        mock_tum_loader.find_files.side_effect = lambda x: (x + marker, None, None, None, None, None)

        root_dirs = TUMManager.find_roots(mock_dataset_root)
        self.assertTrue(mock_tum_loader.find_files.called)
        self.assertEqual(mock.call(teddy_root), mock_tum_loader.find_files.call_args)

        self.assertEqual({'rgbd_dataset_freiburg1_teddy': teddy_root + marker}, root_dirs)

        # Clean up
        rmtree(mock_dataset_root)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_find_roots_skips_roots_that_file_files_raises_exception(self, mock_tum_manager):
        mock_dataset_root = os.path.join(os.path.dirname(__file__), 'mock_tum_dataset')
        teddy_root = os.path.join(mock_dataset_root, 'rgbd_dataset_freiburg1_teddy')
        os.makedirs(teddy_root, exist_ok=True)

        mock_tum_manager.find_files.side_effect = FileNotFoundError()

        root_dirs = TUMManager.find_roots(mock_dataset_root)
        self.assertTrue(mock_tum_manager.find_files.called)
        self.assertEqual(mock.call(teddy_root), mock_tum_manager.find_files.call_args)

        self.assertEqual({}, root_dirs)

        # Clean up
        rmtree(mock_dataset_root)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_find_roots_finds_multiple_folders(self, mock_tum_manager):
        mock_dataset_root = os.path.join(os.path.dirname(__file__), 'mock_tum_dataset')
        marker = "/foobar"
        expected_find_files_call_args = []
        expected_root_dirs = {}
        for name in dataset_names:
            dataset_root = os.path.join(mock_dataset_root, name)
            expected_find_files_call_args.append(dataset_root)
            expected_root_dirs[name] = dataset_root + marker
            os.makedirs(dataset_root, exist_ok=True)

        mock_tum_manager.find_files.side_effect = lambda x: (x + marker, None, None, None, None, None)

        root_dirs = TUMManager.find_roots(mock_dataset_root)
        self.assertEqual(expected_root_dirs, root_dirs)
        self.assertTrue(mock_tum_manager.find_files.called)
        for dataset_root in expected_find_files_call_args:
            self.assertIn(mock.call(dataset_root), mock_tum_manager.find_files.call_args_list)

        # Clean up
        rmtree(mock_dataset_root)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_get_dataset_raises_not_found_for_missing_datasets(self, mock_tum_manager):
        mock_dataset_root = os.path.join(os.path.dirname(__file__), 'mock_tum_dataset')
        os.makedirs(mock_dataset_root, exist_ok=True)
        mock_tum_manager.find_files.side_effect = lambda x: x

        subject = TUMManager(mock_dataset_root)
        with self.assertRaises(NotADirectoryError):
            subject.get_dataset('rgbd_dataset_freiburg1_teddy')

        # Clean up
        rmtree(mock_dataset_root)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_get_dataset_makes_tasks_for_roots_from_find_roots(self, mock_tum_manager, mock_task_manager):
        mock_dataset_root = os.path.join(os.path.dirname(__file__), 'mock_tum_dataset')
        teddy_root = os.path.join(mock_dataset_root, 'rgbd_dataset_freiburg1_teddy')
        os.makedirs(teddy_root, exist_ok=True)

        module_name = 'mymodulename'
        mock_tum_manager.find_files.side_effect = lambda x: (x, None, None, None, None, None)
        mock_tum_manager.__name__ = module_name
        mock_task_manager.get_import_dataset_task.return_value = mock.Mock()

        subject = TUMManager(mock_dataset_root)
        subject.get_dataset('rgbd_dataset_freiburg1_teddy')

        self.assertEqual(mock.call(
            module_name=module_name,
            path=teddy_root,
            additional_args={'dataset_name': 'rgbd_dataset_freiburg1_teddy'},
            num_cpus=mock.ANY,
            num_gpus=mock.ANY,
            memory_requirements=mock.ANY,
            expected_duration=mock.ANY
        ), mock_task_manager.get_import_dataset_task.call_args)

        # Clean up
        rmtree(mock_dataset_root)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_get_dataset_returns_result_from_complete_tasks(self, mock_tum_manager, mock_task_manager):
        mock_dataset_root = os.path.join(os.path.dirname(__file__), 'mock_tum_dataset')
        teddy_root = os.path.join(mock_dataset_root, 'rgbd_dataset_freiburg1_teddy')
        os.makedirs(teddy_root, exist_ok=True)

        mock_tum_manager.find_files.side_effect = lambda x: (x, None, None, None, None, None)
        mock_task = mock.Mock()
        mock_task.is_finished = True
        mock_task_manager.get_import_dataset_task.return_value = mock_task

        subject = TUMManager(mock_dataset_root)
        result = subject.get_dataset('rgbd_dataset_freiburg1_teddy')
        self.assertEqual(mock_task.result, result)

        # Clean up
        rmtree(mock_dataset_root)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_get_dataset_returns_none_for_incomplete_tasks(self, mock_tum_manager, mock_task_manager):
        mock_dataset_root = os.path.join(os.path.dirname(__file__), 'mock_tum_dataset')
        teddy_root = os.path.join(mock_dataset_root, 'rgbd_dataset_freiburg1_teddy')
        os.makedirs(teddy_root, exist_ok=True)

        mock_tum_manager.find_files.side_effect = lambda x: (x, None, None, None, None, None)
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task

        subject = TUMManager(mock_dataset_root)
        result = subject.get_dataset('rgbd_dataset_freiburg1_teddy')
        self.assertIsNone(result)

        # Clean up
        rmtree(mock_dataset_root)
