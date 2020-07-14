# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import tarfile
import bson
from pathlib import Path
from shutil import rmtree
import arvet.database.tests.database_connection as dbconn
from arvet.batch_analysis.task import Task
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
import arvet_slam.dataset.tum.tum_loader as tum_loader
from arvet_slam.dataset.tum.tum_manager import TUMManager, dataset_names


class TestTUMManager(unittest.TestCase):
    mock_dataset_root = Path(__file__).parent / 'mock_tum_dataset'

    def tearDown(self) -> None:
        rmtree(self.mock_dataset_root)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_find_roots_finds_folders_with_the_names_of_datasets(self, mock_tum_loader):
        teddy_root = self.mock_dataset_root / 'rgbd_dataset_freiburg1_teddy'
        teddy_root.mkdir(parents=True, exist_ok=True)

        mock_tum_loader.find_files.side_effect = lambda x: (x / 'foobar', None, None, None, None, None)

        root_dirs = TUMManager.find_roots(self.mock_dataset_root)
        self.assertTrue(mock_tum_loader.find_files.called)
        self.assertEqual(mock.call(teddy_root), mock_tum_loader.find_files.call_args)

        self.assertEqual({'rgbd_dataset_freiburg1_teddy': teddy_root}, root_dirs)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_find_roots_skips_roots_that_file_files_raises_exception(self, mock_tum_loader):
        teddy_root = self.mock_dataset_root / 'rgbd_dataset_freiburg1_teddy'
        teddy_root.mkdir(parents=True, exist_ok=True)

        mock_tum_loader.find_files.side_effect = FileNotFoundError()

        root_dirs = TUMManager.find_roots(self.mock_dataset_root)
        self.assertTrue(mock_tum_loader.find_files.called)
        self.assertIn(mock.call(teddy_root), mock_tum_loader.find_files.call_args_list)

        self.assertEqual({}, root_dirs)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_find_roots_finds_multiple_folders(self, mock_tum_loader):
        expected_find_files_call_args = []
        expected_root_dirs = {}
        for name in dataset_names:
            dataset_root = self.mock_dataset_root / name
            expected_find_files_call_args.append(dataset_root)
            expected_root_dirs[name] = dataset_root
            dataset_root.mkdir(parents=True, exist_ok=True)

        mock_tum_loader.find_files.side_effect = lambda x: (x / 'foobar', None, None, None, None, None)

        root_dirs = TUMManager.find_roots(self.mock_dataset_root)
        self.assertEqual(expected_root_dirs, root_dirs)
        self.assertTrue(mock_tum_loader.find_files.called)
        for dataset_root in expected_find_files_call_args:
            self.assertIn(mock.call(dataset_root), mock_tum_loader.find_files.call_args_list)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_find_roots_finds_tar_files_where_roots_are_unavailable(self, mock_tum_loader):
        tiny_file = self.mock_dataset_root / 'tmp'
        expected_find_files_call_args = []
        expected_root_dirs = {}
        split1 = len(dataset_names) // 3
        split2 = split1 * 2
        folder_only_names = dataset_names[:split1]
        both_names = dataset_names[split1:split2]
        only_tarball_names = dataset_names[split2:]

        for name in dataset_names:
            if name in folder_only_names or name in both_names:
                dataset_root = self.mock_dataset_root / name
                expected_find_files_call_args.append(dataset_root)
                expected_root_dirs[name] = dataset_root
                dataset_root.mkdir(parents=True, exist_ok=True)
            if name in only_tarball_names or name in both_names:
                tarball_file = self.mock_dataset_root / (name + '.tar.tgz')
                if not tiny_file.exists():
                    tiny_file.parent.mkdir(parents=True, exist_ok=True)
                    tiny_file.touch(exist_ok=True)
                with tarfile.open(tarball_file, 'w') as tfp:
                    tfp.add(tiny_file)   # add something to the file, so that it's a valid, non-empty tarball.
                if name in only_tarball_names:
                    # Where both the tarball and root folder are available, prefer the root folder
                    # Tarball files should be missing the extension, so they match the extracted folder path
                    expected_root_dirs[name] = self.mock_dataset_root / name

        mock_tum_loader.find_files.side_effect = lambda x: (x / 'foobar', None, None, None, None, None)

        root_dirs = TUMManager.find_roots(self.mock_dataset_root)
        self.assertEqual(set(expected_root_dirs.keys()), set(root_dirs.keys()))
        for name in expected_root_dirs:
            self.assertEqual(expected_root_dirs[name], root_dirs[name], f"Incorrect root for {name}")
        self.assertTrue(mock_tum_loader.find_files.called)
        for dataset_root in expected_find_files_call_args:
            self.assertIn(mock.call(dataset_root), mock_tum_loader.find_files.call_args_list)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_find_roots_gives_same_path_for_folder_and_tarball(self, mock_tum_loader):
        teddy_root = self.mock_dataset_root / 'rgbd_dataset_freiburg1_teddy'
        teddy_root.mkdir(parents=True, exist_ok=True)
        mock_tum_loader.find_files.side_effect = lambda x: (x / 'foobar', None, None, None, None, None)

        # find the root as a folder
        root_dirs_folder = TUMManager.find_roots(self.mock_dataset_root)
        self.assertEqual({'rgbd_dataset_freiburg1_teddy': teddy_root}, root_dirs_folder)

        # Find it again as a tarball
        rmtree(teddy_root)
        tiny_file = self.mock_dataset_root / 'tmp'
        tarball_file = self.mock_dataset_root / 'rgbd_dataset_freiburg1_teddy.tar.tgz'
        tiny_file.touch(exist_ok=True)
        with tarfile.open(tarball_file, 'w') as tfp:
            tfp.add(tiny_file)  # add something to the file, so that it's a valid, non-empty tarball.

        # find the root as a folder
        root_dirs_tarball = TUMManager.find_roots(self.mock_dataset_root)
        self.assertEqual({'rgbd_dataset_freiburg1_teddy': teddy_root}, root_dirs_tarball)
        self.assertEqual(root_dirs_folder, root_dirs_tarball)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_get_dataset_raises_not_found_for_missing_datasets(self, mock_tum_loader):
        self.mock_dataset_root.mkdir(exist_ok=True)
        mock_tum_loader.find_files.side_effect = lambda x: x

        subject = TUMManager(self.mock_dataset_root)
        with self.assertRaises(NotADirectoryError):
            subject.get_dataset('rgbd_dataset_freiburg1_teddy')

    @mock.patch('arvet_slam.dataset.tum.tum_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_get_dataset_makes_tasks_for_roots_from_find_roots(self, mock_tum_loader, mock_task_manager):
        teddy_root = self.mock_dataset_root / 'rgbd_dataset_freiburg1_teddy'
        teddy_root.mkdir(parents=True, exist_ok=True)

        module_name = 'mymodulename'
        mock_tum_loader.find_files.side_effect = lambda x: (x, None, None, None, None, None)
        mock_tum_loader.__name__ = module_name
        mock_task_manager.get_import_dataset_task.return_value = mock.Mock()

        subject = TUMManager(self.mock_dataset_root)
        subject.get_dataset('rgbd_dataset_freiburg1_teddy')

        self.assertEqual(mock.call(
            module_name=module_name,
            path=str(teddy_root),
            additional_args={'dataset_name': 'rgbd_dataset_freiburg1_teddy'},
            num_cpus=mock.ANY,
            num_gpus=mock.ANY,
            memory_requirements=mock.ANY,
            expected_duration=mock.ANY
        ), mock_task_manager.get_import_dataset_task.call_args)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_get_dataset_returns_result_from_complete_tasks(self, mock_tum_loader, mock_task_manager):
        teddy_root = self.mock_dataset_root / 'rgbd_dataset_freiburg1_teddy'
        teddy_root.mkdir(parents=True, exist_ok=True)

        mock_tum_loader.find_files.side_effect = lambda x: (x, None, None, None, None, None)
        mock_task = mock.Mock()
        mock_task.is_finished = True
        result_id = bson.ObjectId()
        mock_task.get_result.return_value = result_id
        mock_task_manager.get_import_dataset_task.return_value = mock_task

        subject = TUMManager(self.mock_dataset_root)
        result = subject.get_dataset('rgbd_dataset_freiburg1_teddy')
        self.assertEqual(result_id, result)

    @mock.patch('arvet_slam.dataset.tum.tum_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.tum.tum_manager.tum_loader', autospec=True)
    def test_get_dataset_returns_none_for_incomplete_tasks(self, mock_tum_loader, mock_task_manager):
        teddy_root = self.mock_dataset_root / 'rgbd_dataset_freiburg1_teddy'
        teddy_root.mkdir(parents=True, exist_ok=True)

        mock_tum_loader.find_files.side_effect = lambda x: (x, None, None, None, None, None)
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task

        subject = TUMManager(self.mock_dataset_root)
        result = subject.get_dataset('rgbd_dataset_freiburg1_teddy')
        self.assertIsNone(result)


class TestTUMManagerDatabase(unittest.TestCase):
    path_manager = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        Task.objects.all().delete()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        Task._mongometa.collection.drop()

    def test_get_dataset_creates_and_saves_task(self):
        # Really mock this dataset path
        mock_dataset_root = Path(__file__).parent / 'mock_tum_dataset'
        teddy_root = mock_dataset_root / 'rgbd_dataset_freiburg1_teddy'
        teddy_root.mkdir(exist_ok=True, parents=True)
        (teddy_root / 'rgb.txt').touch()
        (teddy_root / 'groundtruth.txt').touch()
        (teddy_root / 'depth.txt').touch()

        subject = TUMManager(mock_dataset_root)
        result = subject.get_dataset('rgbd_dataset_freiburg1_teddy')
        self.assertIsNone(result)

        all_tasks = list(ImportDatasetTask.objects.all())
        self.assertEqual(1, len(all_tasks))
        task = all_tasks[0]
        self.assertEqual(tum_loader.__name__, task.module_name)
        self.assertEqual(str(teddy_root), task.path)

        # Clean up
        rmtree(mock_dataset_root)
