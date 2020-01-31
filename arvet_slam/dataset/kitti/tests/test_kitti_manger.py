import unittest
import unittest.mock as mock
from pathlib import Path
from shutil import rmtree
import arvet.database.tests.database_connection as dbconn
from arvet.batch_analysis.task import Task
from arvet.batch_analysis.tasks.import_dataset_task import ImportDatasetTask
import arvet_slam.dataset.kitti.kitti_loader as kitti_loader
from arvet_slam.dataset.kitti.kitti_manager import KITTIManager, to_sequence_id, sequence_ids


class TestKITTIManager(unittest.TestCase):
    temp_folder = Path(__file__).parent / 'mock_kitti_dataset'

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.temp_folder.exists():
            rmtree(cls.temp_folder)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_find_roots_finds_folders_with_the_names_of_datasets(self, mock_kitti_loader):
        sequence_name = '000001'
        sequence_01_root = self.temp_folder / sequence_name
        sequence_01_root.mkdir(parents=True, exist_ok=True)

        marker = "foobar"
        mock_kitti_loader.find_root.side_effect = lambda x, seq: str(Path(x) / marker)

        root_dirs = KITTIManager.find_roots(str(self.temp_folder))
        self.assertTrue(mock_kitti_loader.find_root.called)
        self.assertEqual(mock.call(str(sequence_01_root), sequence_name), mock_kitti_loader.find_root.call_args)

        self.assertEqual({1: str(sequence_01_root / marker)}, root_dirs)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_find_roots_skips_roots_that_file_files_raises_exception(self, mock_kitti_loader):
        sequence_name = '000001'
        sequence_01_root = self.temp_folder / sequence_name
        sequence_01_root.mkdir(parents=True, exist_ok=True)

        mock_kitti_loader.find_root.side_effect = FileNotFoundError()

        root_dirs = KITTIManager.find_roots(self.temp_folder)
        self.assertTrue(mock_kitti_loader.find_root.called)
        self.assertIn(mock.call(str(sequence_01_root), sequence_name), mock_kitti_loader.find_root.call_args_list)

        self.assertEqual({}, root_dirs)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_find_roots_finds_multiple_folders(self, mock_kitti_loader):
        marker = "foobar"
        expected_find_root_call_args = []
        expected_root_dirs = {}
        for sequence_id in sequence_ids:
            sequence_name = '{0:06}'.format(sequence_id)
            dataset_root = self.temp_folder / sequence_name
            expected_find_root_call_args.append((str(dataset_root), sequence_name))
            expected_root_dirs[sequence_id] = str(dataset_root / marker)
            dataset_root.mkdir(parents=True, exist_ok=True)

        mock_kitti_loader.find_root.side_effect = lambda x, seq: str(Path(x) / marker)

        root_dirs = KITTIManager.find_roots(self.temp_folder)
        self.assertEqual(expected_root_dirs, root_dirs)
        self.assertTrue(mock_kitti_loader.find_root.called)
        for dataset_root, sequence_name in expected_find_root_call_args:
            self.assertIn(mock.call(dataset_root, sequence_name), mock_kitti_loader.find_root.call_args_list)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_get_dataset_raises_not_found_for_missing_datasets(self, mock_kitti_loader, _):
        existing_ids = {2, 5, 9}
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        for sequence_id in existing_ids:
            (self.temp_folder / '{0:06}'.format(sequence_id)).mkdir(parents=True, exist_ok=True)
        mock_kitti_loader.find_root.side_effect = lambda x, seq: x

        # Add any other sequence ids that happen to exist
        for sequence_id in sequence_ids:
            if (self.temp_folder / '{0:06}'.format(sequence_id)).exists():
                existing_ids.add(sequence_id)
        if existing_ids == set(range(11)):
            rmtree(self.temp_folder / '000008')
            existing_ids.remove(8)

        subject = KITTIManager(self.temp_folder)
        for sequence_id in range(11):
            if sequence_id in existing_ids:
                # No exception
                subject.get_dataset(sequence_id)
            else:
                with self.assertRaises(NotADirectoryError):
                    subject.get_dataset(sequence_id)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_get_dataset_raises_not_found_for_indexes_outside_range(self, mock_kitti_loader, _):
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        (self.temp_folder / '000001').mkdir(parents=True, exist_ok=True)
        mock_kitti_loader.find_root.side_effect = lambda x, seq: x

        subject = KITTIManager(self.temp_folder)
        for sequence_idx in range(-20, 20):
            if not 0 <= sequence_idx < 11:
                with self.assertRaises(NotADirectoryError):
                    subject.get_dataset(sequence_idx)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_get_dataset_raises_error_for_sequence_ids_that_are_invalid(self, *_):
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        subject = KITTIManager(self.temp_folder)
        with self.assertRaises(NotADirectoryError) as ex:
            subject.get_dataset('my_sequence_name')
        self.assertIn('my_sequence_name', str(ex.exception))

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_get_dataset_makes_tasks_for_roots_from_find_roots(self, mock_kitti_loader, mock_task_manager):
        sequence_name = '000001'
        sequence_01_root = self.temp_folder / sequence_name
        sequence_01_root.mkdir(parents=True, exist_ok=True)

        module_name = 'mymodulename'
        marker = 'foobar'
        mock_kitti_loader.find_root.side_effect = lambda x, seq: str(Path(x) / marker)
        mock_kitti_loader.__name__ = module_name
        mock_task_manager.get_import_dataset_task.return_value = mock.Mock()

        subject = KITTIManager(self.temp_folder)
        subject.get_dataset(1)

        self.assertEqual(mock.call(
            module_name=module_name,
            path=str(sequence_01_root / marker),
            additional_args={'sequence_number': 1},
            num_cpus=mock.ANY,
            num_gpus=mock.ANY,
            memory_requirements=mock.ANY,
            expected_duration=mock.ANY
        ), mock_task_manager.get_import_dataset_task.call_args)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_get_dataset_returns_result_from_complete_tasks(self, mock_kitti_loader, mock_task_manager):
        sequence_name = '000002'
        (self.temp_folder / sequence_name).mkdir(parents=True, exist_ok=True)

        mock_kitti_loader.find_root.side_effect = lambda x, seq: x
        mock_task = mock.Mock()
        mock_task.is_finished = True
        mock_task_manager.get_import_dataset_task.return_value = mock_task

        subject = KITTIManager(self.temp_folder)
        result = subject.get_dataset(2)
        self.assertEqual(mock_task.result, result)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_get_dataset_returns_none_for_incomplete_tasks(self, mock_kitti_loader, mock_task_manager):
        sequence_name = '000005'
        (self.temp_folder / sequence_name).mkdir(parents=True, exist_ok=True)

        mock_kitti_loader.find_root.side_effect = lambda x, seq: x
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task

        subject = KITTIManager(self.temp_folder)
        result = subject.get_dataset(4)
        self.assertIsNone(result)

    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.kitti.kitti_manager.kitti_loader', autospec=True)
    def test_get_dataset_works_with_string_names(self, mock_kitti_loader, mock_task_manager):
        sequence_name = '000002'
        (self.temp_folder / sequence_name).mkdir(parents=True, exist_ok=True)

        mock_kitti_loader.find_root.side_effect = lambda x, seq: x
        mock_task = mock.Mock()
        mock_task.is_finished = True
        mock_task_manager.get_import_dataset_task.return_value = mock_task

        subject = KITTIManager(self.temp_folder)
        result = subject.get_dataset('2')
        self.assertEqual(mock_task.result, result)

        result = subject.get_dataset('00002')
        self.assertEqual(mock_task.result, result)

        result = subject.get_dataset(sequence_name)
        self.assertEqual(mock_task.result, result)


class TestKITTIManagerDatabase(unittest.TestCase):
    mock_dataset_root = Path(__file__).parent / 'mock_kitti_dataset'

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
        if cls.mock_dataset_root.exists():
            rmtree(cls.mock_dataset_root)

    def test_get_dataset_creates_and_saves_task(self):
        # Really mock this dataset path
        sequence_idx = 3
        sequence_name = "{0:06}".format(sequence_idx)
        short_name = "{0:02}".format(sequence_idx)

        sequence_root = self.mock_dataset_root / 'dataset'
        sequence_root.mkdir(exist_ok=True, parents=True)
        (sequence_root / 'sequences' / short_name / 'image_2').mkdir(parents=True, exist_ok=True)
        (sequence_root / 'sequences' / short_name / 'image_3').mkdir(parents=True, exist_ok=True)
        (sequence_root / 'sequences' / short_name / 'calib.txt').touch()
        (sequence_root / 'sequences' / short_name / 'times.txt').touch()
        (sequence_root / 'poses').mkdir(parents=True, exist_ok=True)
        (sequence_root / 'poses' / ("{0:02}.txt".format(3))).touch()

        subject = KITTIManager(self.mock_dataset_root)
        result = subject.get_dataset(sequence_name)
        self.assertIsNone(result)

        all_tasks = list(ImportDatasetTask.objects.all())
        self.assertEqual(1, len(all_tasks))
        task = all_tasks[0]
        self.assertEqual(kitti_loader.__name__, task.module_name)
        self.assertEqual(str(sequence_root), task.path)


class TestToSequenceId(unittest.TestCase):

    def test_handles_correct_ids(self):
        for idx in range(11):
            self.assertEqual(idx, to_sequence_id(idx))

    def test_handles_correct_names_as_strings(self):
        for idx in range(11):
            self.assertEqual(idx, to_sequence_id(str(idx)))

    def test_raises_value_for_ints_outside_range(self):
        for idx in range(50):
            if 0 <= idx < 11:
                self.assertEqual(idx, to_sequence_id(idx))
            else:
                self.assertEqual(-1, to_sequence_id(idx))

    def test_handles_floats(self):
        for idx in range(11):
            float_idx = idx + (idx % 18) / 90
            self.assertEqual(idx, to_sequence_id(float_idx))

    def test_handles_strings_with_padding(self):
        for idx in range(11):
            self.assertEqual(idx, to_sequence_id("{0:029}".format(idx)))

    def test_returns_negative_one_for_unparsable_strings(self):
        self.assertEqual(-1, to_sequence_id('a'))
        self.assertEqual(-1, to_sequence_id('notanid'))
