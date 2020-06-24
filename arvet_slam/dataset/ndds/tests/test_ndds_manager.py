import unittest
import unittest.mock as mock
import logging
from bson import ObjectId
import shutil
import json
from pathlib import Path
from arvet.metadata.image_metadata import TimeOfDay
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader
import arvet_slam.dataset.ndds.ndds_manager as ndds


class TestNDDSManager(unittest.TestCase):
    root_path = Path(__file__).parent
    sequence_entries = None

    @classmethod
    def setUpClass(cls) -> None:
        root = cls.root_path
        cls.sequence_entries = [
            ndds.SequenceEntry(
                'winter_house',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.MAX_QUALITY,
                TimeOfDay.DAY,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-winter_house-000.tar.gz'
            ),
            ndds.SequenceEntry(
                'winter_house',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.NO_TEXTURE,
                TimeOfDay.DAY,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-winter_house-001.tar.gz'
            ),
            ndds.SequenceEntry(
                'winter_house',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.NO_REFLECTIONS,
                TimeOfDay.DAY,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-winter_house-002.tar.gz'
            ),
            ndds.SequenceEntry(
                'winter_house',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.NO_SMALL_OBJECTS,
                TimeOfDay.DAY,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-winter_house-003.tar.gz'
            ),
            ndds.SequenceEntry(
                'winter_house',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.MIN_QUALITY,
                TimeOfDay.DAY,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-winter_house-004.tar.gz'
            ),
            ndds.SequenceEntry(
                'winter_house',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.MAX_QUALITY,
                TimeOfDay.NIGHT,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-winter_house-005.tar.gz'
            ),
            ndds.SequenceEntry(
                'winter_house',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.MIN_QUALITY,
                TimeOfDay.NIGHT,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-winter_house-006.tar.gz'
            ),
            ndds.SequenceEntry(
                'two_story_apartment',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.MAX_QUALITY,
                TimeOfDay.DAY,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-two_story_apartment-000.tar.gz'
            ),
            ndds.SequenceEntry(
                'two_story_apartment',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                ndds.QualityLevel.NO_REFLECTIONS,
                TimeOfDay.DAY,
                root / 'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop-two_story_apartment-001.tar.gz'
            ),
            ndds.SequenceEntry(
                'winter_house',
                'MH_04_difficult',
                ndds.QualityLevel.MIN_QUALITY,
                TimeOfDay.DAY,
                root / 'MH_04_difficult-winter_house-000.tar.gz'
            ),
            ndds.SequenceEntry(
                'two_story_apartment',
                'MH_04_difficult',
                ndds.QualityLevel.MAX_QUALITY,
                TimeOfDay.DAY,
                root / 'MH_04_difficult-two_story_apartment-000.tar.gz'
            )
        ]

    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.load_sequences', autospec=True)
    def test_get_datasets_returns_all_datasets_by_default(self, mock_load_sequences, mock_task_manager):
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task
        mock_load_sequences.return_value = self.sequence_entries
        manager = ndds.NDDSManager(self.root_path)
        _, pending = manager.get_datasets()

        self.assertEqual(len(self.sequence_entries), pending)
        self.assertEqual(len(self.sequence_entries), mock_task_manager.get_import_dataset_task.call_count)
        for sequence_entry in self.sequence_entries:
            self.assertIn(mock.call(
                module_name=ndds_loader.__name__,
                path=str(sequence_entry.path),
                additional_args={},
                num_cpus=mock.ANY,
                num_gpus=mock.ANY,
                memory_requirements=mock.ANY,
                expected_duration=mock.ANY
            ), mock_task_manager.get_import_dataset_task.call_args_list)

    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.load_sequences', autospec=True)
    def test_get_datasets_returns_only_dataset_from_requested_environment(self, mock_load_sequences, mock_task_manager):
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task
        environment = self.sequence_entries[0].environment
        num_sequences = sum(1 for sequence_entry in self.sequence_entries if sequence_entry.environment == environment)

        mock_load_sequences.return_value = self.sequence_entries
        manager = ndds.NDDSManager(self.root_path)
        _, pending = manager.get_datasets(environment=environment)

        self.assertEqual(num_sequences, pending)
        self.assertEqual(num_sequences, mock_task_manager.get_import_dataset_task.call_count)
        for sequence_entry in self.sequence_entries:
            if sequence_entry.environment == environment:
                self.assertIn(mock.call(
                    module_name=ndds_loader.__name__,
                    path=str(sequence_entry.path),
                    additional_args={},
                    num_cpus=mock.ANY,
                    num_gpus=mock.ANY,
                    memory_requirements=mock.ANY,
                    expected_duration=mock.ANY
                ), mock_task_manager.get_import_dataset_task.call_args_list)

    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.load_sequences', autospec=True)
    def test_get_datasets_returns_only_datasets_with_requested_trajectory(self, mock_load_sequences, mock_task_manager):
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task
        trajectory_id = self.sequence_entries[0].trajectory_id
        num_sequences = sum(1 for sequence_entry in self.sequence_entries
                            if sequence_entry.trajectory_id == trajectory_id)

        mock_load_sequences.return_value = self.sequence_entries
        manager = ndds.NDDSManager(self.root_path)
        _, pending = manager.get_datasets(trajectory_id=trajectory_id)

        self.assertEqual(num_sequences, pending)
        self.assertEqual(num_sequences, mock_task_manager.get_import_dataset_task.call_count)
        for sequence_entry in self.sequence_entries:
            if sequence_entry.trajectory_id == trajectory_id:
                self.assertIn(mock.call(
                    module_name=ndds_loader.__name__,
                    path=str(sequence_entry.path),
                    additional_args={},
                    num_cpus=mock.ANY,
                    num_gpus=mock.ANY,
                    memory_requirements=mock.ANY,
                    expected_duration=mock.ANY
                ), mock_task_manager.get_import_dataset_task.call_args_list)

    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.load_sequences', autospec=True)
    def test_get_datasets_returns_only_datasets_with_requested_quality(self, mock_load_sequences, mock_task_manager):
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task
        quality = ndds.QualityLevel.MIN_QUALITY
        num_sequences = sum(1 for sequence_entry in self.sequence_entries
                            if sequence_entry.quality_level == quality)

        mock_load_sequences.return_value = self.sequence_entries
        manager = ndds.NDDSManager(self.root_path)
        _, pending = manager.get_datasets(quality_level=quality)

        self.assertEqual(num_sequences, pending)
        self.assertEqual(num_sequences, mock_task_manager.get_import_dataset_task.call_count)
        for sequence_entry in self.sequence_entries:
            if sequence_entry.quality_level == quality:
                self.assertIn(mock.call(
                    module_name=ndds_loader.__name__,
                    path=str(sequence_entry.path),
                    additional_args={},
                    num_cpus=mock.ANY,
                    num_gpus=mock.ANY,
                    memory_requirements=mock.ANY,
                    expected_duration=mock.ANY
                ), mock_task_manager.get_import_dataset_task.call_args_list)

    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.load_sequences', autospec=True)
    def test_get_datasets_returns_only_dataset_with_requested_time_of_day(self, mock_load_sequences, mock_task_manager):
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task
        time_of_day = TimeOfDay.NIGHT
        num_sequences = sum(1 for sequence_entry in self.sequence_entries
                            if sequence_entry.time_of_day == time_of_day)

        mock_load_sequences.return_value = self.sequence_entries
        manager = ndds.NDDSManager(self.root_path)
        _, pending = manager.get_datasets(time_of_day=time_of_day)

        self.assertEqual(num_sequences, pending)
        self.assertEqual(num_sequences, mock_task_manager.get_import_dataset_task.call_count)
        for sequence_entry in self.sequence_entries:
            if sequence_entry.time_of_day == time_of_day:
                self.assertIn(mock.call(
                    module_name=ndds_loader.__name__,
                    path=str(sequence_entry.path),
                    additional_args={},
                    num_cpus=mock.ANY,
                    num_gpus=mock.ANY,
                    memory_requirements=mock.ANY,
                    expected_duration=mock.ANY
                ), mock_task_manager.get_import_dataset_task.call_args_list)

    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.load_sequences', autospec=True)
    def test_get_datasets_allows_multiple_conditions(self, mock_load_sequences, mock_task_manager):
        mock_task = mock.Mock()
        mock_task.is_finished = False
        mock_task_manager.get_import_dataset_task.return_value = mock_task
        trajectory_id = self.sequence_entries[0].trajectory_id
        time_of_day = TimeOfDay.DAY
        num_sequences = sum(1 for sequence_entry in self.sequence_entries
                            if sequence_entry.time_of_day == time_of_day and
                            sequence_entry.trajectory_id == trajectory_id)

        mock_load_sequences.return_value = self.sequence_entries
        manager = ndds.NDDSManager(self.root_path)
        _, pending = manager.get_datasets(time_of_day=time_of_day, trajectory_id=trajectory_id)

        self.assertEqual(num_sequences, pending)
        self.assertEqual(num_sequences, mock_task_manager.get_import_dataset_task.call_count)
        for sequence_entry in self.sequence_entries:
            if sequence_entry.time_of_day == time_of_day and sequence_entry.trajectory_id == trajectory_id:
                self.assertIn(mock.call(
                    module_name=ndds_loader.__name__,
                    path=str(sequence_entry.path),
                    additional_args={},
                    num_cpus=mock.ANY,
                    num_gpus=mock.ANY,
                    memory_requirements=mock.ANY,
                    expected_duration=mock.ANY
                ), mock_task_manager.get_import_dataset_task.call_args_list)

    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.task_manager', autospec=True)
    @mock.patch('arvet_slam.dataset.ndds.ndds_manager.load_sequences', autospec=True)
    def test_get_datasets_returns_result_ids_from_complete_tasks(self, mock_load_sequences, mock_task_manager):
        result_id = ObjectId()
        finished_task = mock.Mock()
        finished_task.is_finished = True
        finished_task.get_result.return_value = result_id
        unfinished_task = mock.Mock()
        unfinished_task.is_finished = False
        trajectory_id = self.sequence_entries[0].trajectory_id
        num_sequences = sum(1 for sequence_entry in self.sequence_entries
                            if sequence_entry.trajectory_id == trajectory_id)
        num_pending = sum(1 for sequence_entry in self.sequence_entries
                          if sequence_entry.trajectory_id == trajectory_id and
                          not str(sequence_entry.path).endswith('-000.tar.gz'))

        mock_load_sequences.return_value = self.sequence_entries
        mock_task_manager.get_import_dataset_task.side_effect = \
            lambda path, *args, **kwargs: finished_task if str(path).endswith('-000.tar.gz') else unfinished_task
        manager = ndds.NDDSManager(self.root_path)
        results, pending = manager.get_datasets(trajectory_id=trajectory_id)

        self.assertEqual(num_sequences, mock_task_manager.get_import_dataset_task.call_count)
        self.assertEqual(num_pending, pending)
        self.assertEqual(num_sequences - num_pending, len(results))
        for result in results:
            self.assertEqual(result_id, result)
        for sequence_entry in self.sequence_entries:
            if sequence_entry.trajectory_id == trajectory_id:
                self.assertIn(mock.call(
                    module_name=ndds_loader.__name__,
                    path=str(sequence_entry.path),
                    additional_args={},
                    num_cpus=mock.ANY,
                    num_gpus=mock.ANY,
                    memory_requirements=mock.ANY,
                    expected_duration=mock.ANY
                ), mock_task_manager.get_import_dataset_task.call_args_list)


class TestLoadSequences(unittest.TestCase):
    temp_folder = Path(__file__).parent / 'test_load_sequences'
    sequences_file = temp_folder / 'sequences.json'
    sequences_json = {

    }

    @classmethod
    def setUpClass(cls) -> None:
        # Make a temporary folder
        if not cls.temp_folder.exists():
            cls.temp_folder.mkdir(exist_ok=True, parents=True)
        # disable logging
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls) -> None:
        logging.disable(logging.NOTSET)
        shutil.rmtree(cls.temp_folder)

    @staticmethod
    def make_sequence_json(
            trajectory_id: str = 'rgbd_dataset_freiburg1_360',
            map_name: str = 'scandinavian_house',
            time_of_day: TimeOfDay = TimeOfDay.DAY,
            texture_bias: int = 0,
            disable_reflections: bool = False,
            min_object_volume: float = -1,
            index: int = 0
    ):
        name = f"{trajectory_id}-{map_name}-{index:03}"
        return name, {
                "map": map_name,
                "trajectory_id": trajectory_id,
                "light_level": "NIGHT" if time_of_day is TimeOfDay.NIGHT else "DAY",
                "light_model": "LIT",
                "origin": {
                    "location": [
                        -180,
                        -110,
                        160
                    ],
                    "rotation": [
                        0,
                        0,
                        -50
                    ]
                },
                "left_intrinsics": {
                    "width": 640,
                    "height": 480,
                    "fx": 517.3,
                    "fy": 516.5,
                    "cx": 318.6,
                    "cy": 255.3,
                    "skew": 0
                },
                "right_intrinsics": {
                    "width": 640,
                    "height": 480,
                    "fx": 517.3,
                    "fy": 516.5,
                    "cx": 318.6,
                    "cy": 255.3,
                    "skew": 0
                },
                "texture_bias": texture_bias,
                "disable_reflections": disable_reflections,
                "motion_blur": 0,
                "exposure": None,
                "aperture": 4,
                "min_object_volume": min_object_volume,
                "focal_distance": 233,
                "grain": 0,
                "vignette": 0,
                "lens_flare": 0.5,
                "depth_quality": "KINECT_NOISE"
            }

    def test_raises_exception_if_no_sequences_file(self):
        if self.sequences_file.exists():
            self.sequences_file.unlink()
        with self.assertRaises(FileNotFoundError):
            ndds.load_sequences(self.temp_folder)

    def test_finds_max_quality_sequence_in_a_folder(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name, sequence_data = self.make_sequence_json(trajectory_id=trajectory_id, map_name=environment)
        sequence_path = self.temp_folder / sequence_name
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name: sequence_data}, fp)
        sequence_path.mkdir(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.MAX_QUALITY, results[0].quality_level)
        self.assertEqual(sequence_path, results[0].path)

        # clean up
        self.sequences_file.unlink()
        shutil.rmtree(sequence_path)

    def test_finds_max_quality_sequence_in_a_tar_file(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name, sequence_data = self.make_sequence_json(trajectory_id=trajectory_id, map_name=environment)
        sequence_zip = self.temp_folder / (sequence_name + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name: sequence_data}, fp)
        sequence_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.MAX_QUALITY, results[0].quality_level)
        self.assertEqual(sequence_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_zip.unlink()

    def test_skips_sequence_without_file(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name_1, sequence_data_1 = self.make_sequence_json(
            trajectory_id=trajectory_id, map_name=environment, index=0)
        sequence_name_2, sequence_data_2 = self.make_sequence_json(
            trajectory_id=trajectory_id, map_name=environment, index=1)
        sequence_1_zip = self.temp_folder / (sequence_name_1 + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name_1: sequence_data_1, sequence_name_2: sequence_data_2}, fp)
        sequence_1_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.MAX_QUALITY, results[0].quality_level)
        self.assertEqual(sequence_1_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_1_zip.unlink()

    def test_skips_sequence_with_unknown_environment(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name_1, sequence_data_1 = self.make_sequence_json(trajectory_id=trajectory_id, map_name=environment)
        sequence_name_2, sequence_data_2 = self.make_sequence_json(trajectory_id=trajectory_id, map_name='avalon')
        sequence_1_zip = self.temp_folder / (sequence_name_1 + '.tar.gz')
        sequence_2_zip = self.temp_folder / (sequence_name_2 + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name_1: sequence_data_1, sequence_name_2: sequence_data_2}, fp)
        sequence_1_zip.touch(exist_ok=True)
        sequence_2_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.MAX_QUALITY, results[0].quality_level)
        self.assertEqual(sequence_1_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_1_zip.unlink()
        sequence_2_zip.unlink()

    def test_skips_sequence_with_unknown_trajectory(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name_1, sequence_data_1 = self.make_sequence_json(trajectory_id=trajectory_id, map_name=environment)
        sequence_name_2, sequence_data_2 = self.make_sequence_json(trajectory_id='MH_04_404', map_name=environment)
        sequence_1_zip = self.temp_folder / (sequence_name_1 + '.tar.gz')
        sequence_2_zip = self.temp_folder / (sequence_name_2 + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name_1: sequence_data_1, sequence_name_2: sequence_data_2}, fp)
        sequence_1_zip.touch(exist_ok=True)
        sequence_2_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.MAX_QUALITY, results[0].quality_level)
        self.assertEqual(sequence_1_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_1_zip.unlink()
        sequence_2_zip.unlink()

    def test_skips_sequence_with_unknown_quality(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name_1, sequence_data_1 = self.make_sequence_json(trajectory_id=trajectory_id, map_name=environment)
        sequence_name_2, sequence_data_2 = self.make_sequence_json(
            trajectory_id=trajectory_id, map_name=environment, index=1,
            disable_reflections=True, min_object_volume=0.22
        )
        sequence_1_zip = self.temp_folder / (sequence_name_1 + '.tar.gz')
        sequence_2_zip = self.temp_folder / (sequence_name_2 + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name_1: sequence_data_1, sequence_name_2: sequence_data_2}, fp)
        sequence_1_zip.touch(exist_ok=True)
        sequence_2_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.MAX_QUALITY, results[0].quality_level)
        self.assertEqual(sequence_1_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_1_zip.unlink()
        sequence_2_zip.unlink()

    def test_finds_sequence_without_quality_settings_as_max(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name, sequence_data = self.make_sequence_json(trajectory_id=trajectory_id, map_name=environment)
        # Remove keys for the quality settings from the sequence data. Maximum quality should be assumed.
        del sequence_data['texture_bias']
        del sequence_data['disable_reflections']
        del sequence_data['min_object_volume']
        sequence_zip = self.temp_folder / (sequence_name + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name: sequence_data}, fp)
        sequence_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.MAX_QUALITY, results[0].quality_level)
        self.assertEqual(sequence_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_zip.unlink()

    def test_finds_min_quality_sequence_in_a_tar_file(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name, sequence_data = self.make_sequence_json(
            trajectory_id=trajectory_id, map_name=environment,
            # All 3 settings should be reduced
            texture_bias=15, disable_reflections=True, min_object_volume=0.015
        )
        sequence_zip = self.temp_folder / (sequence_name + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name: sequence_data}, fp)
        sequence_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.MIN_QUALITY, results[0].quality_level)
        self.assertEqual(sequence_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_zip.unlink()

    def test_finds_no_texture_sequence_in_a_tar_file(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name, sequence_data = self.make_sequence_json(
            trajectory_id=trajectory_id,
            map_name=environment,
            texture_bias=15
        )
        sequence_zip = self.temp_folder / (sequence_name + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name: sequence_data}, fp)
        sequence_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.NO_TEXTURE, results[0].quality_level)
        self.assertEqual(sequence_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_zip.unlink()

    def test_finds_no_reflections_sequence_in_a_tar_file(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name, sequence_data = self.make_sequence_json(
            trajectory_id=trajectory_id,
            map_name=environment,
            disable_reflections=True
        )
        sequence_zip = self.temp_folder / (sequence_name + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name: sequence_data}, fp)
        sequence_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.NO_REFLECTIONS, results[0].quality_level)
        self.assertEqual(sequence_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_zip.unlink()

    def test_finds_no_small_objects_sequence_in_a_tar_file(self):
        environment = 'scandinavian_house'
        trajectory_id = 'rgbd_dataset_freiburg1_360'
        sequence_name, sequence_data = self.make_sequence_json(
            trajectory_id=trajectory_id,
            map_name=environment,
            min_object_volume=0.015
        )
        sequence_zip = self.temp_folder / (sequence_name + '.tar.gz')
        with self.sequences_file.open('w') as fp:
            json.dump({sequence_name: sequence_data}, fp)
        sequence_zip.touch(exist_ok=True)

        results = ndds.load_sequences(self.temp_folder)
        self.assertEqual(1, len(results))
        self.assertEqual(trajectory_id, results[0].trajectory_id)
        self.assertEqual(environment, results[0].environment)
        self.assertEqual(TimeOfDay.DAY, results[0].time_of_day)
        self.assertEqual(ndds.QualityLevel.NO_SMALL_OBJECTS, results[0].quality_level)
        self.assertEqual(sequence_zip, results[0].path)

        # clean up
        self.sequences_file.unlink()
        sequence_zip.unlink()


class TestFindFile(unittest.TestCase):

    def test_returns_file(self):
        file_path = Path(__file__)
        result = ndds.find_file(file_path.name, file_path.parent)
        self.assertEqual(file_path, result)

    def test_searches_subfolders(self):
        file_path = Path(__file__)
        result = ndds.find_file(file_path.name, file_path.parents[2])
        self.assertEqual(file_path, result)

    def test_returns_directory(self):
        directory_path = Path(__file__).parent
        result = ndds.find_file(directory_path.name, directory_path.parent)
        self.assertEqual(directory_path, result)

    def test_raises_exception_if_file_is_not_found(self):
        search_root = Path(__file__).parent
        with self.assertRaises(FileNotFoundError):
            ndds.find_file('not_a_real_file', search_root)

    def test_ignores_folders_in_excluded_folders(self):
        file_path = Path(__file__)
        parent_path = file_path.parent
        search_root = parent_path.parent

        # File should be found without excluding the folder it is in
        result = ndds.find_file(file_path.name, search_root)
        self.assertEqual(file_path, result)

        # Excluding the containing folder should mean the file is not found
        with self.assertRaises(FileNotFoundError):
            ndds.find_file(file_path.name, search_root, excluded_folders=[parent_path.name])
