# Copyright (c) 2017, John Skinner
import os
import logging
import typing
import enum
from pathlib import PurePath, Path
import json
from bson import ObjectId
import arvet.metadata.image_metadata as imeta
import arvet.batch_analysis.task_manager as task_manager
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader


ENVIRONMENTS = [
    'two_story_apartment',
    'winter_house',
    'scandinavian_house',
    'two_room_apartment',
    'large_office'
]


TRAJECTORIES = [
    # EuRoC Trajectories
    'MH_01_easy',
    'MH_02_easy',
    'MH_03_medium',
    'MH_04_difficult',
    'MH_05_difficult',
    'V1_01_easy',
    'V1_02_medium',
    'V1_03_difficult',
    'V2_01_easy',
    'V2_02_medium',
    'V2_03_difficult',

    # TUM trajectories
    'rgbd_dataset_freiburg1_360',
    'rgbd_dataset_freiburg1_desk2',
    'rgbd_dataset_freiburg1_desk',
    'rgbd_dataset_freiburg1_floor',
    'rgbd_dataset_freiburg1_plant',
    'rgbd_dataset_freiburg1_room',
    'rgbd_dataset_freiburg1_rpy',
    'rgbd_dataset_freiburg1_teddy',
    'rgbd_dataset_freiburg1_xyz',
    'rgbd_dataset_freiburg2_360_hemisphere',
    'rgbd_dataset_freiburg2_360_kidnap',
    'rgbd_dataset_freiburg2_coke',
    'rgbd_dataset_freiburg2_desk',
    'rgbd_dataset_freiburg2_desk_with_person',
    'rgbd_dataset_freiburg2_dishes',
    'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
    'rgbd_dataset_freiburg2_flowerbouquet',
    'rgbd_dataset_freiburg2_large_no_loop',
    'rgbd_dataset_freiburg2_large_with_loop',
    'rgbd_dataset_freiburg2_metallic_sphere2',
    'rgbd_dataset_freiburg2_metallic_sphere',
    'rgbd_dataset_freiburg2_pioneer_360',
    'rgbd_dataset_freiburg2_pioneer_slam2',
    'rgbd_dataset_freiburg2_pioneer_slam3',
    'rgbd_dataset_freiburg2_pioneer_slam',
    'rgbd_dataset_freiburg2_rpy',
    'rgbd_dataset_freiburg2_xyz',
    'rgbd_dataset_freiburg3_cabinet',
    'rgbd_dataset_freiburg3_large_cabinet',
    'rgbd_dataset_freiburg3_long_office_household',
    'rgbd_dataset_freiburg3_nostructure_notexture_far',
    'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
    'rgbd_dataset_freiburg3_nostructure_texture_far',
    'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
    'rgbd_dataset_freiburg3_sitting_halfsphere',
    'rgbd_dataset_freiburg3_sitting_rpy',
    'rgbd_dataset_freiburg3_sitting_static',
    'rgbd_dataset_freiburg3_sitting_xyz',
    'rgbd_dataset_freiburg3_structure_notexture_far',
    'rgbd_dataset_freiburg3_structure_notexture_near',
    'rgbd_dataset_freiburg3_structure_texture_far',
    'rgbd_dataset_freiburg3_structure_texture_near',
    'rgbd_dataset_freiburg3_teddy',
    'rgbd_dataset_freiburg3_walking_halfsphere',
    'rgbd_dataset_freiburg3_walking_rpy',
    'rgbd_dataset_freiburg3_walking_static',
    'rgbd_dataset_freiburg3_walking_xyz'
]


class QualityLevel(enum.Enum):
    MAX_QUALITY = 4
    NO_REFLECTIONS = 3
    NO_TEXTURE = 2
    NO_SMALL_OBJECTS = 1
    MIN_QUALITY = 0


class SequenceEntry:
    def __init__(
            self,
            environment: str,
            trajectory_id: str,
            quality_level: QualityLevel,
            time_of_day: imeta.TimeOfDay,
            path: PurePath
    ):
        self.environment = str(environment)
        self.trajectory_id = str(trajectory_id)
        self.quality_level = quality_level
        self.time_of_day = time_of_day
        self.path = Path(path)


class NDDSManager:
    """
    Manager for synthetic datasets generated with NDDS.
    Dataset are grouped by quality, time of day, source environment, and reference trajectory
    """

    def __init__(self, root: typing.Union[str, bytes, os.PathLike, PurePath]):
        self._sequence_data = load_sequences(root)

    def get_datasets(
            self,
            environment: str = None,
            trajectory_id: str = None,
            quality_level: QualityLevel = None,
            time_of_day: imeta.TimeOfDay = None,
    ) -> typing.Iterable[ObjectId]:
        """
        Get all image sequences
        :param environment:
        :param trajectory_id:
        :param quality_level:
        :param time_of_day:
        :return:
        """
        sequence_paths = [
            sequence_entry.path
            for sequence_entry in self._sequence_data
            if (
                (environment is None or sequence_entry.environment == environment) and
                (trajectory_id is None or sequence_entry.trajectory_id == trajectory_id) and
                (quality_level is None or sequence_entry.quality_level == quality_level) and
                (time_of_day is None or sequence_entry.time_of_day == time_of_day)
            )
        ]

        sequences = []
        num_pending = 0
        for sequence_path in sequence_paths:
            import_dataset_task = task_manager.get_import_dataset_task(
                module_name=ndds_loader.__name__,
                path=str(sequence_path),
                additional_args={},
                num_cpus=1,
                num_gpus=0,
                memory_requirements='3GB',
                expected_duration='8:00:00',
            )
            if import_dataset_task.is_finished:
                sequences.append(import_dataset_task.get_result())
            else:
                # Make sure the import dataset task gets done
                import_dataset_task.save()
                num_pending += 1
        return sequences, num_pending


def load_sequences(root: typing.Union[str, bytes, os.PathLike, PurePath]) -> typing.List[SequenceEntry]:
    """
    Recursively search for the directories to import from the root folder.
    We're looking for folders with the same names as the
    :param root:
    :return:
    """
    # Find and read data about all the sequencse from sequences.json
    root = Path(root)
    sequences_file = find_file('sequences.json', root)
    with sequences_file.open('r') as fp:
        sequences_data = json.load(fp)

    # For each sequence in the data, loop for the corresponding file/folder and
    excluded_folders = set(sequences_data.keys())
    sequences = []
    for sequence_name, sequence_info in sequences_data.items():
        # First, try and find a base path for that sequence
        try:
            sequence_path = find_file(sequence_name, root, excluded_folders)
        except FileNotFoundError:
            sequence_path = None
        if sequence_path is None:
            try:
                sequence_path = find_file(sequence_name + '.tar.gz', root, excluded_folders)
            except FileNotFoundError:
                sequence_path = None

        # Could not find a file for that sequence, exclude
        if sequence_path is None:
            logging.getLogger(__name__).info(f"Could not find data for sequence {sequence_name}, skipping")
            continue

        # Read the environment and trajectory id, and ensure they are known values
        environment = str(sequence_info['map'])
        trajectory_id = str(sequence_info['trajectory_id'])
        if environment not in ENVIRONMENTS:
            logging.getLogger(__name__).warning(f"Unrecognised environment '{environment}' for {sequence_name}")
            continue
        if trajectory_id not in TRAJECTORIES:
            logging.getLogger(__name__).warning(f"Unrecognised trajectory id '{trajectory_id}' for {sequence_name}")
            continue

        # Parse the light level into the time of day (either day or night)
        if 'light_level' in sequence_info and sequence_info['light_level'].lower() == 'night':
            time_of_day = imeta.TimeOfDay.NIGHT
        else:
            time_of_day = imeta.TimeOfDay.DAY

        # Determine the quality level
        texture_bias = sequence_info.get('texture_bias', 0)
        disable_reflections = sequence_info.get('disable_reflections', False)
        min_object_volume = sequence_info.get('min_object_volume', -1.0)
        if min_object_volume > 0 and disable_reflections and texture_bias > 0:
            quality_level = QualityLevel.MIN_QUALITY
        elif min_object_volume >= 0 and not disable_reflections and texture_bias <= 0:
            quality_level = QualityLevel.NO_SMALL_OBJECTS
        elif min_object_volume < 0 and disable_reflections and texture_bias <= 0:
            quality_level = QualityLevel.NO_REFLECTIONS
        elif min_object_volume < 0 and not disable_reflections and texture_bias > 0:
            quality_level = QualityLevel.NO_TEXTURE
        elif min_object_volume < 0 and not disable_reflections and texture_bias <= 0:
            quality_level = QualityLevel.MAX_QUALITY
        else:
            logging.getLogger(__name__).warning(f"Unrecognised quality combination on {sequence_name}, "
                                                f"(texture_bias: {texture_bias}, "
                                                f"reflections {'dis' if disable_reflections else 'en'}abled, "
                                                f"min object volume {min_object_volume})")
            continue

        # Construct the sequence entry with the information we're using to index it
        sequences.append(SequenceEntry(
            environment,
            trajectory_id,
            quality_level,
            time_of_day,
            sequence_path
        ))
    return sequences


def find_file(file_name: str, base_root: PurePath, excluded_folders: typing.Iterable[str] = None) -> Path:
    """
    Look for a particular file within
    :param file_name: The name of the file to look for
    :param base_root: The root folder to start searching in
    :param excluded_folders: Any folder names to avoid searching in. Used to avoid exploring known file trees.
    :return: The path of the desired file, if it exists
    """
    if excluded_folders is not None:
        excluded_folders = set(excluded_folders)
    else:
        excluded_folders = set()
    to_search = {base_root}
    while len(to_search) > 0:
        candidate_root = to_search.pop()

        if (candidate_root / file_name).exists():
            return candidate_root / file_name

        # File does not exist in this directory, search subfolders
        for child_path in candidate_root.iterdir():
            if child_path.is_dir() and child_path.name not in excluded_folders:
                to_search.add(child_path)

    # Could not find the necessary files to import, raise an exception.
    raise FileNotFoundError(f"Could not find '{file_name}' a valid root directory within '{base_root}'")
