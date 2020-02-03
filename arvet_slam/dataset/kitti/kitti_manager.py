# Copyright (c) 2017, John Skinner
from os import PathLike
import typing
from pathlib import Path
import arvet.batch_analysis.task_manager as task_manager
import arvet_slam.dataset.kitti.kitti_loader as kitti_loader


dataset_names = [
    "{0:06}".format(idx)
    for idx in range(11)
]
sequence_ids = list(range(11))


class KITTIManager:

    def __init__(self, root: typing.Union[str, bytes, PathLike, Path]):
        self._full_paths = self.find_roots(root)

    def __getattr__(self, item):
        sequence_id = to_sequence_id(item)
        if sequence_id in self._full_paths:
            return self.get_dataset(item)
        raise AttributeError("No dataset {0}".format(item))

    def __getitem__(self, item):
        sequence_id = to_sequence_id(item)
        if sequence_id in self._full_paths:
            return self.get_dataset(item)
        raise KeyError("No dataset {0}".format(item))

    def get_missing_datasets(self):
        """
        Get a list of all the datasets that we do not have known roots for.
        Use for debugging
        :return:
        """
        return [dataset_name for dataset_name in dataset_names if int(dataset_name) not in self._full_paths]

    def get_dataset(self, sequence_id: typing.Union[str, int, float]):
        if isinstance(sequence_id, int):
            sequence_id_int = sequence_id
        else:
            sequence_id_int = to_sequence_id(sequence_id)

        if sequence_id_int in self._full_paths:
            import_dataset_task = task_manager.get_import_dataset_task(
                module_name=kitti_loader.__name__,
                path=str(self._full_paths[sequence_id_int]),
                additional_args={'sequence_number': sequence_id_int},
                num_cpus=1,
                num_gpus=0,
                memory_requirements='3GB',
                expected_duration='8:00:00',
            )
            if import_dataset_task.is_finished:
                return import_dataset_task.result
            else:
                # Make sure the import dataset task gets done
                import_dataset_task.save()
                return None
        if 0 <= sequence_id_int < 11:
            raise NotADirectoryError("No root folder for sequence {0:06}, did you download it?".format(sequence_id_int))
        else:
            raise NotADirectoryError(
                "No root folder for sequence {0}, are you sure it's a sequence?".format(sequence_id))

    @classmethod
    def find_roots(cls, root: typing.Union[str, bytes, PathLike, Path]):
        """
        Recursively search for the directories to import from the root folder.
        We're looking for folders with the same names as the
        :param root:
        :return:
        """
        root = Path(root)
        actual_roots = {}
        for sequence_number in range(11):
            try:
                actual_roots[sequence_number] = kitti_loader.find_root(root, sequence_number)
            except FileNotFoundError:
                continue
        return actual_roots


def to_sequence_id(sequence_name) -> int:
    """
    Turn an arbitrary value to a known sequence id.
    Will return an integer in range 0-11 for a valid sequence, or -1 for all other values
    Should handle names as integers, floats, any numeric, or strings.
    :param sequence_name:
    :return:
    """
    try:
        sequence_id = int(sequence_name)
    except ValueError:
        sequence_id = -1
    if 0 <= sequence_id < 11:
        return sequence_id
    return -1
