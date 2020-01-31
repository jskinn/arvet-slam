# Copyright (c) 2017, John Skinner
import os
import typing
import arvet.batch_analysis.task_manager as task_manager
import arvet_slam.dataset.kitti.kitti_loader as kitti_loader


dataset_names = [
    "{0:06}".format(idx)
    for idx in range(11)
]
sequence_ids = list(range(11))


class KITTIManager:

    def __init__(self, root: typing.Union[str, bytes, os.PathLike]):
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
        return [dataset_name for dataset_name in dataset_names if dataset_name not in self._full_paths]

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
    def find_roots(cls, root):
        """
        Recursively search for the directories to import from the root folder.
        We're looking for folders with the same names as the
        :param root:
        :return:
        """
        actual_roots = {}
        to_search = {root}
        while len(to_search) > 0:
            candidate_root = to_search.pop()
            with os.scandir(candidate_root) as dir_iter:
                for dir_entry in dir_iter:
                    if dir_entry.is_dir():
                        dir_name = dir_entry.name
                        if dir_name in dataset_names:
                            # this is a dataset folder, we're not going to search within it
                            try:
                                actual_root = kitti_loader.find_root(dir_entry.path, dir_name)
                            except FileNotFoundError:
                                continue
                            actual_roots[int(dir_name)] = actual_root
                        else:
                            to_search.add(dir_entry.path)
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
