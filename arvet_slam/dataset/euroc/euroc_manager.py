# Copyright (c) 2017, John Skinner
import os
import logging
import typing
from pathlib import PurePath
import arvet.batch_analysis.task_manager as task_manager
import arvet_slam.dataset.euroc.euroc_loader as euroc_loader
import arvet_slam.dataset.euroc.euroc_validator as euroc_validator


dataset_names = [
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
    'V2_03_difficult'
]


class EuRoCManager:

    def __init__(self, root: typing.Union[str, bytes, os.PathLike, PurePath]):
        self._full_paths = self.find_roots(root)

    def __getattr__(self, item):
        if item in dataset_names:
            return self.get_dataset(item)
        raise AttributeError("No dataset {0}".format(item))

    def __getitem__(self, item):
        if item in dataset_names:
            return self.get_dataset(item)
        raise KeyError("No dataset {0}".format(item))

    def get_missing_datasets(self):
        """
        Get a list of all the datasets that we do not have known roots for.
        Use for debugging
        :return:
        """
        return [dataset_name for dataset_name in dataset_names if dataset_name not in self._full_paths]

    def get_dataset(self, name):
        if name in self._full_paths:
            import_dataset_task = task_manager.get_import_dataset_task(
                module_name=euroc_loader.__name__,
                path=str(self._full_paths[name]),
                additional_args={'dataset_name': name},
                num_cpus=1,
                num_gpus=0,
                memory_requirements='3GB',
                expected_duration='8:00:00',
            )
            if import_dataset_task.is_finished:
                return import_dataset_task.get_result()
            else:
                # Make sure the import dataset task gets done
                import_dataset_task.save()
                return None
        raise NotADirectoryError("No root folder for {0}, did you download it?".format(name))

    def verify_dataset(self, name: str, repair: bool = False):
        if name in self._full_paths:
            import_dataset_task = task_manager.get_import_dataset_task(
                module_name=euroc_loader.__name__,
                path=str(self._full_paths[name]),
                additional_args={'dataset_name': name}
            )
            if import_dataset_task.is_finished:
                image_collection = import_dataset_task.get_result()
                return euroc_validator.verify_dataset(image_collection, self._full_paths[name], name, repair)
            else:
                logging.getLogger(__name__).warning(f"Cannot validate {name}, it is not loaded yet")
                return True
        raise NotADirectoryError("No root folder for {0}, did you download it?".format(name))

    @classmethod
    def find_roots(cls, root: typing.Union[str, bytes, os.PathLike, PurePath]):
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
                                actual_root = euroc_loader.find_files(dir_entry.path)
                            except FileNotFoundError:
                                continue
                            # Only want the root path, ignore the other return values
                            actual_roots[dir_name] = actual_root[0]
                        else:
                            to_search.add(dir_entry.path)
        return actual_roots
