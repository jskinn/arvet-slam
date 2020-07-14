# Copyright (c) 2017, John Skinner
import typing
from os import PathLike
import logging
from pathlib import Path, PurePath
import tarfile
import arvet.batch_analysis.task_manager as task_manager
import arvet_slam.dataset.tum.tum_loader as tum_loader
import arvet_slam.dataset.tum.tum_validator as tum_validator


dataset_names = [
    'rgbd_dataset_freiburg1_xyz',
    'rgbd_dataset_freiburg1_rpy',
    'rgbd_dataset_freiburg2_xyz',
    'rgbd_dataset_freiburg2_rpy',
    'rgbd_dataset_freiburg1_360',
    'rgbd_dataset_freiburg1_floor',
    'rgbd_dataset_freiburg1_desk',
    'rgbd_dataset_freiburg1_desk2',
    'rgbd_dataset_freiburg1_room',
    'rgbd_dataset_freiburg2_360_hemisphere',
    'rgbd_dataset_freiburg2_360_kidnap',
    'rgbd_dataset_freiburg2_desk',
    'rgbd_dataset_freiburg2_large_no_loop',
    'rgbd_dataset_freiburg2_large_with_loop',
    'rgbd_dataset_freiburg3_long_office_household',
    'rgbd_dataset_freiburg2_pioneer_360',
    'rgbd_dataset_freiburg2_pioneer_slam',
    'rgbd_dataset_freiburg2_pioneer_slam2',
    'rgbd_dataset_freiburg2_pioneer_slam3',
    'rgbd_dataset_freiburg3_nostructure_notexture_far',
    'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
    'rgbd_dataset_freiburg3_nostructure_texture_far',
    'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
    'rgbd_dataset_freiburg3_structure_notexture_far',
    'rgbd_dataset_freiburg3_structure_notexture_near',
    'rgbd_dataset_freiburg3_structure_texture_far',
    'rgbd_dataset_freiburg3_structure_texture_near',
    'rgbd_dataset_freiburg2_desk_with_person',
    'rgbd_dataset_freiburg3_sitting_static',
    'rgbd_dataset_freiburg3_sitting_xyz',
    'rgbd_dataset_freiburg3_sitting_halfsphere',
    'rgbd_dataset_freiburg3_sitting_rpy',
    'rgbd_dataset_freiburg3_walking_static',
    'rgbd_dataset_freiburg3_walking_xyz',
    'rgbd_dataset_freiburg3_walking_halfsphere',
    'rgbd_dataset_freiburg3_walking_rpy',
    'rgbd_dataset_freiburg1_plant',
    'rgbd_dataset_freiburg1_teddy',
    'rgbd_dataset_freiburg2_coke',
    'rgbd_dataset_freiburg2_dishes',
    'rgbd_dataset_freiburg2_flowerbouquet',
    'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
    'rgbd_dataset_freiburg2_metallic_sphere',
    'rgbd_dataset_freiburg2_metallic_sphere2',
    'rgbd_dataset_freiburg3_cabinet',
    'rgbd_dataset_freiburg3_large_cabinet',
    'rgbd_dataset_freiburg3_teddy'
]


class TUMManager:

    def __init__(self, root: typing.Union[str, bytes, PathLike, PurePath]):
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
                module_name=tum_loader.__name__,
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
                module_name=tum_loader.__name__,
                path=str(self._full_paths[name]),
                additional_args={'dataset_name': name}
            )
            if import_dataset_task.is_finished:
                image_collection = import_dataset_task.get_result()
                return tum_validator.verify_dataset(image_collection, self._full_paths[name], name, repair)
            else:
                # Try looking for an existing tarfile
                for candidate_path in [
                    str(self._full_paths[name]) + '.tar.gz',
                    str(self._full_paths[name]) + '.tgz',
                ]:
                    import_dataset_task = task_manager.get_import_dataset_task(
                        module_name=tum_loader.__name__,
                        path=candidate_path,
                        additional_args={'dataset_name': name}
                    )
                    if import_dataset_task.is_finished:
                        break
                if import_dataset_task.is_finished:
                    if repair:
                        logging.getLogger(__name__).warning(
                            f"Removed suffix from tarball import task for {name}, it should get returned next time")
                        import_dataset_task.path = self._full_paths[name]
                        import_dataset_task.save()
                    image_collection = import_dataset_task.get_result()
                    return tum_validator.verify_dataset(image_collection, self._full_paths[name], name, repair)
                else:
                    # Try looking for an existing task with the actual root from find_files as the path
                    try:
                        actual_root = tum_loader.find_files(self._full_paths[name])
                    except FileNotFoundError:
                        actual_root = None
                    if actual_root is not None and len(actual_root) > 0:
                        import_dataset_task = task_manager.get_import_dataset_task(
                            module_name=tum_loader.__name__,
                            path=actual_root[0],
                            additional_args={'dataset_name': name}
                        )
                    else:
                        import_dataset_task = None
                    if import_dataset_task is not None and import_dataset_task.is_finished:
                        if repair:
                            logging.getLogger(__name__).warning(
                                f"Shortened path for {name}, it should get returned next time")
                            import_dataset_task.path = self._full_paths[name]
                            import_dataset_task.save()
                        image_collection = import_dataset_task.get_result()
                        return tum_validator.verify_dataset(image_collection, self._full_paths[name], name, repair)
                    else:
                        logging.getLogger(__name__).warning(f"Cannot validate {name}, it is not loaded yet? "
                                                            f"(looking for module name \"{tum_loader.__name__}\", "
                                                            f"path \"{str(self._full_paths[name])}\", "
                                                            f"additional args \"{ {'dataset_name': name} }\")")
                        return True
        raise NotADirectoryError("No root folder for {0}, did you download it?".format(name))

    @classmethod
    def find_roots(cls, root: typing.Union[str, bytes, PathLike, PurePath]):
        """
        Recursively search for the directories to import from the root folder.
        We're looking for folders with the same names as the
        :param root: The root folder to search. Search is recursive.
        :return:
        """
        actual_roots = {}
        tarball_roots = {}
        to_search = {Path(root).resolve()}
        while len(to_search) > 0:
            candidate_root = to_search.pop()
            for child_path in candidate_root.iterdir():
                if child_path.is_dir():
                    if child_path.name in dataset_names:
                        # this is could be a dataset folder, look for roots
                        try:
                            tum_loader.find_files(child_path)
                        except FileNotFoundError:
                            continue
                        # Find files worked, store this path
                        actual_roots[child_path.name] = child_path
                    else:
                        # Recursively search this path for more files
                        to_search.add(child_path)
                elif child_path.is_file() and tarfile.is_tarfile(child_path):
                    # the file is a tarball, check if it matches a dataset name
                    file_name = child_path.name
                    period_index = file_name.find('.')
                    if period_index > 0:
                        file_name = file_name[:period_index]    # strip all extensions.
                    if file_name in dataset_names:
                        tarball_roots[file_name] = child_path.parent / file_name

        # for each dataset we found a tarball for, but not a root folder, store the tarball as the root
        for dataset_name in set(tarball_roots.keys()) - set(actual_roots.keys()):
            actual_roots[dataset_name] = tarball_roots[dataset_name]

        return actual_roots
