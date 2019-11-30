# Copyright (c) 2017, John Skinner
import os
import typing
import arvet.batch_analysis.task_manager as task_manager
import arvet_slam.dataset.tum.tum_loader as tum_loader


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

    def __init__(self, root: typing.Union[str, bytes, os.PathLike]):
        self._full_paths = self.find_roots(root)

    def __getattr__(self, item):
        if item in dataset_names:
            return self.get_dataset(item)
        raise AttributeError("No dataset {0}".format(item))

    def __getitem__(self, item):
        if item in dataset_names:
            return self.get_dataset(item)
        raise KeyError("No dataset {0}".format(item))

    def get_dataset(self, name):
        if name in self._full_paths:
            import_dataset_task = task_manager.get_import_dataset_task(
                module_name=tum_loader.__name__,
                path=self._full_paths[name],
                additional_args={'dataset_name': name},
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
        raise NotADirectoryError("No root folder for {0}, did you download it?".format(name))

    @classmethod
    def find_roots(cls, root: typing.Union[str, bytes, os.PathLike]):
        """
        Recursively search for the directories to import from the root folder.
        We're looking for folders with the same names as the
        :param root: The root folder to search. Search is recursive.
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
                                actual_root = tum_loader.find_files(dir_entry.path)
                            except FileNotFoundError:
                                continue
                            # Only want the root path, ignore the other return values
                            actual_roots[dir_name] = actual_root[0]
                        else:
                            to_search.add(dir_entry.path)
        return actual_roots
