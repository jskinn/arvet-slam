# Copyright (c) 2017, John Skinner
import unittest
import shutil
import os.path
from pathlib import Path
import numpy as np
import transforms3d as tf3d
import arvet.util.transform as tf
from arvet.util.test_helpers import ExtendedTestCase
import arvet_slam.dataset.kitti.kitti_loader as kitti_loader


class TestMakeCameraPose(ExtendedTestCase):

    def test_make_camera_pose_returns_transform_object(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = kitti_loader.make_camera_pose(frame_delta)
        self.assertIsInstance(pose, tf.Transform)

    def test_make_camera_pose_rearranges_location_coordinates(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = kitti_loader.make_camera_pose(frame_delta)
        self.assertNPEqual((13.2, -10, 22.4), pose.location)

    def test_make_camera_pose_changes_rotation_each_axis(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        # Roll, rotation around z-axis for kitti
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 0, 1), np.pi / 6, True)
        pose = kitti_loader.make_camera_pose(frame_delta)
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around x-axis for kitti
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((1, 0, 0), np.pi / 6, True)
        pose = kitti_loader.make_camera_pose(frame_delta)
        self.assertNPClose((0, -np.pi / 6, 0), pose.euler)

        # Yaw, rotation around negative y-axis for kitti
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 1, 0), np.pi / 6, True)
        pose = kitti_loader.make_camera_pose(frame_delta)
        self.assertNPClose((0, 0, -np.pi / 6), pose.euler)

    def test_make_camera_pose_combined(self):
        frame_delta = np.identity(4)
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            frame_delta[0:3, 3] = -loc[1], -loc[2], loc[0]
            frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((-rot_axis[1], -rot_axis[2], rot_axis[0]),
                                                              rot_angle, False)
            pose = kitti_loader.make_camera_pose(frame_delta)
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))


class TestFindRoot(unittest.TestCase):
    temp_folder = 'temp_test_kitti_loader_find_files'
    # these are the files find_root looks for
    required_files = ['rgb.txt', 'groundtruth.txt', 'depth.txt']

    @staticmethod
    def make_required_files(root_path, sequence_name):
        (root_path / 'sequences' / sequence_name / 'image_2').mkdir(parents=True, exist_ok=True)
        (root_path / 'sequences' / sequence_name / 'image_3').mkdir(parents=True, exist_ok=True)
        (root_path / 'sequences' / sequence_name / 'calib.txt').touch()
        (root_path / 'sequences' / sequence_name / 'times.txt').touch()
        (root_path / 'poses').mkdir(parents=True, exist_ok=True)
        (root_path / 'poses' / (sequence_name + '.txt')).touch()

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(cls.temp_folder):
            shutil.rmtree(cls.temp_folder)

    def test_finds_root_with_required_directories(self):
        sequence_name = "000007"
        root_path = Path(self.temp_folder) / 'root'
        self.make_required_files(root_path, sequence_name)

        result = kitti_loader.find_root(str(root_path), sequence_name)
        self.assertEqual(str(root_path), result)

        # Clean up after ourselves
        shutil.rmtree(root_path)

    def test_raises_exception_if_a_required_file_is_not_found(self):
        sequence_name = "000005"
        root_path = Path(self.temp_folder) / 'root'
        self.make_required_files(root_path, sequence_name)
        shutil.rmtree(root_path / 'sequences' / sequence_name / 'image_2')
        with self.assertRaises(FileNotFoundError):
            kitti_loader.find_root(str(root_path), sequence_name)

        self.make_required_files(root_path, sequence_name)
        shutil.rmtree(root_path / 'sequences' / sequence_name / 'image_3')
        with self.assertRaises(FileNotFoundError):
            kitti_loader.find_root(str(root_path), sequence_name)

        self.make_required_files(root_path, sequence_name)
        (root_path / 'sequences' / sequence_name / 'calib.txt').unlink()
        with self.assertRaises(FileNotFoundError):
            kitti_loader.find_root(str(root_path), sequence_name)

        self.make_required_files(root_path, sequence_name)
        (root_path / 'sequences' / sequence_name / 'times.txt').unlink()
        with self.assertRaises(FileNotFoundError):
            kitti_loader.find_root(str(root_path), sequence_name)

        self.make_required_files(root_path, sequence_name)
        (root_path / 'poses' / (sequence_name + '.txt')).unlink()
        with self.assertRaises(FileNotFoundError):
            kitti_loader.find_root(str(root_path), sequence_name)

        # Clean up after ourselves
        shutil.rmtree(root_path)

    def test_searches_recursively(self):
        # Create a deeply nested folder structure
        sequence_name = "000003"
        base_root = Path(self.temp_folder)
        true_sequence = 3, 0, 2
        decoy_sequence = 2, 1, 1
        true_path = ''
        for lvl1 in range(5):
            lvl1_path = base_root / "folder_{0}".format(lvl1)
            for lvl2 in range(4):
                lvl2_path = lvl1_path / "folder_{0}".format(lvl2)
                for lvl3 in range(3):
                    path = lvl2_path / "folder_{0}".format(lvl3)
                    path.mkdir(parents=True, exist_ok=True)
                    if (lvl1, lvl2, lvl3) == true_sequence:
                        true_path = path
                        self.make_required_files(true_path, sequence_name)
                    elif (lvl1, lvl2, lvl3) == decoy_sequence:
                        self.make_required_files(path, '000002')
                    else:
                        (path / 'decoy.txt').touch()

        # Search that structure for the one folder that has all we need
        result = kitti_loader.find_root(str(base_root), sequence_name)
        self.assertEqual(str(true_path), result)

        # Clean up after ourselves
        shutil.rmtree(base_root)
