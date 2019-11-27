# Copyright (c) 2017, John Skinner
import unittest
import unittest.mock as mock
import os.path
from pathlib import Path
import shutil
import numpy as np
import transforms3d as tf3d
import arvet.util.transform as tf
from arvet.util.test_helpers import ExtendedTestCase
import arvet_slam.dataset.tum.tum_loader as tum_loader


class TestMakeCameraPose(ExtendedTestCase):

    def test_location(self):
        forward = 51.2
        up = 153.3
        left = -126.07
        pose = tum_loader.make_camera_pose(-1 * left, -1 * up, forward, 0, 0, 0, 1)
        self.assertNPEqual((forward, left, up), pose.location)

    def test_orientation(self):
        angle = np.pi / 7
        forward = 51.2
        up = 153.3
        left = -126.07
        quat = tf3d.quaternions.axangle2quat((-1 * left, -1 * up, forward), angle)
        pose = tum_loader.make_camera_pose(0, 0, 0, quat[0], quat[1], quat[2], quat[3])
        self.assertNPEqual((0, 0, 0), pose.location)
        self.assertNPEqual(tf3d.quaternions.axangle2quat((forward, left, up), angle), pose.rotation_quat(True))

    def test_both(self):
        forward = 51.2
        up = 153.3
        left = -126.07
        angle = np.pi / 7
        o_forward = 1.151325
        o_left = 5.1315
        o_up = -0.2352323
        quat = tf3d.quaternions.axangle2quat((-1 * o_left, -1 * o_up, o_forward), angle)
        pose = tum_loader.make_camera_pose(-1 * left, -1 * up, forward, quat[0], quat[1], quat[2], quat[3])
        self.assertNPEqual((forward, left, up), pose.location)
        self.assertNPEqual(tf3d.quaternions.axangle2quat((o_forward, o_left, o_up), angle), pose.rotation_quat(True))

    def test_randomized(self):
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            quat = tf3d.quaternions.axangle2quat((-rot_axis[1], -rot_axis[2], rot_axis[0]), rot_angle, False)
            pose = tum_loader.make_camera_pose(-loc[1], -loc[2], loc[0], quat[0], quat[1], quat[2], quat[3])
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))


class TestReadImageFilenames(unittest.TestCase):

    def test_reads_filenames_to_mapping(self):
        line_template = "{time} rgb/{time}.png\n"
        data_text = ""
        mapping = {}
        for time in range(0, 10):
            timestamp = time * 49999.36 + 14036381289400.97024
            mapping[timestamp] = "rgb/{0}.png".format(timestamp)
            data_text += line_template.format(time=timestamp)

        mock_open = mock.mock_open(read_data=data_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            result = tum_loader.read_image_filenames('test_filepath')
        self.assertTrue(mock_open.called)
        self.assertEqual('test_filepath', mock_open.call_args[0][0])
        self.assertEqual('r', mock_open.call_args[0][1])
        self.assertEqual(result, mapping)

    def test_ignores_comments_and_empty_lines(self):
        line_template = "{time} {time}.png\n"
        data_text = ""
        mapping = {}
        data_text += """
        # color images
        # file: 'rgbd_dataset_freiburg3_structure_texture_far.bag'
        # timestamp filename
        
        """
        for time in range(5):
            # These lines should be ignored
            timestamp = time * 4999936 + 1
            data_text += "#" + line_template.format(time=timestamp)
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            mapping[timestamp] = "{0}.png".format(timestamp)
            data_text += line_template.format(time=timestamp)

        mock_open = mock.mock_open(read_data=data_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            result = tum_loader.read_image_filenames('test_filepath')
        self.assertEqual(result, mapping)

    def test_ignores_characters_after_comment(self):
        line_template = "{time} rgb/{time}.png  # This is time {time}\n"
        data_text = ""
        mapping = {}
        for time in range(0, 10):
            timestamp = time * 49999.36 + 14036381289400.97024
            mapping[timestamp] = "rgb/{0}.png".format(timestamp)
            data_text += line_template.format(time=timestamp)

        mock_open = mock.mock_open(read_data=data_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            result = tum_loader.read_image_filenames('test_filepath')
        self.assertTrue(mock_open.called)
        self.assertEqual('test_filepath', mock_open.call_args[0][0])
        self.assertEqual('r', mock_open.call_args[0][1])
        self.assertEqual(result, mapping)


class TestReadTrajectory(ExtendedTestCase):
    line_template = "{time} {x} {y} {z} {qx} {qy} {qz} {qw}\n"

    def format_line(self, timestamp, pose, line_template=None):
        """
        Produce a pose encoded as a line, to be read from the trajectory file.
        Provides consistent handling of axis reordering to the expected axis order.
        :param timestamp:
        :param pose:
        :param line_template: The string template for the line to encode
        :return:
        """
        quat = pose.rotation_quat(w_first=True)
        if line_template is None:
            line_template = self.line_template
        return line_template.format(
            time=timestamp,
            x=repr(-1 * pose.location[1]),
            y=repr(-1 * pose.location[2]),
            z=repr(pose.location[0]),
            qw=repr(quat[0]), qx=repr(-1 * quat[2]), qy=repr(-1 * quat[3]), qz=repr(quat[1])
        )

    def test_reads_from_given_file(self):
        trajectory_text = ""
        timestamps = []
        for time in range(0, 10):
            timestamp = time * 1.01 + 100
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            trajectory_text += self.format_line(timestamp, pose)
            timestamps.append(timestamp)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            trajectory = tum_loader.read_trajectory('test_filepath', timestamps)
        self.assertTrue(mock_open.called)
        self.assertEqual('test_filepath', mock_open.call_args[0][0])
        self.assertEqual('r', mock_open.call_args[0][1])
        self.assertEqual(len(trajectory), len(timestamps))

    def test_reads_trajectory_relative_to_first_pose(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        trajectory_text = ""
        for time in range(0, 10):
            timestamp = time * 4999.936 + 1403638128.940097024
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[timestamp] = pose
            absolute_pose = first_pose.find_independent(pose)
            trajectory_text += self.format_line(timestamp, absolute_pose)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            trajectory = tum_loader.read_trajectory('test_filepath', relative_trajectory.keys())
        self.assertEqual(len(trajectory), len(relative_trajectory))
        for time, pose in relative_trajectory.items():
            self.assertIn(time, trajectory)
            self.assertNPClose(pose.location, trajectory[time].location)
            self.assertNPClose(pose.rotation_quat(True), trajectory[time].rotation_quat(True))

    def test_skips_comments_and_blank_lines(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        trajectory_text = "# Starting with a comment\n    # Another comment\n\n"
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[timestamp] = pose
            absolute_pose = first_pose.find_independent(pose)
            trajectory_text += self.format_line(timestamp, absolute_pose)
            # Add incorrect trajectory data, preceeded by a hash to indicate it's a comment
            trajectory_text += "# " + self.format_line(timestamp, pose)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            trajectory = tum_loader.read_trajectory('test_filepath', relative_trajectory.keys())
        self.assertEqual(len(trajectory), len(relative_trajectory))
        for time, pose in relative_trajectory.items():
            self.assertIn(time, trajectory)
            self.assertNPClose(pose.location, trajectory[time].location)
            self.assertNPClose(pose.rotation_quat(True), trajectory[time].rotation_quat(True))

    def test_removes_comments_from_the_end_of_lines(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        trajectory_text = "# Starting with a comment\n    # Another comment\n\n"

        line_template = "{time} {x} {y} {z} {qx} {qy} {qz} {qw} # This is a comment\n"
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[timestamp] = pose
            absolute_pose = first_pose.find_independent(pose)
            trajectory_text += self.format_line(timestamp, absolute_pose, line_template)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            trajectory = tum_loader.read_trajectory('test_filepath', relative_trajectory.keys())
        self.assertEqual(len(trajectory), len(relative_trajectory))
        for time, pose in relative_trajectory.items():
            self.assertIn(time, trajectory)
            self.assertNPClose(pose.location, trajectory[time].location)
            self.assertNPClose(pose.rotation_quat(True), trajectory[time].rotation_quat(True))

    def test_interpolates_trajectory_to_desired_times(self):
        def make_pose(time):
            return tf.Transform(
                location=(time, -10 * time, 0),
                rotation=tf3d.quaternions.axangle2quat((1, 2, 3), (time / 10) * np.pi / 2),
                w_first=True
            )

        encoded_trajectory = {
            time: make_pose(time)
            for time in range(0, 11, 2)
        }
        trajectory_text = ""
        for time, pose in encoded_trajectory.items():
            trajectory_text += self.format_line(time, pose)

        desired_times = [time + 0.1 for time in range(1, 11, 2)]
        first_pose = make_pose(min(desired_times))

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            trajectory = tum_loader.read_trajectory('test_filepath', desired_times)
        self.assertEqual(set(desired_times), set(trajectory.keys()))
        for time in desired_times:
            expected_pose = first_pose.find_relative(make_pose(time))
            self.assertIn(time, trajectory)
            self.assertNPClose(expected_pose.location, trajectory[time].location, atol=1e-14, rtol=0)
            self.assertNPClose(expected_pose.rotation_quat(True), trajectory[time].rotation_quat(True),
                               atol=1e-14, rtol=0)

    def test_interpolates_multiple_times_within_the_same_interval(self):
        def make_pose(time):
            return tf.Transform(
                location=(time, -10 * time, 0),
                rotation=tf3d.quaternions.axangle2quat((1, 2, 3), (time / 10) * np.pi / 2),
                w_first=True
            )

        encoded_trajectory = {
            time: make_pose(time)
            for time in range(0, 11, 2)
        }
        trajectory_text = ""
        for time, pose in encoded_trajectory.items():
            trajectory_text += self.format_line(time, pose)

        desired_times = np.linspace(0, 10, num=50, endpoint=True)   # Lots of desired times for widely spaced samples
        first_pose = make_pose(min(desired_times))

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.tum.tum_loader.open', mock_open, create=True):
            trajectory = tum_loader.read_trajectory('test_filepath', desired_times)
        self.assertEqual(set(desired_times), set(trajectory.keys()))
        for time in desired_times:
            expected_pose = first_pose.find_relative(make_pose(time))
            self.assertIn(time, trajectory)
            self.assertNPEqual(expected_pose.location, trajectory[time].location)
            self.assertNPClose(expected_pose.rotation_quat(True), trajectory[time].rotation_quat(True),
                               atol=1e-14, rtol=0)


class TestAssociateData(unittest.TestCase):

    def test_single_map_returns_sorted(self):
        subject = {(v * 7 + 3) % 10: str(v) for v in range(10)}     # A disordered map
        result = tum_loader.associate_data(subject)
        self.assertEqual([
            [0, '1'], [1, '4'], [2, '7'], [3, '0'], [4, '3'],
            [5, '6'], [6, '9'], [7, '2'], [8, '5'], [9, '8']
        ], result)

    def test_same_keys_associates(self):
        root_map = {(t * 7 + 3) % 10: str(t) for t in range(10)}
        map_float_1 = {t: t * 1.2215 - 0.2234 * t * t + 0.115 for t in range(10)}
        map_float_2 = {t: 4.462 * t * t - 1000.212 for t in range(10)}
        map_filename = {t: "{0}.png".format(t * 4999936 + 1403638128940097024) for t in range(10)}
        result = tum_loader.associate_data(root_map, map_float_1, map_float_2, map_filename)

        self.assertEqual([
            [t, root_map[t], map_float_1[t], map_float_2[t], map_filename[t]] for t in range(10)
        ], result)

    def test_different_keys_associates_keys(self):
        expected_result = []
        root_map = {}
        map_float_1 = {}
        map_float_2 = {}
        map_filename = {}
        for t in range(10):
            root_time = t * 10 + np.random.uniform(-0.1, 0.1)
            float_1 = t * 1.2215 - 0.2234 * t * t + 0.115
            float_2 = 4.462 * t * t - 1000.212
            filename = "{0}.png".format(t * 4999936 + 1403638128940097024)
            expected_result.append([root_time, t, float_1, float_2, filename])
            root_map[root_time] = t
            map_float_1[t * 10 + np.random.uniform(-0.1, 0.1)] = float_1
            map_float_2[t * 10 + np.random.uniform(-0.1, 0.1)] = float_2
            map_filename[t * 10 + np.random.uniform(-0.1, 0.1)] = filename
        result = tum_loader.associate_data(root_map, map_float_1, map_float_2, map_filename)
        self.assertEqual(expected_result, result)


class TestFindFiles(unittest.TestCase):
    temp_folder = 'temp_test_tum_loader_find_files'
    # these are the files find_root looks for
    required_files = ['rgb.txt', 'groundtruth.txt', 'depth.txt']

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(cls.temp_folder):
            shutil.rmtree(cls.temp_folder)

    def test_finds_root_with_required_files(self):
        root_path = Path(self.temp_folder) / 'root'
        root_path.mkdir(parents=True, exist_ok=True)
        for filename in self.required_files:
            (root_path / filename).touch()

        result = tum_loader.find_files(str(root_path))
        self.assertEqual((
            str(root_path),
            str(root_path / 'rgb.txt'),
            str(root_path / 'depth.txt'),
            str(root_path / 'groundtruth.txt'),
        ), result)

        # Clean up after ourselves
        shutil.rmtree(root_path)

    def test_searches_recursively(self):
        # Create a deeply nested folder structure
        base_root = Path(self.temp_folder)
        true_sequence = 3, 0, 2
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
                        for filename in self.required_files:
                            (path / filename).touch()
                    else:
                        (path / 'decoy.txt').touch()

        # Search that structure for the one folder that has all we need
        result = tum_loader.find_files(str(base_root))
        self.assertEqual((
            str(true_path),
            str(true_path / 'rgb.txt'),
            str(true_path / 'depth.txt'),
            str(true_path / 'groundtruth.txt'),
        ), result)

        # Clean up after ourselves
        shutil.rmtree(base_root)

    def test_needs_all_elements(self):
        root_path = Path(self.temp_folder) / 'root'
        root_path.mkdir(parents=True, exist_ok=True)
        for missing_idx in range(len(self.required_files)):
            for filename_idx in range(len(self.required_files)):
                file_path = root_path / self.required_files[filename_idx]
                if filename_idx != missing_idx and not file_path.exists():
                    # Create all the required files except one
                    file_path.touch()
                elif file_path.exists():
                    # Remove the file
                    file_path.unlink()

            with self.assertRaises(FileNotFoundError):
                tum_loader.find_files(str(root_path))

        # Clean up after ourselves
        shutil.rmtree(root_path)


class TestTUMLoader(ExtendedTestCase):

    def test_make_camera_pose_returns_transform_object(self):
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, 0, 0, 0, 1)
        self.assertIsInstance(pose, tf.Transform)

    def test_make_camera_pose_location_coordinates(self):
        # Order here is right, down, forward
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, 0, 0, 0, 1)
        # Change coordinate order to forward, left, up
        self.assertNPEqual((13.2, -10, 22.4), pose.location)

    def test_make_camera_pose_changes_rotation_each_axis(self):
        # Roll, rotation around z-axis
        quat = tf3d.quaternions.axangle2quat((0, 0, 1), np.pi / 6)
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, quat[0], quat[1], quat[2], quat[3])
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around x-axis
        quat = tf3d.quaternions.axangle2quat((1, 0, 0), np.pi / 6)
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, quat[0], quat[1], quat[2], quat[3])
        self.assertNPClose((0, -np.pi / 6, 0), pose.euler)

        # Yaw, rotation around y-axis
        quat = tf3d.quaternions.axangle2quat((0, 1, 0), np.pi / 6)
        pose = tum_loader.make_camera_pose(10, -22.4, 13.2, quat[0], quat[1], quat[2], quat[3])
        self.assertNPClose((0, 0, -np.pi / 6), pose.euler)

    def test_make_camera_pose_combined(self):
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            quat = tf3d.quaternions.axangle2quat((-rot_axis[1], -rot_axis[2], rot_axis[0]), rot_angle)
            pose = tum_loader.make_camera_pose(-loc[1], -loc[2], loc[0], quat[0], quat[1], quat[2], quat[3])
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))

    def test_associate_data_same_keys(self):
        desired_result = sorted(
            [np.random.uniform(0, 100),
             np.random.randint(0, 1000),
             np.random.uniform(-100, 100),
             "test-{0}".format(np.random.randint(0, 1000))]
            for _ in range(20))
        int_map = {stamp: int_val for stamp, int_val, _, _ in desired_result}
        float_map = {stamp: float_val for stamp, _, float_val, _ in desired_result}
        str_map = {stamp: str_val for stamp, _, _, str_val in desired_result}
        self.assertEqual(desired_result, tum_loader.associate_data(int_map, float_map, str_map))

    def test_associate_data_noisy_keys(self):
        random = np.random.RandomState(1531)
        desired_result = sorted(
            [time + random.uniform(0, 0.5),
             random.randint(0, 1000),
             random.uniform(-100, 100),
             "test-{0}".format(random.randint(0, 1000))]
            for time in range(20))
        int_map = {stamp: int_val for stamp, int_val, _, _ in desired_result}
        float_map = {stamp + random.uniform(-0.02, 0.02): float_val for stamp, _, float_val, _ in desired_result}
        str_map = {stamp + random.uniform(-0.02, 0.02): str_val for stamp, _, _, str_val in desired_result}
        self.assertEqual(desired_result, tum_loader.associate_data(int_map, float_map, str_map))

    def test_associate_data_missing_keys(self):
        random = np.random.RandomState()
        original_data = sorted(
            [idx / 2 + random.uniform(0, 0.01),
             random.randint(0, 1000),
             random.uniform(-100, 100),
             "test-{0}".format(random.randint(0, 1000))]
            for idx in range(20))
        int_map = {stamp: int_val for stamp, int_val, _, _ in original_data}
        float_map = {stamp + random.uniform(-0.02, 0.02): float_val for stamp, _, float_val, _ in original_data
                     if stamp > 2}
        str_map = {stamp + random.uniform(-0.02, 0.02): str_val for stamp, _, _, str_val in original_data
                   if stamp < 8}
        self.assertEqual([inner for inner in original_data if 2 < inner[0] < 8],
                         tum_loader.associate_data(int_map, float_map, str_map))


def extend_mock_open(mock_open):
    """
    Extend the mock_open object to allow iteration over the file object.
    :param mock_open:
    :return:
    """
    handle = mock_open.return_value

    def _mock_file_iter():
        nonlocal handle
        for line in handle.readlines():
            yield line

    handle.__iter__.side_effect = _mock_file_iter
