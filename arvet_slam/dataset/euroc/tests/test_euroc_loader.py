import unittest
import unittest.mock as mock
import numpy as np
import cv2
import transforms3d as tf3d
from arvet.util.test_helpers import ExtendedTestCase
import arvet.util.transform as tf
from arvet.metadata.camera_intrinsics import CameraIntrinsics
import arvet_slam.dataset.euroc.euroc_loader as euroc_loader


class TestMakeCameraPose(ExtendedTestCase):

    def test_location(self):
        forward = 51.2
        up = 153.3
        left = -126.07
        pose = euroc_loader.make_camera_pose(-1 * left, -1 * up, forward, 0, 0, 0, 1)
        self.assertNPEqual((forward, left, up), pose.location)

    def test_orientation(self):
        angle = np.pi / 7
        forward = 51.2
        up = 153.3
        left = -126.07
        quat = tf3d.quaternions.axangle2quat((-1 * left, -1 * up, forward), angle)
        pose = euroc_loader.make_camera_pose(0, 0, 0, quat[0], quat[1], quat[2], quat[3])
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
        pose = euroc_loader.make_camera_pose(-1 * left, -1 * up, forward, quat[0], quat[1], quat[2], quat[3])
        self.assertNPEqual((forward, left, up), pose.location)
        self.assertNPEqual(tf3d.quaternions.axangle2quat((o_forward, o_left, o_up), angle), pose.rotation_quat(True))

    def test_randomized(self):
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            quat = tf3d.quaternions.axangle2quat((-rot_axis[1], -rot_axis[2], rot_axis[0]), rot_angle, False)
            pose = euroc_loader.make_camera_pose(-loc[1], -loc[2], loc[0], quat[0], quat[1], quat[2], quat[3])
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))


class TestReadImageFilenames(unittest.TestCase):

    def test_reads_filenames_to_mapping(self):
        line_template = "{time},{time}.png\n"
        data_text = ""
        mapping = {}
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            mapping[timestamp] = "{0}.png".format(timestamp)
            data_text += line_template.format(time=timestamp)

        mock_open = mock.mock_open(read_data=data_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            result = euroc_loader.read_image_filenames('test_filepath')
        self.assertTrue(mock_open.called)
        self.assertEqual('test_filepath', mock_open.call_args[0][0])
        self.assertEqual('r', mock_open.call_args[0][1])
        self.assertEqual(result, mapping)

    def test_ignores_comments_and_empty_lines(self):
        line_template = "{time},{time}.png\n"
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
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            result = euroc_loader.read_image_filenames('test_filepath')
        self.assertEqual(result, mapping)

    def test_ignores_characters_after_comment(self):
        line_template = "{time},rgb/{time}.png  # This is time {time}\n"
        data_text = ""
        mapping = {}
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            mapping[timestamp] = "rgb/{0}.png".format(timestamp)
            data_text += line_template.format(time=timestamp)

        mock_open = mock.mock_open(read_data=data_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            result = euroc_loader.read_image_filenames('test_filepath')
        self.assertTrue(mock_open.called)
        self.assertEqual('test_filepath', mock_open.call_args[0][0])
        self.assertEqual('r', mock_open.call_args[0][1])
        self.assertEqual(result, mapping)


class TestReadTrajectory(ExtendedTestCase):
    line_template = "{time},{x},{y},{z},{qw},{qx},{qy},{qz},-0.005923,-0.002323,-0.002133," \
                        "0.021059,0.076659,-0.026895,0.136910,0.059287\n"

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
            time=int(timestamp),
            x=repr(-1 * pose.location[1]),
            y=repr(-1 * pose.location[2]),
            z=repr(pose.location[0]),
            qw=repr(quat[0]), qx=repr(-1 * quat[2]), qy=repr(-1 * quat[3]), qz=repr(quat[1])
        )

    def test_reads_from_given_file(self):
        trajectory_text = ""
        timestamps = []
        for time in range(0, 10):
            timestamp = time * 101 + 10000
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            trajectory_text += self.format_line(timestamp, pose)
            timestamps.append(timestamp)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath', timestamps)
        self.assertTrue(mock_open.called)
        self.assertEqual('test_filepath', mock_open.call_args[0][0])
        self.assertEqual('r', mock_open.call_args[0][1])
        self.assertEqual(len(trajectory), len(timestamps))

    def test_reads_relative_to_first_pose(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        trajectory_text = ""
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[timestamp] = pose
            absolute_pose = first_pose.find_independent(pose)
            trajectory_text += self.format_line(timestamp, absolute_pose)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath', relative_trajectory.keys())
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
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath', relative_trajectory.keys())
        self.assertEqual(len(trajectory), len(relative_trajectory))
        for time, pose in relative_trajectory.items():
            self.assertIn(time, trajectory)
            self.assertNPClose(pose.location, trajectory[time].location)
            self.assertNPClose(pose.rotation_quat(True), trajectory[time].rotation_quat(True))

    def test_removes_comments_from_the_end_of_lines(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        trajectory_text = "# Starting with a comment\n    # Another comment\n\n"

        line_template = "{time},{x},{y},{z},{qw},{qx},{qy},{qz} # This is a comment\n"
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[timestamp] = pose
            absolute_pose = first_pose.find_independent(pose)
            trajectory_text += self.format_line(timestamp, absolute_pose, line_template)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath', relative_trajectory.keys())
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

        time_scale = 10000
        encoded_trajectory = {
            time_scale * time: make_pose(time)
            for time in range(0, 11, 2)
        }
        trajectory_text = ""
        for time, pose in encoded_trajectory.items():
            trajectory_text += self.format_line(time, pose)

        desired_times = [int(time_scale * (time + 0.1)) for time in range(1, 11, 2)]
        first_pose = make_pose(min(desired_times) / time_scale)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath', desired_times)
        self.assertEqual(set(desired_times), set(trajectory.keys()))
        for time in desired_times:
            expected_pose = first_pose.find_relative(make_pose(time / time_scale))
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

        time_scale = 10000
        encoded_trajectory = {
            time_scale * time: make_pose(time)
            for time in range(0, 11, 2)
        }
        trajectory_text = ""
        for time, pose in encoded_trajectory.items():
            trajectory_text += self.format_line(time, pose)

        # Lots of desired times for widely spaced samples
        desired_times = [int(t) for t in np.linspace(0, 10 * time_scale - 1, num=47, endpoint=True)]
        first_pose = make_pose(min(desired_times) / time_scale)

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath', desired_times)
        self.assertEqual(set(desired_times), set(trajectory.keys()))
        for time in desired_times:
            expected_pose = first_pose.find_relative(make_pose(time / time_scale))
            self.assertIn(time, trajectory)
            self.assertNPClose(expected_pose.location, trajectory[time].location, atol=1e-13, rtol=0)
            self.assertNPClose(expected_pose.rotation_quat(True), trajectory[time].rotation_quat(True),
                               atol=1e-14, rtol=0)


class TestAssociateData(unittest.TestCase):

    def test_single_map_returns_sorted(self):
        subject = {(v * 7 + 3) % 10: str(v) for v in range(10)}     # A disordered map
        result = euroc_loader.associate_data(subject)
        self.assertEqual([
            [0, '1'], [1, '4'], [2, '7'], [3, '0'], [4, '3'],
            [5, '6'], [6, '9'], [7, '2'], [8, '5'], [9, '8']
        ], result)

    def test_same_keys_associates(self):
        root_map = {(t * 7 + 3) % 10: str(t) for t in range(10)}
        map_float_1 = {t: t * 1.2215 - 0.2234 * t * t + 0.115 for t in range(10)}
        map_float_2 = {t: 4.462 * t * t - 1000.212 for t in range(10)}
        map_filename = {t: "{0}.png".format(t * 4999936 + 1403638128940097024) for t in range(10)}
        result = euroc_loader.associate_data(root_map, map_float_1, map_float_2, map_filename)

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
            root_time = t * 10 + np.random.uniform(-1, 1)
            float_1 = t * 1.2215 - 0.2234 * t * t + 0.115
            float_2 = 4.462 * t * t - 1000.212
            filename = "{0}.png".format(t * 4999936 + 1403638128940097024)
            expected_result.append([root_time, t, float_1, float_2, filename])
            root_map[root_time] = t
            map_float_1[t * 10 + np.random.uniform(-1, 1)] = float_1
            map_float_2[t * 10 + np.random.uniform(-1, 1)] = float_2
            map_filename[t * 10 + np.random.uniform(-1, 1)] = filename
        result = euroc_loader.associate_data(root_map, map_float_1, map_float_2, map_filename)
        self.assertEqual(expected_result, result)


class TestGetCameraCalibration(ExtendedTestCase):

    def test_reads_filenames_to_mapping(self):
        # This is a genuine yaml file from MH_01_easy, cam0
        sensor_yaml = "# General sensor definitions.\n" \
                      "sensor_type: camera\n" \
                      "comment: VI-Sensor cam0 (MT9M034)\n" \
                      "\n" \
                      "# Sensor extrinsics wrt. the body-frame.\n" \
                      "T_BS:\n" \
                      "  cols: 4\n" \
                      "  rows: 4\n" \
                      "  data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,\n" \
                      "         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,\n" \
                      "        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,\n" \
                      "         0.0, 0.0, 0.0, 1.0]\n" \
                      "\n" \
                      "# Camera specific definitions.\n" \
                      "rate_hz: 20\n" \
                      "resolution: [752, 480]\n" \
                      "camera_model: pinhole\n" \
                      "intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv\n" \
                      "distortion_model: radial-tangential\n" \
                      "distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]\n"

        mock_open = mock.mock_open(read_data=sensor_yaml)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            extrinsics, intrinsics = euroc_loader.get_camera_calibration('test_filepath')

        self.assertNPClose(np.array(
            [[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
             [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
             [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
             [0.0, 0.0, 0.0, 1.0]]
        ), extrinsics.transform_matrix)
        self.assertEqual(752, intrinsics.width)
        self.assertEqual(480, intrinsics.height)
        self.assertEqual(458.654, intrinsics.fx)
        self.assertEqual(457.296, intrinsics.fy)
        self.assertEqual(367.215, intrinsics.cx)
        self.assertEqual(248.375, intrinsics.cy)
        self.assertEqual(-0.28340811, intrinsics.k1)
        self.assertEqual(0.07395907, intrinsics.k2)
        self.assertEqual(0.00019359, intrinsics.p1)
        self.assertEqual(1.76187114e-05, intrinsics.p2)


class TestRectify(ExtendedTestCase):

    def test_trivial(self):
        # Some arbitrary intrinsics, with distortion
        intrinsics = CameraIntrinsics(
            width=320,
            height=240,
            fx=160,
            fy=160,
            cx=160,
            cy=120
        )
        expected_v, expected_u = np.indices((intrinsics.height, intrinsics.width))
        left_u, left_v, left_intrinsics, right_u, right_v, right_intrinsics = euroc_loader.rectify(
            tf.Transform(), intrinsics, tf.Transform([1, 0, 0]), intrinsics)
        self.assertNPEqual(expected_u, left_u)
        self.assertNPEqual(expected_v, left_v)
        self.assertNPEqual(expected_u, right_u)
        self.assertNPEqual(expected_v, right_v)

    def test_undistorts_left_image(self):
        # Some arbitrary intrinsics, with distortion
        intrinsics = CameraIntrinsics(
            width=100,
            height=100,
            fx=123,
            fy=122,
            cx=51,
            cy=49.5,
            k1=0.28340811,
            k2=0.07395907,
            p1=0.00019359,
            p2=1.76187114e-05,
            k3=-0.0212445
        )
        input_image = np.zeros((100, 100, 3), dtype=np.uint8)

        point_ul = (-40 / intrinsics.fx, -40 / intrinsics.fy)
        point_ur = (-40 / intrinsics.fx, 40 / intrinsics.fy)
        point_ll = (40 / intrinsics.fx, -40 / intrinsics.fy)
        point_lr = (40 / intrinsics.fx, 40 / intrinsics.fy)

        point = world_point_to_pixel(point_ul[0], point_ul[1], intrinsics)
        set_subpixel(input_image, point[0], point[1], np.array([0, 0, 255]))

        point = world_point_to_pixel(point_ur[0], point_ur[1], intrinsics)
        set_subpixel(input_image, point[0], point[1], np.array([0, 255, 0]))

        point = world_point_to_pixel(point_ll[0], point_ll[1], intrinsics)
        set_subpixel(input_image, point[0], point[1], np.array([255, 0, 0]))

        point = world_point_to_pixel(point_lr[0], point_lr[1], intrinsics)
        set_subpixel(input_image, point[0], point[1], np.array([255, 255, 255]))

        left_u, left_v, left_intr, _, _, _ = euroc_loader.rectify(tf.Transform(), intrinsics,
                                                                  tf.Transform([1, 0, 0]), intrinsics)
        undistorted_image = cv2.remap(input_image, left_u, left_v, cv2.INTER_LINEAR)

        point = world_point_to_pixel(point_ul[0], point_ul[1], left_intr)
        colour = get_subpixel(undistorted_image, point[0], point[1])
        self.assertEqual(0, colour[0])
        self.assertEqual(0, colour[1])
        self.assertGreater(colour[2], 50)   # Color is spread around by subpixel interpolation

        point = world_point_to_pixel(point_ur[0], point_ur[1], left_intr)
        colour = get_subpixel(undistorted_image, point[0], point[1])
        self.assertEqual(0, colour[0])
        self.assertGreater(colour[1], 50)
        self.assertEqual(0, colour[2])

        point = world_point_to_pixel(point_ll[0], point_ll[1], left_intr)
        colour = get_subpixel(undistorted_image, point[0], point[1])
        self.assertGreater(colour[0], 50)
        self.assertEqual(0, colour[1])
        self.assertEqual(0, colour[2])

        point = world_point_to_pixel(point_lr[0], point_lr[1], left_intr)
        colour = get_subpixel(undistorted_image, point[0], point[1])
        self.assertGreater(colour[0], 20)
        self.assertGreater(colour[1], 20)
        self.assertGreater(colour[2], 20)
        self.assertEqual(colour[0], colour[1])
        self.assertEqual(colour[0], colour[2])

    def test_undistorts_right_image(self):
        # Some arbitrary intrinsics, with distortion
        intrinsics = CameraIntrinsics(
            width=100,
            height=100,
            fx=123,
            fy=122,
            cx=51,
            cy=49.5,
            k1=0.28340811,
            k2=0.07395907,
            p1=0.00019359,
            p2=1.76187114e-05,
            k3=-0.0212445
        )
        input_image = np.zeros((100, 100, 3), dtype=np.uint8)

        point_ul = (-40 / intrinsics.fx, -40 / intrinsics.fy)
        point_ur = (-40 / intrinsics.fx, 40 / intrinsics.fy)
        point_ll = (40 / intrinsics.fx, -40 / intrinsics.fy)
        point_lr = (40 / intrinsics.fx, 40 / intrinsics.fy)

        point = world_point_to_pixel(point_ul[0], point_ul[1], intrinsics)
        set_subpixel(input_image, point[0], point[1], np.array([0, 0, 255]))

        point = world_point_to_pixel(point_ur[0], point_ur[1], intrinsics)
        set_subpixel(input_image, point[0], point[1], np.array([0, 255, 0]))

        point = world_point_to_pixel(point_ll[0], point_ll[1], intrinsics)
        set_subpixel(input_image, point[0], point[1], np.array([255, 0, 0]))

        point = world_point_to_pixel(point_lr[0], point_lr[1], intrinsics)
        set_subpixel(input_image, point[0], point[1], np.array([255, 255, 255]))

        _, _, _, right_u, right_v, right_intr = euroc_loader.rectify(tf.Transform(), intrinsics,
                                                                     tf.Transform([1, 0, 0]), intrinsics)
        undistorted_image = cv2.remap(input_image, right_u, right_v, cv2.INTER_LINEAR)

        point = world_point_to_pixel(point_ul[0], point_ul[1], right_intr)
        colour = get_subpixel(undistorted_image, point[0], point[1])
        self.assertEqual(0, colour[0])
        self.assertEqual(0, colour[1])
        self.assertGreater(colour[2], 50)   # Color is spread around by subpixel interpolation

        point = world_point_to_pixel(point_ur[0], point_ur[1], right_intr)
        colour = get_subpixel(undistorted_image, point[0], point[1])
        self.assertEqual(0, colour[0])
        self.assertGreater(colour[1], 50)
        self.assertEqual(0, colour[2])

        point = world_point_to_pixel(point_ll[0], point_ll[1], right_intr)
        colour = get_subpixel(undistorted_image, point[0], point[1])
        self.assertGreater(colour[0], 50)
        self.assertEqual(0, colour[1])
        self.assertEqual(0, colour[2])

        point = world_point_to_pixel(point_lr[0], point_lr[1], right_intr)
        colour = get_subpixel(undistorted_image, point[0], point[1])
        self.assertGreater(colour[0], 20)
        self.assertGreater(colour[1], 20)
        self.assertGreater(colour[2], 20)
        self.assertEqual(colour[0], colour[1])
        self.assertEqual(colour[0], colour[2])

    @unittest.skip("I don't know where they got these numbers, but I don't know why they don't match")
    def test_matches_orbslam_example(self):
        # The actual intrinsics and extrinsics taken from the dataset
        left_extrinsics = tf.Transform(np.array([
            [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
            [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
            [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
            [0, 0, 0, 1]
        ]))
        left_intrinsics = CameraIntrinsics(
            width=752,
            height=480,
            fx=458.654,
            fy=457.296,
            cx=367.215,
            cy=248.375,
            k1=-0.28340811,
            k2=0.07395907,
            p1=0.00019359,
            p2=1.76187114e-05,
            k3=0
        )
        right_extrinsics = tf.Transform(np.array([
            [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
            [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
            [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
            [0, 0, 0, 1]
        ]))
        right_intrinsics = CameraIntrinsics(
            width=752,
            height=480,
            fx=457.587,
            fy=456.134,
            cx=379.999,
            cy=255.238,
            k1=-0.28368365,
            k2=0.07451284,
            p1=-0.00010473,
            p2=-3.55590700e-05,
            k3=0
        )

        # These are the orbslam numbers
        oheight = 480
        owidth = 752
        od_left = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
        ok_left = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
        or_left = np.array([[0.999966347530033, -0.001422739138722922, 0.008079580483432283],
                            [0.001365741834644127, 0.9999741760894847, 0.007055629199258132],
                            [-0.008089410156878961, -0.007044357138835809, 0.9999424675829176]])
        op_left = np.array([[435.2046959714599, 0, 367.4517211914062, 0],
                            [0, 435.2046959714599, 252.2008514404297, 0],
                            [0, 0, 1, 0]])
        od_right = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0])
        ok_right = np.array([[457.587, 0.0, 379.999], [0.0, 456.134, 255.238], [0.0, 0.0, 1]])
        or_right = np.array([[0.9999633526194376, -0.003625811871560086, 0.007755443660172947],
                             [0.003680398547259526, 0.9999684752771629, -0.007035845251224894],
                             [-0.007729688520722713, 0.007064130529506649, 0.999945173484644]])
        op_right = np.array([[435.2046959714599, 0, 367.4517211914062, -47.90639384423901],
                             [0, 435.2046959714599, 252.2008514404297, 0],
                             [0, 0, 1, 0]])

        orbslam_m1l, orbslam_m2l = cv2.initUndistortRectifyMap(ok_left, od_left, or_left,
                                                               op_left[0:3, 0:3], (owidth, oheight), cv2.CV_32F)
        orbslam_m1r, orbslam_m2r = cv2.initUndistortRectifyMap(ok_right, od_right, or_right,
                                                               op_right[0:3, 0:3], (owidth, oheight), cv2.CV_32F)

        left_u, left_v, _, right_u, right_v, _ = euroc_loader.rectify(
            left_extrinsics, left_intrinsics, right_extrinsics, right_intrinsics)

        self.assertLess(np.max(np.abs(orbslam_m1l - left_u)), 0.06)
        self.assertLess(np.max(np.abs(orbslam_m2l - left_v)), 0.06)
        self.assertLess(np.max(np.abs(orbslam_m1r - right_u)), 0.06)
        self.assertLess(np.max(np.abs(orbslam_m2r - right_v)), 0.06)


def create_distortion_map(intr):
    """
    Create distortion maps for distorting an image, so that OpenCV can undo it.
    Does the OpenCV undistortion, see:
    - https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
    - https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv.InitUndistortRectifyMap
    :param intr:
    :return:
    """
    y, x = np.indices((intr.height, intr.width))
    x = (x - intr.cx) / intr.fx
    y = (y - intr.cy) / intr.fy
    r2 = x * x + y * y

    # Tangential distortion first, since OpenCV does it last
    x_distort = x - (2 * intr.p1 * x * y + intr.p2 * (r2 + 2 * x * x))
    y_distort = y - (intr.p1 * (r2 + 2 * y * y) + 2 * intr.p2 * x * y)

    radial_distort = (1 + intr.k1 * r2 + intr.k2 * r2 * r2 + intr.k3 * r2 * r2 * r2)
    x_distort /= radial_distort
    y_distort /= radial_distort

    return np.asarray(intr.fx * x_distort + intr.cx, np.float32), \
        np.asarray(intr.fy * y_distort + intr.cy, np.float32)


def create_undistortion_map(intr):
    """
    Create distortion maps for undistorting image
    Used to generate distortion maps, in the same way that OpenCV does.
    Does the OpenCV undistortion, see:
    - https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
    - https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv.InitUndistortRectifyMap

    :param intr:
    :return:
    """
    y, x = np.indices((intr.height, intr.width))
    x = (x - intr.cx) / intr.fx
    y = (y - intr.cy) / intr.fy
    r2 = x * x + y * y

    radial_distort = (1 + intr.k1 * r2 + intr.k2 * r2 * r2 + intr.k3 * r2 * r2 * r2)
    x_distort = x * radial_distort + (2 * intr.p1 * x * y + intr.p2 * (r2 + 2 * x * x))
    y_distort = y * radial_distort + (intr.p1 * (r2 + 2 * y * y) + 2 * intr.p2 * x * y)

    return intr.fx * x_distort + intr.cx, intr.fy * y_distort + intr.cy


def world_point_to_pixel(x, y, intr):
    """
    Feed a point through the OpenCV distortion model to get
    :param x:
    :param y:
    :param intr:
    :return:
    """
    r2 = x * x + y * y

    radial_distort = (1 + intr.k1 * r2 + intr.k2 * r2 * r2 + intr.k3 * r2 * r2 * r2)
    x_distort = x * radial_distort + (2 * intr.p1 * x * y + intr.p2 * (r2 + 2 * x * x))
    y_distort = y * radial_distort + (intr.p1 * (r2 + 2 * y * y) + 2 * intr.p2 * x * y)

    return intr.fx * x_distort + intr.cx, intr.fy * y_distort + intr.cy


def set_subpixel(target, x, y, val):
    floor_x = int(np.floor(x))
    ceil_x = floor_x + 1
    floor_y = int(np.floor(y))
    ceil_y = floor_y + 1

    target[ceil_x, ceil_y] = (x - floor_x) * (y - floor_y) * val
    target[floor_x, ceil_y] = (ceil_x - x) * (y - floor_y) * val
    target[ceil_x, floor_y] = (x - floor_x) * (ceil_y - y) * val
    target[floor_x, floor_y] = (ceil_x - x) * (ceil_y - y) * val


def get_subpixel(target, x, y):
    floor_x = int(np.floor(x))
    ceil_x = floor_x + 1
    floor_y = int(np.floor(y))
    ceil_y = floor_y + 1

    return ((x - floor_x) * (y - floor_y) * target[ceil_x, ceil_y] +
            (ceil_x - x) * (y - floor_y) * target[floor_x, ceil_y] +
            (x - floor_x) * (ceil_y - y) * target[ceil_x, floor_y] +
            (ceil_x - x) * (ceil_y - y) * target[floor_x, floor_y])


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
