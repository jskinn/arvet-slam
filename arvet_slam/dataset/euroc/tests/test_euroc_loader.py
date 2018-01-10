import unittest
import unittest.mock as mock
import numpy as np
import cv2
import transforms3d as tf3d
import arvet_slam.dataset.euroc.euroc_loader as euroc_loader
import arvet.util.transform as tf
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.util.image_utils as im_utils


class TestMakeCameraPose(unittest.TestCase):

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

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


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
        self.assertEqual(result, mapping)


class TestReadTrajectory(unittest.TestCase):

    def test_reads_relative_to_first_pose(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        line_template = "{time},{x},{y},{z},{qw},{qx},{qy},{qz},-0.005923,-0.002323,-0.002133," \
                        "0.021059,0.076659,-0.026895,0.136910,0.059287\n"
        trajectory_text = ""
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[timestamp] = pose
            absolute_pose = first_pose.find_independent(pose)
            quat = absolute_pose.rotation_quat(w_first=True)
            trajectory_text += line_template.format(
                time=timestamp,
                x=repr(-1 * absolute_pose.location[1]),
                y=repr(-1 * absolute_pose.location[2]),
                z=repr(absolute_pose.location[0]),
                qw=repr(quat[0]), qx=repr(-1 * quat[2]), qy=repr(-1 * quat[3]), qz=repr(quat[1])
            )

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath')
        self.assertEqual(len(trajectory), len(relative_trajectory))
        for time, pose in relative_trajectory.items():
            self.assertIn(time, trajectory)
            self.assertNPClose(pose.location, trajectory[time].location)
            self.assertNPClose(pose.rotation_quat(True), trajectory[time].rotation_quat(True))

    def test_skips_comment_lines(self):
        first_pose = tf.Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        trajectory_text = ""
        line_template = "{time},{x},{y},{z},{qw},{qx},{qy},{qz},-0.005923,-0.002323,-0.002133," \
                        "0.021059,0.076659,-0.026895,0.136910,0.059287\n"
        for time in range(0, 10):
            timestamp = time * 4999936 + 1403638128940097024
            pose = tf.Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                                rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[timestamp] = pose
            absolute_pose = first_pose.find_independent(pose)
            quat = absolute_pose.rotation_quat(w_first=True)
            trajectory_text += line_template.format(
                time=timestamp,
                x=repr(-1 * absolute_pose.location[1]),
                y=repr(-1 * absolute_pose.location[2]),
                z=repr(absolute_pose.location[0]),
                qw=repr(quat[0]), qx=repr(-1 * quat[2]), qy=repr(-1 * quat[3]), qz=repr(quat[1])
            )
            # Add incorrect trajectory data, preceeded by a hash to indicate it's a comment
            quat = pose.rotation_quat(w_first=True)
            trajectory_text += "# " + line_template.format(
                time=repr(time),
                x=repr(-1 * pose.location[1]),
                y=repr(-1 * pose.location[2]),
                z=repr(pose.location[0]),
                qw=repr(quat[0]), qx=repr(-1 * quat[2]), qy=repr(-1 * quat[3]), qz=repr(quat[1])
            )

        mock_open = mock.mock_open(read_data=trajectory_text)
        extend_mock_open(mock_open)
        with mock.patch('arvet_slam.dataset.euroc.euroc_loader.open', mock_open, create=True):
            trajectory = euroc_loader.read_trajectory('test_filepath')
        self.assertEqual(len(trajectory), len(relative_trajectory))
        for time, pose in relative_trajectory.items():
            self.assertIn(time, trajectory)
            self.assertNPClose(pose.location, trajectory[time].location)
            self.assertNPClose(pose.rotation_quat(True), trajectory[time].rotation_quat(True))

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(
            str([repr(f) for f in arr1]), str([repr(f) for f in arr2])))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


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


class TestGetCameraCalibration(unittest.TestCase):

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

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


class TestRectify(unittest.TestCase):

    def test_trivial(self):
        # Some arbitrary intrinsics, with distortion
        intrinsics = cam_intr.CameraIntrinsics(
            width=320,
            height=240,
            fx=160,
            fy=160,
            cx=160,
            cy=120
        )
        expected_v, expected_u = np.indices((intrinsics.height, intrinsics.width))
        left_u, left_v, right_u, right_v = euroc_loader.rectify(tf.Transform(), intrinsics,
                                                                tf.Transform([1, 0, 0]), intrinsics)
        self.assertNPEqual(expected_u, left_u)
        self.assertNPEqual(expected_v, left_v)
        self.assertNPEqual(expected_u, right_u)
        self.assertNPEqual(expected_v, right_v)

    def test_uses_k3_correctly(self):
        # Some arbitrary intrinsics, with distortion
        intrinsics = cam_intr.CameraIntrinsics(
            width=752,
            height=480,
            fx=376,
            fy=376,
            cx=376,
            cy=240,
            k1=0.01,
            k3=-0.0212445
        )
        distort_x, distort_y = create_undistortion_map(intrinsics)
        left_u, left_v, right_u, right_v = euroc_loader.rectify(tf.Transform(), intrinsics,
                                                                tf.Transform([1, 0, 0]), intrinsics)
        self.assertNPClose(distort_x, left_u)
        self.assertNPClose(distort_y, left_v)
        self.assertNPClose(distort_x, right_u)
        self.assertNPClose(distort_y, right_v)

    def test_undistorts_left_image(self):
        self.skipTest('Not working')
        # Some arbitrary intrinsics, with distortion
        intrinsics = cam_intr.CameraIntrinsics(
            width=752,
            height=480,
            fx=376,
            fy=376,
            cx=376,
            cy=240,
            k1=0.01,
            # k1=-0.28368365,    # stereoRectify moves the projection plane iff k1<0, for some reason
            k3=-0.0212445
        )
        undistorted_image = np.array([
            [[255 - int(127 * x / intrinsics.width + 127 * y / intrinsics.height),
              int(255 * y / intrinsics.height),
              int(255 * x / intrinsics.width)
              ] for x in range(intrinsics.width)] for y in range(intrinsics.height)
        ], dtype=np.uint8)
        distort_x, distort_y = create_distortion_map(intrinsics)
        distorted_image = cv2.remap(undistorted_image, distort_x, distort_y, cv2.INTER_LINEAR)

        left_u, left_v, _, _ = euroc_loader.rectify(tf.Transform(), intrinsics, tf.Transform([1, 0, 0]), intrinsics)
        result = cv2.remap(distorted_image, left_u, left_v, cv2.INTER_LINEAR)
        self.assertNPClose(undistorted_image, result)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(
            str([repr(f) for f in arr1]), str([repr(f) for f in arr2])))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


def create_distortion_map(intr):
    """
    Create distortion maps for distorting an image, so that OpenCV can undo it.
    Does the OpenCV undistortion, see:
    - https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
    - https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv.InitUndistortRectifyMap

    :param u:
    :param v:
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

    :param u:
    :param v:
    :param intr:
    :return:
    """
    y, x = np.indices((intr.height, intr.width))
    x = (x - intr.cx) / intr.fx
    y = (y - intr.cy) / intr.fy
    r2 = x * x + y * y

    radial_distort = (1 + intr.k1 * r2 + intr.k2 * r2 * r2 + intr.k3 * r2 * r2 * r2)
    x_distort = x * radial_distort
    y_distort = y * radial_distort

    # Tangential distortion first, since OpenCV does it last
    x_distort = x_distort + (2 * intr.p1 * x * y + intr.p2 * (r2 + 2 * x * x))
    y_distort = y_distort + (intr.p1 * (r2 + 2 * y * y) + 2 * intr.p2 * x * y)

    return intr.fx * x_distort + intr.cx, intr.fy * y_distort + intr.cy


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
