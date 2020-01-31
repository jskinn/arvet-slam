# Copyright (c) 2017, John Skinner
import os.path
import typing
import numpy as np
import logging
import cv2
import yaml
try:
    from yaml import CDumper as YamlDumper, CLoader as YamlLoader
except ImportError:
    from yaml import Dumper as YamlDumper, Loader as YamlLoader


import arvet.util.image_utils as image_utils
import arvet.util.associate as ass
import arvet.metadata.image_metadata as imeta
import arvet.util.transform as tf
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import StereoImage
from arvet.core.image_collection import ImageCollection
from arvet_slam.util.trajectory_builder import TrajectoryBuilder


def make_camera_pose(tx: float, ty: float, tz: float, qw: float, qx: float, qy: float, qz: float) -> tf.Transform:
    """
    As far as I can tell, EuRoC uses Z forward coordinates, the same as everything else.
    I need to switch it to X-forward coordinates.

    :param tx: The x coordinate of the location
    :param ty: The y coordinate of the location
    :param tz: The z coordinate of the location
    :param qw: The scalar part of the quaternion orientation
    :param qx: The x imaginary part of the quaternion orientation
    :param qy: The y imaginary part of the quaternion orientation
    :param qz: The z imaginary part of the quaternion orientation
    :return: A Transform object representing the world pose of the current frame
    """
    return tf.Transform(
        location=(tz, -tx, -ty),
        rotation=(qw, qz, -qx, -qy),
        w_first=True
    )


def fix_coordinates(trans: tf.Transform) -> tf.Transform:
    """
    Exchange the coordinates on a transform from camera frame to world frame.
    Used for the camera extrinsics
    :param trans:
    :return:
    """
    x, y, z = trans.location
    qw, qx, qy, qz = trans.rotation_quat(w_first=True)
    return make_camera_pose(x, y, z, qw, qx, qy, qz)


def read_image_filenames(images_file_path: str) -> typing.Mapping[int, str]:
    """
    Read data from a camera sensor, formatted as a csv,
    producing timestamp, filename pairs.
    :param images_file_path:
    :return:
    """
    filename_map = {}
    with open(images_file_path, 'r') as images_file:
        for line in images_file:
            comment_idx = line.find('#')
            if comment_idx >= 0:
                # This line has a comment, ignore everything after it
                line = line[:comment_idx]
            line = line.strip()
            if len(line) > 0:
                parts = line.split(',')
                if len(parts) >= 2:
                    timestamp, relative_path = parts[0:2]
                    filename_map[int(timestamp)] = relative_path.rstrip()  # To remove trailing newlines
    return filename_map


def read_trajectory(trajectory_filepath: str, desired_times: typing.Iterable[float])\
        -> typing.Mapping[float, tf.Transform]:
    """
    Read the ground-truth camera trajectory from file.
    The raw pose information is relative to some world frame, we adjust it to be relative to the initial pose
    of the camera, for standardization.
    This trajectory describes the motion of the robot, combine it with the pose of the camera relative to the robot
    to get the camera trajectory.

    :param trajectory_filepath:
    :return: A map of timestamp to camera pose.
    """
    builder = TrajectoryBuilder(desired_times)
    with open(trajectory_filepath, 'r') as trajectory_file:
        for line in trajectory_file:
            comment_idx = line.find('#')
            if comment_idx >= 0:
                # This line contains a comment, remove everything after it
                line = line[:comment_idx]
            line = line.strip()
            parts = line.split(',')
            if len(parts) >= 8:
                timestamp, tx, ty, tz, qw, qx, qy, qz = parts[0:8]
                pose = make_camera_pose(float(tx), float(ty), float(tz),
                                        float(qw), float(qx), float(qy), float(qz))
                builder.add_trajectory_point(int(timestamp), pose)
    return builder.get_interpolated_trajectory()


def associate_data(root_map: typing.Mapping, *args: typing.Mapping) -> typing.List[typing.List]:
    """
    Convert a number of maps key->value to a list of lists
    [[key, map1[key], map2[key] map3[key] ...] ...]

    The list will be sorted in key order
    Returned inner lists will be in the same order as they are passed as arguments.

    The first map passed is considered the reference point for the list of keys,
    :param root_map: The first map to associate
    :param args: Additional maps to associate to the first one
    :return:
    """
    if len(args) <= 0:
        # Nothing to associate, flatten the root map and return
        return sorted([k, v] for k, v in root_map.items())
    root_keys = set(root_map.keys())
    all_same = True
    # First, check if all the maps have the same list of keys
    for other_map in args:
        if set(other_map.keys()) != root_keys:
            all_same = False
            break
    if all_same:
        # All the maps have the same set of keys, just flatten them
        return sorted([key, root_map[key]] + [other_map[key] for other_map in args]
                      for key in root_keys)
    else:
        # We need to associate the maps, the timestamps are a little out
        rekeyed_maps = []
        for other_map in args:
            matches = ass.associate(root_map, other_map, offset=0, max_difference=3)
            rekeyed_map = {root_key: other_map[other_key] for root_key, other_key in matches}
            root_keys &= set(rekeyed_map.keys())
            rekeyed_maps.append(rekeyed_map)
        return sorted([key, root_map[key]] + [rekeyed_map[key] for rekeyed_map in rekeyed_maps]
                      for key in root_keys)


def get_camera_calibration(sensor_yaml_path: str) -> typing.Tuple[tf.Transform, CameraIntrinsics]:
    with open(sensor_yaml_path, 'r') as sensor_file:
        sensor_data = yaml.load(sensor_file, YamlLoader)

    d = sensor_data['T_BS']['data']
    extrinsics = tf.Transform(np.array([
        [d[0], d[1], d[2], d[3]],
        [d[4], d[5], d[6], d[7]],
        [d[8], d[9], d[10], d[11]],
        [d[12], d[13], d[14], d[15]],
    ]))
    resolution = sensor_data['resolution']
    intrinsics = CameraIntrinsics(
        width=resolution[0],
        height=resolution[1],
        fx=sensor_data['intrinsics'][0],
        fy=sensor_data['intrinsics'][1],
        cx=sensor_data['intrinsics'][2],
        cy=sensor_data['intrinsics'][3],
        k1=sensor_data['distortion_coefficients'][0],
        k2=sensor_data['distortion_coefficients'][1],
        p1=sensor_data['distortion_coefficients'][2],
        p2=sensor_data['distortion_coefficients'][3]
    )
    return extrinsics, intrinsics


def rectify(left_extrinsics: tf.Transform, left_intrinsics: CameraIntrinsics,
            right_extrinsics: tf.Transform, right_intrinsics: CameraIntrinsics) -> \
        typing.Tuple[np.ndarray, np.ndarray, CameraIntrinsics,
                     np.ndarray, np.ndarray, CameraIntrinsics]:
    """
    Compute mapping matrices for performing stereo rectification, from the camera properties.
    Applying the returned transformation to the images using cv2.remap gives us undistorted stereo rectified images

    :param left_extrinsics:
    :param left_intrinsics:
    :param right_extrinsics:
    :param right_intrinsics:
    :return: 4 remapping matrices: left x, left y, right x, right y
    """
    shape = (left_intrinsics.width, left_intrinsics.height)
    left_distortion = np.array([
        left_intrinsics.k1, left_intrinsics.k2,
        left_intrinsics.p1, left_intrinsics.p2,
        left_intrinsics.k3
    ])
    right_distortion = np.array([
        right_intrinsics.k1, right_intrinsics.k2,
        right_intrinsics.p1, right_intrinsics.p2,
        right_intrinsics.k3
    ])

    # We want the transform from the right to the left, which is the position of the left relative to the right
    relative_transform = right_extrinsics.find_relative(left_extrinsics)

    r_left = np.zeros((3, 3))
    r_right = np.zeros((3, 3))
    p_left = np.zeros((3, 4))
    p_right = np.zeros((3, 4))
    cv2.stereoRectify(
        cameraMatrix1=left_intrinsics.intrinsic_matrix(),
        distCoeffs1=left_distortion,
        cameraMatrix2=right_intrinsics.intrinsic_matrix(),
        distCoeffs2=right_distortion,
        imageSize=shape,
        R=relative_transform.rotation_matrix,
        T=relative_transform.location,
        alpha=0,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=shape,
        R1=r_left,
        R2=r_right,
        P1=p_left,
        P2=p_right
    )

    m1l, m2l = cv2.initUndistortRectifyMap(left_intrinsics.intrinsic_matrix(), left_distortion, r_left,
                                           p_left[0:3, 0:3], shape, cv2.CV_32F)
    m1r, m2r = cv2.initUndistortRectifyMap(right_intrinsics.intrinsic_matrix(), right_distortion, r_right,
                                           p_right[0:3, 0:3], shape, cv2.CV_32F)

    # Rectification has changed the camera intrinsics, return the new ones
    rectified_left_intrinsics = CameraIntrinsics(
        width=shape[0],
        height=shape[1],
        fx=p_left[0, 0],
        fy=p_left[1, 1],
        cx=p_left[0, 2],
        cy=p_left[1, 2],
        s=p_left[0, 1]
    )
    rectified_right_intrinsics = CameraIntrinsics(
        width=shape[0],
        height=shape[1],
        fx=p_right[0, 0],
        fy=p_right[1, 1],
        cx=p_right[0, 2],
        cy=p_right[1, 2],
        s=p_right[0, 1]
    )

    return m1l, m2l, rectified_left_intrinsics, m1r, m2r, rectified_right_intrinsics


def find_files(base_root):
    """
    Search the given base directory for the actual dataset root.
    This makes it a little easier for the dataset manager
    :param base_root:
    :return:
    """
    # These are folders we expect within the dataset folder structure. If we hit them, we've gone too far
    excluded_folders = {
        'cam0', 'cam1', 'imu0', 'imu1', 'leica0', 'vicon0', 'pointcloud0', 'state_groudtruth_estimate0',
        'data', '__MACOSX',
    }
    to_search = {base_root}
    while len(to_search) > 0:
        candidate_root = to_search.pop()

        # Make the derivative paths we're looking for. All of these must exist.
        left_rgb_path = os.path.join(candidate_root, 'cam0', 'data.csv')
        left_camera_intrinsics_path = os.path.join(candidate_root, 'cam0', 'sensor.yaml')
        right_rgb_path = os.path.join(candidate_root, 'cam1', 'data.csv')
        right_camera_intrinsics_path = os.path.join(candidate_root, 'cam1', 'sensor.yaml')
        trajectory_path = os.path.join(candidate_root, 'state_groundtruth_estimate0', 'data.csv')

        # If all the required files are present, return that root and the file paths.
        if (os.path.isfile(left_rgb_path) and os.path.isfile(left_camera_intrinsics_path) and
                os.path.isfile(right_rgb_path) and os.path.isfile(right_camera_intrinsics_path) and
                os.path.isfile(trajectory_path)):
            return (candidate_root, left_rgb_path, left_camera_intrinsics_path,
                    right_rgb_path, right_camera_intrinsics_path, trajectory_path)

        # This was not the directory we were looking for, search the subdirectories
        with os.scandir(candidate_root) as dir_iter:
            for dir_entry in dir_iter:
                if dir_entry.is_dir() and dir_entry.name not in excluded_folders:
                    to_search.add(dir_entry.path)

    # Could not find the necessary files to import, raise an exception.
    raise FileNotFoundError("Could not find a valid root directory within '{0}'".format(base_root))


# Different environment types for different datasets
environment_types = {
    'MH_01_easy': imeta.EnvironmentType.INDOOR,
    'MH_02_easy': imeta.EnvironmentType.INDOOR,
    'MH_03_medium': imeta.EnvironmentType.INDOOR,
    'MH_04_difficult': imeta.EnvironmentType.INDOOR,
    'MH_05_difficult': imeta.EnvironmentType.INDOOR,
    'V1_01_easy': imeta.EnvironmentType.INDOOR_CLOSE,
    'V1_02_medium': imeta.EnvironmentType.INDOOR_CLOSE,
    'V1_03_difficult': imeta.EnvironmentType.INDOOR_CLOSE,
    'V2_01_easy': imeta.EnvironmentType.INDOOR_CLOSE,
    'V2_02_medium': imeta.EnvironmentType.INDOOR_CLOSE,
    'V2_03_difficult': imeta.EnvironmentType.INDOOR_CLOSE
}


def import_dataset(root_folder, dataset_name, **_):
    """
    Load an Autonomous Systems Lab dataset into the database.
    See http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#downloads

    Some information drawn from the ethz_asl dataset tools, see: https://github.com/ethz-asl/dataset_tools
    :param root_folder: The body folder, containing body.yaml (i.e. the extracted mav0 folder)
    :param dataset_name: The name of the dataset, see the manager for the list of valid values.
    :return:
    """
    if not os.path.isdir(root_folder):
        raise NotADirectoryError("'{0}' is not a directory".format(root_folder))

    # Step 1: Find the various files containing the data (that's a 6-element tuple unpack for the return value)
    (
        root_folder,
        left_rgb_path, left_camera_intrinsics_path,
        right_rgb_path, right_camera_intrinsics_path,
        trajectory_path
    ) = find_files(root_folder)

    # Step 2: Read the meta-information from the files (that's a 6-element tuple unpack for the return value)
    left_image_files = read_image_filenames(left_rgb_path)
    left_extrinsics, left_intrinsics = get_camera_calibration(left_camera_intrinsics_path)
    right_image_files = read_image_filenames(left_rgb_path)
    right_extrinsics, right_intrinsics = get_camera_calibration(right_camera_intrinsics_path)
    trajectory = read_trajectory(trajectory_path, left_image_files.keys())

    # Step 3: Create stereo rectification matrices from the intrinsics
    left_x, left_y, left_intrinsics, right_x, right_y, right_intrinsics = rectify(
        left_extrinsics, left_intrinsics, right_extrinsics, right_intrinsics)

    # Change the coordinates correctly on the extrinsics. Has to happen after rectification
    left_extrinsics = fix_coordinates(left_extrinsics)
    right_extrinsics = fix_coordinates(right_extrinsics)

    # Step 4: Associate the different data types by timestamp. Trajectory last because it's bigger than the stereo.
    all_metadata = associate_data(left_image_files, right_image_files, trajectory)

    # Step 5: Load the images from the metadata
    first_timestamp = None
    images = []
    timestamps = []
    for timestamp, left_image_file, right_image_file, robot_pose in all_metadata:
        # Timestamps are in POSIX nanoseconds, re-zero them to the start of the dataset, and scale to seconds
        if first_timestamp is None:
            first_timestamp = timestamp
        timestamp = (timestamp - first_timestamp) / 1e9

        left_data = image_utils.read_colour(os.path.join(root_folder, 'cam0', 'data', left_image_file))
        right_data = image_utils.read_colour(os.path.join(root_folder, 'cam1', 'data', right_image_file))

        # Error check the loaded image data
        if left_data is None or left_data.size is 0:
            logging.getLogger(__name__).warning("Could not read left image \"{0}\", result is empty. Skipping.".format(
                os.path.join(root_folder, 'cam0', 'data', left_image_file)))
            continue
        if right_data is None or right_data.size is 0:
            logging.getLogger(__name__).warning("Could not read right image \"{0}\", result is empty. Skipping.".format(
                os.path.join(root_folder, 'cam1', 'data', right_image_file)))
            continue

        left_data = cv2.remap(left_data, left_x, left_y, cv2.INTER_LINEAR)
        right_data = cv2.remap(right_data, right_x, right_y, cv2.INTER_LINEAR)

        left_pose = robot_pose.find_independent(left_extrinsics)
        right_pose = robot_pose.find_independent(right_extrinsics)

        left_metadata = imeta.make_metadata(
            pixels=left_data,
            camera_pose=left_pose,
            intrinsics=left_intrinsics,
            source_type=imeta.ImageSourceType.REAL_WORLD,
            environment_type=environment_types.get(dataset_name, imeta.EnvironmentType.INDOOR_CLOSE),
            light_level=imeta.LightingLevel.WELL_LIT,
            time_of_day=imeta.TimeOfDay.DAY,
        )
        right_metadata = imeta.make_right_metadata(
            pixels=right_data,
            left_metadata=left_metadata,
            camera_pose=right_pose,
            intrinsics=right_intrinsics
        )
        image = StereoImage(
            pixels=left_data,
            right_pixels=right_data,
            metadata=left_metadata,
            right_metadata=right_metadata
        )
        image.save()
        images.append(image)
        timestamps.append(timestamp)

    # Create and save the image collection
    collection = ImageCollection(
        images=images,
        timestamps=timestamps,
        sequence_type=ImageSequenceType.SEQUENTIAL,
        dataset='EuRoC MAV',
        sequence_name=dataset_name,
        trajectory_id=dataset_name
    )
    collection.save()
    return collection
