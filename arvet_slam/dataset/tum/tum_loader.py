# Copyright (c) 2017, John Skinner
import typing
import os.path
import tarfile
from pathlib import Path
import shutil
import arvet.util.image_utils as image_utils
from arvet.util.associate import associate
import arvet.database.image_manager
from arvet.metadata.camera_intrinsics import CameraIntrinsics
import arvet.metadata.image_metadata as imeta
import arvet.util.transform as tf
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import Image
from arvet.core.image_collection import ImageCollection
from arvet_slam.util.trajectory_builder import TrajectoryBuilder


# Different environment types for different datasets
# Determined manually by looking at the sequences
environment_types = {
    'rgbd_dataset_freiburg1_360': imeta.EnvironmentType.INDOOR_CLOSE,
    'rgbd_dataset_freiburg1_desk': imeta.EnvironmentType.INDOOR_CLOSE,
    'rgbd_dataset_freiburg2_dishes': imeta.EnvironmentType.INDOOR,
    'rgbd_dataset_freiburg3_structure_texture_far': imeta.EnvironmentType.INDOOR,
}


def make_camera_pose(tx: float, ty: float, tz: float, qw: float, qx: float, qy: float, qz: float) -> tf.Transform:
    """
    TUM dataset use a different coordinate frame to the one I'm using, which is the same as the Libviso2 frame.
    This function is to convert dataset ground-truth poses to transform objects.
    Thankfully, its still a right-handed coordinate frame, which makes this easier.
    Frame is: z forward, y right, x down

    :param tx: The x coordinate of the location
    :param ty: The y coordinate of the location
    :param tz: The z coordinate of the location
    :param qx: The x part of the quaternion orientation
    :param qy: The y part of the quaternion orientation
    :param qz: The z part of the quaternion orientation
    :param qw: The scalar part of the quaternion orientation
    :return: A Transform object representing the world pose of the current frame
    """
    return tf.Transform(
        location=(tz, -tx, -ty),
        rotation=(qw, qz, -qx, -qy),
        w_first=True
    )


def read_image_filenames(images_file_path: typing.Union[str, bytes, os.PathLike]):
    """
    Read image filenames from a file
    :param images_file_path:
    :return:
    """
    filename_map = {}
    with open(images_file_path, 'r') as images_file:
        for line in images_file:
            comment_idx = line.find('#')
            if comment_idx >= 0:
                # This line contains a comment, remove everything after it
                line = line[:comment_idx]
            line = line.strip()
            if len(line) > 0:
                parts = line.split(' ')
                if len(parts) >= 2:
                    timestamp, relative_path = parts[0:2]
                    filename_map[float(timestamp)] = relative_path.rstrip()  # To remove trailing newlines
    return filename_map


def read_trajectory(trajectory_filepath: typing.Union[str, bytes, os.PathLike], timestamps: typing.Iterable[float]) -> \
        typing.Mapping[float, tf.Transform]:
    """
    Read the ground-truth camera trajectory from file.
    Needs the list of

    :param trajectory_filepath: The path to the trajectory file
    :param timestamps: A list of desired camera timestamps to match against.
    Ground truth timestamps will be interpolated to these times.
    :return: A map of timestamp to camera pose.
    """
    builder = TrajectoryBuilder(timestamps)
    with open(trajectory_filepath, 'r') as trajectory_file:
        for line in trajectory_file:
            comment_idx = line.find('#')
            if comment_idx >= 0:
                # This line contains a comment, remove it
                line = line[:comment_idx]
            line = line.strip()
            if len(line) > 0:
                parts = line.split(' ')
                if len(parts) >= 8:
                    timestamp, tx, ty, tz, qx, qy, qz, qw = parts[0:8]
                    timestamp = float(timestamp)
                    pose = make_camera_pose(float(tx), float(ty), float(tz),
                                            float(qw), float(qx), float(qy), float(qz))
                    builder.add_trajectory_point(timestamp, pose)
    return builder.get_interpolated_trajectory()


def associate_data(root_map, *args):
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
            matches = associate(root_map, other_map, offset=0, max_difference=1)
            rekeyed_map = {root_key: other_map[other_key] for root_key, other_key in matches}
            root_keys &= set(rekeyed_map.keys())
            rekeyed_maps.append(rekeyed_map)
        return sorted([key, root_map[key]] + [rekeyed_map[key] for rekeyed_map in rekeyed_maps]
                      for key in root_keys)


def get_camera_intrinsics(folder_path: typing.Union[str, bytes, os.PathLike]):
    folder_path = str(folder_path).lower()
    if 'freiburg1' in folder_path:
        return CameraIntrinsics(
            width=640,
            height=480,
            fx=517.3,
            fy=516.5,
            cx=318.6,
            cy=255.3,
            k1=0.2624,
            k2=-0.9531,
            k3=1.1633,
            p1=-0.0054,
            p2=0.0026
        )
    elif 'freiburg2' in folder_path:
        return CameraIntrinsics(
            width=640,
            height=480,
            fx=580.8,
            fy=581.8,
            cx=308.8,
            cy=253.0,
            k1=-0.2297,
            k2=1.4766,
            k3=-3.4194,
            p1=0.0005,
            p2=-0.0075
        )
    elif 'freiburg3' in folder_path:
        return CameraIntrinsics(
            width=640,
            height=480,
            fx=535.4,
            fy=539.2,
            cx=320.1,
            cy=247.6
        )
    else:
        # Default to ROS parameters
        return CameraIntrinsics(
            width=640,
            height=480,
            fx=525.0,
            fy=525.0,
            cx=319.5,
            cy=239.5
        )


def find_files(base_root: Path):
    """
    Search the given base directory for the actual dataset root.
    This makes it a little easier for the dataset manager
    :param base_root:
    :return:
    """
    # These are folders we expect within the dataset folder structure. If we hit them, we've gone too far
    excluded_folders = {'depth', 'rgb', '__MACOSX'}
    to_search = {Path(base_root)}
    while len(to_search) > 0:
        candidate_root = to_search.pop()

        # Make the derivative paths we're looking for. All of these must exist.
        rgb_path = candidate_root / 'rgb.txt'
        trajectory_path = candidate_root / 'groundtruth.txt'
        depth_path = candidate_root / 'depth.txt'

        # If all the required files are present, return that root and the file paths.
        if rgb_path.is_file() and trajectory_path.is_file() and depth_path.is_file():
            return candidate_root, rgb_path, depth_path, trajectory_path

        # This was not the directory we were looking for, search the subdirectories
        for child_path in candidate_root.iterdir():
            if child_path.is_dir() and child_path.name not in excluded_folders:
                to_search.add(child_path)

    # Could not find the necessary files to import, raise an exception.
    raise FileNotFoundError("Could not find a valid root directory within '{0}'".format(base_root))


def import_dataset(root_folder, dataset_name, **_):
    """
    Load a TUM RGB-D sequence into the database.


    :return:
    """
    root_folder = Path(root_folder)

    # Step 0: Check the root folder to see if it needs to be extracted from a tarfile
    delete_when_done = None
    if not root_folder.is_dir():
        if (root_folder.parent / dataset_name).is_dir():
            # The root was a tarball, but the extracted data already exists, just use that as the root
            root_folder = root_folder.parent / dataset_name
        else:
            candidate_tar_file = root_folder.parent / (dataset_name + '.tgz')
            if candidate_tar_file.is_file() and tarfile.is_tarfile(candidate_tar_file):
                # Root is actually a tarfile, extract it. find_roots with handle folder structures
                with tarfile.open(candidate_tar_file) as tar_fp:
                    tar_fp.extractall(root_folder.parent / dataset_name)
                root_folder = root_folder.parent / dataset_name
                delete_when_done = root_folder
            else:
                # Could find neither a dir nor a tarfile to extract from
                raise NotADirectoryError("'{0}' is not a directory".format(root_folder))

    # Step 1: Find the relevant metadata files
    root_folder, rgb_path, depth_path, trajectory_path = find_files(root_folder)

    # Step 2: Read the metadata from them
    image_files = read_image_filenames(rgb_path)
    trajectory = read_trajectory(trajectory_path, image_files.keys())
    depth_files = read_image_filenames(depth_path)

    # Step 3: Associate the different data types by timestamp
    all_metadata = associate_data(image_files, trajectory, depth_files)

    # Step 3: Load the images from the metadata
    first_timestamp = None
    image_group = dataset_name
    images = []
    timestamps = []
    with arvet.database.image_manager.get().get_group(image_group, allow_write=True):
        for timestamp, image_file, camera_pose, depth_file in all_metadata:
            # Re-zero the timestamps
            if first_timestamp is None:
                first_timestamp = timestamp
            timestamp = (timestamp - first_timestamp)

            rgb_data = image_utils.read_colour(os.path.join(root_folder, image_file))
            depth_data = image_utils.read_depth(os.path.join(root_folder, depth_file))
            depth_data = depth_data / 5000  # Re-scale depth to meters
            camera_intrinsics = get_camera_intrinsics(root_folder)

            metadata = imeta.make_metadata(
                pixels=rgb_data,
                depth=depth_data,
                camera_pose=camera_pose,
                intrinsics=camera_intrinsics,
                source_type=imeta.ImageSourceType.REAL_WORLD,
                environment_type=environment_types.get(dataset_name, imeta.EnvironmentType.INDOOR_CLOSE),
                light_level=imeta.LightingLevel.WELL_LIT,
                time_of_day=imeta.TimeOfDay.DAY,
            )
            image = Image(
                pixels=rgb_data,
                depth=depth_data,
                image_group=image_group,
                metadata=metadata
            )
            image.save()
            images.append(image)
            timestamps.append(timestamp)

    # Create and save the image collection
    collection = ImageCollection(
        images=images,
        timestamps=timestamps,
        sequence_type=ImageSequenceType.SEQUENTIAL,
        dataset='TUM RGB-D',
        sequence_name=dataset_name,
        trajectory_id=dataset_name
    )
    collection.save()

    if delete_when_done is not None and delete_when_done.exists():
        # We're done and need to clean up after ourselves
        shutil.rmtree(delete_when_done)

    return collection
