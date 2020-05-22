# Copyright (c) 2020, John Skinner
import typing
import os.path
import tarfile
from pathlib import Path
import shutil
from json import loads as json_loads, load as json_load
import numpy as np
import arvet.util.image_utils as image_utils
from arvet.metadata.camera_intrinsics import CameraIntrinsics
import arvet.metadata.image_metadata as imeta
import arvet.util.transform as tf
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import StereoImage
from arvet.core.image_collection import ImageCollection
from arvet_slam.dataset.ndds.depth_noise import create_noisy_depth_image, DepthNoiseQuality


IMG_TEMPLATE = "{0:06}.png"
DATA_TEMPLATE = "{0:06}.json"
DEPTH_TEMPLATE = "{0:06}.depth.png"
INSTANCE_TEMPLATE = "{0:06}.is.png"


def import_dataset(root_folder, **_):
    """
    Load a dataset produced by the Nvidia dataset generator


    :return:
    """
    root_folder = Path(root_folder)

    # Step 0: Check the root folder to see if it needs to be extracted from a tarfile
    delete_when_done = None
    if not root_folder.is_dir():
        if tarfile.is_tarfile(root_folder):
            delete_when_done = root_folder.parent / (root_folder.name.split('.')[0] + '-temp')
            with tarfile.open(root_folder) as tar_fp:
                tar_fp.extractall(delete_when_done)
            root_folder = delete_when_done
        else:
            # Could find neither a dir nor a tarfile to extract from
            raise NotADirectoryError("'{0}' is not a directory or a tarfile we can extract".format(root_folder))

    root_folder, left_path, right_path = find_files(root_folder)
    collection = import_stereo(root_folder, left_path, right_path)

    if delete_when_done is not None and delete_when_done.exists():
        # We're done and need to clean up after ourselves
        shutil.rmtree(delete_when_done)

    return collection


def find_files(base_root: Path) -> typing.Tuple[Path, Path, Path]:
    """
    Search the given base directory for the actual dataset root.
    This makes it a little easier for the dataset manager.
    Handles finding either a stereo root or a single root

    :param base_root: The place to start searching
    :return: The root folder, and the left and right folders
    """
    required_root_files = {
        'settings.json',
        'timestamps.json'
    }
    required_sequence_files = {
        '_object_settings.json',
        '_camera_settings.json',
    }
    excluded_folders = {'left', 'right'}
    to_search = {Path(base_root)}
    while len(to_search) > 0:
        candidate_root = to_search.pop()

        # Check if there is a stereo sequence here
        candidate_left = candidate_root / 'left'
        candidate_right = candidate_root / 'right'
        if candidate_left.exists() and candidate_right.exists() \
                and all((candidate_root / req_file).is_file() for req_file in required_root_files) \
                and all((candidate_left / req_file).is_file() for req_file in required_sequence_files) \
                and all((candidate_right / req_file).is_file() for req_file in required_sequence_files):
            return candidate_root, candidate_left, candidate_right

        # This was not the directory we were looking for, search the subdirectories
        for child_path in candidate_root.iterdir():
            if child_path.is_dir() and child_path.name not in excluded_folders:
                to_search.add(child_path)

    # Could not find the necessary files to import, raise an exception.
    raise FileNotFoundError("Could not find a valid root directory within '{0}'".format(base_root))


def import_stereo(root_folder, left_path: Path, right_path: Path) -> ImageCollection:
    """
    Import the sequence as a stereo
    :param root_folder:
    :param left_path:
    :param right_path:
    :return:
    """
    # Read the timestamps and the generator settings
    # These are saved from python, so need for special loading
    with (root_folder / 'settings.json').open('r') as fp:
        settings = json_load(fp)
    with (root_folder / 'timestamps.json').open('r') as fp:
        timestamps = json_load(fp)

    # Read the camera settings from file
    left_camera_intrinsics = read_camera_intrinsics(left_path / '_camera_settings.json')
    right_camera_intrinsics = read_camera_intrinsics(right_path / '_camera_settings.json')
    left_object_labels = read_object_classes(left_path / '_object_settings.json')
    right_object_labels = read_object_classes(right_path / '_object_settings.json')

    max_img_id = min(
        find_max_img_id(lambda idx: left_path / IMG_TEMPLATE.format(idx)),
        find_max_img_id(lambda idx: right_path / IMG_TEMPLATE.format(idx)),
    )
    if len(timestamps) != max_img_id:
        raise RuntimeError(f"Maximum image id {max_img_id} didn't match the number "
                           f"of available timestamps ({len(timestamps)}), cannot associate.")

    # Read meta-information from the generator, including the timestamps
    sequence_name = root_folder.name
    (
        trajectory_id, environment_type, light_level, time_of_day, simulation_world,
        lighting_model, texture_mipmap_bias, normal_maps_enabled, roughness_enabled, geometry_decimation, depth_quality
    ) = parse_settings(settings)

    # Import all the images
    images = []
    origin = None
    for img_idx in range(max_img_id):
        # Read the raw data for the left image
        left_frame_data = read_json(left_path / DATA_TEMPLATE.format(img_idx))
        left_pixels = image_utils.read_colour(left_path / IMG_TEMPLATE.format(img_idx))
        left_label_image = image_utils.read_colour(left_path / INSTANCE_TEMPLATE.format(img_idx))
        left_ground_truth_depth = image_utils.read_depth(left_path / DEPTH_TEMPLATE.format(img_idx))

        # Read the raw data for the right image
        right_frame_data = read_json(right_path / DATA_TEMPLATE.format(img_idx))
        right_pixels = image_utils.read_colour(right_path / IMG_TEMPLATE.format(img_idx))
        right_label_image = image_utils.read_colour(right_path / INSTANCE_TEMPLATE.format(img_idx))
        right_ground_truth_depth = image_utils.read_depth(right_path / DEPTH_TEMPLATE.format(img_idx))

        # Extract the poses and labels
        left_camera_pose = read_camera_pose(left_frame_data)
        left_labelled_objects = find_labelled_objects(left_label_image, left_frame_data, left_object_labels)
        right_camera_pose = read_camera_pose(right_frame_data)
        right_labelled_objects = find_labelled_objects(right_label_image, right_frame_data, right_object_labels)

        # Compute a noisy depth image
        noisy_depth = create_noisy_depth_image(
            left_ground_truth_depth=left_ground_truth_depth,
            right_ground_truth_depth=right_ground_truth_depth,
            camera_intrinsics=left_camera_intrinsics,
            right_camera_relative_pose=left_camera_pose.find_relative(right_camera_pose),
            quality_level=depth_quality
        )

        # Re-centre the camera poses relative to the first frame
        if origin is None:
            origin = left_camera_pose
        left_camera_pose = origin.find_relative(left_camera_pose)
        right_camera_pose = origin.find_relative(right_camera_pose)

        left_metadata = imeta.make_metadata(
            pixels=left_pixels,
            depth=left_ground_truth_depth,
            camera_pose=left_camera_pose,
            intrinsics=left_camera_intrinsics,
            source_type=imeta.ImageSourceType.SYNTHETIC,
            environment_type=environment_type,
            light_level=light_level,
            time_of_day=time_of_day,
            simulation_world=simulation_world,
            lighting_mode=lighting_model,
            texture_mipmap_bias=texture_mipmap_bias,
            normal_maps_enabled=normal_maps_enabled,
            roughness_enabled=roughness_enabled,
            geometry_decimation=geometry_decimation,
            labelled_objects=left_labelled_objects
        )
        right_metadata = imeta.make_right_metadata(
            pixels=right_pixels,
            depth=right_ground_truth_depth,
            camera_pose=right_camera_pose,
            intrinsics=right_camera_intrinsics,
            labelled_objects=right_labelled_objects,
            left_metadata=left_metadata
        )
        image = StereoImage(
            pixels=left_pixels,
            right_pixels=right_pixels,
            depth=noisy_depth,
            ground_truth_depth=left_ground_truth_depth,
            right_ground_truth_depth=right_ground_truth_depth,
            metadata=left_metadata,
            right_metadata=right_metadata,
        )
        image.save()
        images.append(image)

    # Create and save the image collection
    collection = ImageCollection(
        images=images,
        timestamps=timestamps,
        sequence_type=ImageSequenceType.SEQUENTIAL,
        dataset="generated-SLAM-data",
        sequence_name=sequence_name,
        trajectory_id=trajectory_id
    )
    collection.save()
    return collection


def read_camera_intrinsics(settings_path: Path) -> CameraIntrinsics:
    """
    Read the camera intrinsics from file
    :param settings_path: The path to the camera settings file, output by the NDDS exporter
    :return:
    """
    # TODO: There is some inconsistency in the output settings I've got, particulary between the FOV and intrinsics.
    # Check how they're being generated and output, and ensure that the camera actually has those settings
    camera_settings = read_json(settings_path)
    return CameraIntrinsics(
        width=int(camera_settings['camera_settings']['intrinsic_settings']['resX']),
        height=int(camera_settings['camera_settings']['intrinsic_settings']['resY']),
        fx=float(camera_settings['camera_settings']['intrinsic_settings']['fx']),
        fy=float(camera_settings['camera_settings']['intrinsic_settings']['fy']),
        cx=float(camera_settings['camera_settings']['intrinsic_settings']['cx']),
        cy=float(camera_settings['camera_settings']['intrinsic_settings']['cy']),
        s=float(camera_settings['camera_settings']['intrinsic_settings']['s'])
    )


def read_object_classes(object_settings_path: Path) -> typing.Mapping[int, dict]:
    # Read the object settings for the list of instance ids
    object_settings = read_json(object_settings_path)
    return {
        obj_data['segmentation_instance_id']: obj_data
        for obj_data in object_settings['exported_objects']
    }


def parse_settings(settings: dict) -> typing.Tuple[
    str, imeta.EnvironmentType, imeta.LightingLevel, imeta.TimeOfDay, str, imeta.LightingModel,
    int, bool, bool, int, DepthNoiseQuality
]:
    """
    Read values from the settings saved from the generator.
    Format is:
    {
        'map': map_name,
        'trajectory_id': str(trajectory_id),
        'light_level': lighting_level.name,
        'light_model': lighting_model.name,
        'origin': origin.serialise(),
        'left_intrinsics': left_intrinsics.serialise(),
        'right_intrinsics': right_intrinsics.serialise(),
        'motion_blur': min(1.0, max(0.0, float(motion_blur))),
        'exposure': [float(min(exposure)), float(max(exposure))] if exposure is not None else None,
        'aperture': max(1.0, float(aperture)),
        'focal_distance': max(0.0, float(focal_distance)),
        'grain': max(0.0, float(grain)),
        'vignette': max(0.0, float(vignette)),
        'lens_flare': max(0.0, float(lens_flare)),
        'depth_quality': depth_quality.name
    }

    :param settings:
    :return:
    """
    map_name = settings['map_name']
    trajectory_id = str(settings['trajectory_id'])
    environment_type = imeta.EnvironmentType.INDOOR

    # TODO: We need to hard-code some relationships between map name and some of these settings
    if settings['light_level'].lower() == 'day':
        light_level = imeta.LightingLevel.WELL_LIT
        time_of_day = imeta.TimeOfDay.DAY
    else:
        light_level = imeta.LightingLevel.DIM
        time_of_day = imeta.TimeOfDay.NIGHT

    # TODO: We also need to unify betwen what settings we're pulling here, and what we can affect in the simulator
    if settings['light_model'].lower() == 'unlit':
        lighting_model = imeta.LightingModel.UNLIT
    else:
        lighting_model = imeta.LightingModel.LIT

    # Parse the image image depth quality
    try:
        depth_quality = DepthNoiseQuality[settings['depth_quality']]
    except KeyError:
        depth_quality = DepthNoiseQuality.NO_NOISE

    # TODO: Make these settings do something
    texture_mipmap_bias = 0
    normal_maps_enabled = True
    roughness_enabled = True
    geometry_decimation = 0

    return (
        trajectory_id, environment_type, light_level, time_of_day, map_name,
        lighting_model, texture_mipmap_bias, normal_maps_enabled, roughness_enabled, geometry_decimation, depth_quality
    )


def find_labelled_objects(mask_img: np.ndarray, frame_data: dict, object_data_by_id: typing.Mapping[int, dict]) -> \
        typing.List[imeta.LabelledObject]:
    """

    :param mask_img:
    :param frame_data:
    :param object_data_by_id:
    :return:
    """
    # Get lists of known instance ids and their corresponding colours in the mask image
    known_ids = sorted(object_data_by_id.keys())
    known_colours = np.array([encode_id(instance_id) for instance_id in known_ids])

    # lookup objects frame data by name
    frame_data_by_name = {
        obj_data['name']: obj_data
        for obj_data in frame_data['objects']
    }

    # Now that we can get the masks, build the labels
    labelled_objects = []
    for colour, object_mask in find_unique_colours(mask_img):
        label_id = find_nearest_id_by_colour(colour, known_colours, known_ids)
        if label_id in object_data_by_id:
            object_data = object_data_by_id[label_id]
            bounding_box = generate_bounding_box_from_mask(object_mask)
            if bounding_box is not None:
                x_min, y_min, x_max, y_max = bounding_box
                labelled_object = imeta.MaskedObject(
                    instance_name=object_data['name'],
                    class_names=[object_data['class']],
                    x=x_min,
                    y=y_min,
                    mask=mask_img[x_min:x_max + 1, y_min:y_max + 1]
                )
                if object_data['name'] in frame_data_by_name:
                    labelled_object.relative_pose = read_relative_pose(frame_data_by_name[object_data['name']])
                labelled_objects.append(labelled_object)
    return labelled_objects


def find_max_img_id(make_filename=None):
    """
    Helper to find the largest image id in a given folder.
    We assume that no intermediate ids are missing, and ids start at 0.
    We simply need to find the largest index that exists.
    :param make_filename: A callable to make a filename from an id
    :return:
    """
    min_id = 0
    max_id = 1

    # Expand the search exponentially
    while os.path.isfile(make_filename(max_id)):
        min_id = max_id
        max_id *= 2

    # Binary search between limits
    while max_id - min_id > 1:
        half = (max_id + min_id) // 2
        if os.path.isfile(make_filename(half)):
            min_id = half
        else:
            max_id = half

    return min_id


def encode_id(instance_id):
    """
    Convert an instance id to mask colour
    This matches the encoding done in the dataset renderer, see
    https://github.com/jskinn/Dataset_Synthesizer/blob/local-devel/Source/Plugins/NVSceneCapturer/Source/NVSceneCapturer/Private/NVSceneCapturerUtils.cpp#L673
    :param instance_id:
    :return:
    """
    return [
        (instance_id << 1) & 254,
        (instance_id >> 6) & 254,
        (instance_id >> 13) & 254
    ]


def find_unique_colours(mask_img: np.ndarray) \
        -> typing.Generator[typing.Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator for the
    :param mask_img: The mask image, as a 3-channel colour image
    :return: Generator over unique colours in the image and masks
    """
    # Getting unique colours is much faster if we flatten the channels down to a single 32 bit integer
    # Note that this is distinct from the encode/decode done in encode_id
    mask_im = mask_img.astype(np.uint32)
    integer_image = (mask_im[:, :, 2] << 16) | (mask_im[:, :, 1] << 8) | mask_im[:, :, 0]
    for integer_colour in np.unique(integer_image):
        # Decode the integer back to a colour
        colour = np.array([
            integer_colour & 255,
            (integer_colour >> 8) & 255,
            (integer_colour >> 16) & 255
        ])
        yield colour, (integer_image == integer_colour)


def find_nearest_id_by_colour(search_colour: np.ndarray, known_colours, known_ids):
    """
    Find the nearest colour to
    :param search_colour:
    :param known_colours:
    :param known_ids:
    :return:
    """
    diffs = known_colours - search_colour
    square_dists = [np.dot(diff, diff) for diff in diffs]
    best_idx = int(np.argmin(square_dists))
    if square_dists[best_idx] < 4:
        return known_ids[best_idx]
    return None


def generate_bounding_box_from_mask(mask: np.ndarray) -> typing.Union[typing.Tuple[int, int, int, int], None]:
    """

    :param mask:
    :return:
    """
    flat_x = np.any(mask, axis=0)
    flat_y = np.any(mask, axis=1)
    if not np.any(flat_x) and not np.any(flat_y):
        print("No positive pixels found, returning None")
        return None
    xmin = np.argmax(flat_x)
    ymin = np.argmax(flat_y)
    xmax = len(flat_x) - 1 - np.argmax(flat_x[::-1])
    ymax = len(flat_y) - 1 - np.argmax(flat_y[::-1])
    return int(xmin), int(ymin), int(xmax), int(ymax)


def read_camera_pose(frame_data: dict) -> tf.Transform:
    """
    Read the camera pose from the frame data
    :param frame_data:
    :return:
    """
    camera_data = frame_data['camera_data']
    x, y, z = camera_data['location_worldframe']
    qx, qy, qz, qw = camera_data['quaternion_xyzw_worldframe']
    return make_camera_pose(x, y, z, qw, qx, qy, qz)


def read_relative_pose(object_frame_data: dict) -> tf.Transform:
    """
    Read the pose of an object relative to the camera, from the frame data
    :param object_frame_data: The frame data dict from the matching object in the objects array
    :return: The relative pose of the object, as a Transform
    """
    tx, ty, tz = object_frame_data['location']
    qx, qy, qz, qw = object_frame_data['quaternion_xyzw']
    return make_camera_pose(tx, ty, tz, qw, qx, qy, qz)


def make_camera_pose(tx, ty, tz, qw, qx, qy, qz) -> tf.Transform:
    """
    Swap the coordinate frames from unreal coordinates
    to my standard convention.
    :param tx:
    :param ty:
    :param tz:
    :param qw:
    :param qx:
    :param qy:
    :param qz:
    :return: A Transform object
    """
    # Flip the Y translation and rescale
    location = (tx / 100, -ty / 100, tz / 100)

    # Fiddle the rotation to convert from left handed to right handed
    # This is a combination of flipping the Y axis, and the quaternion inverse (w, -i, -j, -k)
    rotation = np.array([qw, -qx, qy, -qz])
    return tf.Transform(location=location, rotation=rotation, w_first=True)


def read_json(path: Path) -> dict:
    """
    A wrapper around json_load to handle certain string replacements between UE4 json and regular json.
    in particular, there are problems around encoding float NaN and Inf
    :param path:
    :return:
    """
    with path.open('r') as fp:
        json_string = fp.read()
    return json_loads(
        json_string.replace(' -nan(ind)', ' NaN').replace(' nan(ind)', ' NaN')
        .replace(' nan', ' NaN').replace(' -nan', ' NaN')
        .replace(' inf', ' Infinity').replace(' -inf', ' -Infinity')
    )
