# Copyright (c) 2020, John Skinner
import numpy as np
import typing
from pathlib import Path, PurePath
import logging
import cv2
import xxhash
import arvet.util.image_utils as image_utils
import arvet.database.image_manager
from arvet.core.image_collection import ImageCollection
import arvet_slam.dataset.euroc.euroc_loader as euroc_loader


def verify_dataset(image_collection: ImageCollection, root_folder: typing.Union[str, PurePath],
                   dataset_name: str, repair: bool = False):
    """
    Examine an existing Autonomous Systems Lab dataset in the database, and check if for errors
    See http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#downloads

    Some information drawn from the ethz_asl dataset tools, see: https://github.com/ethz-asl/dataset_tools
    :param image_collection: The existing image collection from the loader
    :param root_folder: The body folder, containing body.yaml (i.e. the extracted mav0 folder)
    :param dataset_name: The name of the dataset, see the manager for the list of valid values.
    :param repair: If possible, fix missing images in the dataset
    :return:
    """
    root_folder = Path(root_folder)
    dataset_name = str(dataset_name)
    repair = bool(repair)
    if not root_folder.is_dir():
        raise NotADirectoryError("'{0}' is not a directory".format(root_folder))
    image_group = dataset_name
    valid = True
    irreparable = False

    # Check the image group on the image collection
    if image_collection.image_group != image_group:
        if repair:
            image_collection.image_group = image_group
            image_collection.save()
            logging.getLogger(__name__).info(
                f"Fixed incorrect image group for {image_collection.sequence_name}")
        else:
            logging.getLogger(__name__).warning(
                f"{image_collection.sequence_name} has incorrect image group {image_group}")
            valid = False

    # Find the various files containing the data (that's a 6-element tuple unpack for the return value)
    (
        root_folder,
        left_rgb_path, left_camera_intrinsics_path,
        right_rgb_path, right_camera_intrinsics_path,
        trajectory_path
    ) = euroc_loader.find_files(root_folder)

    # Read the meta-information from the files (that's a 6-element tuple unpack for the return value)
    left_image_files = euroc_loader.read_image_filenames(left_rgb_path)
    left_extrinsics, left_intrinsics = euroc_loader.get_camera_calibration(left_camera_intrinsics_path)
    right_image_files = euroc_loader.read_image_filenames(left_rgb_path)
    right_extrinsics, right_intrinsics = euroc_loader.get_camera_calibration(right_camera_intrinsics_path)

    # Create stereo rectification matrices from the intrinsics
    left_x, left_y, left_intrinsics, right_x, right_y, right_intrinsics = euroc_loader.rectify(
        left_extrinsics, left_intrinsics, right_extrinsics, right_intrinsics)

    # Associate the different data types by timestamp. Trajectory last because it's bigger than the stereo.
    all_metadata = euroc_loader.associate_data(left_image_files, right_image_files)

    # Load the images from the metadata
    total_invalid_images = 0
    total_fixed_images = 0
    image_index = 0
    with arvet.database.image_manager.get().get_group(image_group, allow_write=repair):
        for timestamp, left_image_file, right_image_file in all_metadata:
            changed = False
            img_valid = True
            # Skip if we've hit the end of the data
            if image_index >= len(image_collection):
                logging.getLogger(__name__).error(f"Image {image_index} is missing from the dataset")
                irreparable = True
                valid = False
                total_invalid_images += 1
                continue

            left_img_path = root_folder / 'cam0' / 'data' / left_image_file
            right_img_path = root_folder / 'cam1' / 'data' / right_image_file
            left_pixels = image_utils.read_colour(left_img_path)
            right_pixels = image_utils.read_colour(right_img_path)

            # Error check the loaded image data
            # The EuRoC Sequences MH_04_difficult and V2_03_difficult are missing the first right frame
            # So we actually start loading from
            # In general, frames that are missing are skipped, and do not increment image index
            if left_pixels is None or left_pixels.size is 0:
                logging.getLogger(__name__).warning(
                    f"Could not read left image \"{left_img_path}\", result is empty. Image is skipped.")
                continue
            if right_pixels is None or right_pixels.size is 0:
                logging.getLogger(__name__).warning(
                    f"Could not read right image \"{right_img_path}\", result is empty. Image is skipped.")
                continue

            left_pixels = cv2.remap(left_pixels, left_x, left_y, cv2.INTER_LINEAR)
            right_pixels = cv2.remap(right_pixels, right_x, right_y, cv2.INTER_LINEAR)
            left_hash = bytes(xxhash.xxh64(left_pixels).digest())
            right_hash = bytes(xxhash.xxh64(right_pixels).digest())

            # Load the image from the database
            try:
                _, image = image_collection[image_index]
            except (KeyError, IOError, RuntimeError):
                logging.getLogger(__name__).exception(f"Error loading image object {image_index}")
                valid = False
                image_index += 1    # Index is valid, increment when done
                continue

            # First, check the image group
            if image.image_group != image_group:
                if repair:
                    image.image_group = image_group
                    changed = True
                logging.getLogger(__name__).warning(f"Image {image_index} has incorrect group {image.image_group}")
                valid = False
                img_valid = False

            # Load the pixels from the image
            try:
                left_actual_pixels = image.left_pixels
            except (KeyError, IOError, RuntimeError):
                left_actual_pixels = None
            try:
                right_actual_pixels = image.right_pixels
            except (KeyError, IOError, RuntimeError):
                right_actual_pixels = None

            # Compare the loaded image data to the data read from disk
            if left_actual_pixels is None or not np.array_equal(left_pixels, left_actual_pixels):
                if repair:
                    image.store_pixels(left_pixels)
                    changed = True
                else:
                    logging.getLogger(__name__).error(
                        f"Image {image_index}: Left pixels do not match data read from {left_img_path}")
                img_valid = False
                valid = False
            if left_hash != bytes(image.metadata.img_hash):
                if repair:
                    image.metadata.img_hash = left_hash
                    changed = True
                else:
                    logging.getLogger(__name__).error(f"Image {image_index}: Left hash does not match metadata")
                valid = False
                img_valid = False
            if right_actual_pixels is None or not np.array_equal(right_pixels, right_actual_pixels):
                if repair:
                    image.store_right_pixels(right_pixels)
                    changed = True
                else:
                    logging.getLogger(__name__).error(
                        f"Image {image_index}: Right pixels do not match data read from {right_img_path}")
                valid = False
                img_valid = False
            if right_hash != bytes(image.right_metadata.img_hash):
                if repair:
                    image.right_metadata.img_hash = right_hash
                    changed = True
                else:
                    logging.getLogger(__name__).error(f"Image {image_index}: Right hash does not match metadata")
                valid = False
                img_valid = False
            if changed and repair:
                logging.getLogger(__name__).warning(f"Image {image_index}: repaired")
                image.save()
                total_fixed_images += 1
            if not img_valid:
                total_invalid_images += 1
            image_index += 1

    if irreparable:
        # Images are missing entirely, needs re-import
        logging.getLogger(__name__).error(f"Image Collection {image_collection.pk} for sequence {dataset_name} "
                                          "is IRREPARABLE, invalidate and re-import")
    elif repair:
        # Re-save the modified image collection
        logging.getLogger(__name__).info(f"{image_collection.sequence_name} repaired successfully "
                                         f"({total_fixed_images} image files fixed).")
    elif valid:
        logging.getLogger(__name__).info(f"Verification of {image_collection.sequence_name} successful.")
    else:
        logging.getLogger(__name__).error(
            f"Verification of {image_collection.sequence_name} ({image_collection.pk}) "
            f"FAILED, ({total_invalid_images} images failed)")
    return valid
