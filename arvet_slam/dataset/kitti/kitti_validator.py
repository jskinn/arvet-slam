# Copyright (c) 2020, John Skinner
import numpy as np
import logging
from pathlib import Path
import pykitti
import xxhash
import arvet.database.image_manager
from arvet.core.image_collection import ImageCollection
import arvet_slam.dataset.kitti.kitti_loader as kitti_loader


def verify_dataset(image_collection: ImageCollection, root_folder: Path, sequence_number: int, repair: bool = False):
    """
    Load a KITTI image sequences into the database.
    :return:
    """
    sequence_number = int(sequence_number)
    repair = bool(repair)
    if not 0 <= sequence_number < 11:
        raise ValueError("Cannot import sequence {0}, it is invalid".format(sequence_number))
    root_folder = kitti_loader.find_root(root_folder, sequence_number)
    data = pykitti.odometry(root_folder, sequence="{0:02}".format(sequence_number))
    image_group = f"KITTI_{sequence_number:06}"
    valid = True
    irreparable = False

    # Check the Image Collection
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

    # dataset.calib:      Calibration data are accessible as a named tuple
    # dataset.timestamps: Timestamps are parsed into a list of timedelta objects
    # dataset.poses:      Generator to load ground truth poses T_w_cam0
    # dataset.camN:       Generator to load individual images from camera N
    # dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
    # dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
    # dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]
    total_invalid_images = 0
    total_fixed_images = 0
    with arvet.database.image_manager.get().get_group(image_group, allow_write=repair):
        for img_idx, (left_image, right_image, timestamp, pose) in enumerate(
                zip(data.cam2, data.cam3, data.timestamps, data.poses)):
            changed = False
            img_valid = True
            if img_idx >= len(image_collection):
                logging.getLogger(__name__).error(f"Image {img_idx} is missing from the dataset")
                irreparable = True
                valid = False
                continue

            left_image = np.array(left_image)
            right_image = np.array(right_image)
            left_hash = xxhash.xxh64(left_image).digest()
            right_hash = xxhash.xxh64(right_image).digest()

            # Load the image object from the database
            try:
                _, image = image_collection[img_idx]
            except (KeyError, IOError, RuntimeError):
                logging.getLogger(__name__).exception(f"Error loading image object {img_idx}")
                valid = False
                total_invalid_images += 1
                continue

            # First, check the image group
            if image.image_group != image_group:
                if repair:
                    image.image_group = image_group
                    changed = True
                logging.getLogger(__name__).warning(f"Image {img_idx} has incorrect group {image.image_group}")
                valid = False

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
            if left_actual_pixels is None or not np.array_equal(left_image, left_actual_pixels):
                if repair:
                    image.store_pixels(left_image)
                    changed = True
                else:
                    logging.getLogger(__name__).error(f"Image {img_idx}: Left pixels do not match data read from disk")
                valid = False
                img_valid = False
            if left_hash != image.metadata.img_hash:
                if repair:
                    image.metadata.img_hash = left_hash
                    changed = True
                else:
                    logging.getLogger(__name__).error(f"Image {img_idx}: Left hash does not match metadata")
                valid = False
                img_valid = False
            if right_actual_pixels is None or not np.array_equal(right_image, right_actual_pixels):
                if repair:
                    image.store_right_pixels(right_image)
                    changed = True
                else:
                    logging.getLogger(__name__).error(f"Image {img_idx}: Right pixels do not match data read from disk")
                valid = False
                img_valid = False
            if right_hash != image.right_metadata.img_hash:
                if repair:
                    image.right_metadata.img_hash = right_hash
                    changed = True
                else:
                    logging.getLogger(__name__).error(f"Image {img_idx}: Right hash does not match metadata")
                valid = False
                img_valid = False
            if changed and repair:
                logging.getLogger(__name__).warning(f"Image {img_idx}: repaired")
                image.save()
                total_fixed_images += 1
            if not img_valid:
                total_invalid_images += 1

    if irreparable:
        # Images are missing entirely, needs re-import
        logging.getLogger(__name__).error(f"Image Collection {image_collection.pk} for sequence "
                                          f"{image_collection.sequence_name} is IRREPARABLE, invalidate and re-import")
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
