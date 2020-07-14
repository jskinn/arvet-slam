# Copyright (c) 2020, John Skinner
import logging
import typing
import tarfile
from pathlib import Path
import shutil
import numpy as np
import xxhash
import arvet.util.image_utils as image_utils
import arvet.database.image_manager
from arvet.core.image_collection import ImageCollection
import arvet_slam.dataset.tum.tum_loader as tum_loader


def verify_dataset(image_collection: ImageCollection, root_folder: typing.Union[str, Path],
                   dataset_name: str, repair: bool = False):
    """
    Load a TUM RGB-D sequence into the database.


    :return:
    """
    root_folder = Path(root_folder)
    dataset_name = str(dataset_name)
    repair = bool(repair)
    valid = True
    irreparable = False
    image_group = dataset_name

    # Check the root folder to see if it needs to be extracted from a tarfile
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

    # Check the image group on the image collection
    if image_collection.image_group != image_group:
        if repair:
            image_collection.image_group = image_group
            image_collection.save()
            logging.getLogger(__name__).info(
                f"Fixed incorrect image group for {image_collection.sequence_name}")
        else:
            logging.getLogger(__name__).warning(
                f"{image_collection.sequence_name} has incorrect image group {image_collection.image_group}")
            valid = False

    # Find the relevant metadata files
    root_folder, rgb_path, depth_path, trajectory_path = tum_loader.find_files(root_folder)

    # Step 2: Read the metadata from them
    image_files = tum_loader.read_image_filenames(rgb_path)
    trajectory = tum_loader.read_trajectory(trajectory_path, image_files.keys())
    depth_files = tum_loader.read_image_filenames(depth_path)

    # Step 3: Associate the different data types by timestamp
    all_metadata = tum_loader.associate_data(image_files, trajectory, depth_files)

    # Step 3: Load the images from the metadata
    total_invalid_images = 0
    total_fixed_images = 0
    with arvet.database.image_manager.get().get_group(image_group, allow_write=repair):
        for img_idx, (timestamp, image_file, camera_pose, depth_file) in enumerate(all_metadata):
            changed = False
            img_valid = True
            img_path = root_folder / image_file
            depth_path = root_folder / depth_file
            rgb_data = image_utils.read_colour(img_path)
            depth_data = image_utils.read_depth(depth_path)
            depth_data = depth_data / 5000  # Re-scale depth to meters
            img_hash = xxhash.xxh64(rgb_data).digest()

            # Load the image from the database
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
                img_valid = False

            # Load the pixels from the image
            try:
                actual_pixels = image.pixels
            except (KeyError, IOError, RuntimeError):
                actual_pixels = None
            try:
                actual_depth = image.depth
            except (KeyError, IOError, RuntimeError):
                actual_depth = None

            # Compare the loaded image data to the data read from disk
            if actual_pixels is None or not np.array_equal(rgb_data, actual_pixels):
                if repair:
                    image.store_pixels(rgb_data)
                    changed = True
                else:
                    logging.getLogger(__name__).error(f"Image {img_idx}: Pixels do not match data read from {img_path}")
                valid = False
                img_valid = False
            if img_hash != image.metadata.img_hash:
                if repair:
                    image.metadata.img_hash = img_hash
                    changed = True
                else:
                    logging.getLogger(__name__).error(f"Image {img_idx}: Image hash does not match metadata")
                valid = False
                img_valid = False
            if actual_depth is None or not np.array_equal(depth_data, actual_depth):
                if repair:
                    image.store_depth(depth_data)
                    changed = True
                else:
                    logging.getLogger(__name__).error(
                        f"Image {img_idx}: Depth does not match data read from {depth_path}")
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

    if delete_when_done is not None and delete_when_done.exists():
        # We're done and need to clean up after ourselves
        shutil.rmtree(delete_when_done)

    return valid
