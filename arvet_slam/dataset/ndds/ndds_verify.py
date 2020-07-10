# Copyright (c) 2020, John Skinner
import logging
import tarfile
from pathlib import Path
import shutil
import numpy as np
import arvet.util.image_utils as image_utils
import arvet.database.image_manager
from arvet.core.image_collection import ImageCollection
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader


def verify_sequence(image_collection: ImageCollection, root_folder: Path) -> bool:
    """
    Load a dataset produced by the Nvidia dataset generator
    :return:
    """
    root_folder = Path(root_folder)
    # depth_quality = DepthNoiseQuality(depth_quality)
    valid = True

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

    logging.getLogger(__name__).info(f"Verifying sequence {image_collection.sequence_name}...")
    root_folder, left_path, right_path = ndds_loader.find_files(root_folder)

    # Read the maximum image id, and make sure it matches the collection
    max_img_id = min(
        ndds_loader.find_max_img_id(lambda idx: left_path / ndds_loader.IMG_TEMPLATE.format(idx)),
        ndds_loader.find_max_img_id(lambda idx: right_path / ndds_loader.IMG_TEMPLATE.format(idx)),
    )
    if len(image_collection) != max_img_id + 1:
        logging.getLogger(__name__).warning(f"Maximum image id {max_img_id} did not match the length of the "
                                            f"image collection ({len(image_collection)})")
        # make sure we don't try and get images from the collection it doesn't have
        max_img_id = min(max_img_id, len(image_collection) - 1)
        valid = False

    # Verify the images
    # Open the image manager for writing once, so that we're not constantly opening and closing it with each image
    total_invalid_images = 0
    with arvet.database.image_manager.get():
        for img_idx in range(max_img_id + 1):
            # Expand the file paths for this image
            left_img_path = left_path / ndds_loader.IMG_TEMPLATE.format(img_idx)
            left_depth_path = left_path / ndds_loader.DEPTH_TEMPLATE.format(img_idx)
            right_img_path = right_path / ndds_loader.IMG_TEMPLATE.format(img_idx)
            right_depth_path = right_path / ndds_loader.DEPTH_TEMPLATE.format(img_idx)

            # Read the raw data for the left image
            left_pixels = image_utils.read_colour(left_img_path)
            left_ground_truth_depth = ndds_loader.load_depth_image(left_depth_path)

            # Read the raw data for the right image
            right_pixels = image_utils.read_colour(right_img_path)
            right_ground_truth_depth = ndds_loader.load_depth_image(right_depth_path)

            # Ensure all images are c_contiguous
            if not left_pixels.flags.c_contiguous:
                left_pixels = np.ascontiguousarray(left_pixels)
            if not left_ground_truth_depth.flags.c_contiguous:
                left_ground_truth_depth = np.ascontiguousarray(left_ground_truth_depth)
            if not right_pixels.flags.c_contiguous:
                right_pixels = np.ascontiguousarray(right_pixels)
            if not right_ground_truth_depth.flags.c_contiguous:
                right_ground_truth_depth = np.ascontiguousarray(right_ground_truth_depth)

            # Compute a noisy depth image
            # noisy_depth = create_noisy_depth_image(
            #     left_ground_truth_depth=left_ground_truth_depth,
            #     right_ground_truth_depth=right_ground_truth_depth,
            #     camera_intrinsics=left_camera_intrinsics,
            #     right_camera_relative_pose=left_camera_pose.find_relative(right_camera_pose),
            #     quality_level=depth_quality
            # )

            # Load the image from the database
            try:
                _, image = image_collection[img_idx]
                left_actual_pixels = image.left_pixels
                left_actual_ground_truth_depth = image.left_ground_truth_depth
                right_actual_pixels = image.right_pixels
                right_actual_ground_truth_depth = image.right_ground_truth_depth
            except (KeyError, IOError, RuntimeError):
                logging.getLogger(__name__).exception(f"Error loading image {img_idx}")
                valid = False
                continue

            # Compare the loaded image data to the data read from disk
            if not np.array_equal(left_pixels, left_actual_pixels):
                logging.getLogger(__name__).error(
                    f"Image {img_idx}: Left pixels do not match data read from {left_img_path}")
                total_invalid_images += 1
                valid = False
            if not np.array_equal(left_ground_truth_depth, left_actual_ground_truth_depth):
                logging.getLogger(__name__).error(
                    f"Image {img_idx}: Left depth does not match data read from {left_depth_path}")
                total_invalid_images += 1
                valid = False
            if not np.array_equal(right_pixels, right_actual_pixels):
                logging.getLogger(__name__).error(
                    f"Image {img_idx}: Right pixels do not match data read from {right_img_path}")
                total_invalid_images += 1
                valid = False
            if not np.array_equal(right_ground_truth_depth, right_actual_ground_truth_depth):
                logging.getLogger(__name__).error(
                    f"Image {img_idx}: Right depth does not match data read from {right_depth_path}")
                total_invalid_images += 1
                valid = False

    if delete_when_done is not None and delete_when_done.exists():
        # We're done and need to clean up after ourselves
        shutil.rmtree(delete_when_done)

    logging.getLogger(__name__).info(f"Verification of {image_collection.sequence_name} complete, it is " +
                                     'valid.' if valid else f"INVALID! ({total_invalid_images} images failed)")
    return valid
