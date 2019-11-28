# Copyright (c) 2017, John Skinner
import os.path
import numpy as np
import pykitti
import arvet.metadata.image_metadata as imeta
from arvet.util.transform import Transform
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import StereoImage
from arvet.core.image_collection import ImageCollection


def make_camera_pose(pose):
    """
    KITTI uses a different coordinate frame to the one I'm using, which is the same as the Libviso2 frame.
    This function is to convert dataset ground-truth poses to transform objects.
    Thankfully, its still a right-handed coordinate frame, which makes this easier.
    Frame is: z forward, x right, y down

    :param pose: The raw pose as loaded by pykitti, a 4x4 homgenous transform object.
    :return: A Transform object representing the world pose of the current frame
    """
    pose = np.asmatrix(pose)
    coordinate_exchange = np.array([[0, 0, 1, 0],
                                    [-1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 0, 1]])
    pose = np.dot(np.dot(coordinate_exchange, pose), coordinate_exchange.T)
    return Transform(pose)


def import_dataset(root_folder, sequence_number, **_):
    """
    Load a KITTI image sequences into the database.
    :return:
    """
    sequence_name = "{0:02}".format(sequence_number)
    if not os.path.isdir(root_folder) and os.path.isdir(os.path.join(root_folder, sequence_name)):
        return None

    data = pykitti.odometry(root_folder, sequence=sequence_name)

    # dataset.calib:      Calibration data are accessible as a named tuple
    # dataset.timestamps: Timestamps are parsed into a list of timedelta objects
    # dataset.poses:      Generator to load ground truth poses T_w_cam0
    # dataset.camN:       Generator to load individual images from camera N
    # dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
    # dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
    # dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]
    images = []
    timestamps = []
    for left_image, right_image, timestamp, pose in zip(data.cam2, data.cam3, data.timestamps, data.poses):
        camera_pose = make_camera_pose(pose)
        # camera pose is for cam0, we want cam2, which is 6cm (0.06m) to the left
        # Except that we don't need to control for that, since we want to be relative to the first pose anyway
        # camera_pose = camera_pose.find_independent(tf.Transform(location=(0, 0.06, 0), rotation=(0, 0, 0, 1),
        #                                                         w_first=False))
        # Stereo offset is 0.54m (http://www.cvlibs.net/datasets/kitti/setup.php)
        right_camera_pose = camera_pose.find_independent(Transform(location=(0, -0.54, 0), rotation=(0, 0, 0, 1),
                                                                   w_first=False))
        camera_intrinsics = CameraIntrinsics(
            height=left_image.shape[0],
            width=left_image.shape[1],
            fx=data.calib.K_cam2[0, 0],
            fy=data.calib.K_cam2[1, 1],
            cx=data.calib.K_cam2[0, 2],
            cy=data.calib.K_cam2[1, 2])
        right_camera_intrinsics = CameraIntrinsics(
            height=right_image.shape[0],
            width=right_image.shape[1],
            fx=data.calib.K_cam3[0, 0],
            fy=data.calib.K_cam3[1, 1],
            cx=data.calib.K_cam3[0, 2],
            cy=data.calib.K_cam3[1, 2])
        left_metadata = imeta.make_metadata(
            pixels=left_image,
            camera_pose=camera_pose,
            intrinsics=camera_intrinsics,
            source_type=imeta.ImageSourceType.REAL_WORLD,
            environment_type=imeta.EnvironmentType.OUTDOOR_URBAN,
            light_level=imeta.LightingLevel.WELL_LIT,
            time_of_day=imeta.TimeOfDay.AFTERNOON,
        )
        right_metadata = imeta.make_right_metadata(
            pixels=right_image,
            left_metadata=left_metadata,
            camera_pose=right_camera_pose,
            intrinsics=right_camera_intrinsics
        )
        image = StereoImage(
            pixels=left_image,
            right_pixels=right_image,
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
        dataset='KITTI',
        sequence_name=sequence_name,
        trajectory_id='KITTI_' + sequence_name
    )
    collection.save()
    return collection
