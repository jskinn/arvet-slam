# Copyright (c) 2017, John Skinner
import unittest
import os.path
import shutil
import numpy as np
import transforms3d as tf3d
import arvet.database.tests.test_entity
import arvet.util.transform as tf
import arvet.metadata.image_metadata as imeta
import arvet.metadata.camera_intrinsics as cam_intr
import arvet.core.sequence_type
import arvet.core.image
import arvet.config.path_manager
import arvet_slam.systems.slam.orbslam2
import arvet_slam.trials.slam.visual_slam as slam_trial

_temp_folder = 'temp-test-orbslam2'


class TestORBSLAM2Execution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs(_temp_folder, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_temp_folder)

    def test_simple_trial_run(self):
        # Actually run the system using mocked images
        # We require a vocab file to exist at ../ORBSLAM2/ORBvoc.txt,
        # Download from the ORBSLAM repository and place there.
        subject = arvet_slam.systems.slam.orbslam2.ORBSLAM2(
            vocabulary_file='ORBSLAM2/ORBvoc.txt',
            mode=arvet_slam.systems.slam.orbslam2.SensorMode.STEREO,
            temp_folder=_temp_folder,
            settings={'fps': 3}
        )
        subject.resolve_paths(arvet.config.path_manager.PathManager(['..']))
        subject.set_camera_intrinsics(cam_intr.CameraIntrinsics(
            width=320,
            height=240,
            fx=160,
            fy=160,
            cx=160,
            cy=120
        ))
        subject.set_stereo_baseline(0.05)
        subject.start_trial(arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL)
        num_frames = 100
        for time in range(num_frames):
            image = create_frame(time / num_frames)
            subject.process_image(image, 10 * time / num_frames)
        result = subject.finish_trial()
        self.assertIsInstance(result, slam_trial.SLAMTrialResult)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))


def create_frame(time):
    frame = np.zeros((320, 240), dtype=np.uint8)
    right_frame = np.zeros((320, 240), dtype=np.uint8)
    speed = 200
    f = frame.shape[1] / 2
    cx = frame.shape[1] / 2
    cy = frame.shape[0] / 2
    stars = [{
        'pos': (
            (127 * idx + 34 * idx * idx) % 400 - 200,
            (320 - 17 * idx + 7 * idx * idx) % 400 - 200,
            (183 * idx - speed * time) % 500 + 0.01),
        'width': idx % 31 + 1,
        'height': idx % 27 + 1,
        'colour': idx * 7 % 256
    } for idx in range(300)]
    stars.sort(key=lambda s: s['pos'][2], reverse=True)

    for star in stars:
        x, y, z = star['pos']

        left = int(np.round(f * ((x - star['width'] / 2) / z) + cx))
        right = int(np.round(f * ((x + star['width'] / 2) / z) + cx))

        top = int(np.round(f * ((y - star['height'] / 2) / z) + cy))
        bottom = int(np.round(f * ((y + star['height'] / 2) / z) + cy))

        left = max(0, min(frame.shape[1], left))
        right = max(0, min(frame.shape[1], right))
        top = max(0, min(frame.shape[0], top))
        bottom = max(0, min(frame.shape[0], bottom))

        frame[top:bottom, left:right] = star['colour']

        left = int(np.round(f * ((x + 50 - star['width'] / 2) / z) + cx))
        right = int(np.round(f * ((x + 50 + star['width'] / 2) / z) + cx))

        left = max(0, min(frame.shape[1], left))
        right = max(0, min(frame.shape[1], right))

        right_frame[top:bottom, left:right] = star['colour']

    return arvet.core.image.StereoImage(
        left_data=frame,
        right_data=right_frame,
        metadata=imeta.ImageMetadata(
            source_type=imeta.ImageSourceType.SYNTHETIC,
            hash_=0x0000000,
            camera_pose=tf.Transform(
                location=np.array((1, 2, 3)) + time * np.array((4, 5, 6)),
                rotation=tf3d.quaternions.axangle2quat((1, 2, 34), np.pi / 36 + time * np.pi / 15)
            )
        )
    )
