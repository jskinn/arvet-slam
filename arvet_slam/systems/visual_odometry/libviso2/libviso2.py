# Copyright (c) 2017, John Skinner
import numpy as np
import logging
import pymodm.fields as fields

from .viso2 import Stereo_parameters, VisualOdometryStereo

import arvet.util.image_utils as image_utils
import arvet.core.sequence_type
import arvet.core.system
import arvet.core.trial_result
import arvet.util.transform as tf
from arvet_slam.trials.slam.visual_slam import FrameResult, SLAMTrialResult


class LibVisOSystem(arvet.core.system.VisionSystem):
    """
    Class to run LibVisO2 as a vision system.
    """
    matcher_nms_n = fields.IntegerField(default=3)
    matcher_nms_tau = fields.IntegerField(default=50)
    matcher_match_binsize = fields.IntegerField(default=50)
    matcher_match_radius = fields.IntegerField(default=200)
    matcher_match_disp_tolerance = fields.IntegerField(default=2)
    matcher_outlier_disp_tolerance = fields.IntegerField(default=5)
    matcher_outlier_flow_tolerance = fields.IntegerField(default=5)
    matcher_multi_stage = fields.BooleanField(default=True)
    matcher_half_resolution = fields.BooleanField(default=True)
    matcher_refinement = fields.IntegerField(default=50)
    bucketing_max_features = fields.IntegerField(default=2)
    bucketing_bucket_width = fields.IntegerField(default=50)
    bucketing_bucket_height = fields.IntegerField(default=50)
    ransac_iters = fields.IntegerField(default=200)
    inlier_threshold = fields.FloatField(default=2.0)
    reweighting = fields.BooleanField(default=True)

    def __init__(self, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)

        # These will get overridden by set_camera_intrinisics
        self._focal_distance = 1.0
        self._cu = 320
        self._cv = 240
        self._base = 0.3

        # Ongoing state during a trial that is initialised in start_trial
        self._viso = None
        self._previous_pose = None
        self._estimated_world_pose = None
        self._frame_results = []

    def is_image_source_appropriate(self, image_source):
        return (image_source.sequence_type == arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL and
                image_source.is_stereo_available)

    def is_deterministic(self):
        return True

    def set_camera_intrinsics(self, camera_intrinsics):
        """
        Set the camera intrinisics for libviso2
        :param camera_intrinsics: The camera intrinsics, relative to the image resolution
        :return:
        """
        logging.getLogger(__name__).error("Setting camera intrinsics")
        self._focal_distance = float(camera_intrinsics.fx)
        self._cu = float(camera_intrinsics.cx)
        self._cv = float(camera_intrinsics.cy)

    def set_stereo_baseline(self, baseline):
        """
        Set the stereo baseline
        :param baseline:
        :return:
        """
        logging.getLogger(__name__).error("Setting stereo baseline to {0}".format(baseline))
        self._base = float(baseline)

    def start_trial(self, sequence_type):
        logging.getLogger(__name__).error("Starting LibVisO trial...")
        if not sequence_type == arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL:
            return False
        params = Stereo_parameters()
        logging.getLogger(__name__).error("    Created parameters object, populating ...")

        # Matcher parameters
        params.match.nms_n = self.matcher_nms_n
        params.match.nms_tau = self.matcher_nms_tau
        params.match.match_binsize = self.matcher_match_binsize
        params.match.match_radius = self.matcher_match_radius
        params.match.match_disp_tolerance = self.matcher_match_disp_tolerance
        params.match.outlier_disp_tolerance = self.matcher_outlier_disp_tolerance
        params.match.outlier_flow_tolerance = self.matcher_outlier_flow_tolerance
        params.match.multi_stage = 1 if self.matcher_multi_stage else 0
        params.match.half_resolution = 1 if self.matcher_half_resolution else 0
        params.match.refinement = self.matcher_refinement
        logging.getLogger(__name__).error("    Added matcher parameters ...")

        # Feature bucketing
        params.bucket.max_features = self.bucketing_max_features
        params.bucket.bucket_width = self.bucketing_bucket_width
        params.bucket.bucket_height = self.bucketing_bucket_height
        logging.getLogger(__name__).error("    Added bucket parameters ...")

        # Stereo-specific parameters
        params.ransac_iters = self.ransac_iters
        params.inlier_threshold = self.inlier_threshold
        params.reweighting = self.reweighting
        logging.getLogger(__name__).error("Added stereo specific parameters ...")

        # Camera calibration
        params.calib.f = self._focal_distance
        params.calib.cu = self._cu
        params.calib.cv = self._cv
        params.base = self._base
        logging.getLogger(__name__).error("    Parameters built, creating viso object ...")

        self._viso = VisualOdometryStereo(params)
        self._previous_pose = None
        self._estimated_world_pose = tf.Transform()
        self._frame_results = []
        logging.getLogger(__name__).error("    Started LibVisO trial.")

    def process_image(self, image, timestamp):
        logging.getLogger(__name__).error("Processing image at time {0} ...".format(timestamp))
        left_grey = prepare_image(image.left_pixels)
        right_grey = prepare_image(image.right_pixels)
        logging.getLogger(__name__).error("    prepared images ...")

        self._viso.process_frame(left_grey, right_grey)
        logging.getLogger(__name__).error("    processed frame ...")

        motion = self._viso.getMotion()  # Motion is a 4x4 pose matrix
        np_motion = np.zeros((4, 4))
        motion.toNumpy(np_motion)
        np_motion = np.linalg.inv(np_motion)    # Invert the motion to make it new frame relative to old
        relative_pose = make_relative_pose(np_motion) # This is the pose of the previous pose relative to the next one
        self._estimated_world_pose = self._estimated_world_pose.find_independent(relative_pose)
        logging.getLogger(__name__).error("    got estimated motion ...")

        true_motion = self._previous_pose.find_relative(image.camera_pose) \
            if self._previous_pose is not None else tf.Transform()
        self._previous_pose = image.camera_pose

        self._frame_results.append(FrameResult(
            timestamp=timestamp,
            pose=image.camera_pose,
            motion=true_motion,
            estimated_pose=self._estimated_world_pose,
            estimated_motion=relative_pose,
            num_matches=self._viso.getNumberOfMatches()
        ))
        logging.getLogger(__name__).error("    Processing done.")

    def finish_trial(self):
        logging.getLogger(__name__).error("Finishing LibVisO trial ...")
        result = SLAMTrialResult(
            system=self,
            success=True,
            settings={'key': 'value'},
            results=self._frame_results,
            has_scale=True
        )
        self._frame_results = None
        self._previous_pose = None
        self._estimated_world_pose = None
        self._viso = None
        logging.getLogger(__name__).error("    Created result")
        return result


def make_relative_pose(frame_delta):
    """
    LibVisO2 uses a different coordinate frame to the one I'm using,
    this function is to convert computed frame deltas as libviso estimates them to usable poses.
    Thankfully, its still a right-handed coordinate frame, which makes this easier.
    Frame is: z forward, x right, y down
    Documentation at: http://www.cvlibs.net/software/libviso/

    :param frame_delta: A 4x4 matrix (possibly in list form)
    :return: A Transform object representing the pose of the current frame with respect to the previous frame
    """
    frame_delta = np.asarray(frame_delta)
    coordinate_exchange = np.array([[0, 0, 1, 0],
                                     [-1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])
    pose = np.dot(np.dot(coordinate_exchange, frame_delta), coordinate_exchange.T)
    return tf.Transform(pose)


def prepare_image(image_data):
    """
    Process image data ready to input it to libviso2.
    We expect input images to be greyscale uint8 images, other image types are massaged into that shape
    :param image_data: A numpy array of image data
    :return: An uint8 numpy array, intensity range 0-255
    """
    output_image = image_data
    if output_image.dtype != np.uint8:
        if 0.99 < np.max(output_image) <= 1.001:
            output_image = 255 * output_image
        output_image = np.asarray(output_image, dtype=np.uint8)
    if len(output_image.shape) > 2:
        output_image = image_utils.convert_to_grey(output_image)
    return output_image
