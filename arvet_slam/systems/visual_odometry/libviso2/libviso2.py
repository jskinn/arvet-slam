# Copyright (c) 2017, John Skinner
import numpy as np
import viso2 as libviso2

import arvet.util.image_utils as image_utils
import arvet.core.sequence_type
import arvet.core.system
import arvet.core.trial_result
import arvet_slam.trials.slam.visual_slam as vs
import arvet.util.transform as tf


class LibVisOSystem(arvet.core.system.VisionSystem):
    """
    Class to run LibVisO2 as a vision system.
    """

    def __init__(self,
                 matcher_nms_n: int = 3,
                 matcher_nms_tau: int = 50,
                 matcher_match_binsize: int = 50,
                 matcher_match_radius: int = 200,
                 matcher_match_disp_tolerance: int = 2,
                 matcher_outlier_disp_tolerance: int = 5,
                 matcher_outlier_flow_tolerance: int = 5,
                 matcher_multi_stage: bool = True,
                 matcher_half_resolution: bool = True,
                 matcher_refinement: int = 1,
                 bucketing_max_features: int = 2,
                 bucketing_bucket_width: int = 50,
                 bucketing_bucket_height: int = 50,
                 ransac_iters: int = 200,
                 inlier_threshold: float = 2.0,
                 reweighting: bool = True,
                 id_=None):
        """

        :param matcher_nms_n: non-max-suppression: min. distance between maxima (in pixels)
        :param matcher_nms_tau: non-max-suppression: interest point peakiness threshold
        :param matcher_match_binsize: matching bin width/height (affects efficiency only)
        :param matcher_match_radius: matching radius (du/dv in pixels)
        :param matcher_match_disp_tolerance: dv tolerance for stereo matches (in pixels)
        :param matcher_outlier_disp_tolerance: outlier removal: disparity tolerance (in pixels)
        :param matcher_outlier_flow_tolerance: outlier removal: flow tolerance (in pixels)
        :param matcher_multi_stage: False=disabled, True=multistage matching (denser and faster)
        :param matcher_half_resolution: False=disabled, True=match at half resolution, refine at full resolution
        :param matcher_refinement: refinement (0=none,1=pixel,2=subpixel)

        :param bucketing_max_features: maximal number of features per bucket
        :param bucketing_bucket_width: width of bucket
        :param bucketing_bucket_height: height of bucket

        :param ransac_iters: number of RANSAC iterations
        :param inlier_threshold: fundamental matrix inlier threshold
        :param reweighting: lower border weights (more robust to calibration errors)
        :param id_:
        """
        super().__init__(id_=id_)

        # LibVisO Feature Matcher parameters
        self._matcher_nms_n = int(matcher_nms_n)
        self._matcher_nms_tau = int(matcher_nms_tau)
        self._matcher_match_binsize = int(matcher_match_binsize)
        self._matcher_match_radius = int(matcher_match_radius)
        self._matcher_match_disp_tolerance = int(matcher_match_disp_tolerance)
        self._matcher_outlier_disp_tolerance = int(matcher_outlier_disp_tolerance)
        self._matcher_outlier_flow_tolerance = int(matcher_outlier_flow_tolerance)
        self._matcher_multi_stage = bool(matcher_multi_stage)
        self._matcher_half_resolution = bool(matcher_half_resolution)
        self._matcher_refinement = matcher_refinement if matcher_refinement in (0, 1, 2) else 1

        # Feature bucketing parameters
        self._bucketing_max_features = int(bucketing_max_features)
        self._bucketing_bucket_width = int(bucketing_bucket_width)
        self._bucketing_bucket_height = int(bucketing_bucket_height)

        # Extra stereo parameters
        self._ransac_iters = int(ransac_iters)
        self._inlier_threshold = float(inlier_threshold)
        self._reweighting = bool(reweighting)

        # These will get overridden by set_camera_intrinisics
        self._focal_distance = 1.0
        self._cu = 320
        self._cv = 240
        self._base = 0.3

        self._viso = None
        self._current_pose = None
        self._trajectory = None
        self._gt_poses = None
        self._num_matches = None

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
        self._focal_distance = float(camera_intrinsics.fx)
        self._cu = float(camera_intrinsics.cx)
        self._cv = float(camera_intrinsics.cy)

    def set_stereo_baseline(self, baseline):
        """
        Set the stereo baseline
        :param baseline:
        :return:
        """
        self._base = float(baseline)

    def start_trial(self, sequence_type):
        if not sequence_type == arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL:
            return False
        params = libviso2.Stereo_parameters()

        # Matcher parameters
        params.match.nms_n = self._matcher_nms_n
        params.match.nms_tau = self._matcher_nms_tau
        params.match.match_binsize = self._matcher_match_binsize
        params.match.match_radius = self._matcher_match_radius
        params.match.match_disp_tolerance = self._matcher_match_disp_tolerance
        params.match.outlier_disp_tolerance = self._matcher_outlier_disp_tolerance
        params.match.outlier_flow_tolerance = self._matcher_outlier_flow_tolerance
        params.match.multi_stage = 1 if self._matcher_multi_stage else 0
        params.match.half_resolution = 1 if self._matcher_half_resolution else 0
        params.match.refinement = self._matcher_refinement

        # Feature bucketing
        params.bucket.max_features = self._bucketing_max_features
        params.bucket.bucket_width = self._bucketing_bucket_width
        params.bucket.bucket_height = self._bucketing_bucket_height

        # Stereo-specific parameters
        params.ransac_iters = self._ransac_iters
        params.inlier_threshold = self._inlier_threshold
        params.reweighting = self._reweighting

        # Camera calibration
        params.calib.f = self._focal_distance
        params.calib.cu = self._cu
        params.calib.cv = self._cv
        params.base = self._base

        self._viso = libviso2.VisualOdometryStereo(params)
        self._current_pose = tf.Transform()
        self._trajectory = {}
        self._gt_poses = {}
        self._num_matches = {}

    def process_image(self, image, timestamp):
        left_grey = prepare_image(image.left_data)
        right_grey = prepare_image(image.right_data)
        self._viso.process_frame(left_grey, right_grey)
        motion = self._viso.getMotion()  # Motion is a 4x4 pose matrix
        np_motion = np.zeros((4, 4))
        motion.toNumpy(np_motion)
        relative_pose = make_relative_pose(np_motion)
        self._current_pose = self._current_pose.find_independent(relative_pose)

        self._num_matches[timestamp] = self._viso.getNumberOfMatches()
        self._trajectory[timestamp] = self._current_pose
        self._gt_poses[timestamp] = image.camera_pose

    def finish_trial(self):
        result = vs.SLAMTrialResult(
            system_id=self.identifier,
            sequence_type=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
            system_settings={
                'f': self._focal_distance,
                'cu': self._cu,
                'cv': self._cv,
                'base': self._base
            },
            trajectory=self._trajectory,
            num_matches=self._num_matches,
            ground_truth_trajectory=self._gt_poses
        )
        self._trajectory = None
        self._gt_poses = None
        self._num_matches = None
        self._current_pose = None
        self._viso = None
        return result

    def serialize(self):
        serialized = super().serialize()
        serialized['matcher_nms_n'] = self._matcher_nms_n
        serialized['matcher_nms_tau'] = self._matcher_nms_tau
        serialized['matcher_match_binsize'] = self._matcher_match_binsize
        serialized['matcher_match_radius'] = self._matcher_match_radius
        serialized['matcher_match_disp_tolerance'] = self._matcher_match_disp_tolerance
        serialized['matcher_outlier_disp_tolerance'] = self._matcher_outlier_disp_tolerance
        serialized['matcher_outlier_flow_tolerance'] = self._matcher_outlier_flow_tolerance
        serialized['matcher_multi_stage'] = self._matcher_multi_stage
        serialized['matcher_half_resolution'] = self._matcher_half_resolution
        serialized['matcher_refinement'] = self._matcher_refinement
        serialized['bucketing_max_features'] = self._bucketing_max_features
        serialized['bucketing_bucket_width'] = self._bucketing_bucket_width
        serialized['bucketing_bucket_height'] = self._bucketing_bucket_height
        serialized['ransac_iters'] = self._ransac_iters
        serialized['inlier_threshold'] = self._inlier_threshold
        serialized['reweighting'] = self._reweighting
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'matcher_nms_n' in serialized_representation:
            kwargs['matcher_nms_n'] = serialized_representation['matcher_nms_n']
        if 'matcher_nms_tau' in serialized_representation:
            kwargs['matcher_nms_tau'] = serialized_representation['matcher_nms_tau']
        if 'matcher_match_binsize' in serialized_representation:
            kwargs['matcher_match_binsize'] = serialized_representation['matcher_match_binsize']
        if 'matcher_match_radius' in serialized_representation:
            kwargs['matcher_match_radius'] = serialized_representation['matcher_match_radius']
        if 'matcher_match_disp_tolerance' in serialized_representation:
            kwargs['matcher_match_disp_tolerance'] = serialized_representation['matcher_match_disp_tolerance']
        if 'matcher_outlier_disp_tolerance' in serialized_representation:
            kwargs['matcher_outlier_disp_tolerance'] = serialized_representation['matcher_outlier_disp_tolerance']
        if 'matcher_outlier_flow_tolerance' in serialized_representation:
            kwargs['matcher_outlier_flow_tolerance'] = serialized_representation['matcher_outlier_flow_tolerance']
        if 'matcher_multi_stage' in serialized_representation:
            kwargs['matcher_multi_stage'] = serialized_representation['matcher_multi_stage']
        if 'matcher_half_resolution' in serialized_representation:
            kwargs['matcher_half_resolution'] = serialized_representation['matcher_half_resolution']
        if 'matcher_refinement' in serialized_representation:
            kwargs['matcher_refinement'] = serialized_representation['matcher_refinement']
        if 'bucketing_max_features' in serialized_representation:
            kwargs['bucketing_max_features'] = serialized_representation['bucketing_max_features']
        if 'bucketing_bucket_width' in serialized_representation:
            kwargs['bucketing_bucket_width'] = serialized_representation['bucketing_bucket_width']
        if 'bucketing_bucket_height' in serialized_representation:
            kwargs['bucketing_bucket_height'] = serialized_representation['bucketing_bucket_height']
        if 'ransac_iters' in serialized_representation:
            kwargs['ransac_iters'] = serialized_representation['ransac_iters']
        if 'inlier_threshold' in serialized_representation:
            kwargs['inlier_threshold'] = serialized_representation['inlier_threshold']
        if 'reweighting' in serialized_representation:
            kwargs['reweighting'] = serialized_representation['reweighting']
        return super().deserialize(serialized_representation, db_client, **kwargs)


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
    frame_delta = np.asmatrix(frame_delta)
    coordinate_exchange = np.matrix([[0, 0, 1, 0],
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
