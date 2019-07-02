# Copyright (c) 2017, John Skinner
import numpy as np
import typing
import logging
import time
from operator import attrgetter
import pymodm.fields as fields

from viso2 import Mono_parameters, Stereo_parameters, VisualOdometryStereo, VisualOdometryMono

from arvet.util.column_list import ColumnList
import arvet.util.image_utils as image_utils
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.image import Image
from arvet.core.image_source import ImageSource
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.system import VisionSystem
import arvet.util.transform as tf
from arvet_slam.trials.slam.visual_slam import FrameResult, SLAMTrialResult


class LibVisOStereoSystem(VisionSystem):
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

    # List of available metadata columns, and getters for each
    columns = ColumnList(
        matcher_nms_n=attrgetter('matcher_nms_n'),
        matcher_nms_tau=attrgetter('matcher_nms_tau'),
        matcher_match_binsize=attrgetter('matcher_match_binsize'),
        matcher_match_radius=attrgetter('matcher_match_radius'),
        matcher_match_disp_tolerance=attrgetter('matcher_match_disp_tolerance'),
        matcher_outlier_disp_tolerance=attrgetter('matcher_outlier_disp_tolerance'),
        matcher_outlier_flow_tolerance=attrgetter('matcher_outlier_flow_tolerance'),
        matcher_multi_stage=attrgetter('matcher_multi_stage'),
        matcher_half_resolution=attrgetter('matcher_half_resolution'),
        matcher_refinement=attrgetter('matcher_refinement'),
        bucketing_max_features=attrgetter('bucketing_max_features'),
        bucketing_bucket_width=attrgetter('bucketing_bucket_width'),
        bucketing_bucket_height=attrgetter('bucketing_bucket_height'),
        ransac_iters=attrgetter('ransac_iters'),
        inlier_threshold=attrgetter('inlier_threshold'),
        reweighting=attrgetter('reweighting')
    )

    def __init__(self, *args, **kwargs):
        """

        """
        super(LibVisOStereoSystem, self).__init__(*args, **kwargs)

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

    def is_deterministic(self) -> bool:
        return True

    def is_image_source_appropriate(self, image_source: ImageSource) -> bool:
        return (image_source.sequence_type == ImageSequenceType.SEQUENTIAL and
                image_source.is_stereo_available)

    def set_camera_intrinsics(self, camera_intrinsics: CameraIntrinsics) -> None:
        """
        Set the camera intrinisics for libviso2
        :param camera_intrinsics: The camera intrinsics, relative to the image resolution
        :return:
        """
        logging.getLogger(__name__).error("Setting camera intrinsics")
        self._focal_distance = float(camera_intrinsics.fx)
        self._cu = float(camera_intrinsics.cx)
        self._cv = float(camera_intrinsics.cy)

    def set_stereo_offset(self, offset: tf.Transform) -> None:
        """
        Set the stereo baseline
        :param offset:
        :return:
        """
        baseline = -1 * offset.location[1]   # right is -Y axis
        logging.getLogger(__name__).error("Setting stereo baseline to {0}".format(baseline))
        self._base = float(baseline)

    def start_trial(self, sequence_type: ImageSequenceType) -> None:
        logging.getLogger(__name__).error("Starting LibVisO trial...")
        if not sequence_type == ImageSequenceType.SEQUENTIAL:
            return
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

    def process_image(self, image: Image, timestamp: float) -> None:
        start_time = time.time()
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
        relative_pose = make_relative_pose(np_motion)  # This is the pose of the previous pose relative to the next one
        self._estimated_world_pose = self._estimated_world_pose.find_independent(relative_pose)
        logging.getLogger(__name__).error("    got estimated motion ...")

        true_motion = self._previous_pose.find_relative(image.camera_pose) \
            if self._previous_pose is not None else tf.Transform()
        self._previous_pose = image.camera_pose
        end_time = time.time()

        self._frame_results.append(FrameResult(
            timestamp=timestamp,
            image=image,
            processing_time=end_time - start_time,
            pose=image.camera_pose,
            motion=true_motion,
            estimated_pose=self._estimated_world_pose,
            estimated_motion=relative_pose,
            num_matches=self._viso.getNumberOfMatches()
        ))
        logging.getLogger(__name__).error("    Processing done.")

    def finish_trial(self) -> SLAMTrialResult:
        logging.getLogger(__name__).error("Finishing LibVisO trial ...")
        result = SLAMTrialResult(
            system=self,
            success=True,
            settings={
                'focal_distance': self._focal_distance,
                'cu': self._cu,
                'cv': self._cv,
                'base': self._base
            },
            results=self._frame_results,
            has_scale=True
        )
        self._frame_results = None
        self._previous_pose = None
        self._estimated_world_pose = None
        self._viso = None
        logging.getLogger(__name__).error("    Created result")
        return result

    def get_columns(self) -> typing.Set[str]:
        """
        Get the set of available properties for this system. Pass these to "get_properties", below.
        :return:
        """
        return set(self.columns.keys())

    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        """
        Get the values of the requested properties
        :param columns:
        :return:
        """
        if columns is None:
            columns = self.columns.keys()
        return {
            col_name: self.columns.get_value(self, col_name)
            for col_name in columns
            if col_name in self.columns
        }

    @classmethod
    def preload_image_data(cls, image: Image) -> None:
        """
        Preload the pixel data we use from the images.
        This is a stereo system, load right pixel data as well
        :param image:
        :return:
        """
        super(LibVisOStereoSystem, cls).preload_image_data(image)
        if hasattr(image, 'right_pixels'):
            _ = image.right_pixels

    @classmethod
    def get_instance(
            cls,
            matcher_nms_n=3,
            matcher_nms_tau=50,
            matcher_match_binsize=50,
            matcher_match_radius=200,
            matcher_match_disp_tolerance=2,
            matcher_outlier_disp_tolerance=5,
            matcher_outlier_flow_tolerance=5,
            matcher_multi_stage=True,
            matcher_half_resolution=True,
            matcher_refinement=50,
            bucketing_max_features=2,
            bucketing_bucket_width=50,
            bucketing_bucket_height=50,
            ransac_iters=200,
            inlier_threshold=2.0,
            reweighting=True
    ) -> 'LibVisOStereoSystem':
        """
        Get an instance of this vision system, with some parameters, pulling from the database if possible,
        or construct a new one if needed.
        It is the responsibility of subclasses to ensure that as few instances of each system as possible exist
        within the database.
        Does not save the returned object, you'll usually want to do that straight away.
        :return:
        """
        # Look for existing objects with the same settings
        all_objects = LibVisOStereoSystem.objects.raw({
            'matcher_nms_n': matcher_nms_n,
            'matcher_nms_tau': matcher_nms_tau,
            'matcher_match_binsize': matcher_match_binsize,
            'matcher_match_radius': matcher_match_radius,
            'matcher_match_disp_tolerance': matcher_match_disp_tolerance,
            'matcher_outlier_disp_tolerance': matcher_outlier_disp_tolerance,
            'matcher_outlier_flow_tolerance': matcher_outlier_flow_tolerance,
            'matcher_multi_stage': matcher_multi_stage,
            'matcher_half_resolution': matcher_half_resolution,
            'matcher_refinement': matcher_refinement,
            'bucketing_max_features': bucketing_max_features,
            'bucketing_bucket_width': bucketing_bucket_width,
            'bucketing_bucket_height': bucketing_bucket_height,
            'ransac_iters': ransac_iters,
            'inlier_threshold': inlier_threshold,
            'reweighting': reweighting
        })
        if all_objects.count() > 0:
            return all_objects.first()
        # There isn't an existing system with those settings, make a new one.
        obj = cls(
            matcher_nms_n=matcher_nms_n,
            matcher_nms_tau=matcher_nms_tau,
            matcher_match_binsize=matcher_match_binsize,
            matcher_match_radius=matcher_match_radius,
            matcher_match_disp_tolerance=matcher_match_disp_tolerance,
            matcher_outlier_disp_tolerance=matcher_outlier_disp_tolerance,
            matcher_outlier_flow_tolerance=matcher_outlier_flow_tolerance,
            matcher_multi_stage=matcher_multi_stage,
            matcher_half_resolution=matcher_half_resolution,
            matcher_refinement=matcher_refinement,
            bucketing_max_features=bucketing_max_features,
            bucketing_bucket_width=bucketing_bucket_width,
            bucketing_bucket_height=bucketing_bucket_height,
            ransac_iters=ransac_iters,
            inlier_threshold=inlier_threshold,
            reweighting=reweighting
        )
        return obj


class LibVisOMonoSystem(VisionSystem):
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
    height = fields.FloatField(default=1.0)
    pitch = fields.FloatField(default=0.0)
    ransac_iters = fields.IntegerField(default=2000)
    inlier_threshold = fields.FloatField(default=0.00001)
    motion_threshold = fields.FloatField(default=100.0)

    # List of available metadata columns, and getters for each
    columns = ColumnList(
        matcher_nms_n=attrgetter('matcher_nms_n'),
        matcher_nms_tau=attrgetter('matcher_nms_tau'),
        matcher_match_binsize=attrgetter('matcher_match_binsize'),
        matcher_match_radius=attrgetter('matcher_match_radius'),
        matcher_match_disp_tolerance=attrgetter('matcher_match_disp_tolerance'),
        matcher_outlier_disp_tolerance=attrgetter('matcher_outlier_disp_tolerance'),
        matcher_outlier_flow_tolerance=attrgetter('matcher_outlier_flow_tolerance'),
        matcher_multi_stage=attrgetter('matcher_multi_stage'),
        matcher_half_resolution=attrgetter('matcher_half_resolution'),
        matcher_refinement=attrgetter('matcher_refinement'),
        bucketing_max_features=attrgetter('bucketing_max_features'),
        bucketing_bucket_width=attrgetter('bucketing_bucket_width'),
        bucketing_bucket_height=attrgetter('bucketing_bucket_height'),
        height=attrgetter('height'),
        pitch=attrgetter('pitch'),
        ransac_iters=attrgetter('ransac_iters'),
        inlier_threshold=attrgetter('inlier_threshold'),
        motion_threshold=attrgetter('motion_threshold')
    )

    def __init__(self, *args, **kwargs):
        """

        """
        super(LibVisOMonoSystem, self).__init__(*args, **kwargs)

        # These will get overridden by set_camera_intrinisics
        self._focal_distance = 1.0
        self._cu = 320
        self._cv = 240

        # Ongoing state during a trial that is initialised in start_trial
        self._viso = None
        self._previous_pose = None
        self._estimated_world_pose = None
        self._frame_results = []

    def is_deterministic(self) -> bool:
        return True

    def is_image_source_appropriate(self, image_source: ImageSource) -> bool:
        return image_source.sequence_type == ImageSequenceType.SEQUENTIAL

    def set_camera_intrinsics(self, camera_intrinsics: CameraIntrinsics) -> None:
        """
        Set the camera intrinisics for libviso2
        :param camera_intrinsics: The camera intrinsics, relative to the image resolution
        :return:
        """
        logging.getLogger(__name__).error("Setting camera intrinsics")
        self._focal_distance = float(camera_intrinsics.fx)
        self._cu = float(camera_intrinsics.cx)
        self._cv = float(camera_intrinsics.cy)

    def start_trial(self, sequence_type: ImageSequenceType) -> None:
        logging.getLogger(__name__).error("Starting LibVisO trial...")
        if not sequence_type == ImageSequenceType.SEQUENTIAL:
            return
        params = Mono_parameters()
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

        # Monocular-specific parameters
        params.height = self.height
        params.pitch = self.pitch
        params.ransac_iters = self.ransac_iters
        params.inlier_threshold = self.inlier_threshold
        params.motion_threshold = self.motion_threshold
        logging.getLogger(__name__).error("Added monocular specific parameters ...")

        # Camera calibration
        params.calib.f = self._focal_distance
        params.calib.cu = self._cu
        params.calib.cv = self._cv
        logging.getLogger(__name__).error("    Parameters built, creating viso object ...")

        self._viso = VisualOdometryMono(params)
        self._previous_pose = None
        self._estimated_world_pose = tf.Transform()
        self._frame_results = []
        logging.getLogger(__name__).error("    Started LibVisO trial.")

    def process_image(self, image: Image, timestamp: float) -> None:
        logging.getLogger(__name__).error("Processing image at time {0} ...".format(timestamp))
        start_time = time.time()
        image_greyscale = prepare_image(image.pixels)
        logging.getLogger(__name__).error("    prepared images ...")

        self._viso.process_frame(image_greyscale)
        logging.getLogger(__name__).error("    processed frame ...")

        motion = self._viso.getMotion()  # Motion is a 4x4 pose matrix
        np_motion = np.zeros((4, 4))
        motion.toNumpy(np_motion)
        np_motion = np.linalg.inv(np_motion)    # Invert the motion to make it new frame relative to old
        relative_pose = make_relative_pose(np_motion)  # This is the pose of the previous pose relative to the next one
        self._estimated_world_pose = self._estimated_world_pose.find_independent(relative_pose)
        logging.getLogger(__name__).error("    got estimated motion ...")

        true_motion = self._previous_pose.find_relative(image.camera_pose) \
            if self._previous_pose is not None else tf.Transform()
        self._previous_pose = image.camera_pose
        end_time = time.time()

        self._frame_results.append(FrameResult(
            timestamp=timestamp,
            image=image,
            processing_time=end_time - start_time,
            pose=image.camera_pose,
            motion=true_motion,
            estimated_pose=self._estimated_world_pose,
            estimated_motion=relative_pose,
            num_matches=self._viso.getNumberOfMatches()
        ))
        logging.getLogger(__name__).error("    Processing done.")

    def finish_trial(self) -> SLAMTrialResult:
        logging.getLogger(__name__).error("Finishing LibVisO trial ...")
        result = SLAMTrialResult(
            system=self,
            success=True,
            settings={
                'focal_distance': self._focal_distance,
                'cu': self._cu,
                'cv': self._cv
            },
            results=self._frame_results,
            has_scale=False
        )
        self._frame_results = None
        self._previous_pose = None
        self._estimated_world_pose = None
        self._viso = None
        logging.getLogger(__name__).error("    Created result")
        return result

    def get_columns(self) -> typing.Set[str]:
        """
        Get the set of available properties for this system. Pass these to "get_properties", below.
        :return:
        """
        return set(self.columns.keys())

    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        """
        Get the values of the requested properties
        :param columns:
        :return:
        """
        if columns is None:
            columns = self.columns.keys()
        return {
            col_name: self.columns.get_value(self, col_name)
            for col_name in columns
            if col_name in self.columns
        }

    @classmethod
    def get_instance(
            cls,
            matcher_nms_n=3,
            matcher_nms_tau=50,
            matcher_match_binsize=50,
            matcher_match_radius=200,
            matcher_match_disp_tolerance=2,
            matcher_outlier_disp_tolerance=5,
            matcher_outlier_flow_tolerance=5,
            matcher_multi_stage=True,
            matcher_half_resolution=True,
            matcher_refinement=50,
            bucketing_max_features=2,
            bucketing_bucket_width=50,
            bucketing_bucket_height=50,
            height=1.0,
            pitch=0.0,
            ransac_iters=2000,
            inlier_threshold=0.00001,
            motion_threshold=00.0
    ) -> 'LibVisOMonoSystem':
        """
        Get an instance of this vision system, with some parameters, pulling from the database if possible,
        or construct a new one if needed.
        It is the responsibility of subclasses to ensure that as few instances of each system as possible exist
        within the database.
        Does not save the returned object, you'll usually want to do that straight away.
        :return:
        """
        # Look for existing objects with the same settings
        all_objects = LibVisOMonoSystem.objects.raw({
            'matcher_nms_n': matcher_nms_n,
            'matcher_nms_tau': matcher_nms_tau,
            'matcher_match_binsize': matcher_match_binsize,
            'matcher_match_radius': matcher_match_radius,
            'matcher_match_disp_tolerance': matcher_match_disp_tolerance,
            'matcher_outlier_disp_tolerance': matcher_outlier_disp_tolerance,
            'matcher_outlier_flow_tolerance': matcher_outlier_flow_tolerance,
            'matcher_multi_stage': matcher_multi_stage,
            'matcher_half_resolution': matcher_half_resolution,
            'matcher_refinement': matcher_refinement,
            'bucketing_max_features': bucketing_max_features,
            'bucketing_bucket_width': bucketing_bucket_width,
            'bucketing_bucket_height': bucketing_bucket_height,
            'height': height,
            'pitch': pitch,
            'ransac_iters': ransac_iters,
            'inlier_threshold': inlier_threshold,
            'motion_threshold': motion_threshold
        })
        if all_objects.count() > 0:
            return all_objects.first()
        # There isn't an existing system with those settings, make a new one.
        obj = cls(
            matcher_nms_n=matcher_nms_n,
            matcher_nms_tau=matcher_nms_tau,
            matcher_match_binsize=matcher_match_binsize,
            matcher_match_radius=matcher_match_radius,
            matcher_match_disp_tolerance=matcher_match_disp_tolerance,
            matcher_outlier_disp_tolerance=matcher_outlier_disp_tolerance,
            matcher_outlier_flow_tolerance=matcher_outlier_flow_tolerance,
            matcher_multi_stage=matcher_multi_stage,
            matcher_half_resolution=matcher_half_resolution,
            matcher_refinement=matcher_refinement,
            bucketing_max_features=bucketing_max_features,
            bucketing_bucket_width=bucketing_bucket_width,
            bucketing_bucket_height=bucketing_bucket_height,
            height=height,
            pitch=pitch,
            ransac_iters=ransac_iters,
            inlier_threshold=inlier_threshold,
            motion_threshold=motion_threshold
        )
        return obj


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
