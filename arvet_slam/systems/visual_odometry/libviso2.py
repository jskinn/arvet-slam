# Copyright (c) 2017, John Skinner
import abc
import numpy as np
import typing
import logging
import time
from enum import Enum
from operator import attrgetter
import pymodm.fields as fields

from viso2 import Mono_parameters, Stereo_parameters, VisualOdometryStereo, VisualOdometryMono

from arvet.util.column_list import ColumnList
import arvet.util.image_utils as image_utils
from arvet.database.enum_field import EnumField
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.database.pymodm_abc import ABCModelMeta
from arvet.core.image import Image
from arvet.core.image_source import ImageSource
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.system import VisionSystem, StochasticBehaviour
import arvet.util.transform as tf
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import FrameResult, SLAMTrialResult


class MatcherRefinement(Enum):
    NONE = 0
    PIXEL = 1
    SUBPIXEL = 2


class LibVisOSystem(VisionSystem, metaclass=ABCModelMeta):
    """
    Class to run LibVisO2 as a vision system.
    A generic base class, the specific types are below (LibVisOStereoSystem, LibVisOMonoSystem)
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
    matcher_refinement = EnumField(MatcherRefinement, default=MatcherRefinement.PIXEL)
    bucketing_max_features = fields.IntegerField(default=2)
    bucketing_bucket_width = fields.IntegerField(default=50)
    bucketing_bucket_height = fields.IntegerField(default=50)

    # List of available metadata columns, and getters for each
    columns = ColumnList(
        seed=None,
        in_height=None,
        in_width=None,
        in_fx=None,
        in_fy=None,
        in_cx=None,
        in_cy=None,

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
        bucketing_bucket_height=attrgetter('bucketing_bucket_height')
    )

    def __init__(self, *args, **kwargs):
        """

        """
        super(LibVisOSystem, self).__init__(*args, **kwargs)

        # These will get overridden by set_camera_intrinisics
        self._focal_distance = 1.0
        self._cu = 320
        self._cv = 240
        self._width = 0     # These are not actually used, only stored
        self._height = 0

        # Ongoing state during a trial that is initialised in start_trial
        self._viso = None
        self._seed = None
        self._start_time = None
        self._has_chosen_origin = False
        self._frame_results = []

    @classmethod
    def is_deterministic(cls) -> StochasticBehaviour:
        """
        LibVisO2 is controlled with a seed
        :return: StochasticBehaviour.SEEDED
        """
        return StochasticBehaviour.SEEDED

    def is_image_source_appropriate(self, image_source: ImageSource) -> bool:
        return image_source.sequence_type == ImageSequenceType.SEQUENTIAL

    def set_camera_intrinsics(self, camera_intrinsics: CameraIntrinsics, average_timestep: float) -> None:
        """
        Set the camera intrinisics for libviso2
        :param camera_intrinsics: The camera intrinsics, relative to the image resolution
        :param average_timestep: The average time between frames. Not relevant to libviso.
        :return:
        """
        logging.getLogger(__name__).debug("Setting camera intrinsics")
        self._focal_distance = float(camera_intrinsics.fx)
        self._cu = float(camera_intrinsics.cx)
        self._cv = float(camera_intrinsics.cy)
        self._width = float(camera_intrinsics.width)
        self._height = float(camera_intrinsics.height)

    def start_trial(self, sequence_type: ImageSequenceType, seed: int = 0) -> None:
        logging.getLogger(__name__).debug("Starting LibVisO trial...")
        self._start_time = time.time()
        if not sequence_type == ImageSequenceType.SEQUENTIAL:
            return

        self._viso = self.make_viso_instance()
        self._seed = seed
        self._viso.seed(seed)
        self._has_chosen_origin = False
        self._frame_results = []
        logging.getLogger(__name__).debug("    Started LibVisO trial.")

    def process_image(self, image: Image, timestamp: float) -> None:
        start_time = time.time()
        logging.getLogger(__name__).debug("Processing image at time {0} ...".format(timestamp))

        # This is the pose of the previous pose relative to the next one
        tracking, estimated_motion = self.handle_process_image(self._viso, image, timestamp)
        logging.getLogger(__name__).debug("    got estimated motion ...")
        end_time = time.time()

        frame_result = FrameResult(
            timestamp=timestamp,
            image=image.pk,
            processing_time=end_time - start_time,
            pose=image.camera_pose,
            tracking_state=TrackingState.OK if tracking else
            TrackingState.LOST if self._has_chosen_origin else TrackingState.NOT_INITIALIZED,
            estimated_motion=estimated_motion,
            num_matches=self._viso.getNumberOfMatches()
        )
        if tracking and not self._has_chosen_origin:
            # set the intial pose estimate to 0, so we can infer the later ones from the motions
            self._has_chosen_origin = True
            frame_result.estimated_pose = tf.Transform()
            frame_result.estimated_motion = None    # This will always be the identity on the first valid frame
        self._frame_results.append(frame_result)
        logging.getLogger(__name__).debug("    Processing done.")

    def finish_trial(self) -> SLAMTrialResult:
        logging.getLogger(__name__).debug("Finishing LibVisO trial ...")
        result = SLAMTrialResult(
            system=self,
            success=True,
            settings=self.get_settings(),
            results=self._frame_results,
            has_scale=self.has_scale
        )
        self._frame_results = None
        self._viso = None
        result.run_time = time.time() - self._start_time
        self._start_time = None
        logging.getLogger(__name__).debug("    Created result")
        return result

    def get_columns(self) -> typing.Set[str]:
        """
        Get the set of available properties for this system. Pass these to "get_properties", below.
        :return:
        """
        return set(self.columns.keys())

    def get_properties(self, columns: typing.Iterable[str] = None,
                       settings: typing.Mapping[str, typing.Any] = None) -> typing.Mapping[str, typing.Any]:
        """
        Get the values of the requested properties
        :param columns:
        :param settings:
        :return:
        """
        if columns is None:
            columns = self.columns.keys()
        if settings is None:
            settings = {}
        return {
            col_name: settings[col_name] if col_name in settings else self.columns.get_value(self, col_name)
            for col_name in columns
            if col_name in self.columns
        }

    @abc.abstractmethod
    def make_viso_instance(self):
        """
        Make the viso object. Stereo mode will make a stereo object, monocular a monocular object
        :return:
        """
        pass

    @abc.abstractmethod
    def handle_process_image(self, viso, image: Image, timestamp: float) -> \
            typing.Tuple[bool, typing.Union[tf.Transform, None]]:
        """
        Send the image to the viso object.
        In stereo mode, we need to send left and right frames, in monocular only one frame.
        :param viso: The viso object, created by 'make_viso_instance'
        :param image: The image object
        :param timestamp: The timestamp for this frame
        :return: True and a transform if the estimate is successful, False and None otherwise
        """
        pass

    @property
    @abc.abstractmethod
    def has_scale(self):
        pass

    def get_settings(self):
        return {
            'seed': self._seed,
            'in_fx': self._focal_distance,
            'in_fy': self._focal_distance,
            'in_cu': self._cu,
            'in_cv': self._cv,
            'in_height': self._height,
            'in_width': self._width
        }


class LibVisOStereoSystem(LibVisOSystem):
    """

    """

    ransac_iters = fields.IntegerField(default=200)
    inlier_threshold = fields.FloatField(default=2.0)
    reweighting = fields.BooleanField(default=True)

    # List of available metadata columns, and getters for each
    columns = ColumnList(
        LibVisOSystem.columns,
        base=None,
        ransac_iters=attrgetter('ransac_iters'),
        inlier_threshold=attrgetter('inlier_threshold'),
        reweighting=attrgetter('reweighting')
    )

    def __init__(self, *args, **kwargs):
        super(LibVisOStereoSystem, self).__init__(*args, **kwargs)

        # These will get overridden by set_stereo_offset
        self._base = 0.3

    @property
    def has_scale(self):
        return True

    def is_image_source_appropriate(self, image_source: ImageSource) -> bool:
        return (super(LibVisOStereoSystem, self).is_image_source_appropriate(image_source) and
                image_source.is_stereo_available)

    def set_stereo_offset(self, offset: tf.Transform) -> None:
        """
        Set the stereo baseline
        :param offset:
        :return:
        """
        baseline = -1 * offset.location[1]   # right is -Y axis
        logging.getLogger(__name__).debug("Setting stereo baseline to {0}".format(baseline))
        self._base = float(baseline)

    def make_viso_instance(self):
        """
        Construct a stereo libviso system
        :return:
        """
        params = Stereo_parameters()
        logging.getLogger(__name__).debug("    Created parameters object, populating ...")

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
        params.match.refinement = self.matcher_refinement.value
        logging.getLogger(__name__).debug("    Added matcher parameters ...")

        # Feature bucketing
        params.bucket.max_features = self.bucketing_max_features
        params.bucket.bucket_width = self.bucketing_bucket_width
        params.bucket.bucket_height = self.bucketing_bucket_height
        logging.getLogger(__name__).debug("    Added bucket parameters ...")

        # Stereo-specific parameters
        params.ransac_iters = self.ransac_iters
        params.inlier_threshold = self.inlier_threshold
        params.reweighting = self.reweighting
        logging.getLogger(__name__).debug("Added stereo specific parameters ...")

        # Camera calibration
        params.calib.f = self._focal_distance
        params.calib.cu = self._cu
        params.calib.cv = self._cv
        params.base = self._base
        logging.getLogger(__name__).debug("    Parameters built, creating viso object ...")

        return VisualOdometryStereo(params)

    def handle_process_image(self, viso, image: Image, timestamp: float) -> \
            typing.Tuple[bool, typing.Union[tf.Transform, None]]:
        """
        Send a frame to LibViso2, and get back the estimated motion
        :param viso: The visual odometry object. Will be a stereo object.
        :param image: The image object. Will be a stereo image
        :param timestamp: The timestamp
        :return: True and a transform if the estimate is successful, False and None otherwise
        """
        left_grey = prepare_image(image.left_pixels)
        right_grey = prepare_image(image.right_pixels)
        logging.getLogger(__name__).debug("    prepared images ...")

        success = viso.process_frame(left_grey, right_grey)
        logging.getLogger(__name__).debug("    processed frame ...")

        if success:
            motion = viso.getMotion()  # Motion is a 4x4 pose matrix
            np_motion = np.zeros((4, 4))
            motion.toNumpy(np_motion)
            np_motion = np.linalg.inv(np_motion)  # Invert the motion to make it new frame relative to old
            # This is the pose of the previous pose relative to the next one
            return True, make_relative_pose(np_motion)
        return False, None

    def get_settings(self):
        settings = super(LibVisOStereoSystem, self).get_settings()
        settings['base'] = self._base
        return settings

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
            matcher_nms_n: int = 3,
            matcher_nms_tau: int = 50,
            matcher_match_binsize: int = 50,
            matcher_match_radius: int = 200,
            matcher_match_disp_tolerance: int = 2,
            matcher_outlier_disp_tolerance: int = 5,
            matcher_outlier_flow_tolerance: int = 5,
            matcher_multi_stage: bool = True,
            matcher_half_resolution: bool = True,
            matcher_refinement: MatcherRefinement = MatcherRefinement.PIXEL,
            bucketing_max_features: int = 2,
            bucketing_bucket_width: int = 50,
            bucketing_bucket_height: int = 50,
            ransac_iters: int = 200,
            inlier_threshold: float = 2.0,
            reweighting: bool = True
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
            'matcher_nms_n': int(matcher_nms_n),
            'matcher_nms_tau': int(matcher_nms_tau),
            'matcher_match_binsize': int(matcher_match_binsize),
            'matcher_match_radius': int(matcher_match_radius),
            'matcher_match_disp_tolerance': int(matcher_match_disp_tolerance),
            'matcher_outlier_disp_tolerance': int(matcher_outlier_disp_tolerance),
            'matcher_outlier_flow_tolerance': int(matcher_outlier_flow_tolerance),
            'matcher_multi_stage': bool(matcher_multi_stage),
            'matcher_half_resolution': bool(matcher_half_resolution),
            'matcher_refinement': matcher_refinement.name,
            'bucketing_max_features': int(bucketing_max_features),
            'bucketing_bucket_width': int(bucketing_bucket_width),
            'bucketing_bucket_height': int(bucketing_bucket_height),
            'ransac_iters': int(ransac_iters),
            'inlier_threshold': float(inlier_threshold),
            'reweighting': bool(reweighting)
        })
        if all_objects.count() > 0:
            return all_objects.first()
        # There isn't an existing system with those settings, make a new one.
        obj = cls(
            matcher_nms_n=int(matcher_nms_n),
            matcher_nms_tau=int(matcher_nms_tau),
            matcher_match_binsize=int(matcher_match_binsize),
            matcher_match_radius=int(matcher_match_radius),
            matcher_match_disp_tolerance=int(matcher_match_disp_tolerance),
            matcher_outlier_disp_tolerance=int(matcher_outlier_disp_tolerance),
            matcher_outlier_flow_tolerance=int(matcher_outlier_flow_tolerance),
            matcher_multi_stage=bool(matcher_multi_stage),
            matcher_half_resolution=bool(matcher_half_resolution),
            matcher_refinement=matcher_refinement,
            bucketing_max_features=int(bucketing_max_features),
            bucketing_bucket_width=int(bucketing_bucket_width),
            bucketing_bucket_height=int(bucketing_bucket_height),
            ransac_iters=int(ransac_iters),
            inlier_threshold=float(inlier_threshold),
            reweighting=bool(reweighting)
        )
        return obj


class LibVisOMonoSystem(LibVisOSystem):
    """
    Class to run LibVisO2 as a vision system in monocular mode.
    """
    height = fields.FloatField(default=1.0)
    pitch = fields.FloatField(default=0.0)
    ransac_iters = fields.IntegerField(default=2000)
    inlier_threshold = fields.FloatField(default=0.00001)
    motion_threshold = fields.FloatField(default=100.0)

    # List of available metadata columns, and getters for each
    columns = ColumnList(
        LibVisOSystem.columns,
        height=attrgetter('height'),
        pitch=attrgetter('pitch'),
        ransac_iters=attrgetter('ransac_iters'),
        inlier_threshold=attrgetter('inlier_threshold'),
        motion_threshold=attrgetter('motion_threshold')
    )

    @property
    def has_scale(self):
        return False

    def make_viso_instance(self):
        """
        Make a monocular libviso instance
        :return:
        """
        params = Mono_parameters()
        logging.getLogger(__name__).debug("    Created parameters object, populating ...")

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
        params.match.refinement = self.matcher_refinement.value
        logging.getLogger(__name__).debug("    Added matcher parameters ...")

        # Feature bucketing
        params.bucket.max_features = self.bucketing_max_features
        params.bucket.bucket_width = self.bucketing_bucket_width
        params.bucket.bucket_height = self.bucketing_bucket_height
        logging.getLogger(__name__).debug("    Added bucket parameters ...")

        # Monocular-specific parameters
        params.height = self.height
        params.pitch = self.pitch
        params.ransac_iters = self.ransac_iters
        params.inlier_threshold = self.inlier_threshold
        params.motion_threshold = self.motion_threshold
        logging.getLogger(__name__).debug("Added monocular specific parameters ...")

        # Camera calibration
        params.calib.f = self._focal_distance
        params.calib.cu = self._cu
        params.calib.cv = self._cv
        logging.getLogger(__name__).debug("    Parameters built, creating viso object ...")

        return VisualOdometryMono(params)

    def handle_process_image(self, viso, image: Image, timestamp: float) -> \
            typing.Tuple[bool, typing.Union[tf.Transform, None]]:
        """
        Send a frame to LibViso2, and get back the estimated motion
        :param viso: The visual odometry object. Will be a stereo object.
        :param image: The image object. Will be a stereo image
        :param timestamp: The timestamp
        :return: True and a transform if the estimate is successful, False and None otherwise
        """
        image_greyscale = prepare_image(image.pixels)
        logging.getLogger(__name__).debug("    prepared images ...")

        success = self._viso.process_frame(image_greyscale)
        logging.getLogger(__name__).debug("    processed frame ...")

        if success:
            motion = self._viso.getMotion()  # Motion is a 4x4 pose matrix
            np_motion = np.zeros((4, 4))
            motion.toNumpy(np_motion)
            np_motion = np.linalg.inv(np_motion)  # Invert the motion to make it new frame relative to old
            # This is the pose of the previous pose relative to the next one
            return True, make_relative_pose(np_motion)
        return False, None

    @classmethod
    def get_instance(
            cls,
            matcher_nms_n: int = 3,
            matcher_nms_tau: int = 50,
            matcher_match_binsize: int = 50,
            matcher_match_radius: int = 200,
            matcher_match_disp_tolerance: int = 2,
            matcher_outlier_disp_tolerance: int = 5,
            matcher_outlier_flow_tolerance: int = 5,
            matcher_multi_stage: bool = True,
            matcher_half_resolution: bool = True,
            matcher_refinement: MatcherRefinement = MatcherRefinement.PIXEL,
            bucketing_max_features: int = 2,
            bucketing_bucket_width: int = 50,
            bucketing_bucket_height: int = 50,
            height: float = 1.0,
            pitch: float = 0.0,
            ransac_iters: int = 2000,
            inlier_threshold: float = 0.00001,
            motion_threshold: float = 100.0
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
            'matcher_nms_n': int(matcher_nms_n),
            'matcher_nms_tau': int(matcher_nms_tau),
            'matcher_match_binsize': int(matcher_match_binsize),
            'matcher_match_radius': int(matcher_match_radius),
            'matcher_match_disp_tolerance': int(matcher_match_disp_tolerance),
            'matcher_outlier_disp_tolerance': int(matcher_outlier_disp_tolerance),
            'matcher_outlier_flow_tolerance': int(matcher_outlier_flow_tolerance),
            'matcher_multi_stage': bool(matcher_multi_stage),
            'matcher_half_resolution': bool(matcher_half_resolution),
            'matcher_refinement': matcher_refinement.name,
            'bucketing_max_features': int(bucketing_max_features),
            'bucketing_bucket_width': int(bucketing_bucket_width),
            'bucketing_bucket_height': int(bucketing_bucket_height),
            'height': float(height),
            'pitch': float(pitch),
            'ransac_iters': int(ransac_iters),
            'inlier_threshold': float(inlier_threshold),
            'motion_threshold': float(motion_threshold)
        })
        if all_objects.count() > 0:
            return all_objects.first()
        # There isn't an existing system with those settings, make a new one.
        obj = cls(
            matcher_nms_n=int(matcher_nms_n),
            matcher_nms_tau=int(matcher_nms_tau),
            matcher_match_binsize=int(matcher_match_binsize),
            matcher_match_radius=int(matcher_match_radius),
            matcher_match_disp_tolerance=int(matcher_match_disp_tolerance),
            matcher_outlier_disp_tolerance=int(matcher_outlier_disp_tolerance),
            matcher_outlier_flow_tolerance=int(matcher_outlier_flow_tolerance),
            matcher_multi_stage=bool(matcher_multi_stage),
            matcher_half_resolution=bool(matcher_half_resolution),
            matcher_refinement=matcher_refinement,
            bucketing_max_features=int(bucketing_max_features),
            bucketing_bucket_width=int(bucketing_bucket_width),
            bucketing_bucket_height=int(bucketing_bucket_height),
            height=float(height),
            pitch=float(pitch),
            ransac_iters=int(ransac_iters),
            inlier_threshold=float(inlier_threshold),
            motion_threshold=float(motion_threshold)
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
