# Copyright (c) 2017, John Skinner
import time
import logging
import typing
from os import PathLike

import numpy as np
import re
import queue
import multiprocessing
import enum
from operator import attrgetter
import tempfile
from pathlib import Path
import pymodm.fields as fields

import arvet.util.transform as tf
import arvet.util.image_utils as image_utils
from arvet.util.associate import associate
from arvet.util.column_list import ColumnList
from arvet.config.path_manager import PathManager
from arvet.database.enum_field import EnumField
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.system import VisionSystem, StochasticBehaviour
from arvet.core.image_source import ImageSource
from arvet.core.image import Image
from arvet.core.sequence_type import ImageSequenceType
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult


# Try and use LibYAML where available, fall back to the python implementation
from yaml import dump as yaml_dump
try:
    from yaml import CDumper as YamlDumper
except ImportError:
    from yaml import Dumper as YamlDumper


class SensorMode(enum.Enum):
    MONOCULAR = 0
    STEREO = 1
    RGBD = 2


class OrbSlam2(VisionSystem):
    """
    Python wrapper for ORB_SLAM2
    """
    vocabulary_file = fields.CharField(required=True)
    mode = EnumField(SensorMode, required=True)
    depth_threshold = fields.FloatField(required=True, default=40.0)
    orb_num_features = fields.IntegerField(required=True, default=2000)
    orb_scale_factor = fields.FloatField(required=True, default=1.2)
    orb_num_levels = fields.IntegerField(required=True, default=8)
    orb_ini_threshold_fast = fields.IntegerField(required=True, default=12)
    orb_min_threshold_fast = fields.IntegerField(required=True, default=7)

    # List of available metadata columns, and getters for each
    columns = ColumnList(
        vocabulary_file=attrgetter('vocabulary_file'),
        mode=attrgetter('mode'),
        in_height=None,
        in_width=None,
        in_fx=None,
        in_fy=None,
        in_cx=None,
        in_cy=None,

        in_p1=None,
        in_p2=None,
        in_k1=None,
        in_k2=None,
        in_k3=None,
        depth_threshold=attrgetter('depth_threshold'),
        orb_num_features=attrgetter('orb_num_features'),
        orb_scale_factor=attrgetter('orb_scale_factor'),
        orb_num_levels=attrgetter('orb_num_levels'),
        orb_ini_threshold_fast=attrgetter('orb_ini_threshold_fast'),
        orb_min_threshold_fast=attrgetter('orb_min_threshold_fast')
    )

    def __init__(self, *args, **kwargs):
        super(OrbSlam2, self).__init__(*args, **kwargs)

        self._intrinsics = None
        self._framerate = 30
        self._stereo_baseline = None

        self._expected_completion_timeout = 3600     # This is how long we wait after the dataset is finished
        self._temp_folder = None
        self._actual_vocab_file = None

        self._settings_file = None
        self._child_process = None
        self._input_queue = None
        self._output_queue = None
        self._start_time = None
        self._partial_frame_results = None

    @classmethod
    def is_deterministic(cls) -> StochasticBehaviour:
        """
        ORB_SLAM2 is non-deterministic, it will always give different results.
        :return: StochasticBehaviour.NON_DETERMINISTIC
        """
        return StochasticBehaviour.NON_DETERMINISTIC

    def is_image_source_appropriate(self, image_source: ImageSource) -> bool:
        """
        Is the dataset appropriate for testing this vision system.
        This will depend on which sensor mode ORB_SLAM is configured in,
        stereo mode will require stereo to be available, while RGB-D mode will require depth to be available.
        Also checks the ORB feature pyramid settings against the

        :param image_source: The source for images that this system will potentially be run with.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        return (image_source.sequence_type == ImageSequenceType.SEQUENTIAL and (
            self.mode == SensorMode.MONOCULAR or
            (self.mode == SensorMode.STEREO and image_source.is_stereo_available) or
            (self.mode == SensorMode.RGBD and image_source.is_depth_available)
        ) and check_feature_pyramid_settings(
            img_width=image_source.camera_intrinsics.width,
            img_height=image_source.camera_intrinsics.height,
            orb_scale_factor=self.orb_scale_factor,
            orb_num_levels=self.orb_num_levels
        ))

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
        properties = {
            col_name: settings[col_name] if col_name in settings else self.columns.get_value(self, col_name)
            for col_name in columns
            if col_name in self.columns
        }
        if 'mode' in properties and not isinstance(properties['mode'], SensorMode):
            properties['mode'] = SensorMode[properties['mode']]
        return properties

    def set_camera_intrinsics(self, camera_intrinsics: CameraIntrinsics, average_timestep: float) -> None:
        """
        Set the intrinsics of the camera using
        :param camera_intrinsics: A metadata.camera_intrinsics.CameraIntriniscs object
        :param average_timestep: The average time interval between frames. Used to configure ORB_SLAM2
        :return:
        """
        if self._child_process is None:
            self._intrinsics = camera_intrinsics
            self._framerate = 1 / average_timestep

    def set_stereo_offset(self, offset: tf.Transform) -> None:
        """
        Set the stereo baseline configuration.
        :param offset:
        :return:
        """
        # ORBSLAM expects cameras to be coplanar, only offset to the right (-Y)
        self._stereo_baseline = -1 * offset.location[1]

    def resolve_paths(self, path_manager: PathManager):
        self._temp_folder = path_manager.get_temp_folder()
        self._actual_vocab_file = path_manager.find_file(self.vocabulary_file)

    def start_trial(self, sequence_type: ImageSequenceType, seed: int = 0) -> None:
        """
        Start a trial with this system.
        After calling this, we can feed images to the system.
        When the trial is complete, call finish_trial to get the result.
        :param sequence_type: Are the provided images part of a sequence, or just unassociated pictures.
        :param seed: A random seed. Ignored.
        :return: void
        """
        if sequence_type is not ImageSequenceType.SEQUENTIAL:
            return

        logging.getLogger(__name__).debug(
            "Starting ORBSLAM with the following settings:\n"
            "  vocab path: '{0}'\n"
            "  temp folder: '{1}'\n"
            "  stereo baseline: {2}\n"
            "  intrinsics: {3}\n"
            "  framerate: {4}".format(
                self._actual_vocab_file, self._temp_folder, self._stereo_baseline, self._intrinsics, self._framerate
            ))

        self._start_time = time.time()
        self.save_settings()  # we have to save the settings, so that orb-slam can load them
        self._partial_frame_results = {}
        self._input_queue = multiprocessing.Queue()
        self._output_queue = multiprocessing.Queue()
        self._child_process = multiprocessing.Process(target=run_orbslam,
                                                      args=(self._output_queue,
                                                            self._input_queue,
                                                            str(self._actual_vocab_file),
                                                            str(self._settings_file),
                                                            self.mode))
        self._child_process.daemon = True
        self._child_process.start()
        try:
            started = self._output_queue.get(block=True, timeout=self._expected_completion_timeout)
        except queue.Empty:
            started = None

        if started is None:
            # Failed to start, clean up and then raise exception
            self._stop_subprocess(terminate=True)
            self.remove_settings()
            self._input_queue = None
            self._output_queue = None
            self._partial_frame_results = None
            raise RuntimeError("Failed to start ORBSLAM2, timed out after {0} seconds".format(
                self._expected_completion_timeout))

    def process_image(self, image: Image, timestamp: float) -> None:
        """
        Process an image as part of the current run.
        Should automatically start a new trial if none is currently started.
        :param image: The image object for this frame
        :param timestamp: A timestamp or index associated with this image. Sometimes None.
        :return: void
        """
        if self._input_queue is not None:
            # Wait here, to throttle the input rate to the queue, and prevent it from growing too large
            # delay_time = 0
            # while self._input_queue.qsize() > 30 and delay_time < 10:
            #     time.sleep(1)
            #     delay_time += 1
            logging.getLogger(__name__).debug("Sending frame {0}...".format(len(self._partial_frame_results)))

            # Add the camera pose to the ground-truth trajectory
            self._partial_frame_results[timestamp] = FrameResult(
                timestamp=timestamp,
                image=image.pk,
                pose=image.camera_pose
            )

            # Send different input based on the running mode
            if self.mode == SensorMode.MONOCULAR:
                self._input_queue.put((image_utils.to_uint_image(image.pixels), None, timestamp))
            elif self.mode == SensorMode.STEREO:
                self._input_queue.put((image_utils.to_uint_image(image.left_pixels),
                                       image_utils.to_uint_image(image.right_pixels), timestamp))
            elif self.mode == SensorMode.RGBD:
                self._input_queue.put((image_utils.to_uint_image(image.pixels), image.depth, timestamp))

    def finish_trial(self) -> SLAMTrialResult:
        """
        End the current trial, returning a trial result.
        Return none if no trial is started.
        :return:
        :rtype TrialResult:
        """
        if self._input_queue is None:
            raise RuntimeError("Cannot finish ORBSLAM trial, failed to start.")

        # This will end the main loop, see run_orbslam, below
        self._input_queue.put(None)

        # Get the results from the subprocess
        timeout = (len(self._partial_frame_results) + 1) * self._expected_completion_timeout / 10
        try:
            frame_statistics = self._output_queue.get(
                block=True, timeout=timeout
            )
        except queue.Empty:
            frame_statistics = None

        # First, clean up (regardless of whether we got results)
        self._stop_subprocess(terminate=frame_statistics is None)
        self.remove_settings()
        self._input_queue = None
        self._output_queue = None

        # If we couldn't get the results, raise an exception
        if frame_statistics is None:
            raise RuntimeError("Failed to stop ORBSLAM2, timed out after {0} seconds".format(timeout))

        # Merge the frame statistics with the partial frame results
        matches = associate(self._partial_frame_results, frame_statistics, offset=0, max_difference=0.1)
        unrecognised_timestamps = set(frame_statistics.keys())
        for local_stamp, subprocess_stamp in matches:
            frame_result = self._partial_frame_results[local_stamp]
            frame_stats = frame_statistics[subprocess_stamp]
            if frame_stats[0] is not None:
                frame_result.processing_time = frame_stats[0]
                frame_result.num_features = frame_stats[1]
                frame_result.num_matches = frame_stats[2]
                frame_result.tracking_state = frame_stats[3]
                if frame_stats[4] is not None:
                    estimated_pose = np.identity(4)
                    estimated_pose[0:3, :] = frame_stats[4]
                    frame_result.estimated_pose = make_relative_pose(estimated_pose)
                unrecognised_timestamps.remove(subprocess_stamp)
        if len(unrecognised_timestamps) > 0:
            valid_timestamps = np.array(list(self._partial_frame_results.keys()))
            logging.getLogger(__name__).warning("Got inconsistent timestamps:\n" + '\n'.join(
                '{0} (closest was {1})'.format(
                    unrecognised_timestamp,
                    _find_closest(unrecognised_timestamp, valid_timestamps)
                )
                for unrecognised_timestamp in unrecognised_timestamps
            ))

        result = SLAMTrialResult(
            system=self.pk,
            success=len(self._partial_frame_results) > 0,
            results=[self._partial_frame_results[timestamp]
                     for timestamp in sorted(self._partial_frame_results.keys())],
            has_scale=(self.mode != SensorMode.MONOCULAR),
            settings={
                'in_fx': self._intrinsics.fx,
                'in_fy': self._intrinsics.fy,
                'in_cx': self._intrinsics.cx,
                'in_cy': self._intrinsics.cy,
                'in_k1': self._intrinsics.k1,
                'in_k2': self._intrinsics.k2,
                'in_p1': self._intrinsics.p1,
                'in_p2': self._intrinsics.p2,
                'in_k3': self._intrinsics.k3,
                'in_width': self._intrinsics.width,
                'in_height': self._intrinsics.height,
                'vocabulary_file': str(self.vocabulary_file),
                'mode': str(self.mode.name),
                'depth_threshold': self.depth_threshold,
                'orb_num_features': self.orb_num_features,
                'orb_scale_factor': self.orb_scale_factor,
                'orb_num_levels': self.orb_num_levels,
                'orb_ini_threshold_fast': self.orb_ini_threshold_fast,
                'orb_min_threshold_fast': self.orb_min_threshold_fast
            }
        )
        result.run_time = time.time() - self._start_time
        self._partial_frame_results = None
        self._start_time = None
        return result

    @classmethod
    def get_instance(
            cls,
            vocabulary_file: str = None,
            mode: SensorMode = None,
            depth_threshold: float = 40.0,
            orb_num_features: int = 2000,
            orb_scale_factor: float = 1.2,
            orb_num_levels: int = 8,
            orb_ini_threshold_fast: int = 12,
            orb_min_threshold_fast: int = 7
    ) -> 'OrbSlam2':
        """
        Get an instance of this vision system, with some parameters, pulling from the database if possible,
        or construct a new one if needed.
        It is the responsibility of subclasses to ensure that as few instances of each system as possible exist
        within the database.
        Does not save the returned object, you'll usually want to do that straight away.
        :return:
        """
        if vocabulary_file is None:
            raise ValueError("Cannot search for None vocabulary file, please specify a vocab file")
        if mode is None:
            raise ValueError("Cannot search for ORBSLAM without a mode, please specify a sensor mode")
        # Look for existing objects with the same settings
        all_objects = OrbSlam2.objects.raw({
            'vocabulary_file': str(vocabulary_file),
            'mode': str(mode.name),
            'depth_threshold': float(depth_threshold),
            'orb_num_features': int(orb_num_features),
            'orb_scale_factor': float(orb_scale_factor),
            'orb_num_levels': int(orb_num_levels),
            'orb_ini_threshold_fast': int(orb_ini_threshold_fast),
            'orb_min_threshold_fast': int(orb_min_threshold_fast)
        })
        if all_objects.count() > 0:
            return all_objects.first()
        # There isn't an existing system with those settings, make a new one.
        obj = cls(
            vocabulary_file=str(vocabulary_file),
            mode=mode,
            depth_threshold=float(depth_threshold),
            orb_num_features=int(orb_num_features),
            orb_scale_factor=float(orb_scale_factor),
            orb_num_levels=int(orb_num_levels),
            orb_ini_threshold_fast=int(orb_ini_threshold_fast),
            orb_min_threshold_fast=int(orb_min_threshold_fast)
        )
        return obj

    def save_settings(self):
        if self._settings_file is None:
            if self._temp_folder is None:
                raise RuntimeError("Cannot save settings, no configured temporary directory")
            if self._intrinsics is None:
                raise RuntimeError("Cannot save settings without the camera intrinsics")

            # Build the settings object
            orbslam_settings = {
                'Camera': {
                    # Camera calibration and distortion parameters (OpenCV)
                    # Most of these get overridden with the camera intrinsics at the start of the run.
                    'fx': self._intrinsics.fx,
                    'fy': self._intrinsics.fy,
                    'cx': self._intrinsics.cx,
                    'cy': self._intrinsics.cy,

                    'k1': self._intrinsics.k1,
                    'k2': self._intrinsics.k2,
                    'p1': self._intrinsics.p1,
                    'p2': self._intrinsics.p2,
                    'k3': self._intrinsics.k3,

                    'width': self._intrinsics.width,
                    'height': self._intrinsics.height,

                    # Camera frames per second
                    'fps': self._framerate,

                    # Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
                    # All the images in this system will be greyscale anyway
                    'RGB': 1
                },

                # Close/Far threshold. Baseline times. I don't know what this does.
                'ThDepth': self.depth_threshold,

                # Depthmap values factor (all my depth is in meters, rescaling is handled elsewhere)
                'DepthMapFactor': 1.0,

                'ORBextractor': {
                    # ORB Extractor: Number of features per image
                    'nFeatures': self.orb_num_features,

                    # ORB Extractor: Scale factor between levels in the scale pyramid
                    'scaleFactor': self.orb_scale_factor,

                    # ORB Extractor: Number of levels in the scale pyramid
                    'nLevels': self.orb_num_levels,

                    # ORB Extractor: Fast threshold
                    # Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
                    # Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
                    # You can lower these values if your images have low contrast
                    'iniThFAST': self.orb_ini_threshold_fast,
                    'minThFAST': self.orb_min_threshold_fast
                },
                # Viewer configuration expected by ORB_SLAM2
                # Since the viewer is disabled, these values don't matter, but need to exist
                'Viewer': {
                    'KeyFrameSize': 0.05,
                    'KeyFrameLineWidth': 1,
                    'GraphLineWidth': 0.9,
                    'PointSize': 2,
                    'CameraSize': 0.08,
                    'CameraLineWidth': 3,
                    'ViewpointX': 0,
                    'ViewpointY': -0.7,
                    'ViewpointZ': -1.8,
                    'ViewpointF': 500
                }
            }
            if self.mode is SensorMode.STEREO:
                if self._stereo_baseline is not None:
                    # stereo baseline times fx
                    orbslam_settings['Camera']['bf'] = float(self._stereo_baseline * self._intrinsics.fx)
                else:
                    raise RuntimeError("Cannot save stereo settings without a stereo baseline")

            # Choose a new settings file, using mkstemp to avoid collisions
            _, self._settings_file = tempfile.mkstemp(
                prefix='orb-slam2-settings-{0}-'.format(self.pk if self.pk is not None else 'unregistered'),
                suffix='.yaml',
                dir=self._temp_folder
            )
            self._settings_file = Path(self._settings_file)
            dump_config(self._settings_file, orbslam_settings)

    def remove_settings(self) -> None:
        """
        Get rid of the settings file after creating it using save_settings
        :return:
        """
        if self._settings_file is not None:
            if self._settings_file.exists():
                self._settings_file.unlink()
            self._settings_file = None

    def _stop_subprocess(self, terminate: bool = False, timeout: float = 5.0) -> None:
        """
        Stop the subprocess, by any means necessary.
        :param terminate: Whether to open with SIGTERM before trying to join, do when you know it's crashed.
        :return:
        """
        if self._child_process:
            if terminate:
                self._child_process.terminate()
            self._child_process.join(timeout=timeout)
            if not terminate and self._child_process.is_alive():
                # we didn't terminate before, but we've been unable to join, send sig-term
                self._child_process.terminate()
                self._child_process.join(timeout=timeout)
            if self._child_process.is_alive():
                # We've timed out after a terminate, kill it with fire
                self._child_process.kill()
            self._child_process = None


def load_config(filename):
    """
    Load an opencv yaml FileStorage file, accounting for a couple of inconsistencies in syntax.
    :param filename: The file to load from
    :return: A python object constructed from the config, or an empty dict if not found
    """
    config = {}
    with open(filename, 'r') as config_file:
        re_comment_split = re.compile('[%#]')
        for line in config_file:
            line = re_comment_split.split(line, 1)[0]
            if len(line) <= 0:
                continue
            else:
                key, value = line.split(':', 1)
                key = key.strip('"\' \t')
                value = value.strip()
                value_lower = value.lower()
                if value_lower == 'true':
                    actual_value = True
                elif value_lower == 'false':
                    actual_value = False
                else:
                    try:
                        actual_value = float(value)
                    except ValueError:
                        actual_value = value
                config[key] = actual_value
    return config


def dump_config(filename: typing.Union[str, PathLike], data: dict,
                dumper=YamlDumper, default_flow_style=False, **kwargs):
    """
    Dump the ORB_SLAM config to file,
    There's some fiddling with the format here so that OpenCV will read it on the other end.
    :param filename:
    :param data:
    :param dumper:
    :param default_flow_style:
    :param kwargs:
    :return:
    """
    with open(filename, 'w') as config_file:
        config_file.write("%YAML:1.0\n")
        return yaml_dump(nested_to_dotted(data), config_file, Dumper=dumper,
                         default_flow_style=default_flow_style, **kwargs)


def nested_to_dotted(data):
    """
    Change a nested dictionary to one with dot-separated keys
    This is for working with the weird YAML1.0 format expected by ORBSLAM config files
    :param data:
    :return:
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for inner_key, inner_value in nested_to_dotted(value).items():
                result[key + '.' + inner_key] = inner_value
        else:
            result[key] = value
    return result


def make_relative_pose(pose_matrix: np.ndarray) -> tf.Transform:
    """
    ORBSLAM2 is using the common CV coordinate frame Z forward, X right, Y down (I think)
    this function handles the coordinate frame

    Frame is: z forward, x right, y down
    Not documented, worked out by trial and error

    :param pose_matrix: The homogenous pose matrix, as a 4x4 matrix
    :return: A Transform object representing the pose of the current frame with respect to the previous frame
    """
    coordinate_exchange = np.array([[0, 0, 1, 0],
                                    [-1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 0, 1]])
    pose = np.dot(np.dot(coordinate_exchange, pose_matrix), coordinate_exchange.T)
    return tf.Transform(pose)


def _find_closest(value, options):
    """
    A quick helper for finding the closest timestamp to the
    :param value:
    :param options:
    :return:
    """
    if options.size <= 0:
        return None
    idx = (np.abs(options - value)).argmin()
    return options[idx]


def check_feature_pyramid_settings(img_width: int, img_height: int, orb_scale_factor: float, orb_num_levels: int) \
        -> bool:
    """
    Check that the ORB feature pyramid settings support the given image resolution.
    If the images are too small, or the settings wrong, the feature pyramid will produce levels that are
    either 0 or negative, resulting in crashes or undefined behaviour.

    This is reverse-engineered from the ORBSLAM2 source, particularly ORBextractor::ComputeKeyPointsOctTree
    in ORBExtractor.cc

    :param img_width: The width of the input image
    :param img_height: The height of the input image
    :param orb_scale_factor: The scale factor between layers of the feature pyramid
    :param orb_num_levels: The number of levels in the feature pyramid
    :return: True
    """
    # First, find the scale factor of the smallest level in the pyramid
    max_pyramid_scale = orb_scale_factor ** (orb_num_levels - 1)
    if img_width / max_pyramid_scale < 0.5 or img_height / max_pyramid_scale < 0.5:
        # First, this will definitely crash if any of the feature pyramid levels have a resolution < 1
        return False

    # ORBSLAM2 uses a 16 pixel edge margin on all sides (32 from width/height), and then splits into grids of 30px
    # If a feature pyramid level has dimension less than or equal to 0, the system will crash.
    # The 32 comes from EDGE_THRESHOLD (19), as EDGE_THRESHOLD-3 is added/subtracted from the min/max dimensions in
    # void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints) in ORBExtractor.cc
    # Thus, these are the width/height of the lowest resolution level in the feature pyramid
    smallest_layer_width = np.round(img_width / max_pyramid_scale) - 32
    smallest_layer_height = np.round(img_height / max_pyramid_scale) - 32
    if smallest_layer_width <= 0 or smallest_layer_height <= 0:
        # If either of these dimensions are less than or equal to 0, the ORB extraction will behave undefined
        return False

    # Next, check each level individually for bad values
    # There is an odd dependency to the aspect ratio in DistributeOctTree, so some layer width/height combinations
    # will cause problems.
    for level in range(orb_num_levels - 1, -1, -1):
        layer_width = np.round(img_width / (orb_scale_factor ** level)) - 32
        layer_height = np.round(img_height / (orb_scale_factor ** level)) - 32

        # This is initial number of nodes used in ORBextractor::DistributeOctTree
        # if it happens to be 0, then a derivative value hX will be inf, and things will break unpredictably
        # Negative values at least run, but will cause strange behaviour.
        n_ini = np.round(layer_width / layer_height)
        if n_ini <= 0:
            return False
        # A derivative value of n_ini, which is part of why 0 is not allowed
        # I don't know if it is possible to still hit this after passing previous checks, but it's bad if it happens
        hx = layer_width / n_ini
        if hx <= 0:
            return False

    # everything seems ok, proceed.
    return True


def run_orbslam(output_queue, input_queue, vocab_file, settings_file, mode):
    """
    Actually run the orbslam system. This is done in a separate process to isolate memory leaks,
    and in case it crashes.
    :param output_queue:
    :param input_queue:
    :param vocab_file:
    :param settings_file:
    :param mode:
    :return:
    """
    from orbslam2 import System, Tracking
    from arvet_slam.trials.slam.tracking_state import TrackingState

    # Map tracking states as output from ORB_SLAM to our tracking state
    tracking_mapping = {
        Tracking.SYSTEM_NOT_READY: TrackingState.NOT_INITIALIZED,
        Tracking.NO_IMAGES_YET: TrackingState.NOT_INITIALIZED,
        Tracking.NOT_INITIALIZED: TrackingState.NOT_INITIALIZED,
        Tracking.LOST: TrackingState.LOST,
        Tracking.OK: TrackingState.OK
    }

    sensor_mode = System.MONOCULAR
    if mode == SensorMode.RGBD:
        sensor_mode = System.RGBD
    elif mode == SensorMode.STEREO:
        sensor_mode = System.STEREO
    logging.getLogger(__name__).info("Starting ORBSLAM2 in {0} mode...".format(mode.name.lower()))

    orbslam_system = System(str(vocab_file), str(settings_file), sensor_mode, bUseViewer=False)
    output_queue.put('ORBSLAM started!')  # Tell the parent process we've set-up correctly and are ready to go.
    logging.getLogger(__name__).info("ORBSLAM2 Ready.")

    frame_statistics = {}
    running = True
    # prev_timestamp = None
    # prev_actual_time = 0
    while running:
        in_data = input_queue.get(block=True)
        if isinstance(in_data, tuple) and len(in_data) == 3:
            logging.getLogger(__name__).debug("... ORBSLAM processing frame {0}.".format(len(frame_statistics)))
            img1, img2, timestamp = in_data

            # Wait for the timestamp after the first frame
            # if prev_timestamp is not None:
            #     delay = max(0, timestamp - prev_timestamp - time.time() + prev_actual_time)
            #     if delay > 0:
            #         time.sleep(delay)
            # prev_timestamp = timestamp
            # prev_actual_time = time.time()

            processing_start = time.time()
            if mode == SensorMode.MONOCULAR:
                orbslam_system.process_mono(img1, timestamp)
            elif mode == SensorMode.STEREO:
                orbslam_system.process_stereo(img1, img2, timestamp)
            elif mode == SensorMode.RGBD:
                orbslam_system.process_rgbd(img1, img2, timestamp)
            processing_time = time.time() - processing_start

            # Record statistics about the current frame.
            num_features = orbslam_system.get_num_features()
            num_matches = orbslam_system.get_num_matches()
            current_state = orbslam_system.get_tracking_state()
            frame_statistics[timestamp] = [
                processing_time,
                num_features,
                num_matches,
                tracking_mapping[current_state],
                None
            ]
        else:
            # Non-matching input indicates the end of processing, stop the main loop
            logging.getLogger(__name__).info("Got terminate input, finishing up and sending results.")
            running = False

    # Get the trajectory from orbslam
    trajectory = orbslam_system.get_trajectory_points()

    # Associate the trajectory with the collected frame statistics
    trajectory = {est.timestamp: make_pose_mat(est) for est in trajectory}
    matches = associate(frame_statistics, trajectory, 0, max_difference=0.1)
    for frame_stamp, traj_stamp in matches:
        frame_statistics[frame_stamp][4] = trajectory[traj_stamp]

    # send the final trajectory to the parent
    output_queue.put(frame_statistics)

    # shut down the system. This is going to crash it, but that's ok, because it's a subprocess
    orbslam_system.shutdown()
    logging.getLogger(__name__).info("Finished running ORBSLAM2")


def make_pose_mat(est):
    """
    From the orbslam output, build a matrix
    :param est:
    :return:
    """
    return [
        [est.r00, est.r01, est.r02, est.t0],
        [est.r10, est.r11, est.r12, est.t1],
        [est.r20, est.r21, est.r22, est.t2]
    ]
