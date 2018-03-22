# Copyright (c) 2017, John Skinner
import os
import time
import logging
import numpy as np
import copy
import re
import signal
import queue
import multiprocessing
import enum
import tempfile
import arvet.core.system
import arvet.core.sequence_type
import arvet.core.trial_result
import arvet_slam.trials.slam.visual_slam
import arvet.util.transform as tf
import arvet.util.dict_utils as du
import arvet.util.image_utils as image_utils


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


class ORBSLAM2(arvet.core.system.VisionSystem):
    """
    Python wrapper for ORB_SLAM2
    """

    def __init__(self, vocabulary_file, settings, mode=SensorMode.RGBD, temp_folder='temp', id_=None):
        super().__init__(id_=id_)
        self._vocabulary_file = vocabulary_file

        self._mode = mode if isinstance(mode, SensorMode) else SensorMode.RGBD
        # Default settings based on UE4 calibration results
        self._orbslam_settings = du.defaults({}, settings, {
            'Camera': {
                # Camera calibration and distortion parameters (OpenCV)
                # Most of these get overridden with the camera intrinsics at the start of the run.
                'fx': 640,
                'fy': 480,
                'cx': 320,
                'cy': 240,

                'k1': 0,
                'k2': 0,
                'p1': 0,
                'p2': 0,
                'k3': 0,

                # Camera frames per second
                'fps': 30.0,

                # stereo baseline times fx
                'bf': 0,

                # Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
                # All the images in this system will be RGB order
                'RGB': 1
            },

            # Close/Far threshold. Baseline times. I don't know what this does.
            'ThDepth': 35.0,

            # Depthmap values factor (all my depth is in meters, rescaling is handled elsewhere)
            'DepthMapFactor': 1.0,

            'ORBextractor': {
                # ORB Extractor: Number of features per image
                'nFeatures': 2000,

                # ORB Extractor: Scale factor between levels in the scale pyramid
                'scaleFactor': 1.2,

                # ORB Extractor: Number of levels in the scale pyramid
                'nLevels': 8,

                # ORB Extractor: Fast threshold
                # Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
                # Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
                # You can lower these values if your images have low contrast
                'iniThFAST': 20,
                'minThFAST': 7
            },
            # Viewer configuration expected by ORB_SLAM2
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
        })
        self._temp_folder = temp_folder

        self._expected_completion_timeout = 3600     # This is how long we wait after the dataset is finished
        self._actual_vocab_file = None
        self._settings_file = None
        self._child_process = None
        self._input_queue = None
        self._output_queue = None
        self._gt_trajectory = None

    @property
    def mode(self):
        return self._mode

    @property
    def is_deterministic(self):
        """
        Is the visual system deterministic.

        If this is false, it will have to be tested multiple times, because the performance will be inconsistent
        between runs.

        :return: True iff the algorithm will produce the same results each time.
        :rtype: bool
        """
        return False

    def is_image_source_appropriate(self, image_source):
        """
        Is the dataset appropriate for testing this vision system.
        :param image_source: The source for images that this system will potentially be run with.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        return (image_source.sequence_type == arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL and (
            self._mode == SensorMode.MONOCULAR or
            (self._mode == SensorMode.STEREO and image_source.is_stereo_available) or
            (self._mode == SensorMode.RGBD and image_source.is_depth_available)))

    def set_camera_intrinsics(self, camera_intrinsics):
        """
        Set the intrinsics of the camera using
        :param camera_intrinsics: A metadata.camera_intrinsics.CameraIntriniscs object
        :return:
        """
        if self._child_process is None:
            self._orbslam_settings['Camera']['width'] = camera_intrinsics.width
            self._orbslam_settings['Camera']['height'] = camera_intrinsics.height
            self._orbslam_settings['Camera']['bf'] = (camera_intrinsics.fx * self._orbslam_settings['Camera']['bf']
                                                      / self._orbslam_settings['Camera']['fx'])
            self._orbslam_settings['Camera']['fx'] = camera_intrinsics.fx
            self._orbslam_settings['Camera']['fy'] = camera_intrinsics.fy
            self._orbslam_settings['Camera']['cx'] = camera_intrinsics.cx
            self._orbslam_settings['Camera']['cy'] = camera_intrinsics.cy
            self._orbslam_settings['Camera']['k1'] = camera_intrinsics.k1
            self._orbslam_settings['Camera']['k2'] = camera_intrinsics.k2
            self._orbslam_settings['Camera']['k3'] = camera_intrinsics.k3
            self._orbslam_settings['Camera']['p1'] = camera_intrinsics.p1
            self._orbslam_settings['Camera']['p2'] = camera_intrinsics.p2

    def set_stereo_baseline(self, baseline):
        """
        Set the stereo baseline configuration.
        :param baseline:
        :return:
        """
        self._orbslam_settings['Camera']['bf'] = float(baseline) * self._orbslam_settings['Camera']['fx']

    def resolve_paths(self, path_manager):
        self._actual_vocab_file = path_manager.find_file(self._vocabulary_file)

    def start_trial(self, sequence_type):
        """
        Start a trial with this system.
        After calling this, we can feed images to the system.
        When the trial is complete, call finish_trial to get the result.
        :param sequence_type: Are the provided images part of a sequence, or just unassociated pictures.
        :return: void
        """
        if sequence_type is not arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL:
            return

        self.save_settings()  # we have to save the settings, so that orb-slam can load them
        self._gt_trajectory = {}
        self._input_queue = multiprocessing.Queue()
        self._output_queue = multiprocessing.Queue()
        self._child_process = multiprocessing.Process(target=run_orbslam,
                                                      args=(self._output_queue,
                                                            self._input_queue,
                                                            self._actual_vocab_file,
                                                            self._settings_file,
                                                            self._mode))
        self._child_process.daemon = True
        self._child_process.start()
        self._output_queue.close()  # We're not putting data into that, close our thread.
        try:
            started = self._output_queue.get(block=True, timeout=self._expected_completion_timeout)
        except queue.Empty:
            logging.getLogger(__name__).error("Failed to start ORBSLAM2, timed out after {0} seconds".format(
                self._expected_completion_timeout))
            started = None
        if started is None:
            self._input_queue.close()
            self._output_queue.close()
            self._child_process.terminate()
            self._child_process.join(timeout=10)
            if self._child_process.is_alive():
                os.kill(self._child_process.pid, signal.SIGKILL)  # Definitely kill the process.
            if os.path.isfile(self._settings_file):
                os.remove(self._settings_file)  # Delete the settings file
            self._settings_file = None
            self._child_process = None
            self._input_queue = None
            self._output_queue = None
            self._gt_trajectory = None
        return started is not None

    def process_image(self, image, timestamp):
        """
        Process an image as part of the current run.
        Should automatically start a new trial if none is currently started.
        :param image: The image object for this frame
        :param timestamp: A timestamp or index associated with this image. Sometimes None.
        :return: void
        """
        if self._input_queue is not None:
            # Wait here, to throttle the input rate to the queue, and prevent it from growing too large
            delay_time = 0
            while self._input_queue.qsize() > 30 and delay_time < 10:
                time.sleep(1)
                delay_time += 1

            # Add the camera pose to the ground-truth trajectory
            self._gt_trajectory[timestamp] = image.camera_pose

            # Send different input based on the running mode
            if self._mode == SensorMode.MONOCULAR:
                self._input_queue.put((image_utils.to_uint_image(image.data), None, timestamp))
            elif self._mode == SensorMode.STEREO:
                self._input_queue.put((image_utils.to_uint_image(image.left_data),
                                       image_utils.to_uint_image(image.right_data), timestamp))
            elif self._mode == SensorMode.RGBD:
                self._input_queue.put((image_utils.to_uint_image(image.data),
                                       image_utils.to_uint_image(image.depth_data), timestamp))

    def finish_trial(self):
        """
        End the current trial, returning a trial result.
        Return none if no trial is started.
        :return:
        :rtype TrialResult:
        """
        if self._input_queue is None:
            logging.getLogger(__name__).warning("Cannot finish ORBSLAM trial, failed to start.")
            return None

        # This will end the main loop, see run_orbslam, below
        self._input_queue.put(None)
        self._input_queue.close()

        # First, get the length of the outputs, this affects how long we will wait for the future output
        output_size = get_with_default(self._output_queue, self._expected_completion_timeout,
                                       default=self._expected_completion_timeout / 10)
        trajectory_list = get_with_default(self._output_queue, output_size * 10, None)
        tracking_stats = get_with_default(self._output_queue, output_size * 10, {})
        num_features = get_with_default(self._output_queue, output_size * 10, {})
        num_matches = get_with_default(self._output_queue, output_size * 10, {})

        if isinstance(trajectory_list, list):
            # completed successfully, return the trajectory
            self._child_process.join()    # explicitly join

            # Build the trajectory from raw data
            trajectory = {}
            for (timestamp, r00, r01, r02, t0,
                 r10, r11, r12, t1,
                 r20, r21, r22, t2) in trajectory_list:
                trajectory[timestamp] = make_relative_pose(
                    np.array([
                        [r00, r01, r02, t0],
                        [r10, r11, r12, t1],
                        [r20, r21, r22, t2],
                        [0, 0, 0, 1],
                    ])
                )

            result = arvet_slam.trials.slam.visual_slam.SLAMTrialResult(
                system_id=self.identifier,
                trajectory=trajectory,
                ground_truth_trajectory=self._gt_trajectory,
                tracking_stats=tracking_stats,
                num_features=num_features,
                num_matches=num_matches,
                sequence_type=arvet.core.sequence_type.ImageSequenceType.SEQUENTIAL,
                system_settings=self.get_settings()
            )
        else:
            # something went wrong, kill it with fire
            logging.getLogger(__name__).error("Failed to stop ORBSLAM2, timed out after {0} seconds".format(
                self._expected_completion_timeout))
            result = None
            self._child_process.terminate()
            self._child_process.join(timeout=5)
            if self._child_process.is_alive():
                os.kill(self._child_process.pid, signal.SIGKILL)    # Definitely kill the process.

        if os.path.isfile(self._settings_file):
            os.remove(self._settings_file)  # Delete the settings file
        self._settings_file = None
        self._child_process = None
        self._input_queue = None
        self._output_queue = None
        self._gt_trajectory = None
        return result

    def get_settings(self):
        return self._orbslam_settings

    def save_settings(self):
        if self._settings_file is None:
            # Choose a new settings file, using mkstemp to avoid collisions
            fp = tempfile.NamedTemporaryFile(
                prefix='orb-slam2-settings-{0}-'.format(
                    self.identifier if self.identifier is not None else 'unregistered'),
                suffix='.yaml',
                dir=self._temp_folder,
                mode='w',
                delete=False
            )
            self._settings_file = fp.name
            fp.write('%')
            fp.close()  # close the open file handle, file should now exist
            dump_config(self._settings_file, self._orbslam_settings)

    def validate(self):
        valid = super().validate()
        if not os.path.isfile(self._vocabulary_file):
            valid = False
        return valid

    def serialize(self):
        serialized = super().serialize()
        serialized['vocabulary_file'] = self._vocabulary_file
        serialized['mode'] = self._mode.value

        # Need to clean and minimize the settings we're saving, specifically omitting camera and viewer settings
        settings = copy.deepcopy(self.get_settings())
        del settings['Camera']
        del settings['Viewer']
        serialized['settings'] = settings
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'vocabulary_file' in serialized_representation:
            kwargs['vocabulary_file'] = serialized_representation['vocabulary_file']
        if 'mode' in serialized_representation:
            kwargs['mode'] = SensorMode(serialized_representation['mode'])
        if 'settings' in serialized_representation:
            kwargs['settings'] = serialized_representation['settings']
        kwargs['temp_folder'] = db_client.temp_folder
        return super().deserialize(serialized_representation, db_client, **kwargs)


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


def dump_config(filename, data, dumper=YamlDumper, default_flow_style=False, **kwargs):
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


def get_with_default(queue_: multiprocessing.Queue, timeout: float, default=None):
    """
    A tiny helper to retrieve from a multiprocessing queue,
    returning the given default value if we time out.
    :param queue_:
    :param timeout:
    :param default:
    :return:
    """
    try:
        result = queue_.get(block=True, timeout=timeout)
    except queue.Empty:
        result = default
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
    coordinate_exchange = np.matrix([[0, 0, 1, 0],
                                     [-1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])
    pose = np.dot(np.dot(coordinate_exchange, pose_matrix), coordinate_exchange.T)
    return tf.Transform(pose)


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
    import orbslam2
    import arvet_slam.trials.slam.tracking_state as tracking_state
    import time

    tracking_stats = {}
    num_features = {}
    num_matches = {}
    input_queue.close()  # We're not putting data into that, close our thread to it.

    sensor_mode = orbslam2.Sensor.RGBD
    if mode == SensorMode.MONOCULAR:
        sensor_mode = orbslam2.Sensor.MONOCULAR
    elif mode == SensorMode.STEREO:
        sensor_mode = orbslam2.Sensor.STEREO
    logging.getLogger(__name__).info("Starting ORBSLAM2 in {0} mode...".format(sensor_mode.name.lower()))

    orbslam_system = orbslam2.System(vocab_file, settings_file, sensor_mode)
    orbslam_system.set_use_viewer(True)
    orbslam_system.initialize()
    output_queue.put('ORBSLAM started!')  # Tell the parent process we've set-up correctly and are ready to go.
    logging.getLogger(__name__).info("ORBSLAM2 Ready.")

    running = True
    prev_timestamp = None
    prev_actual_time = 0
    while running:
        in_data = input_queue.get(block=True)
        if isinstance(in_data, tuple) and len(in_data) == 3:
            img1, img2, timestamp = in_data

            # Wait for the timestamp after the first frame
            if prev_timestamp is not None:
                time.sleep(max(0, timestamp - prev_timestamp - time.time() + prev_actual_time))
            prev_timestamp = timestamp
            prev_actual_time = time.time()

            if mode == SensorMode.MONOCULAR:
                orbslam_system.process_image_mono(img1, timestamp)
            elif mode == SensorMode.STEREO:
                orbslam_system.process_image_stereo(img1, img2, timestamp)
            elif mode == SensorMode.RGBD:
                orbslam_system.process_image_rgbd(img1, img2, timestamp)

            # Record statistics about the current frame.
            num_features[timestamp] = orbslam_system.get_num_features()
            num_matches[timestamp] = orbslam_system.get_num_matched_features()
            current_state = orbslam_system.get_tracking_state()
            if (current_state == orbslam2.TrackingState.SYSTEM_NOT_READY or
                    current_state == orbslam2.TrackingState.NO_IMAGES_YET or
                    current_state == orbslam2.TrackingState.NOT_INITIALIZED):
                tracking_stats[timestamp] = tracking_state.TrackingState.NOT_INITIALIZED
            elif current_state == orbslam2.TrackingState.OK:
                tracking_stats[timestamp] = tracking_state.TrackingState.OK
            else:
                tracking_stats[timestamp] = tracking_state.TrackingState.LOST
        else:
            # Non-matching input indicates the end of processing, stop the main loop
            logging.getLogger(__name__).info("Got terminate input, finishing up and sending results.")
            running = False

    # send the final trajectory to the parent
    output_queue.put(len(tracking_stats))
    output_queue.put(orbslam_system.get_trajectory_points())
    output_queue.put(tracking_stats)
    output_queue.put(num_features)
    output_queue.put(num_matches)
    output_queue.close()

    # shut down the system. This is going to crash it, but that's ok, because it's a subprocess
    orbslam_system.shutdown()
    logging.getLogger(__name__).info("Finished running ORBSLAM2")
