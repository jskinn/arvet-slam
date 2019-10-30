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
import tempfile
from pathlib import Path
import pymodm.fields as fields
from dso import DSOSystem, Undistort, UndistortPinhole, UndistortRadTan, Output3DWrapper, FrameShell, CalibHessian, configure as dso_configure

import arvet.util.transform as tf
import arvet.util.image_utils as image_utils
from arvet.config.path_manager import PathManager
from arvet.database.enum_field import EnumField
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.image import Image
from arvet.core.sequence_type import ImageSequenceType
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult, TrackingState


# Try and use LibYAML where available, fall back to the python implementation
from yaml import dump as yaml_dump
try:
    from yaml import CDumper as YamlDumper
except ImportError:
    from yaml import Dumper as YamlDumper


class RectificationMode(enum.Enum):
    NONE = 0
    CALIB = 1
    CROP = 2
    # FULL = 3  # I think FULL rectification mode is broken, from reading their code. The K Matrix is missing.


class DSO(VisionSystem):
    """
    Python wrapper for Direct Sparse Odometry (DSO)
    See https://github.com/JakobEngel/dso
    Bound to python using SWIG
    """
    rectification_mode = EnumField(RectificationMode, required=True)
    rectification_intrinsics = fields.EmbeddedDocumentField(CameraIntrinsics, required=True)

    def __init__(self, *args, **kwargs):
        super(DSO, self).__init__(*args, **kwargs)

        self._intrinsics = None
        self._framerate = 30
        self._stereo_baseline = None
        self._has_photometric_calibration = False

        self._undistorter = None
        self._output_wrapper = None
        self._system = None

        self._start_time = None
        self._image_index = 0
        self._frame_results = None
        self._processing_start_times = None

    @property
    def is_deterministic(self) -> bool:
        """
        Is the visual system deterministic.

        If this is false, it will have to be tested multiple times, because the performance will be inconsistent
        between runs.
        I don't think DSO is deterministic, LSD-SLAM apparently wasn't

        :return: True iff the algorithm will produce the same results each time.
        :rtype: bool
        """
        return False

    def is_image_source_appropriate(self, image_source: ImageSource) -> bool:
        """
        Is the dataset appropriate for testing this vision system.
        This will depend on which sensor mode ORB_SLAM is configured in,
        stereo mode will require stereo to be available, while RGB-D mode will require depth to be available.

        :param image_source: The source for images that this system will potentially be run with.
        :return: True iff the particular dataset is appropriate for this vision system.
        :rtype: bool
        """
        return image_source.sequence_type == ImageSequenceType.SEQUENTIAL

    def get_columns(self) -> typing.Set[str]:
        pass

    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        pass

    def set_camera_intrinsics(self, camera_intrinsics: CameraIntrinsics, average_timestep: float) -> None:
        """
        Set the intrinsics of the camera using
        :param camera_intrinsics: A metadata.camera_intrinsics.CameraIntriniscs object
        :param average_timestep: The average time interval between frames. Used to configure ORB_SLAM2
        :return:
        """
        if self._system is None:
            self._intrinsics = camera_intrinsics
            self._framerate = 1 / average_timestep

    def start_trial(self, sequence_type: ImageSequenceType) -> None:
        """
        Start a trial with this system.
        After calling this, we can feed images to the system.
        When the trial is complete, call finish_trial to get the result.
        :param sequence_type: Are the provided images part of a sequence, or just unassociated pictures.
        :return: void
        """
        if sequence_type is not ImageSequenceType.SEQUENTIAL:
            raise RuntimeError("Cannot start trial with {0} image source".format(sequence_type.name))
        if self._intrinsics is None:
            raise RuntimeError("Cannot start trial, intrinsics have not been provided yet")

        self._start_time = time.time()
        self._frame_results = {}
        self._processing_start_times = {}

        # Figure out mode and preset for DSO
        # mode:
        #   mode = 0 - use iff a photometric calibration exists(e.g.TUM monoVO dataset).
        #   mode = 1 - use iff NO photometric calibration exists(e.g.ETH EuRoC MAV dataset).
        #   mode = 2 - use iff images are not photometrically distorted(e.g.syntheticdatasets).
        # preset:
        #   preset = 0 - default settings (2k pts etc.), not enforcing real - time execution
        #   preset = 1 - default settings (2k pts etc.), enforcing 1x real - time execution
        # WARNING: These two overwrite image resolution with 424 x 320.
        #   preset = 2 - fast settings (800 pts etc.), not enforcing real - time execution.
        #   preset = 3 - fast settings (800 pts etc.), enforcing 5x real - time execution
        mode = 1
        preset = 0

        dso_configure(preset=preset, mode=mode, quiet=True, nolog=True)

        # Build the undistorter, this will preprocess images and remove distortion
        if self.rectification_mode is RectificationMode.CALIB:
            self._undistorter = make_undistort_from_out_intrinsics(self._intrinsics, self.rectification_intrinsics)
        else:
            self._undistorter = make_undistort_from_mode(self._intrinsics, self.rectification_mode,
                                                         self.rectification_intrinsics.width,
                                                         self.rectification_intrinsics.height)
        if mode is not 0:
            self._undistorter.setNoPhotometricCalibration()
        self._undistorter.applyGlobalConfig()   # Need to do this to set camera intrinsics

        # Make an output wrapper to accumulate output information
        self._output_wrapper = DSOOutputWrapper()

        # Build the system itself.
        self._system = DSOSystem()
        self._system.outputWrapper.append(self._output_wrapper)

        # TODO: Build a listener for getting out values
        self._start_time = time.time()
        self._image_index = 0

    def process_image(self, image: Image, timestamp: float) -> None:
        """
        Process an image as part of the current run.
        Should automatically start a new trial if none is currently started.
        :param image: The image object for this frame
        :param timestamp: A timestamp or index associated with this image. Sometimes None.
        :return: void
        """
        if self._undistorter is None:
            raise RuntimeError("Cannot process image, trial has not started yet. Call 'start_trial'")
        image_data = image_utils.to_uint_image(image_utils.convert_to_grey(image.pixels))
        dso_img = self._undistorter.undistort_greyscale(image_data, 0, timestamp, 1.0)
        self._processing_start_times[timestamp] = time.time()
        self._system.addActiveFrame(dso_img, self._image_index)
        self._image_index += 1

        self._frame_results[timestamp] = FrameResult(
            timestamp=timestamp,
            image=image.pk,
            pose=image.camera_pose,
            tracking_state=TrackingState.NOT_INITIALIZED,
            processing_time=np.nan
        )

    def finish_trial(self) -> SLAMTrialResult:
        """
        End the current trial, returning a trial result.
        Return none if no trial is started.
        :return:
        :rtype TrialResult:
        """
        if self._system is None:
            raise RuntimeError("Cannot finish trial, no trial started. Call 'start_trial'")

        # Wait for the system to finish
        self._system.blockUntilMappingIsFinished()

        # Collate the frame results
        unrecognised_timestamps = set()
        for timestamp, trans, rot, finish_time in self._output_wrapper.frame_deltas:
            if timestamp in self._frame_results:
                self._frame_results[timestamp].estimated_pose = make_pose(trans, rot)
                self._frame_results[timestamp].processing_time = finish_time - self._processing_start_times[timestamp]
                self._frame_results[timestamp].tracking_state = TrackingState.OK
            else:
                unrecognised_timestamps.add(timestamp)
        if len(unrecognised_timestamps) > 0:
            valid_timestamps = np.array(list(self._frame_results.keys()))
            logging.getLogger(__name__).warning("Got inconsistent timestamps:\n" + '\n'.join(
                '{0} (closest was {1})'.format(
                    unrecognised_timestamp,
                    _find_closest(unrecognised_timestamp, valid_timestamps)
                )
                for unrecognised_timestamp in unrecognised_timestamps
            ))

        # Organize the tracking state, it is NOT_INITIALIZED until we are first found, then it is LOST
        found = False
        for timestamp in sorted(self._frame_results.keys()):
            if self._frame_results[timestamp].tracking_state is TrackingState.OK:
                found = True
            elif found and self._frame_results[timestamp].tracking_state is TrackingState.NOT_INITIALIZED:
                self._frame_results[timestamp].tracking_state = TrackingState.LOST

        # Clean up
        self._undistorter = None
        self._system = None
        self._output_wrapper = None

        result = SLAMTrialResult(
            system=self.pk,
            success=len(self._frame_results) > 0,
            results=[self._frame_results[timestamp]
                     for timestamp in sorted(self._frame_results.keys())],
            has_scale=False,
            settings=self.make_settings()
        )
        result.run_time = time.time() - self._start_time
        self._frame_results = None
        self._start_time = None
        return result

    def make_settings(self):
        settings = {
            'rectification_mode': self.rectification_mode.name,
            'width': self.rectification_intrinsics.width,
            'height': self.rectification_intrinsics.height
        }
        if self.rectification_mode is RectificationMode.CALIB:
            settings['out_fx'] = self.rectification_intrinsics.fx
            settings['out_fy'] = self.rectification_intrinsics.fy
            settings['out_cx'] = self.rectification_intrinsics.cx
            settings['out_cy'] = self.rectification_intrinsics.cy
        return settings


class DSOOutputWrapper(Output3DWrapper):

    def __init__(self):
        super().__init__()
        self.frame_deltas = []

    # def publishGraph(self, connectivity: 'std::map< uint64_t,Eigen::Vector2i,std::less< uint64_t >,Eigen::aligned_allocator< std::pair< uint64_t const,Eigen::Vector2i > > > const &') -> None:
    #     print("Got a graph? Ok I guess...")

    # def needPushDepthImage(self) -> bool:
    #     print("Asked about depth images")
    #     return False

    # def publishKeyframes(self, frames: 'FrameHessianVec', final: 'bool', HCalib: 'CalibHessian') -> None:
    #     print("GOT A KEYFRAME!!!!!")

    def publishCamPose(self, frame: FrameShell, HCalib: CalibHessian) -> None:
        # print("OUT: Current Frame {0} (time {1}, internal ID {2}). CameraToWorld:".format(
        #     frame.incoming_id,
        #     frame.timestamp,
        #     frame.id
        # ))
        # print("Translation: {0}, Rotation: {1}".format(
        #     frame.get_cam_to_world_translation(),
        #     frame.get_cam_to_world_rotation()
        # ))

        self.frame_deltas.append([
            frame.timestamp,
            np.squeeze(frame.get_cam_to_world_translation()),
            np.squeeze(frame.get_cam_to_world_rotation()),
            time.time()
        ])


def make_pose(translation, rotation) -> tf.Transform:
    """
    ORBSLAM2 is using the common CV coordinate frame Z forward, X right, Y down (I think)
    this function handles the coordinate frame

    Frame is: z forward, x right, y down
    Not documented, worked out by trial and error

    :param translation: The translation 3-vector
    :param rotation: The rotation quaternion, w-last
    :return: A Transform object representing the pose of the current frame with respect to the previous frame
    """
    # coordinate_exchange = np.array([[0, 0, 1, 0],
    #                                 [-1, 0, 0, 0],
    #                                 [0, -1, 0, 0],
    #                                 [0, 0, 0, 1]])
    # pose = np.dot(np.dot(coordinate_exchange, pose_matrix), coordinate_exchange.T)
    x, y, z = translation
    qx, qy, qz, qw = rotation
    return tf.Transform(
        location=(z, -x, -y),
        rotation=(qw, qz, -qx, -qy),
        w_first=True
    )


def make_undistort_from_mode(intrinsics: CameraIntrinsics, rectification_mode: RectificationMode,
                             out_width: int, out_height: int) -> Undistort:
    """
    Make a DSO undistorter using a constant rectification mode, and fixed output size.
    Will produce either UndistortPinhole or UndistortRadTan, depending on whether there is any actual distortion.
    See https://github.com/JakobEngel/dso#geometric-calibration-file
    this function handles cases where the third line is "none", "full" or "crop", see below for the other case

    :param intrinsics: The intrinsics of the input images
    :param rectification_mode: The rectification mode to use, will default to 'crop'
    :param out_width: The width of the output images
    :param out_height: The height of the output images
    :return: A new DSO undistort object.
    """
    rect_mode = Undistort.RECT_CROP
    if rectification_mode is RectificationMode.NONE:
        rect_mode = Undistort.RECT_NONE
    elif rectification_mode is RectificationMode.CALIB:
        raise ValueError("Cannot build Undistort with CALIB mode without intrinsics, "
                         "use 'make_undistort_from_out_intrinsics'")

    if intrinsics.k1 == 0 and intrinsics.k2 == 0 and \
            intrinsics.p1 == 0 and intrinsics.p2 == 0:
        return UndistortPinhole(
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            intrinsics.width,
            intrinsics.height,
            rect_mode,
            out_width,
            out_height
        )
    return UndistortRadTan(
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        intrinsics.k1,
        intrinsics.k2,
        intrinsics.p1,
        intrinsics.p2,
        intrinsics.width,
        intrinsics.height,
        rect_mode,
        out_width,
        out_height
    )


def make_undistort_from_out_intrinsics(intrinsics: CameraIntrinsics, out_intrinsics: CameraIntrinsics) -> Undistort:
    """
    Make a DSO undistorter using a different intrinsics matrix to rectify
    Will produce either UndistortPinhole or UndistortRadTan, depending on whether there is any actual distortion.
    See https://github.com/JakobEngel/dso#geometric-calibration-file
    this function handles cases where the third line is fx fy cx cy

    :param intrinsics: Intrinsics of the input images, including distortion
    :param out_intrinsics: Specification of the output images. Note that distortion values here will be ignored.
    :return: A new DSO Undistort object.
    """
    if intrinsics.k1 == 0 and intrinsics.k2 == 0 and \
            intrinsics.p1 == 0 and intrinsics.p2 == 0:
        return UndistortPinhole(
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            intrinsics.width,
            intrinsics.height,
            out_intrinsics.fx / out_intrinsics.width,
            out_intrinsics.fy / out_intrinsics.height,
            out_intrinsics.cx / out_intrinsics.width,
            out_intrinsics.cy / out_intrinsics.height,
            out_intrinsics.width,
            out_intrinsics.height
        )
    return UndistortRadTan(
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        intrinsics.k1,
        intrinsics.k2,
        intrinsics.p1,
        intrinsics.p2,
        intrinsics.width,
        intrinsics.height,
        out_intrinsics.fx / out_intrinsics.width,
        out_intrinsics.fy / out_intrinsics.height,
        out_intrinsics.cx / out_intrinsics.width,
        out_intrinsics.cy / out_intrinsics.height,
        out_intrinsics.width,
        out_intrinsics.height
    )


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
