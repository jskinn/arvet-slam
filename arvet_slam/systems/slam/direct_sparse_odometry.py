# Copyright (c) 2017, John Skinner
import time
import logging
import typing
import numpy as np
import enum
from operator import attrgetter
import pymodm.fields as fields
from dso import DSOSystem, Undistort, UndistortPinhole, UndistortRadTan, Output3DWrapper, \
    FrameShell, CalibHessian, configure as dso_configure

import arvet.util.transform as tf
import arvet.util.image_utils as image_utils
from arvet.util.column_list import ColumnList
from arvet.database.enum_field import EnumField
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.system import VisionSystem, StochasticBehaviour
from arvet.core.image_source import ImageSource
from arvet.core.image import Image
from arvet.core.sequence_type import ImageSequenceType
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult, TrackingState


class RectificationMode(enum.Enum):
    NONE = 0
    CALIB = 1
    CROP = 2
    # FULL = 3 # I think FULL rectification mode is broken, from reading their code. The K Matrix is missing.


class DSO(VisionSystem):
    """
    Python wrapper for Direct Sparse Odometry (DSO)
    See https://github.com/JakobEngel/dso
    Bound to python using SWIG
    """
    rectification_mode = EnumField(RectificationMode, required=True)
    rectification_intrinsics = fields.EmbeddedDocumentField(CameraIntrinsics, required=True)

    columns = ColumnList(
        rectification_mode=attrgetter('rectification_mode'),
        height=attrgetter('rectification_intrinsics.height'),
        width=attrgetter('rectification_intrinsics.width'),
        fx=lambda obj: obj.rectification_intrinsics.fx if obj.rectification_mode is RectificationMode.CALIB else np.nan,
        fy=lambda obj: obj.rectification_intrinsics.fy if obj.rectification_mode is RectificationMode.CALIB else np.nan,
        cx=lambda obj: obj.rectification_intrinsics.cx if obj.rectification_mode is RectificationMode.CALIB else np.nan,
        cy=lambda obj: obj.rectification_intrinsics.cx if obj.rectification_mode is RectificationMode.CALIB else np.nan
    )

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

    @classmethod
    def is_deterministic(cls) -> StochasticBehaviour:
        """
        I don't think DSO is deterministic, LSD-SLAM apparently wasn't.
        This may change in response to testing or reading the source.

        :return: StochasticBehaviour.NON_DETERMINISTIC
        """
        return StochasticBehaviour.NON_DETERMINISTIC

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

    def start_trial(self, sequence_type: ImageSequenceType, seed: int = 0) -> None:
        """
        Start a trial with this system.
        After calling this, we can feed images to the system.
        When the trial is complete, call finish_trial to get the result.
        :param sequence_type: Are the provided images part of a sequence, or just unassociated pictures.
        :param seed: A random seed. Not used, but may be given.
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
        if self.rectification_mode is RectificationMode.NONE:
            # For no undistortion, simply pass through, out resolution is always
            self._undistorter = make_undistort_from_mode(
                self._intrinsics, self.rectification_mode, self._intrinsics.width, self._intrinsics.height)
        elif self.rectification_mode is RectificationMode.CALIB:
            # CALIB rectification uses the full intrinsics
            self._undistorter = make_undistort_from_out_intrinsics(self._intrinsics, self.rectification_intrinsics)
        else:
            # Otherwise, build an undistorter that crops to the configured fixed resolution
            self._undistorter = make_undistort_from_mode(
                self._intrinsics, self.rectification_mode,
                self.rectification_intrinsics.width, self.rectification_intrinsics.height
            )
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
            'rectification_mode': self.rectification_mode.name
        }
        if self.rectification_mode is RectificationMode.NONE:
            settings['width'] = self._intrinsics.width
            settings['height'] = self._intrinsics.height
        else:
            settings['width'] = self.rectification_intrinsics.width
            settings['height'] = self.rectification_intrinsics.height
        if self.rectification_mode is RectificationMode.CALIB:
            settings['out_fx'] = self.rectification_intrinsics.fx
            settings['out_fy'] = self.rectification_intrinsics.fy
            settings['out_cx'] = self.rectification_intrinsics.cx
            settings['out_cy'] = self.rectification_intrinsics.cy
        return settings

    @classmethod
    def get_instance(
            cls,
            rectification_mode: RectificationMode = None,
            rectification_intrinsics: CameraIntrinsics = None
    ) -> 'DSO':
        """
        Get an instance of this vision system, with some parameters, pulling from the database if possible,
        or construct a new one if needed.
        It is the responsibility of subclasses to ensure that as few instances of each system as possible exist
        within the database.
        Does not save the returned object, you'll usually want to do that straight away.
        :return:
        """
        if rectification_mode is None:
            raise ValueError("Cannot search for DSO without rectification mode")
        if rectification_intrinsics is None:
            raise ValueError("Cannot search for DSO without intrinsics")
        # Look for existing objects with the same settings
        query = {
            'rectification_mode': rectification_mode.name,
            'rectification_intrinsics.width': rectification_intrinsics.width,
            'rectification_intrinsics.height': rectification_intrinsics.height
        }
        if rectification_mode is RectificationMode.CALIB:
            # When using CALIB rectification, the other intrinsics matter
            query['rectification_intrinsics.fx'] = rectification_intrinsics.fx
            query['rectification_intrinsics.fy'] = rectification_intrinsics.fy
            query['rectification_intrinsics.cx'] = rectification_intrinsics.cx
            query['rectification_intrinsics.cy'] = rectification_intrinsics.cy
        all_objects = DSO.objects.raw(query)
        if all_objects.count() > 0:
            return all_objects.first()
        # There isn't an existing system with those settings, make a new one.
        obj = cls(
            rectification_mode=rectification_mode,
            rectification_intrinsics=rectification_intrinsics
        )
        return obj


class DSOOutputWrapper(Output3DWrapper):
    """
    A simple grabber for collecting frame estimates output from DSO
    """

    def __init__(self):
        super().__init__()
        self.frame_deltas = []

    def publishCamPose(self, frame: FrameShell, HCalib: CalibHessian) -> None:
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
