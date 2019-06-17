# Copyright (c) 2018, John Skinner
import typing
import pymodm
import pymodm.fields as fields
import numpy as np
from arvet.database.enum_field import EnumField
from arvet.core.image import Image
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult, check_trial_collection
import arvet.util.associate
import arvet.util.transform as tf
import arvet.util.trajectory_helpers as th
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult


class PoseError(pymodm.EmbeddedMongoModel):
    """
    Errors in a pose estimate. Given two poses, these are the ways we measure the difference between them.
    We have 5 different numbers for the translation error:
    x, y, z (cartesian coordinates), length & direction (polar coordinates)
    The orientation error has a single value: "rot"
    """
    x = fields.FloatField()
    y = fields.FloatField()
    z = fields.FloatField()
    length = fields.FloatField()
    direction = fields.FloatField()
    rot = fields.FloatField()


class FrameError(pymodm.EmbeddedMongoModel):
    """
    All the errors from a single frame
    One of these gets created for each frame for each trial
    """
    timestamp = fields.FloatField(required=True)
    image = fields.ReferenceField(Image, required=True)
    tracking = EnumField(TrackingState)
    absolute_error = fields.EmbeddedDocumentField(PoseError)
    relative_error = fields.EmbeddedDocumentField(PoseError)
    noise = fields.EmbeddedDocumentField(PoseError)
    num_features = fields.IntegerField(default=0)
    num_matches = fields.IntegerField(default=0)

    columns = {
        'timestamp': lambda self: self.timestamp,

        # Image metadata
        'lens_focal_distance': lambda self: self.image.metadata.lens_focal_distance,
        'aperture': lambda self: self.image.metadata.aperture,

        'red_mean': lambda self: self.image.metadata.red_mean,
        'red_std': lambda self: self.image.metadata.red_std,
        'green_mean': lambda self: self.image.metadata.green_mean,
        'green_std': lambda self: self.image.metadata.green_std,
        'blue_mean': lambda self: self.image.metadata.blue_mean,
        'blue_std': lambda self: self.image.metadata.blue_std,
        'depth_mean': lambda self: self.image.metadata.depth_mean,
        'depth_std': lambda self: self.image.metadata.depth_std,

        'environment_type': lambda self: self.image.metadata.environment_type,
        'light_level': lambda self: self.image.metadata.light_level.value(),
        'time_of_day': lambda self: self.image.metadata.time_of_day,

        'simulation_world': lambda self: self.image.metadata.simulation_world,
        'lighting_model': lambda self: self.image.metadata.lighting_model,
        'texture_mipmap_bias': lambda self: self.image.metadata.texture_mipmap_bias,
        'normal_maps_enabled': lambda self: self.image.metadata.normal_maps_enabled,
        'roughness_enabled': lambda self: self.image.metadata.roughness_enabled,
        'geometry_decimation': lambda self: self.image.metadata.geometry_decimation,

        # Estimate results
        'tracking': lambda self: 1.0 if self.tracking is TrackingState.OK else 0.0,

        'abs_error_x': lambda self: self.absolute_error.x,
        'abs_error_y': lambda self: self.absolute_error.y,
        'abs_error_z': lambda self: self.absolute_error.z,
        'abs_error_length': lambda self: self.absolute_error.length,
        'abs_error_direction': lambda self: self.absolute_error.direction,
        'abs_rot_error': lambda self: self.absolute_error.rot,

        'trans_error_x': lambda self: self.relative_error.x,
        'trans_error_y': lambda self: self.relative_error.y,
        'trans_error_z': lambda self: self.relative_error.z,
        'trans_error_length': lambda self: self.relative_error.length,
        'trans_error_direction': lambda self: self.relative_error.direction,
        'rot_error': lambda self: self.relative_error.rot,

        'trans_noise_x': lambda self: self.noise.x,
        'trans_noise_y': lambda self: self.noise.y,
        'trans_noise_z': lambda self: self.noise.z,
        'trans_noise_length': lambda self: self.noise.length,
        'trans_noise_direction': lambda self: self.noise.direction,
        'rot_noise': lambda self: self.noise.rot,

        'num_features': lambda self: self.num_features,
        'num_matches': lambda self: self.num_matches
    }

    def to_dict(self, columns: typing.Iterable[str] = None):
        """
        Flatten the frame error to a dictionary.
        This is used to construct rows in a Pandas data frame, so the keys are column names
        :return:
        """
        if columns is None:
            columns = self.columns.keys()
        return {
            column_name: self.columns[column_name](self)
            for column_name in columns
            if column_name in self.columns
        }


class FrameErrorResult(MetricResult):
    """
    Error observations per estimate of a pose
    """
    errors = pymodm.EmbeddedDocumentListField(FrameError)

    def get_results(self, columns: typing.Iterable[str] = None) -> typing.List[dict]:
        return [frame_error.to_dict(columns) for frame_error in self.errors]


class FrameErrorMetric(Metric):

    def is_trial_appropriate(self, trial_result):
        return isinstance(SLAMTrialResult, trial_result)

    def measure_results(self, trial_results: typing.Iterable[TrialResult]) -> 'FrameErrorResult':
        """
        Collect the errors
        :param trial_results: The results of several trials to aggregate
        :return:
        :rtype BenchmarkResult:
        """
        trial_results = list(trial_results)
        invalid_reason = check_trial_collection(trial_results)
        if invalid_reason is not None:
            return MetricResult(
                metric=self,
                trial_results=trial_results,
                success=False,
                message=invalid_reason
            )

        # First, we need to find the average computed trajectory, so we can estimate noise
        if len(trial_results) > 1:
            mean_computed_motions = th.compute_average_trajectory([
                trial_result.get_computed_camera_motions() for trial_result in trial_results
            ])
        else:
            # We don't want to estimate noise for a single trajectory, in that case it should always be NaN
            mean_computed_motions = {}

        # Then, tally all the errors for all the computed trajectories
        estimate_errors = []
        for trial_result in trial_results:
            # Find a mapping from the timestamps in the frame results to the timestamps in the average trajectory
            to_average = {
                k: v for k, v in arvet.util.associate.associate(
                    [frame_result.timestamp for frame_result in trial_result.results], mean_computed_motions,
                    offset=0, max_difference=0.1
                )
            }
            for frame_result in trial_result.results:
                # Collect together the error statistics for this frame result
                frame_error = FrameError(
                    timestamp=frame_result.timestamp,
                    image=frame_result.image,
                    tracking=frame_result.tracking_state,
                    num_features=frame_result.num_features,
                    num_matches=frame_result.num_matches
                )
                if frame_result.estimated_pose is not None:
                    frame_result.absolute_error = make_pose_error(frame_result.estimated_pose, frame_result.pose)
                if frame_result.estimated_motion is not None:
                    frame_error.relative_error = make_pose_error(frame_result.estimated_motion, frame_result.motion)
                    if frame_result.timestamp in to_average:
                        frame_error.noise = make_pose_error(
                            frame_result.estimated_motion,
                            mean_computed_motions[to_average[frame_result.timestamp]]
                        )
                estimate_errors.append(frame_error)

        # Once we've tallied all the results, either succeed or fail based on the number of results.
        if len(estimate_errors) <= 0:
            return MetricResult(
                metric=self,
                trial_results=trial_results,
                success=False,
                message="No measurable errors for these trajectories"
            )
        return FrameErrorResult(
            metric=self,
            trial_results=trial_results,
            success=True,
            estimate_errors=estimate_errors
        )


def make_pose_error(estimated_pose: tf.Transform, reference_pose: tf.Transform) -> PoseError:
    """

    :param estimated_pose:
    :param reference_pose:
    :return:
    """
    trans_error = estimated_pose.location - reference_pose.location
    trans_error_length = np.linalg.norm(trans_error)

    trans_error_direction = 0   # No direction if the vectors are the same
    if trans_error_length > 0:
        # Get the unit vector in the direction of the true location
        unit_reference = reference_pose.location / np.linalg.norm(reference_pose.location)
        # Find the angle between the trans error and the true location
        trans_error_direction = np.arccos(
            # Clip to arccos range to avoid errors
            np.clip(
                # a \dot b = |a||b|cos theta
                np.dot(trans_error / trans_error_length, unit_reference),
                -1.0, 1.0
            )
        )
    # Different to the trans_direction, this is the angle between the estimated orientation and true orientation
    rot_error = tf.quat_diff(estimated_pose.rotation_quat(w_first=True), reference_pose.rotation_quat(w_first=True))
    return PoseError(
        x=trans_error[0],
        y=trans_error[1],
        z=trans_error[2],
        length=trans_error_length,
        direction=trans_error_direction,
        rot=rot_error
    )
