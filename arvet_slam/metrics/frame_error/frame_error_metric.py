# Copyright (c) 2018, John Skinner
import typing
from operator import attrgetter
import pymodm
import pymodm.fields as fields
import numpy as np
from arvet.database.enum_field import EnumField
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.image import Image
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult, check_trial_collection
from arvet.util.column_list import ColumnList
import arvet.util.associate
import arvet.util.transform as tf
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult


class PoseError(pymodm.EmbeddedMongoModel):
    """
    Errors in a pose estimate. Given two poses, these are the ways we measure the difference between them.
    We have 5 different numbers for the translation error:
    x, y, z (cartesian coordinates), length & direction (polar coordinates)
    The orientation error has a single value: "rot"
    """
    x = fields.FloatField(required=True)
    y = fields.FloatField(required=True)
    z = fields.FloatField(required=True)
    length = fields.FloatField(required=True)
    direction = fields.FloatField(required=True)
    rot = fields.FloatField(required=True)


class FrameError(pymodm.EmbeddedMongoModel):
    """
    All the errors from a single frame
    One of these gets created for each frame for each trial
    """
    repeat = fields.IntegerField(required=True)
    timestamp = fields.FloatField(required=True)
    image = fields.ReferenceField(Image, required=True, on_delete=fields.ReferenceField.CASCADE)
    tracking = EnumField(TrackingState, default=TrackingState.OK)
    absolute_error = fields.EmbeddedDocumentField(PoseError)
    relative_error = fields.EmbeddedDocumentField(PoseError)
    noise = fields.EmbeddedDocumentField(PoseError)
    num_features = fields.IntegerField(default=0)
    num_matches = fields.IntegerField(default=0)

    columns = ColumnList(
        repeat=attrgetter('repeat'),
        timestamp=attrgetter('timestamp'),
        tracking=attrgetter('is_tracking'),

        abs_error_x=attrgetter('absolute_error.x'),
        abs_error_y=attrgetter('absolute_error.y'),
        abs_error_z=attrgetter('absolute_error.z'),
        abs_error_length=attrgetter('absolute_error.length'),
        abs_error_direction=attrgetter('absolute_error.direction'),
        abs_rot_error=attrgetter('absolute_error.rot'),

        trans_error_x=attrgetter('relative_error.x'),
        trans_error_y=attrgetter('relative_error.y'),
        trans_error_z=attrgetter('relative_error.z'),
        trans_error_length=attrgetter('relative_error.length'),
        trans_error_direction=attrgetter('relative_error.direction'),
        rot_error=attrgetter('relative_error.rot'),

        trans_noise_x=lambda obj: obj.noise.x if obj.noise is not None else None,
        trans_noise_y=lambda obj: obj.noise.y if obj.noise is not None else None,
        trans_noise_z=lambda obj: obj.noise.z if obj.noise is not None else None,
        trans_noise_length=lambda obj: obj.noise.length if obj.noise is not None else None,
        trans_noise_direction=lambda obj: obj.noise.direction if obj.noise is not None else None,
        rot_noise=lambda obj: obj.noise.rot if obj.noise is not None else None,

        num_features=attrgetter('num_features'),
        num_matches=attrgetter('num_matches')
    )

    @property
    def is_tracking(self) -> bool:
        return self.tracking is TrackingState.OK

    def get_properties(self, columns: typing.Iterable[str] = None, other_properties: dict = None):
        """
        Flatten the frame error to a dictionary.
        This is used to construct rows in a Pandas data frame, so the keys are column names
        :return:
        """
        if other_properties is None:
            other_properties = {}
        image_properties = self.image.get_properties(columns)
        if columns is None:
            columns = set(self.columns.keys())
        error_properties = {
            column_name: self.columns.get_value(self, column_name)
            for column_name in columns
            if column_name in self.columns
        }
        return {
            **other_properties,
            **image_properties,
            **error_properties
        }


class FrameErrorResult(MetricResult):
    """
    Error observations per estimate of a pose
    """
    system = fields.ReferenceField(VisionSystem, required=True, on_delete=pymodm.ReferenceField.CASCADE)
    image_source = fields.ReferenceField(ImageSource, required=True, on_delete=pymodm.ReferenceField.CASCADE)
    errors = fields.EmbeddedDocumentListField(FrameError, required=True, blank=True)
    image_columns = fields.ListField(fields.CharField())

    def get_columns(self) -> typing.Set[str]:
        columns = set(FrameError.columns.keys()) | set(self.image_columns) \
                  | self.system.get_columns() | self.image_source.get_columns() | self.metric.get_columns()
        return columns

    def get_results(self, columns: typing.Iterable[str] = None) -> typing.List[dict]:
        """
        Collate together the results for this metric result
        Can capture many different independent variables, including those from the system
        :param columns:
        :return:
        """
        system_properties = self.system.get_properties(columns)
        image_source_properties = self.image_source.get_properties(columns)
        metric_properties = self.metric.get_properties(columns)
        other_properties = {**system_properties, **image_source_properties, **metric_properties}
        return [frame_error.get_properties(columns, other_properties) for frame_error in self.errors]

    @classmethod
    def visualize_results(cls, results: typing.Iterable[MetricResult], output_folder: str,
                          plots: typing.Iterable[str] = None) -> None:
        pass


class FrameErrorMetric(Metric):

    def get_columns(self) -> typing.Set[str]:
        """
        The frame error metric has no parameters, and provides no properties
        :return:
        """
        return set()

    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        """
        The frame error metric has no parameters, and provides no properties
        :param columns:
        :return:
        """
        return {}

    def is_trial_appropriate(self, trial_result):
        return isinstance(SLAMTrialResult, trial_result)

    def measure_results(self, trial_results: typing.Iterable[TrialResult]) -> FrameErrorResult:
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

        # First, we need to find the average computed motions, so we can estimate noise
        if len(trial_results) > 1:
            mean_computed_motions = find_average_motions([
                trial_result.get_computed_camera_motions() for trial_result in trial_results
            ])
        else:
            # We don't want to estimate noise for a single trajectory, in that case it should always be NaN
            mean_computed_motions = {}

        # Then, tally all the errors for all the computed trajectories
        estimate_errors = []
        image_columns = set()
        for repeat, trial_result in enumerate(trial_results):
            # Find a mapping from the timestamps in the frame results to the timestamps in the average trajectory
            to_average = {
                k: v for k, v in arvet.util.associate.associate(
                    {frame_result.timestamp: True for frame_result in trial_result.results}, mean_computed_motions,
                    offset=0, max_difference=0.1
                )
            }
            scaled_motions = trial_result.get_computed_camera_motions()
            scaled_trajectory = trial_result.get_computed_camera_poses()
            gt_origin = None
            estimate_origin = None
            for frame_result in trial_result.results:
                # Collect together the error statistics for this frame result
                image_columns |= frame_result.image.get_columns()
                frame_error = FrameError(
                    repeat=repeat,
                    timestamp=frame_result.timestamp,
                    image=frame_result.image,
                    tracking=frame_result.tracking_state,
                    num_features=frame_result.num_features,
                    num_matches=frame_result.num_matches
                )
                if frame_result.timestamp in scaled_trajectory and \
                        scaled_trajectory[frame_result.timestamp] is not None:
                    if estimate_origin is None:
                        gt_origin = frame_result.pose
                        estimate_origin = scaled_trajectory[frame_result.timestamp]
                    frame_error.absolute_error = make_pose_error(
                        estimate_origin.find_relative(scaled_trajectory[frame_result.timestamp]),
                        gt_origin.find_relative(frame_result.pose)
                    )
                if frame_result.timestamp in scaled_motions and scaled_motions[frame_result.timestamp] is not None:
                    frame_error.relative_error = make_pose_error(
                        scaled_motions[frame_result.timestamp], frame_result.motion)
                    if frame_result.timestamp in to_average and \
                            mean_computed_motions[to_average[frame_result.timestamp]] is not None:
                        frame_error.noise = make_pose_error(
                            scaled_motions[frame_result.timestamp],
                            mean_computed_motions[to_average[frame_result.timestamp]]
                        )
                estimate_errors.append(frame_error)

        # Once we've tallied all the results, either succeed or fail based on the number of results.
        if len(estimate_errors) <= 0:
            return FrameErrorResult(
                metric=self,
                trial_results=trial_results,
                success=False,
                message="No measurable errors for these trajectories"
            )
        return FrameErrorResult(
            metric=self,
            trial_results=trial_results,
            system=trial_results[0].system,
            image_source=trial_results[0].image_source,
            success=True,
            image_columns=list(image_columns),
            errors=estimate_errors
        )


def make_pose_error(estimated_pose: tf.Transform, reference_pose: tf.Transform) -> PoseError:
    """

    :param estimated_pose:
    :param reference_pose:
    :return:
    """
    trans_error = estimated_pose.location - reference_pose.location
    trans_error_length = np.linalg.norm(trans_error)

    trans_error_direction = np.nan   # No direction if the vectors are the same
    if trans_error_length > 0:
        # Get the unit vector in the direction of the true location
        reference_norm = np.linalg.norm(reference_pose.location)
        if reference_norm > 0:
            unit_reference = reference_pose.location / reference_norm
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


def find_average_motions(trajectories: typing.Iterable[typing.Mapping[float, tf.Transform]]) \
        -> typing.Mapping[float, tf.Transform]:
    """
    Find the average motions from a number of estimated motions
    We use a custom implementation here because we only want the average to exist when there is uncertainty.
    If there is only one estimate for a given time, we should return None, rather than that single estimate.
    Can handle small variations in timestamps, but privileges timestamps from earlier trajectories for association
    :param trajectories:
    :return:
    """
    associated_times = {}
    associated_poses = {}
    for traj in trajectories:
        traj_times = set(traj.keys())
        # First, add all the times that can be associated to an existing time
        matches = arvet.util.associate.associate(associated_times, traj, offset=0, max_difference=0.1)
        for match in matches:
            associated_times[match[0]].append(match[1])
            if traj[match[1]] is not None:
                associated_poses[match[0]].append(traj[match[1]])
            traj_times.remove(match[1])
        # Add all the times in this trajectory that don't have associations yet
        for time in traj_times:
            associated_times[time] = [time]
            associated_poses[time] = [traj[time]] if traj[time] is not None else []
    # Take the median associated time and pose together
    return {
        np.median(associated_times[time]): tf.compute_average_pose(associated_poses[time])
        if time in associated_poses and len(associated_poses[time]) > 1 else None
        for time in associated_times.keys()
    }
