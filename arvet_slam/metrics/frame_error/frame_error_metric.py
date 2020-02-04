# Copyright (c) 2018, John Skinner
import typing
import logging
from operator import attrgetter
import bson
import numpy as np

import pymodm
import pymodm.fields as fields
from pymodm.queryset import QuerySet
from pymodm.manager import Manager
from pymodm.context_managers import no_auto_dereference

from arvet.database.autoload_modules import autoload_modules
from arvet.database.reference_list_field import ReferenceListField
from arvet.database.enum_field import EnumField
from arvet.database.transform_field import TransformField
from arvet.core.system import VisionSystem
from arvet.core.image_source import ImageSource
from arvet.core.image import Image
from arvet.core.trial_result import TrialResult
from arvet.core.metric import Metric, MetricResult, T_MetricResult, check_trial_collection
from arvet.util.column_list import ColumnList
import arvet.util.associate
import arvet.util.transform as tf
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult
import arvet_slam.metrics.frame_error.plots.plot_abs_error_distrubution as abs_error_plot


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


class FrameError(pymodm.MongoModel):
    """
    All the errors from a single frame
    One of these gets created for each frame for each trial
    """
    trial_result = fields.ReferenceField(TrialResult, required=True)
    repeat = fields.IntegerField(required=True)
    timestamp = fields.FloatField(required=True)
    image = fields.ReferenceField(Image, required=True, on_delete=fields.ReferenceField.CASCADE)
    motion = TransformField(required=True)
    processing_time = fields.FloatField(default=np.nan)

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
        processing_time=attrgetter('processing_time'),
        motion_x=attrgetter('motion.x'),
        motion_y=attrgetter('motion.y'),
        motion_z=attrgetter('motion.z'),
        motion_roll=lambda obj: obj.motion.euler[0],
        motion_pitch=lambda obj: obj.motion.euler[1],
        motion_yaw=lambda obj: obj.motion.euler[2],
        num_features=attrgetter('num_features'),
        num_matches=attrgetter('num_matches'),

        abs_error_x=lambda obj: obj.absolute_error.x if obj.absolute_error is not None else np.nan,
        abs_error_y=lambda obj: obj.absolute_error.y if obj.absolute_error is not None else np.nan,
        abs_error_z=lambda obj: obj.absolute_error.z if obj.absolute_error is not None else np.nan,
        abs_error_length=lambda obj: obj.absolute_error.length if obj.absolute_error is not None else np.nan,
        abs_error_direction=lambda obj: obj.absolute_error.direction if obj.absolute_error is not None else np.nan,
        abs_rot_error=lambda obj: obj.absolute_error.rot if obj.absolute_error is not None else np.nan,

        trans_error_x=lambda obj: obj.relative_error.x if obj.relative_error is not None else np.nan,
        trans_error_y=lambda obj: obj.relative_error.y if obj.relative_error is not None else np.nan,
        trans_error_z=lambda obj: obj.relative_error.z if obj.relative_error is not None else np.nan,
        trans_error_length=lambda obj: obj.relative_error.length if obj.relative_error is not None else np.nan,
        trans_error_direction=lambda obj: obj.relative_error.direction if obj.relative_error is not None else np.nan,
        rot_error=lambda obj: obj.relative_error.rot if obj.relative_error is not None else np.nan,

        trans_noise_x=lambda obj: obj.noise.x if obj.noise is not None else np.nan,
        trans_noise_y=lambda obj: obj.noise.y if obj.noise is not None else np.nan,
        trans_noise_z=lambda obj: obj.noise.z if obj.noise is not None else np.nan,
        trans_noise_length=lambda obj: obj.noise.length if obj.noise is not None else np.nan,
        trans_noise_direction=lambda obj: obj.noise.direction if obj.noise is not None else np.nan,
        rot_noise=lambda obj: obj.noise.rot if obj.noise is not None else np.nan
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


class FrameErrorResultQuerySet(QuerySet):
    
    def delete(self):
        """
        When a frame error result is deleted, also delete the frame errors it refers to
        :return:
        """
        frame_error_ids = set(err_id for doc in self.values() for err_id in doc['errors'])
        FrameError.objects.raw({'_id': {'$in': list(frame_error_ids)}}).delete()
        super(FrameErrorResultQuerySet, self).delete()


FrameErrorResultManger = Manager.from_queryset(FrameErrorResultQuerySet)


class FrameErrorResult(MetricResult):
    """
    Error observations per estimate of a pose
    """
    system = fields.ReferenceField(VisionSystem, required=True, on_delete=pymodm.ReferenceField.CASCADE)
    image_source = fields.ReferenceField(ImageSource, required=True, on_delete=pymodm.ReferenceField.CASCADE)
    errors = ReferenceListField(FrameError, required=True, blank=True, on_delete=pymodm.ReferenceField.CASCADE)
    image_columns = fields.ListField(fields.CharField())

    objects = FrameErrorResultManger()

    def save(self, cascade=None, full_clean=True, force_insert=False):
        """
        When saving, also save the
        :param cascade:
        :param full_clean:
        :param force_insert:
        :return:
        """
        for frame_error in self.errors:
            frame_error.save(cascade, full_clean, force_insert)
        super(FrameErrorResult, self).save(cascade, full_clean, force_insert)

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
    def get_available_plots(cls) -> typing.Set[str]:
        """
        Get the set of available plots for this metric.
        That is, these are the values that when passed to visualise_results, actually do something.
        They should also be human-readable names.
        :return: A set of valid plot names
        """
        return {
            abs_error_plot.NAME
        }

    @classmethod
    def visualize_results(cls: typing.Type[T_MetricResult],
                          results: typing.Iterable[T_MetricResult],
                          plots: typing.Collection[str],
                          display: bool = True, output: str = '') -> None:
        """
        Visualize
        :param results:
        :param plots:
        :param display:
        :param output:
        :return:
        """
        # Plotting imports, which we don't want to bother with most of the time.
        import pandas as pd
        import matplotlib.pyplot as plt

        plots = set(plots)

        # Work out which data we need to make the requested plots
        columns = set()
        if abs_error_plot.NAME in plots:
            columns |= abs_error_plot.get_required_columns()

        # We don't need any data, and won't produce any plots. return.
        if len(columns) <= 0:
            return

        # Collect the relevant data from the metric results and build the dataframe
        logging.getLogger(__name__).info("Collating results...")
        data = []
        for metric_result in results:
            data.extend(metric_result.get_results(columns))
        dataframe = pd.DataFrame(data)

        # Delegate plotting the results
        logging.getLogger(__name__).info("Plotting...")
        if abs_error_plot.NAME in plots:
            abs_error_plot.plot(dataframe, output)

        # Show the generated plots
        if display:
            plt.show()


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
        return isinstance(trial_result, SLAMTrialResult)

    def measure_results(self, trial_results: typing.Iterable[TrialResult]) -> FrameErrorResult:
        """
        Collect the errors
        TODO: Track the error introduced by a loop closure, somehow. Might need to track loop closures in the FrameResult
        :param trial_results: The results of several trials to aggregate
        :return:
        :rtype BenchmarkResult:
        """
        trial_results = list(trial_results)

        # preload model types for the models linked to the trial results.
        with no_auto_dereference(SLAMTrialResult):
            model_ids = set(tr.system for tr in trial_results if isinstance(tr.system, bson.ObjectId))
            autoload_modules(VisionSystem, list(model_ids))
            model_ids = set(tr.image_source for tr in trial_results if isinstance(tr.image_source, bson.ObjectId))
            autoload_modules(ImageSource, list(model_ids))

        # Check if the set of trial results is valid. Loads the models.
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
                    trial_result=trial_result,
                    repeat=repeat,
                    timestamp=frame_result.timestamp,
                    image=frame_result.image,
                    tracking=frame_result.tracking_state,
                    processing_time=frame_result.processing_time,
                    motion=frame_result.motion,
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


def quat_cosine(q1: typing.Union[typing.Sequence, np.ndarray], q2: typing.Union[typing.Sequence, np.ndarray]) -> float:
    """
    Find the cosine of the  angle between the two quaternions
    This is similar to the quat-diff, but without the arccos, which makes it more stable around zero
    :param q1: A quaternion, [w, x, y, z]
    :param q2: A quaternion, [w, x, y, z]
    :return:
    """
    q1 = np.asarray(q1)
    if np.dot(q1, q2) < 0:
        # Quaternions have opposite handedness, flip q1 since it's already an ndarray
        q1 = -1 * q1
    q_inv = q1 * np.array([1.0, -1.0, -1.0, -1.0])
    q_inv = q_inv / np.linalg.norm(q_inv)

    # We only care about the scalar component, compose only that
    return q_inv[0] * q2[0] - q_inv[1] * q2[1] - q_inv[2] * q2[2] - q_inv[3] * q2[3]
