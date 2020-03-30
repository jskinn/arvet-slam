# Copyright (c) 2018, John Skinner
import typing
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
from arvet.core.metric import Metric, MetricResult, check_trial_collection
from arvet.util.column_list import ColumnList
# import arvet.util.associate
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
    absolute_error = fields.EmbeddedDocumentField(PoseError, blank=True)
    relative_error = fields.EmbeddedDocumentField(PoseError, blank=True)
    noise = fields.EmbeddedDocumentField(PoseError, blank=True)
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
        Handles pulling data from the linked system and linked image
        :return:
        """
        system = self.trial_result.system
        if other_properties is None:
            other_properties = {}
        if columns is None:
            columns = set(self.columns.keys()) | system.get_columns() | self.image.get_columns()
        image_properties = self.image.get_properties(columns)
        error_properties = {
            column_name: self.columns.get_value(self, column_name)
            for column_name in columns
            if column_name in self.columns
        }
        system_properties = system.get_properties(columns, self.trial_result.settings)
        return {
            **other_properties,
            **image_properties,
            **error_properties,
            **system_properties
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
        image_source_properties = self.image_source.get_properties(columns)
        metric_properties = self.metric.get_properties(columns)
        other_properties = {**image_source_properties, **metric_properties}
        return [frame_error.get_properties(columns, other_properties) for frame_error in self.errors]


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
        TODO: Track the error introduced by a loop closure, somehow.
        Might need to track loop closures in the FrameResult
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

        # Make sure we have a non-zero number of trials to measure
        if len(trial_results) <= 0:
            return MetricResult(
                metric=self,
                trial_results=trial_results,
                success=False,
                message="Cannot measure zero trials."
            )

        # Ensure the trials all have the same number of results
        for repeat, trial_result in enumerate(trial_results[1:]):
            if len(trial_result.results) != len(trial_results[0].results):
                return MetricResult(
                    metric=self,
                    trial_results=trial_results,
                    success=False,
                    message=f"Repeat {repeat + 1} has a different number of frames "
                            f"({len(trial_result.results)} != {len(trial_results[0].results)})"
                )

        # Then, tally all the errors for all the computed trajectories
        estimate_errors = []
        image_columns = set()
        estimate_origins = [None for _ in range(len(trial_results))]
        ground_truth_origins = [None for _ in range(len(trial_results))]
        for frame_idx, frame_results in enumerate(zip(*(trial_result.results for trial_result in trial_results))):
            # Get the estimated motions and absolute poses for each trial
            # The trial result handles rescaling them to the ground truth if the scale was not available.
            scaled_motions = [trial_result.get_scaled_motion(frame_idx) for trial_result in trial_results]
            scaled_poses = [trial_result.get_scaled_pose(frame_idx) for trial_result in trial_results]

            # Find the average estimated motion for this frame across all the different trials
            # The average is not available for frames with only a single estimate
            non_null_motions = [motion for motion in scaled_motions if motion is not None]
            if len(non_null_motions) > 1:
                average_motion = tf.compute_average_pose(non_null_motions)
            else:
                average_motion = None

            # Union the image columns for all the images for all the frame results
            image_columns |= set(
                column for frame_result in frame_results for column in frame_result.image.get_columns())

            # Look for the first frame for each trial where the estimated pose is defined, record it as the origin
            # This aligns the two maps, even if it didn't start at the same place
            for repeat_idx, estimated_pose in enumerate(scaled_poses):
                if estimate_origins[repeat_idx] is None and estimated_pose is not None:
                    estimate_origins[repeat_idx] = estimated_pose
                    ground_truth_origins[repeat_idx] = frame_results[repeat_idx].pose

            estimate_errors.extend(FrameError(
                trial_result=trial_results[repeat_idx],
                repeat=repeat_idx,
                timestamp=frame_result.timestamp,
                image=frame_result.image,
                tracking=frame_result.tracking_state,
                processing_time=frame_result.processing_time,
                motion=frame_result.motion,
                num_features=frame_result.num_features,
                num_matches=frame_result.num_matches,
                # Compute the error in the absolute estimated pose (if available)
                absolute_error=make_pose_error(
                    estimate_origins[repeat_idx].find_relative(scaled_poses[repeat_idx]),
                    ground_truth_origins[repeat_idx].find_relative(frame_result.pose)
                ) if scaled_poses[repeat_idx] is not None else None,
                # Compute the error of the motion relative to the true motion
                relative_error=make_pose_error(
                    scaled_motions[repeat_idx],
                    frame_result.motion
                ) if scaled_motions[repeat_idx] is not None else None,
                # Compute the error between the motion and the average estimated motion
                noise=make_pose_error(
                    scaled_motions[repeat_idx],
                    average_motion
                ) if scaled_motions[repeat_idx] is not None and average_motion is not None else None
            ) for repeat_idx, frame_result in enumerate(frame_results))

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
        matches = associate(associated_times, traj, offset=0, max_difference=0.1)
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


def associate(first_list, second_list, offset, max_difference, window=3):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    # first_keys = list(first_list.keys())  # copy both keys lists, so we can remove from them later
    # second_keys = list(second_list.keys())
    # potential_matches = [(abs(a - (b + offset)), a, b)
    #                      for a in first_keys
    #                      for b in second_keys
    #                      if abs(a - (b + offset)) < max_difference]

    first_keys = sorted(first_list.keys())
    second_keys = sorted(second_list.keys())
    if first_keys == second_keys:
        return [(a, a) for a in first_keys]
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for idx, a in enumerate(first_keys)
        for b in second_keys[max(0, idx - window):min(len(second_keys), idx + window)]
        if abs(a - (b + offset)) < max_difference
    ]

    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches
