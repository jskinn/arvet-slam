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
        :return:        """
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


class TrialErrors(pymodm.EmbeddedMongoModel):
    frame_errors = ReferenceListField(FrameError, required=True, blank=True)
    frames_lost = fields.ListField(fields.IntegerField(), blank=True)
    frames_found = fields.ListField(fields.IntegerField(), blank=True)
    times_lost = fields.ListField(fields.FloatField(), blank=True)
    times_found = fields.ListField(fields.FloatField(), blank=True)
    distances_lost = fields.ListField(fields.FloatField(), blank=True)
    distances_found = fields.ListField(fields.FloatField(), blank=True)


class FrameErrorResultQuerySet(QuerySet):
    
    def delete(self):
        """
        When a frame error result is deleted, also delete the frame errors it refers to
        :return:
        """
        frame_error_ids = set(err_id for doc in self.values()
                              for trial_errors in doc['errors'] for err_id in trial_errors['frame_errors'])
        FrameError.objects.raw({'_id': {'$in': list(frame_error_ids)}}).delete()
        super(FrameErrorResultQuerySet, self).delete()


FrameErrorResultManger = Manager.from_queryset(FrameErrorResultQuerySet)


class FrameErrorResult(MetricResult):
    """
    Error observations per estimate of a pose
    """
    system = fields.ReferenceField(VisionSystem, required=True, on_delete=pymodm.ReferenceField.CASCADE)
    image_source = fields.ReferenceField(ImageSource, required=True, on_delete=pymodm.ReferenceField.CASCADE)
    errors = fields.EmbeddedDocumentListField(TrialErrors, required=True, blank=True)
    image_columns = fields.ListField(fields.CharField())

    objects = FrameErrorResultManger()

    def save(self, cascade=None, full_clean=True, force_insert=False):
        """
        When saving, also save the frame results
        :param cascade:
        :param full_clean:
        :param force_insert:
        :return:
        """
        # Cascade the save to the frame errors
        if cascade or (self._mongometa.cascade and cascade is not False):
            frame_errors_to_create = [
                frame_error
                for trial_errors in self.errors
                for frame_error in trial_errors.frame_errors
                if frame_error.pk is None
            ]
            frame_errors_to_save = [
                frame_error
                for trial_errors in self.errors
                for frame_error in trial_errors.frame_errors
                if frame_error.pk is not None
            ]
            # Do error creation in bulk. Updates still happen individually.
            new_ids = FrameError.objects.bulk_create(frame_errors_to_create, full_clean=full_clean)
            for new_id, model in zip(new_ids, frame_errors_to_create):
                model.pk = new_id
            for frame_error in frame_errors_to_save:
                frame_error.save(cascade, full_clean, force_insert)
        super(FrameErrorResult, self).save(cascade, full_clean, force_insert)

    def get_columns(self) -> typing.Set[str]:
        funcs = ['min', 'max', 'mean', 'median', 'std']
        values = ['frames_lost', 'frames_found', 'times_lost', 'times_found', 'distance_lost', 'distance_found']

        columns = (
            set(FrameError.columns.keys()) |
            set(self.image_columns) |
            self.system.get_columns() |
            self.image_source.get_columns() |
            self.metric.get_columns()
        )
        columns |= set(func + '_' + val for func in funcs for val in values)
        return columns

    def get_results(self, columns: typing.Iterable[str] = None) -> typing.List[dict]:
        """
        Collate together the results for this metric result
        Can capture many different independent variables, including those from the system
        :param columns:
        :return:
        """
        if columns is None:
            # If no columns, do all columns
            columns = self.get_columns()
        columns = set(columns)
        image_source_properties = self.image_source.get_properties(columns)
        metric_properties = self.metric.get_properties(columns)
        other_properties = {**image_source_properties, **metric_properties}

        # Find column values that need to be computed for this object
        # Possibilities are any of 5 aggregate functions (min, max, mean, median, std)
        # applied to any of frames_lost, frames_found, time_lost, time_found, distance_lost, or distance_found.
        # These will be evaluated for each separate trial, and aggregated with the results from that trial.
        # we pre-compute to avoid checking which columns are actually specified every repeat
        funcs = [('min', np.min), ('max', np.max), ('mean', np.mean), ('median', np.median), ('std', np.std)]
        values = [('frames_lost', 'frames_lost'), ('frames_found', 'frames_found'),
                  ('times_lost', 'times_lost'), ('times_found', 'times_found'),
                  ('distance_lost', 'distances_lost'), ('distance_found', 'distances_found')]
        columns_to_compute = [
            (func_name + '_' + col_name, func, data)
            for col_name, data in values
            for func_name, func in funcs
            if func_name + '_' + col_name in columns
        ]

        results = []
        for trial_errors in self.errors:
            # Compute the values of certain columns available from this result
            for column, func, attribute in columns_to_compute:
                data = getattr(trial_errors, attribute)
                other_properties[column] = func(data) if len(data) > 0 else np.nan

            results.extend([
                frame_error.get_properties(columns, other_properties)
                for frame_error in trial_errors.frame_errors
            ])
        return results


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
        estimate_errors = [[] for _ in range(len(trial_results))]
        image_columns = set()
        distances_lost = [[] for _ in range(len(trial_results))]
        times_lost = [[] for _ in range(len(trial_results))]
        frames_lost = [[] for _ in range(len(trial_results))]
        distances_found = [[] for _ in range(len(trial_results))]
        times_found = [[] for _ in range(len(trial_results))]
        frames_found = [[] for _ in range(len(trial_results))]

        estimate_origins = [None for _ in range(len(trial_results))]
        ground_truth_origins = [None for _ in range(len(trial_results))]
        is_tracking = [False for _ in range(len(trial_results))]
        tracking_frames = [0 for _ in range(len(trial_results))]
        tracking_distances = [0 for _ in range(len(trial_results))]
        prev_tracking_time = [0 for _ in range(len(trial_results))]
        current_tracking_time = [0 for _ in range(len(trial_results))]

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

            for repeat_idx, frame_result in enumerate(frame_results):

                # Look for the first frame for each trial where the estimated pose is defined, record it as the origin
                # This aligns the two maps, even if it didn't start at the same place
                if estimate_origins[repeat_idx] is None and scaled_poses[repeat_idx] is not None:
                    estimate_origins[repeat_idx] = scaled_poses[repeat_idx]
                    ground_truth_origins[repeat_idx] = frame_results[repeat_idx].pose

                # Record how long the current tracking state has persisted
                if frame_idx <= 0:
                    # Cannot change to or from tracking on the first frame
                    is_tracking[repeat_idx] = (frame_result.tracking_state is TrackingState.OK)
                    prev_tracking_time[repeat_idx] = frame_result.timestamp
                elif is_tracking[repeat_idx] and frame_result.tracking_state is not TrackingState.OK:
                    # This trial has become lost, add to the list and reset the counters
                    frames_found[repeat_idx].append(tracking_frames[repeat_idx])
                    distances_found[repeat_idx].append(tracking_distances[repeat_idx])
                    times_found[repeat_idx].append(current_tracking_time[repeat_idx] - prev_tracking_time[repeat_idx])
                    tracking_frames[repeat_idx] = 0
                    tracking_distances[repeat_idx] = 0
                    prev_tracking_time[repeat_idx] = current_tracking_time[repeat_idx]
                    is_tracking[repeat_idx] = False
                elif not is_tracking[repeat_idx] and frame_result.tracking_state is TrackingState.OK:
                    # This trial has started to track, record how long it was lost for
                    frames_lost[repeat_idx].append(tracking_frames[repeat_idx])
                    distances_lost[repeat_idx].append(tracking_distances[repeat_idx])
                    times_lost[repeat_idx].append(current_tracking_time[repeat_idx] - prev_tracking_time[repeat_idx])
                    tracking_frames[repeat_idx] = 0
                    tracking_distances[repeat_idx] = 0
                    prev_tracking_time[repeat_idx] = current_tracking_time[repeat_idx]
                    is_tracking[repeat_idx] = True

                # Update the current tracking information
                tracking_frames[repeat_idx] += 1
                tracking_distances[repeat_idx] += np.linalg.norm(frame_result.motion.location)
                current_tracking_time[repeat_idx] = frame_result.timestamp

                # Build the frame error
                estimate_errors[repeat_idx].append(FrameError(
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
                ))

        # Add any accumulated tracking information left over at the end
        if len(trial_results[0].results) > 0:
            for repeat_idx, tracking in enumerate(is_tracking):
                if tracking:
                    frames_found[repeat_idx].append(tracking_frames[repeat_idx])
                    distances_found[repeat_idx].append(tracking_distances[repeat_idx])
                    times_found[repeat_idx].append(current_tracking_time[repeat_idx] - prev_tracking_time[repeat_idx])
                else:
                    frames_lost[repeat_idx].append(tracking_frames[repeat_idx])
                    distances_lost[repeat_idx].append(tracking_distances[repeat_idx])
                    times_lost[repeat_idx].append(current_tracking_time[repeat_idx] - prev_tracking_time[repeat_idx])

        # Once we've tallied all the results, either succeed or fail based on the number of results.
        if len(estimate_errors) <= 0 or any(len(trial_errors) <= 0 for trial_errors in estimate_errors):
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
            errors=[
                TrialErrors(
                    frame_errors=estimate_errors[repeat],
                    frames_lost=frames_lost[repeat],
                    frames_found=frames_found[repeat],
                    times_lost=times_lost[repeat],
                    times_found=times_found[repeat],
                    distances_lost=distances_lost[repeat],
                    distances_found=distances_found[repeat]
                )
                for repeat, trial_result in enumerate(trial_results)
            ]
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
            dot_product = np.dot(trans_error / trans_error_length, unit_reference)
            trans_error_direction = np.arccos(
                # Clip to arccos range to avoid errors
                min(1.0, max(0.0, dot_product))
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
