# Copyright (c) 2017, John Skinner
import typing
import pymodm
import pymodm.fields as fields
from arvet.core.trial_result import TrialResult
from arvet.core.sequence_type import ImageSequenceType
import arvet.util.transform as tf
import arvet.util.trajectory_helpers as th
from arvet.database.transform_field import TransformField
from arvet.database.enum_field import EnumField
from arvet_slam.trials.slam.tracking_state import TrackingState


class FrameResult(pymodm.EmbeddedMongoModel):
    timestamp = fields.FloatField(required=True)
    pose = TransformField(required=True)
    motion = TransformField(required=True)
    estimated_pose = TransformField()
    estimated_motion = TransformField()
    tracking_state = EnumField(TrackingState, default=TrackingState.OK)
    num_features = fields.IntegerField(default=0)
    num_matches = fields.IntegerField(default=0)


class SLAMTrialResult(TrialResult):
    """
    The results of running a visual SLAM system.
    Has the ground truth and computed trajectories,
    the tracking statistics, and the number of features detected
    """
    results = fields.EmbeddedDocumentListField(FrameResult, required=True)
    has_scale = fields.BooleanField(default=True)
    sequence_type = EnumField(ImageSequenceType)

    @property
    def trajectory(self) -> typing.Mapping[float, tf.Transform]:
        return {result.timestamp: result.estimated_pose for result in self.results}

    @property
    def tracking_stats(self) -> typing.Mapping[float, TrackingState]:
        return {result.timestamp: result.tracking_state for result in self.results}

    @property
    def num_features(self) -> typing.Mapping[float, int]:
        return {result.timestamp: result.num_features for result in self.results}

    @property
    def num_matches(self) -> typing.Mapping[float, int]:
        return {result.timestamp: result.num_matches for result in self.results}

    @property
    def ground_truth_trajectory(self) -> typing.Mapping[float, tf.Transform]:
        return {result.timestamp: result.pose for result in self.results}

    def get_computed_camera_poses(self) -> typing.Mapping[float, tf.Transform]:
        return self.trajectory

    def get_computed_camera_motions(self) -> typing.Mapping[float, tf.Transform]:
        return th.trajectory_to_motion_sequence(self.trajectory)

    def get_ground_truth_camera_poses(self) -> typing.Mapping[float, tf.Transform]:
        return self.ground_truth_trajectory

    def get_ground_truth_motions(self) -> typing.Mapping[float, tf.Transform]:
        return th.trajectory_to_motion_sequence(self.ground_truth_trajectory)

    def get_tracking_states(self) -> typing.Mapping[float, TrackingState]:
        return self.tracking_stats
