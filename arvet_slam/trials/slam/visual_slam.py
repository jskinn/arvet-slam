# Copyright (c) 2017, John Skinner
import typing
import pickle
import bson
import arvet.core.trial_result
import arvet.core.sequence_type
import arvet.util.transform as tf
import arvet.util.trajectory_helpers as th
import arvet_slam.trials.slam.tracking_state as ts


class SLAMTrialResult(arvet.core.trial_result.TrialResult):
    """
    The results of running a visual SLAM system.
    Has the ground truth and computed trajectories,
    the tracking statistics, and the number of features detected
    """
    def __init__(self, system_id: bson.ObjectId,
                 trajectory: typing.Mapping[float, tf.Transform],
                 ground_truth_trajectory: typing.Mapping[float, tf.Transform],
                 system_settings: dict,
                 tracking_stats: typing.Mapping[float, ts.TrackingState] = None,
                 num_features: typing.Mapping[float, int] = None,
                 num_matches: typing.Mapping[float, int] = None,
                 has_scale: bool = True,
                 sequence_type: arvet.core.sequence_type.ImageSequenceType = None,
                 id_: bson.ObjectId = None,
                 **kwargs):
        kwargs['success'] = True
        super().__init__(system_id=system_id, sequence_type=sequence_type,
                         system_settings=system_settings, id_=id_, **kwargs)
        self._trajectory = trajectory
        self._ground_truth_trajectory = ground_truth_trajectory
        self._tracking_stats = tracking_stats if tracking_stats is not None else {}
        self._num_features = num_features if num_features is not None else {}
        self._num_matches = num_matches if num_matches is not None else {}
        self._has_scale = bool(has_scale)

    @property
    def has_scale(self) -> bool:
        return self._has_scale

    @property
    def trajectory(self) -> typing.Mapping[float, tf.Transform]:
        return self._trajectory

    @property
    def tracking_stats(self) -> typing.Mapping[float, ts.TrackingState]:
        return self._tracking_stats

    @property
    def num_features(self) -> typing.Mapping[float, int]:
        return self._num_features

    @property
    def num_matches(self) -> typing.Mapping[float, int]:
        return self._num_matches

    @property
    def ground_truth_trajectory(self) -> typing.Mapping[float, tf.Transform]:
        return self._ground_truth_trajectory

    def get_computed_camera_poses(self) -> typing.Mapping[float, tf.Transform]:
        return self.trajectory

    def get_computed_camera_motions(self) -> typing.Mapping[float, tf.Transform]:
        return th.trajectory_to_motion_sequence(self.trajectory)

    def get_ground_truth_camera_poses(self) -> typing.Mapping[float, tf.Transform]:
        return self.ground_truth_trajectory

    def get_ground_truth_motions(self) -> typing.Mapping[float, tf.Transform]:
        return th.trajectory_to_motion_sequence(self.ground_truth_trajectory)

    def get_tracking_states(self) -> typing.Mapping[float, ts.TrackingState]:
        return self.tracking_stats

    def serialize(self):
        serialized = super().serialize()
        serialized['ground_truth_trajectory'] = bson.Binary(pickle.dumps(self.ground_truth_trajectory,
                                                                         protocol=pickle.HIGHEST_PROTOCOL))
        serialized['trajectory'] = bson.Binary(pickle.dumps(self.trajectory, protocol=pickle.HIGHEST_PROTOCOL))
        serialized['tracking_stats'] = bson.Binary(pickle.dumps(self.tracking_stats, protocol=pickle.HIGHEST_PROTOCOL))
        serialized['num_features'] = [(stamp, features) for stamp, features in self.num_features.items()]
        serialized['num_matches'] = [(stamp, matches) for stamp, matches in self.num_matches.items()]
        serialized['has_scale'] = self.has_scale
        return serialized

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'ground_truth_trajectory' in serialized_representation:
            kwargs['ground_truth_trajectory'] = pickle.loads(serialized_representation['ground_truth_trajectory'])
        if 'trajectory' in serialized_representation:
            kwargs['trajectory'] = pickle.loads(serialized_representation['trajectory'])
        if 'tracking_stats' in serialized_representation:
            kwargs['tracking_stats'] = pickle.loads(serialized_representation['tracking_stats'])
        if 'num_features' in serialized_representation:
            kwargs['num_features'] = {stamp: features for stamp, features in serialized_representation['num_features']}
        if 'num_matches' in serialized_representation:
            kwargs['num_matches'] = {stamp: features for stamp, features in serialized_representation['num_matches']}
        if 'has_scale' in serialized_representation:
            kwargs['has_scale'] = serialized_representation['has_scale']
        return super().deserialize(serialized_representation, db_client, **kwargs)
