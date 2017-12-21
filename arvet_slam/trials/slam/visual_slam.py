# Copyright (c) 2017, John Skinner
import pickle
import bson
import arvet.core.trial_result


class SLAMTrialResult(arvet.core.trial_result.TrialResult):
    """
    The results of running a visual SLAM system.
    Has the ground truth and computed trajectories,
    and the tracking statistics.
    """
    def __init__(self, system_id, trajectory, ground_truth_trajectory, tracking_stats, num_features, num_matches,
                 sequence_type, system_settings, id_=None, **kwargs):
        kwargs['success'] = True
        super().__init__(system_id=system_id, sequence_type=sequence_type,
                         system_settings=system_settings, id_=id_, **kwargs)
        self._trajectory = trajectory
        self._ground_truth_trajectory = ground_truth_trajectory
        self._tracking_stats = tracking_stats
        self._num_features = num_features
        self._num_matches = num_matches

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def tracking_stats(self):
        return self._tracking_stats

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_matches(self):
        return self._num_matches

    @property
    def ground_truth_trajectory(self):
        return self._ground_truth_trajectory

    def get_ground_truth_camera_poses(self):
        return self.ground_truth_trajectory

    def get_computed_camera_poses(self):
        return self.trajectory

    def get_tracking_states(self):
        return self.tracking_stats

    def serialize(self):
        serialized = super().serialize()
        serialized['ground_truth_trajectory'] = bson.Binary(pickle.dumps(self.ground_truth_trajectory,
                                                                         protocol=pickle.HIGHEST_PROTOCOL))
        serialized['trajectory'] = bson.Binary(pickle.dumps(self.trajectory, protocol=pickle.HIGHEST_PROTOCOL))
        serialized['tracking_stats'] = bson.Binary(pickle.dumps(self.tracking_stats, protocol=pickle.HIGHEST_PROTOCOL))
        serialized['num_features'] = [(stamp, features) for stamp, features in self.num_features.items()]
        serialized['num_matches'] = [(stamp, matches) for stamp, matches in self.num_matches.items()]
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
        return super().deserialize(serialized_representation, db_client, **kwargs)
