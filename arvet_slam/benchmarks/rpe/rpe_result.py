# Copyright (c) 2017, John Skinner
import typing
import numpy as np
import pickle
import bson
import arvet.core.benchmark


class BenchmarkRPEResult(arvet.core.benchmark.BenchmarkResult):
    """
    Relative Pose Error results.
    There are a lot of different ways to slice the result intervals

    That is, what is the error between the calculated and ground truth trajectories at as many points as possible.
    This can be represented by several values, but is probably best encapsulated by the
    Root Mean-Squared Error, in the rmse property.
    """

    def __init__(self, benchmark_id: bson.ObjectId, trial_result_ids: typing.Iterable[bson.ObjectId],
                 timestamps: typing.Iterable[float], errors: typing.Iterable[typing.Iterable[float]],
                 rpe_settings: dict, id_: bson.ObjectId = None, **kwargs):
        """
        Construct an RPE result
        :param benchmark_id:
        :param trial_result_ids:
        :param timestamps:
        :param errors:
        :param rpe_settings:
        :param id_:
        :param kwargs:
        """
        kwargs['success'] = True
        super().__init__(benchmark_id=benchmark_id, trial_result_ids=trial_result_ids, id_=id_, **kwargs)
        self._timestamps = list(timestamps)
        self._errors_observations = np.array(errors)
        self._rpe_settings = rpe_settings

    @property
    def timestamps(self):
        """
        Get the ground-truth timestamps for the trajectories that produced this result.
        All the interval values will be in this list, although not all timestamps may necessarily appear
        in intervals.

        Use this to iterate over or perform logic on all the timestamps in the trajectory.
        :return:
        """
        return self._timestamps

    @property
    def settings(self):
        return self._rpe_settings

    @property
    def translational_errors(self):
        """
        Get all the translational errors for all intervals, as a big list.
        Use this for fitting distributions
        :return:
        """
        return self._errors_observations[:, 2]

    @property
    def rotational_errors(self):
        """
        Get all the rotational errors for all intervals, as a big list.
        Use this for fitting distributions
        :return:
        """
        return self._errors_observations[:, 3]

    @property
    def translational_noise(self):
        """
        Get all the translational noise for all intervals, as a big list.
        Use this for fitting distributions
        :return:
        """
        return self._errors_observations[:, 4]

    @property
    def rotational_noise(self):
        """
        Get the rotational noise for all intervals, as a big list.
        :return:
        """
        return self._errors_observations[:, 5]

    def collect_observations(self, criteria: typing.Callable[[np.ndarray], bool],
                             reduce: typing.Callable[[np.ndarray], typing.Any]) -> typing.List[typing.Any]:
        """
        Collect all error observations matching a certain criteria,
        and extract a particular value from them.

        An error observation is a 6x1 numpy array, containing elements in the following order:
        [interval start, interval end, translation error, rotation error, translation noise, rotation noise]

        This is a generic helper for reading error observations, and can be used to collect many
        different statistics about the errors.
        For instance, to find the translation error for all intervals including a particular time, you could do:
        ```
        collect_observations(
            lambda obs: obs[0] <= time <= obs[1],
            lambda obs: obs[2]
        )
        ```
        which would give you a list of all the errors across intervals including the specified time

        :param criteria: Criteria to determine the observations that are included.
        Take an observation and return True if it should be included in the output
        :param reduce: Reduce an observation to a particular value
        :return: A list of all the reduced values for all observations that match the criteria
        """
        return [
            reduce(observation)
            for observation in self._errors_observations
            if criteria(observation)
        ]

    def get_aggregate_translational_error(self):
        """
        Find the mean translational error, for all the computed intervals.
        From experimentation, this should form an exponential distribution,
        P(err) = (1/mean) e^-(err / mean), and mean ~= std

        This result discards the context of the intervals over which the data was collected.
        If not using a fixed delta, then this result may be strange.
        :return:
        """
        trans_errors = self.translational_errors
        rmse = np.sqrt(np.dot(trans_errors, trans_errors) / len(trans_errors))
        return rmse, np.mean(trans_errors), np.std(trans_errors), np.min(trans_errors), np.max(trans_errors)

    def get_aggregate_rotational_error(self):
        """
        Find the mean translational error, for all the computed intervals.
        We expect this to form a beta distribution over the range (0, pi).
        This can be parameterized as: P(err) is proportional to \theta^(\alpha - 1) * (1 - \theta)^(\beta - 1)
        for some parameters \alpha and \beta

        :return:
        """
        rot_errors = self.rotational_errors
        rmse = np.sqrt(np.dot(rot_errors, rot_errors) / len(rot_errors))
        return rmse, np.mean(rot_errors), np.std(rot_errors), np.min(rot_errors), np.max(rot_errors)

    def serialize(self):
        output = super().serialize()
        output['timestamps'] = self.timestamps
        output['errors'] = bson.Binary(pickle.dumps(self._errors_observations.tolist(),
                                                    protocol=pickle.HIGHEST_PROTOCOL))
        output['settings'] = self.settings
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'timestamps' in serialized_representation:
            kwargs['timestamps'] = serialized_representation['timestamps']
        if 'errors' in serialized_representation:
            kwargs['errors'] = pickle.loads(serialized_representation['errors'])
        if 'settings' in serialized_representation:
            kwargs['rpe_settings'] = serialized_representation['settings']
        return super().deserialize(serialized_representation, db_client, **kwargs)
