#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
This is based on the code in evaluate_rpe.py distributed in the TUM RGBD benchmark tools.
See: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
Original Comment:
This script computes the relative pose error from the ground truth trajectory
and the estimated trajectory.
"""

import random
import typing
import numpy
import arvet.core.trial_result
import arvet.core.benchmark
import arvet.util.trajectory_helpers as th
import arvet_slam.benchmarks.rpe.rpe_result


class BenchmarkRPE(arvet.core.benchmark.Benchmark):

    def __init__(self, max_pairs: int = 10000, fixed_delta: bool = False, delta: float = 1.0,
                 delta_unit: str = 's', offset: float = 0.0, scale_: float = 1.0, id_=None):
        """
        The Relative Pose Error benchmark, which computes the difference between ground truth and estimated motion
        over some set of intervals of the trajectory.
        Depending on how it is configured, these intervals may be a fixed distance, angle change, or time appart,
        or might be every possible frame to frame comparison along the trajectory.
        Think about what measure you want to use, because this benchmark can produce a lot of different results.
        In particular, think about the interval size. Larger intervals accumulate error from multiple estimates,
        so have a greater variation.

        Uses the cartesian distance and angle between estimate and ground truth as the error.

        :param max_pairs: maximum number of pose comparisons (default: 10000, set to zero to disable downsampling
        :param fixed_delta: only consider pose pairs that have a distance of delta delta_unit
        (e.g., for evaluating the drift per second/meter/radian)
        :param delta: delta for evaluation (default: 1.0)
        :param delta_unit: unit of delta
        (options: \'s\' for seconds, \'m\' for meters, \'rad\' for radians, \'f\' for frames; default: \'s\')
        :param offset: time offset between ground-truth and estimated trajectory (default: 0.0)
        :param scale_: scaling factor for the estimated trajectory (default: 1.0)
        """
        super().__init__(id_=id_)
        self.max_pairs = int(max_pairs)
        self.fixed_delta = fixed_delta
        self.delta = delta
        self._delta_unit = 's'  # Set a default value
        self.delta_unit = delta_unit
        self.offset = offset
        self.scale = scale_

    @property
    def delta_unit(self) -> str:
        return self._delta_unit

    @delta_unit.setter
    def delta_unit(self, delta_unit: str):
        if delta_unit is 's' or delta_unit is 'm' or delta_unit is 'rad' or delta_unit is 'f':
            self._delta_unit = delta_unit

    def get_settings(self):
        return {
            'offset': self.offset,
            'scale': self.scale,
            'max_pairs': self.max_pairs,
            'fixed_delta': self.fixed_delta,
            'delta': self.delta,
            'delta_unit': self.delta_unit
        }

    def serialize(self):
        output = super().serialize()
        output['offset'] = self.offset
        output['scale'] = self.scale
        output['max_pairs'] = self.max_pairs
        output['fixed_delta'] = self.fixed_delta
        output['delta'] = self.delta
        output['delta_unit'] = self.delta_unit
        return output

    @classmethod
    def deserialize(cls, serialized_representation, db_client, **kwargs):
        if 'offset' in serialized_representation:
            kwargs['offset'] = serialized_representation['offset']
        if 'scale' in serialized_representation:
            kwargs['scale_'] = serialized_representation['scale']
        if 'max_pairs' in serialized_representation:
            kwargs['max_pairs'] = serialized_representation['max_pairs']
        if 'fixed_delta' in serialized_representation:
            kwargs['fixed_delta'] = serialized_representation['fixed_delta']
        if 'delta' in serialized_representation:
            kwargs['delta'] = serialized_representation['delta']
        if 'delta_unit' in serialized_representation:
            kwargs['delta_unit'] = serialized_representation['delta_unit']
        return super().deserialize(serialized_representation, db_client, **kwargs)

    @classmethod
    def get_trial_requirements(cls):
        return {'sucess': True}

    def is_trial_appropriate(self, trial_result):
        return (hasattr(trial_result, 'identifier') and
                hasattr(trial_result, 'get_ground_truth_camera_poses') and
                hasattr(trial_result, 'get_computed_camera_poses'))

    def benchmark_results(self, trial_results: typing.Iterable[arvet.core.trial_result.TrialResult]) \
            -> arvet.core.benchmark.BenchmarkResult:
        """
        Perform the RPE benchmark.
        :param trial_results: The results of several trials to aggregate
        :return:
        :rtype BenchmarkResult:
        """
        # First, we're going to get all the trajectories
        trial_result_ids = []
        ground_truth_trajectory = None
        computed_trajectories = []
        for trial_result in trial_results:
            if ground_truth_trajectory is None:
                ground_truth_trajectory = trial_result.get_ground_truth_camera_poses()
            computed_trajectories.append(trial_result.get_computed_camera_poses())
            trial_result_ids.append(trial_result.identifier)

        # Find the mean computed trajectory, which gives us noise
        mean_computed_trajectory = th.compute_average_trajectory(computed_trajectories)

        # convert to pose matricies for evaluate
        mean_computed_trajectory = {stamp: pose.transform_matrix for stamp, pose in mean_computed_trajectory.items()}
        ground_truth_trajectory = {stamp: pose.transform_matrix for stamp, pose in ground_truth_trajectory.items()}

        # Then, tally all the errors for all the computed trajectoreis
        all_errors = []
        for computed_trajectory in computed_trajectories:
            computed_trajectory = {stamp: pose.transform_matrix for stamp, pose in computed_trajectory.items()}
            errors = evaluate_trajectory(
                traj_gt=ground_truth_trajectory,
                traj_est=computed_trajectory,
                traj_est_mean=mean_computed_trajectory,
                param_max_pairs=int(self.max_pairs),
                param_fixed_delta=self.fixed_delta,
                param_delta=float(self.delta),
                param_delta_unit=self.delta_unit,
                param_offset=float(self.offset),
                param_scale=self.scale
            )
            if errors is not None:
                all_errors += errors

        if len(all_errors) < 2:
            return arvet.core.benchmark.FailedBenchmark(
                benchmark_id=self.identifier,
                trial_result_ids=trial_result_ids,
                reason="Couldn't find matching timestamp pairs between groundtruth and estimated trajectory"
                       "for any of the estimated trajectories"
            )
        return arvet_slam.benchmarks.rpe.rpe_result.BenchmarkRPEResult(
            benchmark_id=self.identifier,
            trial_result_ids=trial_result_ids,
            timestamps=sorted(ground_truth_trajectory.keys()),
            errors=all_errors,
            rpe_settings=self.get_settings()
        )


def find_closest_index(index_list, t):
    """
    Find the index of the closest value in a list.

    Input:
    L -- the list
    t -- value to be found

    Output:
    index of the closest element
    """
    beginning = 0
    difference = abs(index_list[0] - t)
    best = 0
    end = len(index_list)
    while beginning < end:
        middle = int((end + beginning) / 2)
        if abs(index_list[middle] - t) < difference:
            difference = abs(index_list[middle] - t)
            best = middle
        if t == index_list[middle]:
            return middle
        elif index_list[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best


def ominus(a, b):
    """
    Compute the relative 3D transformation between a and b.

    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)

    Output:
    Relative 3D transformation from a to b.
    """
    return numpy.dot(numpy.linalg.inv(a), b)


def scale(a, scalar):
    """
    Scale the translational components of a 4x4 homogeneous matrix by a scale factor.
    """
    return numpy.array(
        [[a[0, 0], a[0, 1], a[0, 2], a[0, 3] * scalar],
         [a[1, 0], a[1, 1], a[1, 2], a[1, 3] * scalar],
         [a[2, 0], a[2, 1], a[2, 2], a[2, 3] * scalar],
         [a[3, 0], a[3, 1], a[3, 2], a[3, 3]]]
    )


def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return numpy.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return numpy.arccos(min(1, max(-1, (numpy.trace(transform[0:3, 0:3]) - 1) / 2)))


def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    total = 0
    for t in motion:
        total += compute_distance(t)
        distances.append(total)
    return distances


def rotations_along_trajectory(traj, scale_):
    """
    Compute the angular rotations along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    total = 0
    for t in motion:
        total += compute_angle(t) * scale_
        distances.append(total)
    return distances


def evaluate_trajectory(traj_gt, traj_est, traj_est_mean=None, param_max_pairs=10000, param_fixed_delta=False,
                        param_delta=1.00, param_delta_unit="s", param_offset=0.00, param_scale=1.00):
    """
    Compute the relative pose error between two trajectories.

    Input:
    traj_gt -- the first trajectory (ground truth)
    traj_est -- the second trajectory (estimated trajectory)
    traj_est_mean -- The mean estimated trajectory for this benchmark
    param_max_pairs -- number of relative poses to be evaluated
    param_fixed_delta -- false: evaluate over all possible pairs
                         true: only evaluate over pairs with a given distance (delta)
    param_delta -- distance between the evaluated pairs
    param_delta_unit -- unit for comparison:
                        "s": seconds
                        "m": meters
                        "rad": radians
                        "deg": degrees
                        "f": frames
    param_offset -- time offset between two trajectories (to model the delay)
    param_scale -- scale to be applied to the second trajectory

    Output:
    list of compared poses and the resulting translation and rotation error
    """
    stamps_gt = sorted(traj_gt.keys())
    stamps_est = sorted(traj_est.keys())
    stamps_est_mean = sorted(traj_est_mean.keys()) if traj_est_mean is not None else []

    stamps_est_return = []
    for t_est in stamps_est:
        t_gt = stamps_gt[find_closest_index(stamps_gt, t_est + param_offset)]
        t_est_return = stamps_est[find_closest_index(stamps_est, t_gt - param_offset)]
        # t_gt_return = stamps_gt[find_closest_index(stamps_gt, t_est_return + param_offset)]
        if t_est_return not in stamps_est_return:
            stamps_est_return.append(t_est_return)
    if len(stamps_est_return) < 2:
        return None

    if param_delta_unit == "s":
        index_est = list(traj_est.keys())
        index_est.sort()
    elif param_delta_unit == "m":
        index_est = distances_along_trajectory(traj_est)
    elif param_delta_unit == "rad":
        index_est = rotations_along_trajectory(traj_est, 1)
    elif param_delta_unit == "deg":
        index_est = rotations_along_trajectory(traj_est, 180 / numpy.pi)
    elif param_delta_unit == "f":
        index_est = range(len(traj_est))
    else:
        raise Exception("Unknown unit for delta: '%s'" % param_delta_unit)

    if not param_fixed_delta:
        if param_max_pairs == 0 or len(traj_est) < numpy.sqrt(param_max_pairs):
            pairs = [(i, j) for i in range(len(traj_est)) for j in range(len(traj_est))]
        else:
            pairs = [(random.randint(0, len(traj_est) - 1), random.randint(0, len(traj_est) - 1))
                     for _ in range(param_max_pairs)]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = find_closest_index(index_est, index_est[i] + param_delta)
            if j != len(traj_est) - 1:
                pairs.append((i, j))
        if param_max_pairs != 0 and len(pairs) > param_max_pairs:
            pairs = random.sample(pairs, param_max_pairs)

    gt_interval = numpy.median([s - t for s, t in zip(stamps_gt[1:], stamps_gt[:-1])])
    gt_max_time_difference = 2 * gt_interval

    result = []
    for i, j in pairs:
        stamp_est_0 = stamps_est[i]
        stamp_est_1 = stamps_est[j]

        stamp_gt_0 = stamps_gt[find_closest_index(stamps_gt, stamp_est_0 + param_offset)]
        stamp_gt_1 = stamps_gt[find_closest_index(stamps_gt, stamp_est_1 + param_offset)]

        if (abs(stamp_gt_0 - (stamp_est_0 + param_offset)) > gt_max_time_difference or
                abs(stamp_gt_1 - (stamp_est_1 + param_offset)) > gt_max_time_difference):
            continue

        error44 = ominus(scale(
            ominus(traj_est[stamp_est_1], traj_est[stamp_est_0]), param_scale),
            ominus(traj_gt[stamp_gt_1], traj_gt[stamp_gt_0]))
        trans = compute_distance(error44)
        rot = compute_angle(error44)

        # Additioanlly, measure variation from the average estimated trajectory over the same window if it is provided
        trans_noise = numpy.nan
        rot_noise = numpy.nan
        if traj_est_mean is not None:
            stamp_est_mean_0 = stamps_est_mean[find_closest_index(stamps_est_mean, stamp_est_0 + param_offset)]
            stamp_est_mean_1 = stamps_est_mean[find_closest_index(stamps_est_mean, stamp_est_1 + param_offset)]
            noise44 = ominus(
                ominus(traj_est[stamp_est_1], traj_est[stamp_est_0]),
                ominus(traj_est_mean[stamp_est_mean_1], traj_est_mean[stamp_est_mean_0])
            )
            trans_noise = compute_distance(noise44)
            rot_noise = compute_angle(noise44)

        # Don't add the estimated times, we don't want to care about those
        result.append([stamp_gt_0, stamp_gt_1, trans, rot, trans_noise, rot_noise])
    return result
