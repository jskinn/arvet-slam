# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import bson
import copy
import transforms3d as tf3d
import arvet.util.transform as tf
import arvet.util.dict_utils as du
import arvet.database.tests.test_entity
import arvet.core.benchmark
import arvet_slam.benchmarks.rpe.relative_pose_error as rpe


def create_random_trajectory(random_state, duration=600, length=10):
    trajectory = {}
    current_pose = tf.Transform(
        random_state.uniform(-1000, 1000, 3),
        random_state.uniform(-1, 1, 4)
    )
    velocity = random_state.uniform(-10, 10, 3)
    angular_velocity = tf3d.quaternions.axangle2quat(
        vector=random_state.uniform(-1, 1, 3),
        theta=random_state.uniform(-np.pi / 30, np.pi / 30)
    )
    for time in range(duration):
        current_pose = tf.Transform(
            location=current_pose.location + velocity,
            rotation=tf3d.quaternions.qmult(current_pose.rotation_quat(w_first=True), angular_velocity)
        )
        velocity += random_state.normal(0, 1, 3)
        angular_velocity = tf3d.quaternions.qmult(angular_velocity, tf3d.quaternions.axangle2quat(
            vector=random_state.uniform(-1, 1, 3),
            theta=random_state.normal(0, np.pi / 30)
        ))
        trajectory[time + random_state.normal(0, 0.1)] = current_pose

    return {random_state.uniform(0, duration):
            tf.Transform(location=random_state.uniform(-1000, 1000, 3), rotation=random_state.uniform(0, 1, 4))
            for _ in range(length)}


def create_noise(trajectory, random_state, time_offset=0, time_noise=0.01, loc_noise=10, rot_noise=np.pi/64):
    if not isinstance(loc_noise, np.ndarray):
        loc_noise = np.array([loc_noise, loc_noise, loc_noise])

    noise = {}
    for time, pose in trajectory.items():
        noise[time] = tf.Transform(location=random_state.uniform(-loc_noise, loc_noise),
                                   rotation=tf3d.quaternions.axangle2quat(random_state.uniform(-1, 1, 3),
                                                                          random_state.uniform(-rot_noise, rot_noise)),
                                   w_first=True)

    relative_frame = tf.Transform(location=random_state.uniform(-1000, 1000, 3),
                                  rotation=random_state.uniform(0, 1, 4))

    changed_trajectory = {}
    for time, pose in trajectory.items():
        relative_pose = relative_frame.find_relative(pose)
        noisy_time = time + time_offset + random_state.uniform(-time_noise, time_noise)
        noisy_pose = relative_pose.find_independent(noise[time])
        changed_trajectory[noisy_time] = noisy_pose

    return changed_trajectory, noise


class MockTrialResult:

    def __init__(self, gt_trajectory, comp_trajectory):
        self._id = bson.ObjectId()
        self._gt_traj = gt_trajectory
        self._comp_traj = comp_trajectory

    @property
    def identifier(self):
        return self._id

    @property
    def ground_truth_trajectory(self):
        return self._gt_traj

    @ground_truth_trajectory.setter
    def ground_truth_trajectory(self, ground_truth_trajectory):
        self._gt_traj = ground_truth_trajectory

    @property
    def computed_trajectory(self):
        return self._comp_traj

    @computed_trajectory.setter
    def computed_trajectory(self, computed_trajectory):
        self._comp_traj = computed_trajectory

    def get_ground_truth_camera_poses(self):
        return self._gt_traj

    def get_computed_camera_poses(self):
        return self._comp_traj


class TestBenchmarkRPE(arvet.database.tests.test_entity.EntityContract, unittest.TestCase):

    def setUp(self):
        self.random = np.random.RandomState(1311)   # Use a random stream to make the results consistent
        trajectory = create_random_trajectory(self.random)
        self.trial_results = []
        self.noise = []
        for _ in range(10):
            noisy_trajectory, noise = create_noise(trajectory, self.random)
            self.trial_results.append(MockTrialResult(gt_trajectory=trajectory, comp_trajectory=noisy_trajectory))
            self.noise.append(noise)

    def get_class(self):
        return rpe.BenchmarkRPE

    def make_instance(self, *args, **kwargs):
        kwargs = du.defaults(kwargs, {
            'max_pairs': self.random.randint(100, 100000),
            'fixed_delta': self.random.randint(0, 1) == 1,
            'delta': self.random.normal(1, 0.1),
            'delta_unit': self.random.choice(['s', 'm', 'rad', 'f']),
            'offset': self.random.uniform(0, 100),
            'scale_': self.random.normal(1, 0.1),
        })
        return rpe.BenchmarkRPE(*args, **kwargs)

    def assert_models_equal(self, benchmark1, benchmark2):
        """
        Helper to assert that two benchmarks are equal
        :param benchmark1: BenchmarkRPE
        :param benchmark2: BenchmarkRPE
        :return:
        """
        if (not isinstance(benchmark1, rpe.BenchmarkRPE) or
                not isinstance(benchmark2, rpe.BenchmarkRPE)):
            self.fail('object was not a BenchmarkRPE')
        self.assertEqual(benchmark1.identifier, benchmark2.identifier)
        self.assertEqual(benchmark1.offset, benchmark2.offset)
        self.assertEqual(benchmark1.scale, benchmark2.scale)
        self.assertEqual(benchmark1.max_pairs, benchmark2.max_pairs)
        self.assertEqual(benchmark1.fixed_delta, benchmark2.fixed_delta)
        self.assertEqual(benchmark1.delta, benchmark2.delta)
        self.assertEqual(benchmark1.delta_unit, benchmark2.delta_unit)

    def test_benchmark_results_returns_a_benchmark_result(self):
        benchmark = rpe.BenchmarkRPE(max_pairs=0)
        result = benchmark.benchmark_results(self.trial_results)
        self.assertIsInstance(result, arvet.core.benchmark.BenchmarkResult)
        self.assertNotIsInstance(result, arvet.core.benchmark.FailedBenchmark)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(set(trial_result.identifier for trial_result in self.trial_results), set(result.trial_results))

    def test_benchmark_results_fails_for_no_matching_timestaps(self):
        # Adjust the computed timestamps so none of them match
        for trial_result in self.trial_results:
            trial_result.computed_trajectory = {
                time + 10000: pose
                for time, pose in trial_result.computed_trajectory.items()
            }

        # Perform the benchmark
        benchmark = rpe.BenchmarkRPE(max_pairs=0)
        result = benchmark.benchmark_results(self.trial_results)
        self.assertIsInstance(result, arvet.core.benchmark.FailedBenchmark)

    def test_benchmark_results_estimates_no_error_for_identical_trajectory(self):
        # Copy the ground truth exactly
        for trial_result in self.trial_results:
            trial_result.computed_trajectory = copy.deepcopy(trial_result.ground_truth_trajectory)

        benchmark = rpe.BenchmarkRPE(max_pairs=0)
        result = benchmark.benchmark_results(self.trial_results)

        if isinstance(result, arvet.core.benchmark.FailedBenchmark):
            print(result.reason)

        # Check all the errors are zero
        self.assertTrue(np.all(np.isclose(np.zeros(result.translational_errors.shape), result.translational_errors)))
        # We need more tolerance for the rotational error, because of the way the arccos
        # results in the smallest possible change producing a value around 2e-8
        self.assertTrue(np.all(np.isclose(np.zeros(result.rotational_errors.shape),
                                          result.rotational_errors, atol=1e-7)))

    def test_benchmark_results_estimates_no_error_for_noiseless_trajectory(self):
        # Create a new computed trajectory with no noise, but a fixed offset from the real trajectory
        # That is, the relative motions are the same, but the start point is different
        for trial_result in self.trial_results:
            comp_traj, _ = create_noise(
                trajectory=trial_result.ground_truth_trajectory,
                random_state=self.random,
                time_offset=0,
                time_noise=0,
                loc_noise=0,
                rot_noise=0
            )
            trial_result.computed_trajectory = comp_traj

        benchmark = rpe.BenchmarkRPE(max_pairs=0)
        result = benchmark.benchmark_results(self.trial_results)

        self.assertTrue(np.all(np.isclose(np.zeros(result.translational_errors.shape), result.translational_errors)))
        self.assertTrue(np.all(np.isclose(np.zeros(result.rotational_errors.shape),
                                          result.rotational_errors, atol=1e-7)))

    def test_benchmark_results_estimates_reasonable_trajectory_error_per_frame(self):
        benchmark = rpe.BenchmarkRPE(max_pairs=0, fixed_delta=True, delta=1, delta_unit='f')
        result = benchmark.benchmark_results(self.trial_results)
        # This is the max noise added to the trajectory during generation, it should be less than that
        self.assertLessEqual(10, np.max(result.translational_errors))
        self.assertLessEqual(np.pi/64, np.max(result.rotational_errors))

    def test_offset_shifts_query_trajectory_time(self):
        # Create a new noise trajectory with a large time offset for each trajectory but no noise
        for trial_result in self.trial_results:
            comp_traj, _ = create_noise(
                trajectory=trial_result.ground_truth_trajectory,
                random_state=self.random,
                time_offset=1000,
                time_noise=0,
                loc_noise=0,
                rot_noise=0
            )
            trial_result.computed_trajectory = comp_traj

        # This should fail due to the offset
        benchmark = rpe.BenchmarkRPE(max_pairs=0)
        result = benchmark.benchmark_results(self.trial_results)
        self.assertIsInstance(result, arvet.core.benchmark.FailedBenchmark)

        # This one should work, since the offset brings things back close together
        benchmark.offset = -1000
        result = benchmark.benchmark_results(self.trial_results)
        self.assertNotIsInstance(result, arvet.core.benchmark.FailedBenchmark)
        self.assertTrue(np.all(np.isclose(np.zeros(result.translational_errors.shape), result.translational_errors)))
        self.assertTrue(np.all(np.isclose(np.zeros(result.rotational_errors.shape),
                                          result.rotational_errors, atol=1e-7)))

    def test_scale_affects_trajectory_position(self):
        # Change the computed trajectories to be the same as the ground truth but scaled uniformly
        # This is an absolute scale, rather than a scale of motions, but it's what the scale parameter does.
        scale = 4243
        for trial_result in self.trial_results:
            trial_result.computed_trajectory = {
                time: tf.Transform(
                    location=pose.location / scale,
                    rotation=pose.rotation_quat(w_first=True),
                    w_first=True
                )
                for time, pose in trial_result.ground_truth_trajectory.items()
            }

        # This should have a large error due to the bad scale
        benchmark = rpe.BenchmarkRPE(max_pairs=0)
        result = benchmark.benchmark_results(self.trial_results)
        self.assertFalse(np.all(np.isclose(np.zeros(result.translational_errors.shape), result.translational_errors)))

        # This one should have no error
        benchmark.scale = scale
        result = benchmark.benchmark_results(self.trial_results)
        self.assertTrue(np.all(np.isclose(np.zeros(result.translational_errors.shape), result.translational_errors)))
        # We don't test rotation error, it isn't affected by scale
