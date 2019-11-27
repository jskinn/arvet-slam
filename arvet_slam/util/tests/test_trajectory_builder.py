import numpy as np
import transforms3d as tf3d
from arvet.util.transform import Transform, linear_interpolate
from arvet.util.test_helpers import ExtendedTestCase
from arvet_slam.util.trajectory_builder import TrajectoryBuilder


class TestTrajectoryLoader(ExtendedTestCase):

    def test_normalises_to_first_pose(self):
        first_pose = Transform(location=(15.2, -1167.9, -1.2), rotation=(0.535, 0.2525, 0.11, 0.2876))
        relative_trajectory = {}
        absolute_trajectory = {}
        for time in range(0, 10):
            timestamp = time * 4999.936 + 1403638128.940097024
            pose = Transform(location=(0.122 * time, -0.53112 * time, 1.893 * time),
                             rotation=(0.772 * time, -0.8627 * time, -0.68782 * time))
            relative_trajectory[timestamp] = pose
            absolute_trajectory[timestamp] = first_pose.find_independent(pose)

        builder = TrajectoryBuilder(relative_trajectory.keys())
        for time in sorted(absolute_trajectory.keys()):
            builder.add_trajectory_point(time, absolute_trajectory[time])
        result = builder.get_interpolated_trajectory()
        self.assertEqual(set(relative_trajectory.keys()), set(result.keys()))
        for time in relative_trajectory.keys():
            self.assertNPClose(relative_trajectory[time].location, result[time].location, atol=1e-13, rtol=0)
            self.assertNPClose(relative_trajectory[time].rotation_quat(True), result[time].rotation_quat(True),
                               atol=1e-14, rtol=0)

    def test_interpolates_pose(self):
        time_offset = 0.332
        times = [time + time_offset for time in range(0, 10)]
        builder = TrajectoryBuilder(times)
        for time in range(0, 11):   # includes 10, to make sure we get either end of the range
            # Create sample points at the given times
            builder.add_trajectory_point(time, Transform(
                location=(10 * time, -time, 0),
                rotation=tf3d.quaternions.axangle2quat((4, -3, 2), time * np.pi / 20),
                w_first=True
            ))
        result = builder.get_interpolated_trajectory()
        self.assertEqual(set(times), set(result.keys()))

        first_pose = Transform(
            location=(10 * time_offset, -time_offset, 0),
            rotation=tf3d.quaternions.axangle2quat((4, -3, 2), time_offset * np.pi / 20),
            w_first=True
        )
        for time in times:
            # This is the true pose at that time, relative to the true pose at the first time.
            # need to do it this way, because of the rotation, which shifts things
            # the build has never seen this transform, the timestamps are different
            expected_pose = first_pose.find_relative(Transform(
                location=(10 * time, -time, 0),
                rotation=tf3d.quaternions.axangle2quat((4, -3, 2), time * np.pi / 20),
                w_first=True
            ))
            interpolated_pose = result[time]
            self.assertNPClose(expected_pose.location, interpolated_pose.location, atol=1e-13, rtol=0)
            self.assertNPClose(expected_pose.rotation_quat(True), interpolated_pose.rotation_quat(True),
                               atol=1e-14, rtol=0)

    def test_interpolates_pose_over_long_range(self):
        builder = TrajectoryBuilder(range(1, 10))
        # Only add the start and end times, rely on the LERP for all other poses
        builder.add_trajectory_point(0, Transform())
        builder.add_trajectory_point(10, Transform(     # the pose at time 10
            location=(10 * 10, -10, 0),
            rotation=tf3d.quaternions.axangle2quat((4, -3, 2), 10 * np.pi / 20),
            w_first=True
        ))
        result = builder.get_interpolated_trajectory()
        self.assertEqual(set(range(1, 10)), set(result.keys()))

        first_pose = Transform(
            location=(10 * 1, -1, 0),
            rotation=tf3d.quaternions.axangle2quat((4, -3, 2), 1 * np.pi / 20),
            w_first=True
        )
        for time in range(1, 10):
            # This is the true pose at that time, relative to the true pose at the first time.
            # need to do it this way, because of the rotation, which shifts things
            # the build has never seen this transform, the timestamps are different
            expected_pose = first_pose.find_relative(Transform(
                location=(10 * time, -time, 0),
                rotation=tf3d.quaternions.axangle2quat((4, -3, 2), time * np.pi / 20),
                w_first=True
            ))
            interpolated_pose = result[time]
            self.assertNPClose(expected_pose.location, interpolated_pose.location, atol=1e-13, rtol=0)
            self.assertNPClose(expected_pose.rotation_quat(True), interpolated_pose.rotation_quat(True),
                               atol=1e-14, rtol=0)

    def test_linearly_interpolates_from_nearest_pose(self):
        def make_pose(time):
            # Make times that depend non-linearly on time so that linear interpolation is inaccurate
            return Transform(
                location=(10 * time - 0.1 * time * time, 10 * np.cos(time * np.pi / 50), 0),
                rotation=tf3d.quaternions.axangle2quat((-2, -3, 2), np.log(time + 1) * np.pi / (4 * np.log(10))),
                w_first=True
            )

        time_offset = 0.332
        times = [time + time_offset for time in range(0, 100, 5)]
        builder = TrajectoryBuilder(times)
        for time in range(0, 101):   # includes 100, to make sure we get either end of the range
            # Create sample points at the given times, all integers so we can round
            builder.add_trajectory_point(time, make_pose(time))
        result = builder.get_interpolated_trajectory()
        self.assertEqual(set(times), set(result.keys()))

        first_pose = linear_interpolate(make_pose(0), make_pose(1), min(times))
        for time in times:
            # This is the true pose at that time, relative to the true pose at the first time.
            # need to do it this way, because of the rotation, which shifts things
            # the build has never seen this transform, the timestamps are different
            true_pose = first_pose.find_relative(make_pose(time))

            # We expect a linear interpolation between the poses from the previous and next integer seconds
            # Since the sample times we gave it were on the integers.
            # Interpolation should also be scaled to the progress between the two.
            expected_pose = first_pose.find_relative(linear_interpolate(
                make_pose(np.floor(time)), make_pose(np.ceil(time)), time - np.floor(time)))
            interpolated_pose = result[time]
            self.assertNPClose(expected_pose.location, interpolated_pose.location, atol=1e-13, rtol=0)
            self.assertNPClose(expected_pose.rotation_quat(True), interpolated_pose.rotation_quat(True),
                               atol=1e-14, rtol=0)

    def test_raises_exception_if_not_enough_data_for_time(self):
        builder = TrajectoryBuilder(time + 0.4311 for time in range(0, 10))
        for time in range(0, 10):   # No times before 0.5, but min desired timestamp is 0.4331
            builder.add_trajectory_point(time + 0.5, Transform((time * 10, -time, 0)))
        with self.assertRaises(RuntimeError) as ctx:
            builder.get_interpolated_trajectory()
        message = str(ctx.exception)
        self.assertIn('0.4311', message)  # Check includes the timestamp in the error message

        builder = TrajectoryBuilder(time + 0.229 for time in range(0, 10))
        for time in range(0, 10):   # No times after 8.5, but max desired time is 9.229
            builder.add_trajectory_point(time - 0.5, Transform((time * 10, -time, 0)))
        with self.assertRaises(RuntimeError) as ctx:
            builder.get_interpolated_trajectory()
        message = str(ctx.exception)
        self.assertIn('9.229', message)  # Check includes the timestamp in the error message
