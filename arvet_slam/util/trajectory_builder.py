import typing
from arvet.util.transform import Transform, linear_interpolate


class TrajectoryBuilder:
    """
    Simple helper to load trajectories, handling re-centering to the first frame as the origin and
    matching to image timestamps.
    """

    def __init__(self, desired_timestamps: typing.Iterable[float]):
        """
        Create a new loader for trajectories.
        Doesn't actually reproduce the trajectory as given, instead it seeks to find the pose at a given list
        of existing timestamps, using linear interpolation where they fall between known values.
        Use this to estimate the pose at a given image time, when the IMU records positions/orientations around it.
        Also normalises the trajectory to be relative to the pose of the very first image, to account for
        different origins.

        :param desired_timestamps: A list of image timestamps. Doesn't store poses for times except those in this list
        """
        self._point_estimators = {
            timestamp: PointEstimate(timestamp)
            for timestamp in desired_timestamps
        }

    def add_trajectory_point(self, timestamp: float, pose: Transform) -> None:
        """
        Record
        :param timestamp:
        :param pose:
        :return:
        """
        for point_estimate in self._point_estimators.values():
            point_estimate.add_sample(timestamp, pose)

    def get_interpolated_trajectory(self) -> typing.Mapping[float, Transform]:
        """
        Get the resulting
        :return:
        """
        # get the estimated pose for the first timestamp in the trajectory
        first_time = min(self._point_estimators.keys())
        if not self._point_estimators[first_time].can_estimate():
            raise RuntimeError("Not enough timestamps read to determine pose for time {0}".format(first_time))
        first_pose = self._point_estimators[first_time].get_estimate()

        # Build the rest of the trajectory relative to the first pose
        result = {first_time: Transform()}
        for time, point_estimator in self._point_estimators.items():
            if time != first_time:
                if not point_estimator.can_estimate():
                    raise RuntimeError("Not enough trajectory points read to determine pose for time {0}".format(time))
                result[time] = first_pose.find_relative(point_estimator.get_estimate())

        return result


class PointEstimate:
    """
    Track nearest and furthest estimates for a given timestamp.
    Use this to estimate the best
    It's only a class to bundle up the 6 different state variables.
    """

    def __init__(self, timestamp: float):
        self.timestamp = timestamp
        self._lesser_time = None
        self._lesser_pose = None
        self._lesser_diff = None
        self._greater_time = None
        self._greater_pose = None
        self._greater_diff = None

    def add_sample(self, timestamp: float, pose: Transform) -> None:
        """
        Across all the calls to 'add_sample', track the closest timestamp under this one, and the closest over.
        These provide the two closest
        :param timestamp:
        :param pose:
        :return:
        """
        diff = abs(timestamp - self.timestamp)
        if timestamp <= self.timestamp and (self._lesser_diff is None or diff < self._lesser_diff):
            self._lesser_time = timestamp
            self._lesser_pose = pose
            self._lesser_diff = diff
        if timestamp >= self.timestamp and (self._greater_diff is None or diff < self._greater_diff):
            self._greater_time = timestamp
            self._greater_pose = pose
            self._greater_diff = diff

    def can_estimate(self) -> bool:
        """
        Do we have enough information to produce an estimate for this time?
        If not, get_estimate may error
        :return:
        """
        return self._lesser_pose is not None and self._greater_pose is not None

    def get_estimate(self) -> Transform:
        """
        Get the current best estimate for the pose at the given time.
        :return:
        """
        if self._lesser_diff == 0:
            # Got the desired timestamp exactly, return it
            return self._lesser_pose
        alpha = (self.timestamp - self._lesser_time) / (self._greater_time - self._lesser_time)
        return linear_interpolate(self._lesser_pose, self._greater_pose, alpha)


