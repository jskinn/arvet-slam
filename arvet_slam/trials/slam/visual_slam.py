# Copyright (c) 2017, John Skinner
import typing
from operator import attrgetter
import bson
import numpy as np
import pymodm
import pymodm.fields as fields
from pymodm.errors import ValidationError
from arvet.core.image import Image
from arvet.core.trial_result import TrialResult
from arvet.util.transform import Transform
from arvet.database.transform_field import TransformField
from arvet.database.enum_field import EnumField
from .tracking_state import TrackingState


class FrameResult(pymodm.EmbeddedMongoModel):
    """
    SLAM results for a single frame.
    Each frame, make one of these. At the end of the trial, list them up and store in the SLAM trial result.
    We expect these to be in ascending order of timestamp
    """
    timestamp = fields.FloatField(required=True)
    image = fields.ReferenceField(Image, required=True, on_delete=fields.ReferenceField.PULL)
    processing_time = fields.FloatField(required=True)
    pose = TransformField(required=True)
    motion = TransformField(required=True)
    estimated_pose = TransformField(blank=True)
    estimated_motion = TransformField(blank=True)
    tracking_state = EnumField(TrackingState, default=TrackingState.OK, required=True)
    loop_edges = fields.ListField(fields.FloatField(), blank=True)
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

    def __init__(self, *args, **kwargs):
        super(SLAMTrialResult, self).__init__(*args, **kwargs)

        # Lazy-compute the scale
        self._ground_truth_scale = None
        self._estimated_scale = None

        # Check that all loop closure timestamps refer to known frames
        known_timestamps = set(result.timestamp for result in self.results)
        if not all(all(timestamp in known_timestamps for timestamp in result.loop_edges) for result in self.results):
            missing_timestamps = {
                result.timestamp: [timestamp for timestamp in result.loop_edges if timestamp not in known_timestamps]
                for result in self.results
                if any(timestamp not in known_timestamps for timestamp in result.loop_edges)
            }
            raise ValueError(f"Some frames had loop closures that did't correspond to a frame: {missing_timestamps}")

        # If the results aren't sorted, re-sort them
        if not all(res2.timestamp >= res1.timestamp for res1, res2 in zip(self.results[:-1], self.results[1:])):
            self.results = sorted(self.results, key=attrgetter('timestamp'))

        # Find poses or motions, whichever we don't have
        self.infer_missing_poses_and_motions()

    def infer_missing_poses_and_motions(self):
        """
        Based on the current results, infer the values of motions from poses and vice versa.
        Basically, the both the true and estimated trajectories are stored twice, once as absolute poses,
        and one as relative motions. One can be inferred from the other, but to save calculation time later,
        we store both.

        In practice, construct with either a set of motion or a set of poses for each, and let this function
        handle building the other.

        Run automatically on construction, you can call it manually afterwards if the results change.
        The trial will be invalid if there are missing values that can be inferred, or they are inconsistent.
        :return:
        """
        if len(self.results) > 0:
            # Fill in missing pose and motions for the first frame
            # Defaults are usually 0
            self.results[0].motion = Transform()
            if self.results[0].pose is None:
                self.results[0].pose = Transform()

            # Fill in missing pose or motion
            to_backpropagate_motions = []
            for idx in range(1, len(self.results)):
                if self.results[idx].pose is None and self.results[idx].motion is not None:
                    # We have motion but no pose, compute pose
                    self.results[idx].pose = self.results[idx - 1].pose.find_independent(self.results[idx].motion)
                if self.results[idx].pose is not None and self.results[idx].motion is None:
                    # We have pose but no motion, compute motion
                    self.results[idx].motion = self.results[idx - 1].pose.find_relative(self.results[idx].pose)

                if self.results[idx - 1].estimated_pose is None:
                    if self.results[idx].estimated_pose is not None and self.results[idx].estimated_motion is not None:
                        # We have a pose and a motion for this frame, but no pose for the previous frame, we can go back
                        to_backpropagate_motions.append(idx)
                else:
                    if self.results[idx].estimated_pose is not None and self.results[idx].estimated_motion is None:
                        # We have estimated poses, but no estimated motion, infer estimated motion
                        self.results[idx].estimated_motion = self.results[idx - 1].estimated_pose.find_relative(
                            self.results[idx].estimated_pose
                        )
                    if self.results[idx].estimated_pose is None and self.results[idx].estimated_motion is not None:
                        # We have the previous estimated pose, and the estimated motion,
                        # we can combine into the next estimated motion
                        self.results[idx].estimated_pose = self.results[idx - 1].estimated_pose.find_independent(
                            self.results[idx].estimated_motion)

            # Go back and infer earlier estimated poses from later ones and motions
            for start_idx in reversed(to_backpropagate_motions):
                for idx in range(start_idx, 0, -1):
                    if self.results[idx].estimated_pose is not None and self.results[idx].estimated_motion is not None \
                            and self.results[idx - 1].estimated_pose is None:
                        self.results[idx - 1].estimated_pose = self.results[idx].estimated_pose.find_independent(
                            self.results[idx].estimated_motion.inverse()
                        )
                    else:
                        break

    def clean(self):
        """
        Custom validation. Checks that estimated motions and trajectories match.
        Raises validation errors if the stored motion does not match the change in pose,
        or the estimated motion does not match the change in estimated pose
        Uses is_close to avoid floating point variation
        :return:
        """
        if self.results[0].motion != Transform():
            raise ValidationError("The true motion for the first frame must be zero")
        if self.results[0].estimated_motion is not None:
            raise ValidationError("The estimated motion for the first frame must be None")

        for idx in range(1, len(self.results)):
            # Check for 0 change in timestamp, will cause divide by zeros when calculating speed
            if self.results[idx].timestamp - self.results[idx - 1].timestamp <= 0:
                raise ValidationError("Frame {0} has timestamp less than or equal to the frame before it".format(idx))

            # Validate the true motion
            pose_motion = self.results[idx - 1].pose.find_relative(self.results[idx].pose)
            if not np.all(np.isclose(pose_motion.location, self.results[idx].motion.location)) or not \
                    np.all(np.isclose(pose_motion.rotation_quat(), self.results[idx].motion.rotation_quat())):
                raise ValidationError(
                    "Ground truth motion does not match change in position at frame {0} ({1} != {2})".format(
                        idx, pose_motion, self.results[idx].motion
                    ))

            # Validate the estimated motion
            # Firs off, we need to know the estimated
            if self.results[idx - 1].estimated_pose is not None:
                if self.results[idx].estimated_pose is not None and self.results[idx].estimated_motion is None:
                    raise ValidationError("Estimated motion for frame {0} can be inferred, but is missing".format(idx))
                elif self.results[idx].estimated_pose is None and self.results[idx].estimated_motion is not None:
                    raise ValidationError("Estimated pose for frame {0} can be inferred, but is missing".format(idx))
                elif self.results[idx].estimated_pose is not None and self.results[idx].estimated_motion is not None:
                    pose_motion = self.results[idx - 1].estimated_pose.find_relative(self.results[idx].estimated_pose)
                    if not np.all(np.isclose(pose_motion.location, self.results[idx].estimated_motion.location)) or \
                            not np.all(np.isclose(pose_motion.rotation_quat(),
                                                  self.results[idx].estimated_motion.rotation_quat())):
                        raise ValidationError(
                            "Ground truth motion does not match change in position at frame {0} ({1} != {2})".format(
                                idx, pose_motion, self.results[idx].estimated_motion
                            ))

    @property
    def trajectory(self) -> typing.Mapping[float, Transform]:
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
    def ground_truth_trajectory(self) -> typing.Mapping[float, Transform]:
        return {result.timestamp: result.pose for result in self.results}

    @property
    def ground_truth_scale(self) -> float:
        """
        Get the scale of the ground truth motions where the estimated motion is not None.
        That is, the average speed of the ground truth, over the same set of motions used to compute estimated_scale.
        When dealing with monocular trajectories, we re-scale so that the average speed is the same,
        across the same set of frames so that the scales are consistent.
        :return:
        """
        if self._ground_truth_scale is None:
            # Compute it once, but not until we need it
            # Compute the average distance between points where an estimate is available.
            # The estimated scale will be the average over the same distances, and so the scale will work
            points = [result.pose.location for result in self.results if result.estimated_pose is not None]
            distances = []
            if len(points) >= 2:
                distances = [
                    np.linalg.norm(point - points[0])
                    for point in points[1:]
                ]
            # Collect distances between points based on the motions where the estimated pose is unavailable
            distances.extend([
                np.linalg.norm(result.motion.location)
                for result in self.results
                if result.estimated_motion is not None and result.estimated_pose is None
            ])
            if len(distances) > 0:
                self._ground_truth_scale = np.mean(distances)
            else:
                self._ground_truth_scale = 0
        return self._ground_truth_scale

    @property
    def estimated_scale(self) -> float:
        """
        Get the scale of the estimated motions, that is, the average estimated speed.
        When dealing with monocular trajectories, we re-scale so that the average speed is the same as the ground truth
        :return:
        """
        if self._estimated_scale is None:
            # Compute it once, but not until we need it
            # Compute the average distance between estimated points.
            # The ground truth will only use the frames where an estimate is available to compute the scale
            points = [result.estimated_pose.location for result in self.results if result.estimated_pose is not None]
            distances = []
            if len(points) >= 2:
                distances = [
                    np.linalg.norm(point - points[0])
                    for point in points[1:]
                ]
            # Collect distances between points based on the motions where the estimated pose is unavailable
            distances.extend([
                np.linalg.norm(result.estimated_motion.location)
                for result in self.results
                if result.estimated_motion is not None and result.estimated_pose is None
            ])
            if len(distances) > 0:
                self._estimated_scale = np.mean(distances)
            else:
                self._estimated_scale = 0
        return self._estimated_scale

    def get_scaled_motion(self, index: int) -> typing.Union[Transform, None]:
        """
        Get the motion of the camera from a particular frame, re-scaled to the ground truth motion
        if the system didn't have scale available when it ran.
        May return None if there is no available estimate
        :param index:
        :return:
        """
        if not 0 <= index < len(self.results):
            return None
        base_motion = self.results[index].estimated_motion
        if self.has_scale or base_motion is None:
            return base_motion
        # Handle the system producing 0 motion estimates for every frame, so scale is 0
        scale = (self.ground_truth_scale / self.estimated_scale) if self.estimated_scale != 0 else 1
        return Transform(
            location=scale * base_motion.location,
            rotation=base_motion.rotation_quat(w_first=True),
            w_first=True
        )

    def get_scaled_pose(self, index: int) -> typing.Union[Transform, None]:
        """
        Get the estimated pose of the camera.
        If the system couldn't infer scale (i.e., it was monocular), then the estimated poses will be re-scaled
        so that the average distances between poses is the same as the average distance between poses
        in the ground truth.
        That is an affine transformation, so it doesn't
        :param index:
        :return:
        """
        if not 0 <= index < len(self.results):
            return None
        base_pose = self.results[index].estimated_pose
        if self.has_scale or base_pose is None:
            return base_pose
        # Handle the system producing 0 motion estimates for every frame, so scale is 0
        scale = (self.ground_truth_scale / self.estimated_scale) if self.estimated_scale != 0 else 1
        return Transform(
            location=scale * base_pose.location,
            rotation=base_pose.rotation_quat(w_first=True),
            w_first=True
        )

    def get_ground_truth_camera_poses(self) -> typing.Mapping[float, Transform]:
        return self.ground_truth_trajectory

    def get_ground_truth_motions(self) -> typing.Mapping[float, Transform]:
        return {result.timestamp: result.motion for result in self.results}

    def get_tracking_states(self) -> typing.Mapping[float, TrackingState]:
        return self.tracking_stats

    @classmethod
    def load_minimal(cls, object_id: bson.ObjectId) -> 'SLAMTrialResult':
        """
        Load a minimal SLAM trial result object, which lets us query information about the trial result,
        as for is_trial_appropriate, but does not contain the full result data.
        :param object_id: The database id of the trial result
        :return: A partial SLAMTrialResult object.
        """
        return cls.objects.only(
            'system',
            'image_source',
            'success',
            'has_scale'
        ).get({'_id': object_id})
