# Copyright (c) 2017, John Skinner
import typing
from operator import attrgetter
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
    estimated_pose = TransformField()
    estimated_motion = TransformField()
    tracking_state = EnumField(TrackingState, default=TrackingState.OK, required=True)
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

        # If the results aren't sorted, re-sort them
        if not all(self.results[idx].timestamp >= self.results[idx - 1].timestamp
                   for idx in range(1, len(self.results))):
            self.results = sorted(self.results, key=attrgetter('timestamp'))

        # check consistency of estimated pose and motion
        if len(self.results) > 0:
            # Fill in missing pose and motions for the first frame
            # Defaults are usually 0
            if self.results[0].pose is None:
                self.results[0].pose = Transform()
            if self.results[0].motion is None:
                self.results[0].motion = Transform()

            # Fill in missing pose or motion
            for idx in range(1, len(self.results)):
                if self.results[idx].pose is None and self.results[idx].motion is not None:
                    # We have motion but no pose, compute pose
                    self.results[idx].pose = self.results[idx - 1].pose.find_independent(self.results[idx].motion)
                if self.results[idx].pose is not None and self.results[idx].motion is None:
                    # We have pose but no motion, compute motion
                    self.results[idx].motion = self.results[idx - 1].pose.find_relative(self.results[idx].pose)

                if self.results[idx].estimated_motion is None and self.results[idx].estimated_pose is not None \
                        and self.results[idx - 1].estimated_pose is not None:
                    # We have estimated poses, but no estimated motion, infer estimated motion
                    self.results[idx].estimated_motion = self.results[idx - 1].estimated_pose.find_relative(
                        self.results[idx].estimated_pose
                    )
                if self.results[idx].estimated_pose is None and self.results[idx].estimated_motion is not None \
                        and self.results[idx - 1].estimated_pose is not None:
                    # We have the previous estimated pose, and the estimated motion,
                    # we can combine into the next estimated motion
                    self.results[idx].estimated_pose = self.results[idx - 1].estimated_pose.find_independent(
                        self.results[idx].estimated_motion)

    def clean(self):
        """
        Custom validation. Checks that estimated motions and trajectories match.
        Raises validation errors if the stored motion does not match the change in pose,
        or the estimated motion does not match the change in estimated pose
        Uses is_close to avoid floating point variation
        :return:
        """
        for idx in range(1, len(self.results)):
            # Validate the true motion
            pose_motion = self.results[idx - 1].pose.find_relative(self.results[idx].pose)
            if not np.all(np.isclose(pose_motion.location, self.results[idx].motion.location)) or not \
                    np.all(np.isclose(pose_motion.rotation_quat(), self.results[idx].motion.rotation_quat())):
                raise ValidationError(
                    "Ground truth motion does not match change in position at frame {0} ({1} != {2})".format(
                        idx, pose_motion, self.results[idx].motion
                    ))

            # Validate the estimated motion
            if self.results[idx - 1].estimated_pose is not None and self.results[idx].estimated_pose is not None \
                    and self.results[idx].estimated_motion is not None:
                pose_motion = self.results[idx - 1].estimated_pose.find_relative(self.results[idx].estimated_pose)
                if not np.all(np.isclose(pose_motion.location, self.results[idx].estimated_motion.location)) or not \
                        np.all(np.isclose(pose_motion.rotation_quat(),
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

    def get_computed_camera_poses(self) -> typing.Mapping[float, Transform]:
        return self.trajectory

    def get_computed_camera_motions(self) -> typing.Mapping[float, Transform]:
        return {result.timestamp: result.estimated_motion for result in self.results}

    def get_ground_truth_camera_poses(self) -> typing.Mapping[float, Transform]:
        return self.ground_truth_trajectory

    def get_ground_truth_motions(self) -> typing.Mapping[float, Transform]:
        return {result.timestamp: result.motion for result in self.results}

    def get_tracking_states(self) -> typing.Mapping[float, TrackingState]:
        return self.tracking_stats
