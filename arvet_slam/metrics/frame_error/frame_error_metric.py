# Copyright (c) 2018, John Skinner
import typing

import bson
import numpy as np
import transforms3d as t3
import arvet.util.transform as tf
from arvet.core.image_source import ImageSource
from arvet.core.metric import Metric, MetricResult, check_trial_collection
from arvet.core.system import VisionSystem
from arvet.core.trial_result import TrialResult
from arvet.database.autoload_modules import autoload_modules
from pymodm.context_managers import no_auto_dereference

from arvet_slam.metrics.frame_error.frame_error_result import make_pose_error, make_frame_error, \
    TrialErrors, make_frame_error_result, FrameErrorResult
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult


class FrameErrorMetric(Metric):

    def get_columns(self) -> typing.Set[str]:
        """
        The frame error metric has no parameters, and provides no properties
        :return:
        """
        return set()

    def get_properties(self, columns: typing.Iterable[str] = None) -> typing.Mapping[str, typing.Any]:
        """
        The frame error metric has no parameters, and provides no properties
        :param columns:
        :return:
        """
        return {}

    def is_trial_appropriate(self, trial_result):
        return isinstance(trial_result, SLAMTrialResult)

    def measure_results(self, trial_results: typing.Iterable[TrialResult]) -> FrameErrorResult:
        """
        Collect the errors
        TODO: Track the error introduced by a loop closure, somehow.
        Might need to track loop closures in the FrameResult
        :param trial_results: The results of several trials to aggregate
        :return:
        :rtype BenchmarkResult:
        """
        trial_results = list(trial_results)

        # preload model types for the models linked to the trial results.
        with no_auto_dereference(SLAMTrialResult):
            model_ids = set(tr.system for tr in trial_results if isinstance(tr.system, bson.ObjectId))
            autoload_modules(VisionSystem, list(model_ids))
            model_ids = set(tr.image_source for tr in trial_results if isinstance(tr.image_source, bson.ObjectId))
            autoload_modules(ImageSource, list(model_ids))

        # Check if the set of trial results is valid. Loads the models.
        invalid_reason = check_trial_collection(trial_results)
        if invalid_reason is not None:
            return MetricResult(
                metric=self,
                trial_results=trial_results,
                success=False,
                message=invalid_reason
            )

        # Make sure we have a non-zero number of trials to measure
        if len(trial_results) <= 0:
            return MetricResult(
                metric=self,
                trial_results=trial_results,
                success=False,
                message="Cannot measure zero trials."
            )

        # Ensure the trials all have the same number of results
        for repeat, trial_result in enumerate(trial_results[1:]):
            if len(trial_result.results) != len(trial_results[0].results):
                return MetricResult(
                    metric=self,
                    trial_results=trial_results,
                    success=False,
                    message=f"Repeat {repeat + 1} has a different number of frames "
                            f"({len(trial_result.results)} != {len(trial_results[0].results)})"
                )

        # Load the system, it must be the same for all trials (see check_trial_collection)
        system = trial_results[0].system

        # Pre-load the image objects in a batch, to avoid loading them piecemeal later
        images = [image for _, image in trial_results[0].image_source]

        # Build mappings between frame result timestamps and poses for each trial
        timestamps_to_pose = [{
            frame_result.timestamp: frame_result.pose
            for frame_result in trial_result.results
        } for trial_result in trial_results]

        # Then, tally all the errors for all the computed trajectories
        estimate_errors = [[] for _ in range(len(trial_results))]
        image_columns = set()
        distances_lost = [[] for _ in range(len(trial_results))]
        times_lost = [[] for _ in range(len(trial_results))]
        frames_lost = [[] for _ in range(len(trial_results))]
        distances_found = [[] for _ in range(len(trial_results))]
        times_found = [[] for _ in range(len(trial_results))]
        frames_found = [[] for _ in range(len(trial_results))]

        estimate_origins = [None for _ in range(len(trial_results))]
        ground_truth_origins = [None for _ in range(len(trial_results))]
        is_tracking = [False for _ in range(len(trial_results))]
        tracking_frames = [0 for _ in range(len(trial_results))]
        tracking_distances = [0 for _ in range(len(trial_results))]
        prev_tracking_time = [0 for _ in range(len(trial_results))]
        current_tracking_time = [0 for _ in range(len(trial_results))]

        for frame_idx, frame_results in enumerate(zip(*(trial_result.results for trial_result in trial_results))):
            # Get the estimated motions and absolute poses for each trial
            # The trial result handles rescaling them to the ground truth if the scale was not available.
            scaled_motions = [trial_result.get_scaled_motion(frame_idx) for trial_result in trial_results]
            scaled_poses = [trial_result.get_scaled_pose(frame_idx) for trial_result in trial_results]

            # Find the average estimated motion for this frame across all the different trials
            # The average is not available for frames with only a single estimate
            non_null_motions = [motion for motion in scaled_motions if motion is not None]
            if len(non_null_motions) > 1:
                average_motion = tf.compute_average_pose(non_null_motions)
            else:
                average_motion = None

            # Union the image columns for all the images for all the frame results
            image_columns |= set(
                column for frame_result in frame_results for column in frame_result.image.get_columns())

            for repeat_idx, frame_result in enumerate(frame_results):

                # Look for the first frame for each trial where the estimated pose is defined, record it as the origin
                # This aligns the two maps, even if it didn't start at the same place
                if estimate_origins[repeat_idx] is None and scaled_poses[repeat_idx] is not None:
                    estimate_origins[repeat_idx] = scaled_poses[repeat_idx]
                    ground_truth_origins[repeat_idx] = frame_result.pose

                # Record how long the current tracking state has persisted
                if frame_idx <= 0:
                    # Cannot change to or from tracking on the first frame
                    is_tracking[repeat_idx] = (frame_result.tracking_state is TrackingState.OK)
                    prev_tracking_time[repeat_idx] = frame_result.timestamp
                elif is_tracking[repeat_idx] and frame_result.tracking_state is not TrackingState.OK:
                    # This trial has become lost, add to the list and reset the counters
                    frames_found[repeat_idx].append(tracking_frames[repeat_idx])
                    distances_found[repeat_idx].append(tracking_distances[repeat_idx])
                    times_found[repeat_idx].append(current_tracking_time[repeat_idx] - prev_tracking_time[repeat_idx])
                    tracking_frames[repeat_idx] = 0
                    tracking_distances[repeat_idx] = 0
                    prev_tracking_time[repeat_idx] = current_tracking_time[repeat_idx]
                    is_tracking[repeat_idx] = False
                elif not is_tracking[repeat_idx] and frame_result.tracking_state is TrackingState.OK:
                    # This trial has started to track, record how long it was lost for
                    frames_lost[repeat_idx].append(tracking_frames[repeat_idx])
                    distances_lost[repeat_idx].append(tracking_distances[repeat_idx])
                    times_lost[repeat_idx].append(current_tracking_time[repeat_idx] - prev_tracking_time[repeat_idx])
                    tracking_frames[repeat_idx] = 0
                    tracking_distances[repeat_idx] = 0
                    prev_tracking_time[repeat_idx] = current_tracking_time[repeat_idx]
                    is_tracking[repeat_idx] = True

                # Update the current tracking information
                tracking_frames[repeat_idx] += 1
                tracking_distances[repeat_idx] += np.linalg.norm(frame_result.motion.location)
                current_tracking_time[repeat_idx] = frame_result.timestamp

                # Turn loop closures into distances. We don't need to worry about origins because everything is GT frame
                if len(frame_result.loop_edges) > 0:
                    loop_distances, loop_angles = compute_loop_distances_and_angles(
                        frame_result.pose,
                        (
                            timestamps_to_pose[repeat_idx][timestamp]
                            for timestamp in frame_result.loop_edges
                            if timestamp in timestamps_to_pose[repeat_idx]  # they should all be in there, but for safety, check
                        )
                    )
                else:
                    loop_distances, loop_angles = [], []

                # Build the frame error
                frame_error = make_frame_error(
                    trial_result=trial_results[repeat_idx],
                    frame_result=frame_result,
                    image=images[frame_idx],
                    system=system,
                    repeat_index=repeat_idx,
                    loop_distances=loop_distances,
                    loop_angles=loop_angles,
                    # Compute the error in the absolute estimated pose (if available)
                    absolute_error=make_pose_error(
                        estimate_origins[repeat_idx].find_relative(scaled_poses[repeat_idx]),
                        ground_truth_origins[repeat_idx].find_relative(frame_result.pose)
                    ) if scaled_poses[repeat_idx] is not None else None,
                    # Compute the error of the motion relative to the true motion
                    relative_error=make_pose_error(
                        scaled_motions[repeat_idx],
                        frame_result.motion
                    ) if scaled_motions[repeat_idx] is not None else None,
                    # Compute the error between the motion and the average estimated motion
                    noise=make_pose_error(
                        scaled_motions[repeat_idx],
                        average_motion
                    ) if scaled_motions[repeat_idx] is not None and average_motion is not None else None,
                    systemic_error=make_pose_error(
                        average_motion,
                        frame_result.motion
                    ) if average_motion is not None else None
                )
                estimate_errors[repeat_idx].append(frame_error)

        # Add any accumulated tracking information left over at the end
        if len(trial_results[0].results) > 0:
            for repeat_idx, tracking in enumerate(is_tracking):
                if tracking:
                    frames_found[repeat_idx].append(tracking_frames[repeat_idx])
                    distances_found[repeat_idx].append(tracking_distances[repeat_idx])
                    times_found[repeat_idx].append(current_tracking_time[repeat_idx] - prev_tracking_time[repeat_idx])
                else:
                    frames_lost[repeat_idx].append(tracking_frames[repeat_idx])
                    distances_lost[repeat_idx].append(tracking_distances[repeat_idx])
                    times_lost[repeat_idx].append(current_tracking_time[repeat_idx] - prev_tracking_time[repeat_idx])

        # Once we've tallied all the results, either succeed or fail based on the number of results.
        if len(estimate_errors) <= 0 or any(len(trial_errors) <= 0 for trial_errors in estimate_errors):
            return FrameErrorResult(
                metric=self,
                trial_results=trial_results,
                success=False,
                message="No measurable errors for these trajectories"
            )
        return make_frame_error_result(
            metric=self,
            trial_results=trial_results,
            errors=[
                TrialErrors(
                    frame_errors=estimate_errors[repeat],
                    frames_lost=frames_lost[repeat],
                    frames_found=frames_found[repeat],
                    times_lost=times_lost[repeat],
                    times_found=times_found[repeat],
                    distances_lost=distances_lost[repeat],
                    distances_found=distances_found[repeat]
                )
                for repeat, trial_result in enumerate(trial_results)
            ]
        )


def compute_loop_distances_and_angles(current_pose: tf.Transform, linked_poses: typing.Iterable[tf.Transform]) -> \
        typing.Tuple[typing.List[float], typing.List[float]]:
    relative_poses = [
        current_pose.find_relative(linked_pose)
        for linked_pose in linked_poses
    ]
    return [
        np.linalg.norm(relative_pose.location)
        for relative_pose in relative_poses
    ], [
        tf.quat_angle(relative_pose.rotation_quat(w_first=True))
        for relative_pose in relative_poses
    ]


def align_trajectory_to_ground_truth(
        trajectory: typing.Iterable[tf.Transform],
        ground_truth: typing.Iterable[tf.Transform],
        compute_scale: bool = False,
        use_symmetric_scale: bool = True
) -> typing.Tuple[tf.Transform, float]:
    """
    Compute a single Transform and scale to optimally fit an estimated trajectory to a ground truth trajectory

    Based on
    "Closed-form solution of absolute orientation using unit quaternions" by Berthold Horn (1987)
    as recommended by the TUM ATE benchmark

    :param trajectory: The list of estimated points to transform
    :param ground_truth: The list of ground truth points
    :param compute_scale: Whether to compute a scale. Some trajectories should not need it
    :param use_symmetric_scale: Whether to use the scale from the symmetric error as in Section 2E.
    Synthetic data has more precise ground truth, which should be preferred for scale calculation.
    :return: A Transform and scale such that taking each trajectory point relative to the output, and multiplying by
    the scale transforms the trajectory point into the ground truth frame.
    """
    estimated_points = [pose.location for pose in trajectory]
    gt_points = [pose.location for pose in ground_truth]
    if not len(estimated_points) == len(gt_points):
        # The two trajectories must consist of corresponding points
        raise RuntimeError(f"Cannot resolve together trajectories of different lengths, points must correspond")

    # Normalise the two sets of points to have origin 0, 0, 0; as in Section 2C of the paper
    estimated_centroid = np.mean(estimated_points, axis=0)
    gt_centroid = np.mean(gt_points, axis=0)

    # Compute the sum of outer products (matrix M in Section 4A)
    prod_sum = sum(
        np.outer(p1 - gt_centroid, p2 - estimated_centroid)
        for p1, p2 in zip(gt_points, estimated_points)
    )

    # Rearrange the elements of prod_sum into the matrix N from Section 4A
    quat_product_matrix = np.array([
        [prod_sum[0, 0] + prod_sum[1, 1] + prod_sum[2, 2], prod_sum[1, 2] - prod_sum[2, 1], prod_sum[2, 0] - prod_sum[0, 2], prod_sum[0, 1] - prod_sum[1, 0]],
        [prod_sum[1, 2] - prod_sum[2, 1], prod_sum[0, 0] - prod_sum[1, 1] - prod_sum[2, 2], prod_sum[0, 1] + prod_sum[1, 0], prod_sum[2, 0] + prod_sum[0, 2]],
        [prod_sum[2, 0] - prod_sum[0, 2], prod_sum[0, 1] + prod_sum[1, 0], prod_sum[1, 1] - prod_sum[0, 0] - prod_sum[2, 2], prod_sum[1, 2] + prod_sum[2, 1]],
        [prod_sum[0, 1] - prod_sum[1, 0], prod_sum[2, 0] + prod_sum[0, 2], prod_sum[1, 2] + prod_sum[2, 1], prod_sum[2, 2] - prod_sum[0, 0] - prod_sum[1, 1]]
    ])

    # The optimum rotation quat is the largest eigenvector corresponding to the largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(quat_product_matrix)
    max_idx = np.argmax(eigenvalues)
    rotation_quat = eigenvectors[:, max_idx]   # Optimum rotation, w-first

    scale = 1.0
    if compute_scale:
        if use_symmetric_scale:
            # Use the symmetric definition in Section 2E
            # Flipped so that pose * scale = unscaled pose
            scale = np.sqrt(
                sum(np.dot(point - gt_centroid, point - gt_centroid) for point in gt_points) /
                sum(np.dot(point - estimated_centroid, point - estimated_centroid) for point in estimated_points)
            )
        else:
            # Base the scale on the rotation, as in Section 2D
            # s = D / S_{l}
            # Except that we want to multiply by the scale to transform to the gt frame, so we invert
            scale = sum(np.dot(point - gt_centroid, point - gt_centroid) for point in gt_points)
            scale = scale / sum(
                np.dot(estimated_point - estimated_centroid,
                       t3.quaternions.rotate_vector(gt_point - gt_centroid, rotation_quat))
                for gt_point, estimated_point in zip(gt_points, estimated_points)
            )

    # Optimum translation is the difference in the centroids, once corrected for rotation and scale
    translation = estimated_centroid - t3.quaternions.rotate_vector(gt_centroid / scale, rotation_quat)
    return tf.Transform(
        location=translation,
        rotation=rotation_quat,
        w_first=True
    ), scale
