# Copyright (c) 2018, John Skinner
import typing
import logging
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

        # Choose transforms between each trajectory and the ground truth
        estimate_origins_and_scales = [
            robust_align_trajectory_to_ground_truth(
                [frame_result.estimated_pose
                 for frame_result in trial_result.results if frame_result.estimated_pose is not None],
                [frame_result.pose
                 for frame_result in trial_result.results if frame_result.estimated_pose is not None],
                compute_scale=not bool(trial_result.has_scale),
                use_symmetric_scale=True
            )
            for trial_result in trial_results
        ]
        motion_scales = [1.0] * len(trial_results)
        for idx in range(len(trial_results)):
            if not trial_results[idx].has_scale:
                motion_scales[idx] = robust_compute_motions_scale(
                    [frame_result.estimated_motion
                     for frame_result in trial_results[idx].results if frame_result.estimated_motion is not None],
                    [frame_result.motion
                     for frame_result in trial_results[idx].results if frame_result.estimated_motion is not None],
                )

        # Then, tally all the errors for all the computed trajectories
        estimate_errors = [[] for _ in range(len(trial_results))]
        image_columns = set()
        distances_lost = [[] for _ in range(len(trial_results))]
        times_lost = [[] for _ in range(len(trial_results))]
        frames_lost = [[] for _ in range(len(trial_results))]
        distances_found = [[] for _ in range(len(trial_results))]
        times_found = [[] for _ in range(len(trial_results))]
        frames_found = [[] for _ in range(len(trial_results))]

        is_tracking = [False for _ in range(len(trial_results))]
        tracking_frames = [0 for _ in range(len(trial_results))]
        tracking_distances = [0 for _ in range(len(trial_results))]
        prev_tracking_time = [0 for _ in range(len(trial_results))]
        current_tracking_time = [0 for _ in range(len(trial_results))]

        for frame_idx, frame_results in enumerate(zip(*(trial_result.results for trial_result in trial_results))):
            # Get the estimated motions and absolute poses for each trial,
            # And convert them to the ground truth coordinate frame using
            # the scale, translation and rotation we chose
            scaled_motions = [
                tf.Transform(
                    location=frame_results[idx].estimated_motion.location * motion_scales[idx],
                    rotation=frame_results[idx].estimated_motion.rotation_quat(True), w_first=True
                ) if frame_results[idx].estimated_motion is not None else None
                for idx in range(len(frame_results))
            ]
            scaled_poses = [
                align_point(
                    pose=frame_results[idx].estimated_pose,
                    shift=estimate_origins_and_scales[idx][0],
                    rotation=estimate_origins_and_scales[idx][1],
                    scale=estimate_origins_and_scales[idx][2]
                ) if frame_results[idx].estimated_pose is not None else None
                for idx in range(len(frame_results))
            ]

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
                        scaled_poses[repeat_idx],   # The
                        frame_result.pose
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


def robust_align_trajectory_to_ground_truth(
        trajectory: typing.Iterable[tf.Transform],
        ground_truth: typing.Iterable[tf.Transform],
        compute_scale: bool = False,
        use_symmetric_scale: bool = True,
        outlier_sigma_threshold: float = 2.0
) -> typing.Tuple[np.ndarray, np.ndarray, float]:
    """
    Choose an optimal transformation of the given trajectory onto the ground truth.
    We do this twice to be robust to outliers.
    We choose an initial transform, and then discard points
    The final transform is based on this filtered set of points.

    Outliers are filtered based on the distribution of residual errors after transformation
    Points with a residual error more than the configured number of standard deviations (outlier_sigma_threshold)
    away from the mean are rejected.
    If the residuals are small or have small variance (<10^-15),
    or if we would reject more than half the original points, we return the initial estimated transform rather
    than reject outliers.

    See: https://en.wikipedia.org/wiki/Robust_regression
    https://en.wikipedia.org/wiki/Huber_loss
    :param trajectory: The estimated poses that will be transformed.
    :param ground_truth: The true poses onto which the trajectory must be transformed
    :param compute_scale: Whether to adjust the scale of the given trajectory. Monocular sequences have this freedom.
    :param use_symmetric_scale: Whether to use the symmetric calculation of scale from the Horn paper.
    :param outlier_sigma_threshold: The distance from the mean residual that will be excluded as an error.
    :return:
    """
    trajectory = list(trajectory)
    ground_truth = list(ground_truth)
    if not len(trajectory) == len(ground_truth):
        # The two trajectories must consist of corresponding points
        raise RuntimeError("Cannot resolve together trajectories of different lengths "
                           f"({len(trajectory)} vs {len(ground_truth)}), points must correspond")
    elif len(trajectory) <= 0 or len(ground_truth) <= 0:
        logging.getLogger(__name__).info(f"No optimal transform for an empty trajectory")
        return np.array([0, 0, 0]), np.array([1, 0, 0, 0]), 1.0

    # Estimate an initial trajectory
    translation, rotation, scale = align_trajectory_to_ground_truth(
        trajectory, ground_truth, compute_scale, use_symmetric_scale)

    # Measure the resulting error after transformation, and reject points
    shifted_trajectory = [align_point(pose, translation, rotation, scale) for pose in trajectory]
    residuals = [pose.location - true_pose.location for pose, true_pose in zip(shifted_trajectory, ground_truth)]
    mean_error = np.mean(residuals, axis=0)
    std_error = np.std(residuals, axis=0)
    if np.all(np.abs(mean_error) < 1e-14) and np.all(np.abs(std_error) < 1e-14):
        # We nailed it with the first transform, there is minimal error. Just return what we've got
        return translation, rotation, scale
    to_include = [np.all(np.abs(residual - mean_error) <= outlier_sigma_threshold * std_error) for residual in residuals]
    num_retained = sum(to_include)
    if num_retained < 0.5 * len(to_include) or num_retained == len(to_include):
        # Too many points are outliers, we can't work like this. Just return our initial best guess.
        # Or our first guess was perfect, there are no outliers.
        return translation, rotation, scale

    filtered_trajectory = [pose for pose, include in zip(trajectory, to_include) if include]
    filtered_ground_truth = [pose for pose, include in zip(ground_truth, to_include) if include]
    return align_trajectory_to_ground_truth(
        filtered_trajectory, filtered_ground_truth, compute_scale, use_symmetric_scale)


def align_trajectory_to_ground_truth(
        trajectory: typing.Iterable[tf.Transform],
        ground_truth: typing.Iterable[tf.Transform],
        compute_scale: bool = False,
        use_symmetric_scale: bool = True
) -> typing.Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a single Transform and scale to optimally fit an estimated trajectory to a ground truth trajectory

    Based on
    "Closed-form solution of absolute orientation using unit quaternions" by Berthold Horn (1987)
    as recommended by the TUM ATE benchmark.
    We're using the ground truth as the "right" set of points, and the estimated as the "left".
    Thus we find a transformation of the estimated points onto the ground truth points.

    There is a degenerate case, where all the points are co-linear.
    In this case, we cannot estimate a rotation around the central axis that minimised rotation error.
    However, few real trajectories will be in a flawlessly straight line, right?

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
        raise RuntimeError("Cannot resolve together trajectories of different lengths "
                           f"({len(estimated_points)} vs {len(gt_points)}), points must correspond")
    elif len(estimated_points) <= 0 or len(gt_points) <= 0:
        logging.getLogger(__name__).info(f"No optimal transform for an empty trajectory")
        return np.array([0, 0, 0]), np.array([1, 0, 0, 0]), 1.0

    # Normalise the two sets of points to have origin 0, 0, 0; as in Section 2C of the paper
    estimated_centroid = np.mean(estimated_points, axis=0)
    gt_centroid = np.mean(gt_points, axis=0)

    # Compute the sum of outer products (matrix M in Section 4A)
    prod_sum = sum(
        np.outer(p1 - estimated_centroid, p2 - gt_centroid)
        for p1, p2 in zip(estimated_points, gt_points)
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
    if sum(1 for val in eigenvalues if val == eigenvalues[max_idx]) > 1:
        # In the case where there are multiple equal max eigenvalues,
        # We choose the eigenvector which represents the smallest rotation, that is
        # is the closest to [1, 0, 0, 0]
        diff = eigenvectors - np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        diff = np.sum(diff * diff, axis=0)
        _, max_idx = min((diff[idx], idx) for idx in range(len(eigenvalues))
                         if eigenvalues[idx] == eigenvalues[max_idx])
    rotation_quat = eigenvectors[:, max_idx]   # Optimum rotation, w-first

    scale = 1.0
    if compute_scale:
        if use_symmetric_scale:
            # Use the symmetric definition in Section 2E
            gt_variance = sum(np.dot(point - gt_centroid, point - gt_centroid) for point in gt_points)
            est_variance = sum(
                np.dot(point - estimated_centroid, point - estimated_centroid) for point in estimated_points)
            if gt_variance > 1e-15 and est_variance > 1e-15:
                scale = np.sqrt(
                    gt_variance /
                    est_variance
                )
            elif est_variance <= 1e-15:
                logging.getLogger(__name__).info(
                    f"Estimated variation is too small to estimate scale ({est_variance})")
            elif gt_variance <= 1e-15:
                logging.getLogger(__name__).info(
                    f"Ground truth variation is too small to estimate scale ({gt_variance})")
        else:
            # Base the scale on the rotation, as in Section 2E
            # We're using the ground truth in the denominator because we assume those points are more precise
            # \bar{s} = \bar{D} / S_{r}
            # and then flip it to return to converting from estimated to gt:
            # s ~= 1 / \bar{s} = S_{r} / \bar{D}
            inv_quat = rotation_quat * [1, -1, -1, -1]
            gt_variance = sum(np.dot(point - gt_centroid, point - gt_centroid) for point in gt_points)
            square_errors = sum(
                np.dot(estimated_point - estimated_centroid,
                       t3.quaternions.rotate_vector(gt_point - gt_centroid, inv_quat))
                for gt_point, estimated_point in zip(gt_points, estimated_points)
            )
            if gt_variance > 1e-15 and square_errors > 1e-15:
                scale = gt_variance / square_errors
            elif square_errors <= 1e-15:
                logging.getLogger(__name__).info(
                    f"Square difference between projected vectors is too small to estimate scale ({square_errors})")
            elif gt_variance <= 1e-15:
                logging.getLogger(__name__).info(
                    f"Ground truth variance is too low to estimate scale ({gt_variance})")

    # Optimum translation is the difference in the centroids, once corrected for rotation and scale
    translation = gt_centroid - scale * t3.quaternions.rotate_vector(estimated_centroid, rotation_quat)
    return translation, rotation_quat, scale


def align_point(pose: tf.Transform, shift: np.ndarray, rotation: np.ndarray, scale: float = 1.0) -> tf.Transform:
    """
    Transform an estimated point
    :param pose:
    :param shift:
    :param rotation:
    :param scale:
    :return:
    """
    return tf.Transform(
        location=shift + scale * t3.quaternions.rotate_vector(pose.location, rotation),
        rotation=t3.quaternions.qmult(rotation, pose.rotation_quat(w_first=True)),
        w_first=True
    )


def robust_compute_motions_scale(
        motions: typing.Iterable[tf.Transform],
        ground_truth: typing.Iterable[tf.Transform],
        outlier_sigma_threshold: float = 2.0
) -> float:
    """
    Estimate the optimal scale for estimated motions relative to the true motions.
    A monocular system has no reference for scale, and thus when evaluating,
    we choose an optimal re-scale against the ground truth that minimises errors.
    Only the scale is changed, motion direction and estimated orientation are unaffected.

    To account for outlier motions that might skew our results, we

    :param motions: The estimated motions.
    :param ground_truth: True motions for the same frames.
    :param outlier_sigma_threshold: Motions with a residual more than this many standard devations away from the mean
    will be rejected as outliers.
    :return:
    """
    # Estimate an initial scale
    scale = compute_motions_scale(motions, ground_truth)

    # Measure the resulting error after transformation, and reject points
    residuals = [scale * pose.location - true_pose.location for pose, true_pose in zip(motions, ground_truth)]
    mean_error = np.mean(residuals, axis=0)
    std_error = np.std(residuals, axis=0)
    if np.all(np.abs(mean_error) < 1e-14) and np.all(np.abs(std_error) < 1e-14):
        # We nailed it with the first scale, there is minimal error. Just return what we've got
        return scale
    to_include = [np.all(np.abs(residual - mean_error) <= outlier_sigma_threshold * std_error) for residual in
                  residuals]
    num_retained = sum(to_include)
    if num_retained < 0.5 * len(to_include) or num_retained == len(to_include):
        # Too many points are outliers, we can't work like this. Just return our initial best guess.
        # Or our first guess was perfect, there are no outliers.
        return scale

    filtered_motions = [motion for motion, include in zip(motions, to_include) if include]
    filtered_ground_truth = [motion for motion, include in zip(ground_truth, to_include) if include]
    return compute_motions_scale(filtered_motions, filtered_ground_truth)


def compute_motions_scale(
        motions: typing.Iterable[tf.Transform],
        ground_truth: typing.Iterable[tf.Transform]
) -> float:
    """
    Compute an optimial uniform scaling between estimated motions and ground truth.
    This does not include translation or rotation, so the optimal scale is simply
    min || true - scale * estimated ||
    which has optima at scale = sum(true dot estimated) / sum(estimated dot estimated)
    the sum of the dot produces of the points, over the sum of the

    Similar to the scale component in Section 2D and 2E of
    "Closed-form solution of absolute orientation using unit quaternions" by Berthold Horn (1987)

    :param motions: Estimated motions for some number of frames
    :param ground_truth: True motions for the same frames.
    :return: A uniform scale to transform the estimated motions onto the true motions
    """
    estimated_motions = [pose.location for pose in motions]
    true_motions = [pose.location for pose in ground_truth]
    if not len(estimated_motions) == len(true_motions):
        # The two trajectories must consist of corresponding points
        raise RuntimeError(f"Cannot resolve together motions of different lengths, points must correspond")
    # Choose an optimal scaling factor, without translation or rotation applied
    square_error = sum(
        np.dot(estimated_motion, true_motion)
        for estimated_motion, true_motion in zip(estimated_motions, true_motions)
    )
    estimated_variance = sum(
        np.dot(estimated_point, estimated_point)
        for estimated_point in estimated_motions
    )
    if square_error / len(estimated_motions) > 1e-14 and estimated_variance / len(estimated_motions) > 1e-14:
        # Use the absolute scale, so that the system cannot point the opposite direction, and still be valid
        return abs(square_error / estimated_variance)
    elif square_error / len(estimated_motions) <= 1e-14:
        logging.getLogger(__name__).warning(
            f"Motions are purpendicular to true motions ({square_error}), cannot choose scale")
    elif estimated_variance / len(estimated_motions) <= 1e-14:
        logging.getLogger(__name__).warning(
            f"Absolute motions are almost zero ({estimated_variance}), cannot scale")
    return 1.0
