# Copyright (c) 2017, John Skinner
import unittest
import os
import numpy as np
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager

from arvet.util.test_helpers import ExtendedTestCase
from arvet.core.system import VisionSystem
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image import Image
from arvet.core.image_collection import ImageCollection

from arvet_slam.systems.visual_odometry.libviso2 import LibVisOMonoSystem, MatcherRefinement
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult


class TestLibVisOMonoDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        VisionSystem._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        VisionSystem._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)

    def test_stores_and_loads(self):
        kwargs = {
            'matcher_nms_n': np.random.randint(0, 5),
            'matcher_nms_tau': np.random.randint(0, 100),
            'matcher_match_binsize': np.random.randint(0, 100),
            'matcher_match_radius': np.random.randint(0, 500),
            'matcher_match_disp_tolerance': np.random.randint(0, 10),
            'matcher_outlier_disp_tolerance': np.random.randint(0, 10),
            'matcher_outlier_flow_tolerance': np.random.randint(0, 10),
            'matcher_multi_stage': np.random.choice([True, False]),
            'matcher_half_resolution': np.random.choice([True, False]),
            'matcher_refinement': np.random.choice(MatcherRefinement),
            'bucketing_max_features': np.random.randint(0, 10),
            'bucketing_bucket_width': np.random.randint(0, 100),
            'bucketing_bucket_height': np.random.randint(0, 100),
            'height': np.random.uniform(0.0, 3.0),
            'pitch': np.random.uniform(0.0, 3.0),
            'ransac_iters': np.random.randint(0, 100),
            'inlier_threshold': np.random.uniform(0.0, 3.0),
            'motion_threshold': np.random.uniform(0.0, 100.0)
        }
        obj = LibVisOMonoSystem(**kwargs)
        obj.save()

        # Load all the entities
        all_entities = list(VisionSystem.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()

    def test_result_saves(self):
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)

        # Make an image collection with some number of images
        images = []
        image_builder = DemoImageBuilder(mode=ImageMode.MONOCULAR, width=160, height=120)
        num_images = 10
        for time in range(num_images):
            image = image_builder.create_frame(time / num_images)
            image.save()
            images.append(image)
        image_collection = ImageCollection(
            images=images,
            timestamps=list(range(len(images))),
            sequence_type=ImageSequenceType.SEQUENTIAL
        )
        image_collection.save()

        subject = LibVisOMonoSystem()
        subject.save()

        # Actually run the system using mocked images
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), 1 / 10)
        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for time, image in enumerate(images):
            subject.process_image(image, time)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)
        result.image_source = image_collection
        result.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], result)
        all_entities[0].delete()

        SLAMTrialResult._mongometa.collection.drop()
        ImageCollection._mongometa.collection.drop()
        Image._mongometa.collection.drop()

    def test_get_instance_can_create_an_instance(self):
        matcher_nms_n = np.random.randint(1, 5)
        matcher_nms_tau = np.random.randint(20, 100)
        matcher_match_binsize = np.random.randint(20, 100)
        matcher_match_radius = np.random.randint(20, 100)
        matcher_match_disp_tolerance = np.random.randint(1, 5)
        matcher_outlier_disp_tolerance = np.random.randint(2, 10)
        matcher_outlier_flow_tolerance = np.random.randint(2, 10)
        matcher_multi_stage = np.random.choice([True, False])
        matcher_half_resolution = np.random.choice([True, False])
        matcher_refinement = np.random.choice(MatcherRefinement)
        bucketing_max_features = np.random.randint(2, 10)
        bucketing_bucket_width = np.random.randint(20, 100)
        bucketing_bucket_height = np.random.randint(20, 100)
        height = np.random.uniform(0.8, 1.2)
        pitch = np.random.uniform(-0.1, 0.1)
        ransac_iters = np.random.randint(500, 3000)
        inlier_threshold = np.random.uniform(0, 0.0001)
        motion_threshold = np.random.uniform(20.0, 300.0)
        obj = LibVisOMonoSystem.get_instance(
            matcher_nms_n=matcher_nms_n,
            matcher_nms_tau=matcher_nms_tau,
            matcher_match_binsize=matcher_match_binsize,
            matcher_match_radius=matcher_match_radius,
            matcher_match_disp_tolerance=matcher_match_disp_tolerance,
            matcher_outlier_disp_tolerance=matcher_outlier_disp_tolerance,
            matcher_outlier_flow_tolerance=matcher_outlier_flow_tolerance,
            matcher_multi_stage=matcher_multi_stage,
            matcher_half_resolution=matcher_half_resolution,
            matcher_refinement=matcher_refinement,
            bucketing_max_features=bucketing_max_features,
            bucketing_bucket_width=bucketing_bucket_width,
            bucketing_bucket_height=bucketing_bucket_height,
            height=height,
            pitch=pitch,
            ransac_iters=ransac_iters,
            inlier_threshold=inlier_threshold,
            motion_threshold=motion_threshold
        )
        self.assertEqual(matcher_nms_n, obj.matcher_nms_n)
        self.assertEqual(matcher_nms_tau, obj.matcher_nms_tau)
        self.assertEqual(matcher_match_binsize, obj.matcher_match_binsize)
        self.assertEqual(matcher_match_radius, obj.matcher_match_radius)
        self.assertEqual(matcher_match_disp_tolerance, obj.matcher_match_disp_tolerance)
        self.assertEqual(matcher_outlier_disp_tolerance, obj.matcher_outlier_disp_tolerance)
        self.assertEqual(matcher_outlier_flow_tolerance, obj.matcher_outlier_flow_tolerance)
        self.assertEqual(matcher_multi_stage, obj.matcher_multi_stage)
        self.assertEqual(matcher_half_resolution, obj.matcher_half_resolution)
        self.assertEqual(matcher_refinement, obj.matcher_refinement)
        self.assertEqual(bucketing_max_features, obj.bucketing_max_features)
        self.assertEqual(bucketing_bucket_width, obj.bucketing_bucket_width)
        self.assertEqual(bucketing_bucket_height, obj.bucketing_bucket_height)
        self.assertEqual(height, obj.height)
        self.assertEqual(pitch, obj.pitch)
        self.assertEqual(ransac_iters, obj.ransac_iters)
        self.assertEqual(inlier_threshold, obj.inlier_threshold)
        self.assertEqual(motion_threshold, obj.motion_threshold)

    def test_creates_an_instance_with_defaults_by_default(self):
        obj = LibVisOMonoSystem()
        result = LibVisOMonoSystem.get_instance()
        self.assertEqual(obj.matcher_nms_n, result.matcher_nms_n)
        self.assertEqual(obj.matcher_nms_tau, result.matcher_nms_tau)
        self.assertEqual(obj.matcher_match_binsize, result.matcher_match_binsize)
        self.assertEqual(obj.matcher_match_radius, result.matcher_match_radius)
        self.assertEqual(obj.matcher_match_disp_tolerance, result.matcher_match_disp_tolerance)
        self.assertEqual(obj.matcher_outlier_disp_tolerance, result.matcher_outlier_disp_tolerance)
        self.assertEqual(obj.matcher_outlier_flow_tolerance, result.matcher_outlier_flow_tolerance)
        self.assertEqual(obj.matcher_multi_stage, result.matcher_multi_stage)
        self.assertEqual(obj.matcher_half_resolution, result.matcher_half_resolution)
        self.assertEqual(obj.matcher_refinement, result.matcher_refinement)
        self.assertEqual(obj.bucketing_max_features, result.bucketing_max_features)
        self.assertEqual(obj.bucketing_bucket_width, result.bucketing_bucket_width)
        self.assertEqual(obj.bucketing_bucket_height, result.bucketing_bucket_height)
        self.assertEqual(obj.height, result.height)
        self.assertEqual(obj.pitch, result.pitch)
        self.assertEqual(obj.ransac_iters, result.ransac_iters)
        self.assertEqual(obj.inlier_threshold, result.inlier_threshold)
        self.assertEqual(obj.motion_threshold, result.motion_threshold)

    def test_get_instance_returns_an_existing_instance_simple(self):
        obj = LibVisOMonoSystem()
        obj.save()

        result = LibVisOMonoSystem.get_instance()
        self.assertEqual(obj.pk, result.pk)
        self.assertEqual(obj, result)

    def test_get_instance_returns_an_existing_instance_complex(self):
        matcher_nms_n = np.random.randint(1, 5)
        matcher_nms_tau = np.random.randint(20, 100)
        matcher_match_binsize = np.random.randint(20, 100)
        matcher_match_radius = np.random.randint(20, 100)
        matcher_match_disp_tolerance = np.random.randint(1, 5)
        matcher_outlier_disp_tolerance = np.random.randint(2, 10)
        matcher_outlier_flow_tolerance = np.random.randint(2, 10)
        matcher_multi_stage = np.random.choice([True, False])
        matcher_half_resolution = np.random.choice([True, False])
        matcher_refinement = np.random.choice(MatcherRefinement)
        bucketing_max_features = np.random.randint(2, 10)
        bucketing_bucket_width = np.random.randint(20, 100)
        bucketing_bucket_height = np.random.randint(20, 100)
        height = np.random.uniform(0.8, 1.2)
        pitch = np.random.uniform(-0.1, 0.1)
        ransac_iters = np.random.randint(500, 3000)
        inlier_threshold = np.random.uniform(0, 0.0001)
        motion_threshold = np.random.uniform(20.0, 300.0)

        obj = LibVisOMonoSystem(
            matcher_nms_n=matcher_nms_n,
            matcher_nms_tau=matcher_nms_tau,
            matcher_match_binsize=matcher_match_binsize,
            matcher_match_radius=matcher_match_radius,
            matcher_match_disp_tolerance=matcher_match_disp_tolerance,
            matcher_outlier_disp_tolerance=matcher_outlier_disp_tolerance,
            matcher_outlier_flow_tolerance=matcher_outlier_flow_tolerance,
            matcher_multi_stage=matcher_multi_stage,
            matcher_half_resolution=matcher_half_resolution,
            matcher_refinement=matcher_refinement,
            bucketing_max_features=bucketing_max_features,
            bucketing_bucket_width=bucketing_bucket_width,
            bucketing_bucket_height=bucketing_bucket_height,
            height=height,
            pitch=pitch,
            ransac_iters=ransac_iters,
            inlier_threshold=inlier_threshold,
            motion_threshold=motion_threshold
        )
        obj.save()

        result = LibVisOMonoSystem.get_instance(
            matcher_nms_n=matcher_nms_n,
            matcher_nms_tau=matcher_nms_tau,
            matcher_match_binsize=matcher_match_binsize,
            matcher_match_radius=matcher_match_radius,
            matcher_match_disp_tolerance=matcher_match_disp_tolerance,
            matcher_outlier_disp_tolerance=matcher_outlier_disp_tolerance,
            matcher_outlier_flow_tolerance=matcher_outlier_flow_tolerance,
            matcher_multi_stage=matcher_multi_stage,
            matcher_half_resolution=matcher_half_resolution,
            matcher_refinement=matcher_refinement,
            bucketing_max_features=bucketing_max_features,
            bucketing_bucket_width=bucketing_bucket_width,
            bucketing_bucket_height=bucketing_bucket_height,
            height=height,
            pitch=pitch,
            ransac_iters=ransac_iters,
            inlier_threshold=inlier_threshold,
            motion_threshold=motion_threshold
        )
        self.assertEqual(obj.pk, result.pk)
        self.assertEqual(obj, result)


class TestLibVisOMono(unittest.TestCase):

    def test_can_start_and_stop_trial(self):
        subject = LibVisOMonoSystem()
        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        result = subject.finish_trial()
        self.assertIsInstance(result, SLAMTrialResult)

    def test_get_columns_returns_column_list(self):
        subject = LibVisOMonoSystem()
        self.assertEqual({
            'seed',
            'in_width',
            'in_height',
            'in_fx',
            'in_fy',
            'in_cx',
            'in_cy',
            'matcher_nms_n',
            'matcher_nms_tau',
            'matcher_match_binsize',
            'matcher_match_radius',
            'matcher_match_disp_tolerance',
            'matcher_outlier_disp_tolerance',
            'matcher_outlier_flow_tolerance',
            'matcher_multi_stage',
            'matcher_half_resolution',
            'matcher_refinement',
            'bucketing_max_features',
            'bucketing_bucket_width',
            'bucketing_bucket_height',
            'height',
            'pitch',
            'ransac_iters',
            'inlier_threshold',
            'motion_threshold',
        }, subject.get_columns())

    def test_get_properties_returns_the_value_of_all_columns(self):
        matcher_nms_n = 10
        matcher_nms_tau = 35
        matcher_match_binsize = 16
        matcher_match_radius = 155
        matcher_match_disp_tolerance = 4
        matcher_outlier_disp_tolerance = 3
        matcher_outlier_flow_tolerance = 6
        matcher_multi_stage = False
        matcher_half_resolution = False
        matcher_refinement = MatcherRefinement.SUBPIXEL
        bucketing_max_features = 6
        bucketing_bucket_width = 45
        bucketing_bucket_height = 66
        height = 0.8
        pitch = 0.22
        ransac_iters = 1444
        inlier_threshold = 0.0006
        motion_threshold = 78.4

        subject = LibVisOMonoSystem(
            matcher_nms_n=matcher_nms_n,
            matcher_nms_tau=matcher_nms_tau,
            matcher_match_binsize=matcher_match_binsize,
            matcher_match_radius=matcher_match_radius,
            matcher_match_disp_tolerance=matcher_match_disp_tolerance,
            matcher_outlier_disp_tolerance=matcher_outlier_disp_tolerance,
            matcher_outlier_flow_tolerance=matcher_outlier_flow_tolerance,
            matcher_multi_stage=matcher_multi_stage,
            matcher_half_resolution=matcher_half_resolution,
            matcher_refinement=matcher_refinement,
            bucketing_max_features=bucketing_max_features,
            bucketing_bucket_width=bucketing_bucket_width,
            bucketing_bucket_height=bucketing_bucket_height,
            height=height,
            pitch=pitch,
            ransac_iters=ransac_iters,
            inlier_threshold=inlier_threshold,
            motion_threshold=motion_threshold
        )
        properties = subject.get_properties()
        for key, value in {
            'seed': np.nan,
            'in_width': np.nan,
            'in_height': np.nan,
            'in_fx': np.nan,
            'in_fy': np.nan,
            'in_cx': np.nan,
            'in_cy': np.nan,
            'matcher_nms_n': matcher_nms_n,
            'matcher_nms_tau': matcher_nms_tau,
            'matcher_match_binsize': matcher_match_binsize,
            'matcher_match_radius': matcher_match_radius,
            'matcher_match_disp_tolerance': matcher_match_disp_tolerance,
            'matcher_outlier_disp_tolerance': matcher_outlier_disp_tolerance,
            'matcher_outlier_flow_tolerance': matcher_outlier_flow_tolerance,
            'matcher_multi_stage': matcher_multi_stage,
            'matcher_half_resolution': matcher_half_resolution,
            'matcher_refinement': matcher_refinement,
            'bucketing_max_features': bucketing_max_features,
            'bucketing_bucket_width': bucketing_bucket_width,
            'bucketing_bucket_height': bucketing_bucket_height,
            'height': height,
            'pitch': pitch,
            'ransac_iters': ransac_iters,
            'inlier_threshold': inlier_threshold,
            'motion_threshold': motion_threshold
        }.items():
            self.assertIn(key, properties)
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(properties[key]))
            else:
                self.assertEqual(value, properties[key])

    def test_get_properties_returns_only_requested_columns_that_exist(self):
        matcher_nms_n = 10
        matcher_nms_tau = 35
        matcher_match_binsize = 16
        matcher_match_radius = 155
        matcher_match_disp_tolerance = 4
        matcher_outlier_disp_tolerance = 3
        matcher_outlier_flow_tolerance = 6
        matcher_multi_stage = False
        matcher_half_resolution = False
        matcher_refinement = MatcherRefinement.SUBPIXEL
        bucketing_max_features = 6
        bucketing_bucket_width = 45
        bucketing_bucket_height = 66
        height = 0.8
        pitch = 0.22
        ransac_iters = 1444
        inlier_threshold = 0.0006
        motion_threshold = 78.4

        subject = LibVisOMonoSystem(
            matcher_nms_n=matcher_nms_n,
            matcher_nms_tau=matcher_nms_tau,
            matcher_match_binsize=matcher_match_binsize,
            matcher_match_radius=matcher_match_radius,
            matcher_match_disp_tolerance=matcher_match_disp_tolerance,
            matcher_outlier_disp_tolerance=matcher_outlier_disp_tolerance,
            matcher_outlier_flow_tolerance=matcher_outlier_flow_tolerance,
            matcher_multi_stage=matcher_multi_stage,
            matcher_half_resolution=matcher_half_resolution,
            matcher_refinement=matcher_refinement,
            bucketing_max_features=bucketing_max_features,
            bucketing_bucket_width=bucketing_bucket_width,
            bucketing_bucket_height=bucketing_bucket_height,
            height=height,
            pitch=pitch,
            ransac_iters=ransac_iters,
            inlier_threshold=inlier_threshold,
            motion_threshold=motion_threshold
        )
        self.assertEqual({
            'matcher_match_binsize': matcher_match_binsize,
            'matcher_match_disp_tolerance': matcher_match_disp_tolerance,
            'matcher_outlier_disp_tolerance': matcher_outlier_disp_tolerance,
            'matcher_refinement': matcher_refinement,
            'pitch': pitch,
        }, subject.get_properties({
            'matcher_match_binsize', 'matcher_match_disp_tolerance', 'matcher_outlier_disp_tolerance',
            'not_a_column',
            'matcher_refinement', 'pitch',
            'also_not_a_column', 'sir_not_appearing_in_these_columns'
        }))


class TestLibVisOMonoExecution(ExtendedTestCase):

    def test_simple_trial_run(self):
        # Actually run the system using mocked images
        num_frames = 50
        max_time = 50
        speed = 0.1
        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            width=640, height=480, num_stars=150,
            length=max_time * speed, speed=speed,
            close_ratio=0.6, min_size=1, max_size=50
        )

        subject = LibVisOMonoSystem(motion_threshold=1000)
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL, seed=0)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertEqual(subject, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertEqual({
            'seed': 0,
            'in_fx': image_builder.focal_length,
            'in_fy': image_builder.focal_length,
            'in_cu': image_builder.width / 2,
            'in_cv': image_builder.height / 2,
            'in_width': image_builder.width,
            'in_height': image_builder.height
        }, result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        has_been_lost = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)

            # If we're lost, our tracking state should depend of if we've been lost before
            is_first_frame = False
            if frame_result.tracking_state != TrackingState.OK:
                if has_been_found:
                    has_been_lost = True
                    self.assertEqual(frame_result.tracking_state, TrackingState.LOST)
                else:
                    self.assertEqual(frame_result.tracking_state, TrackingState.NOT_INITIALIZED)
            elif has_been_found is False:
                is_first_frame = True
                has_been_found = True

            # Motion should be none when we are lost, and on the first found frame
            if is_first_frame or frame_result.tracking_state != TrackingState.OK:
                self.assertIsNone(frame_result.estimated_motion)
            else:
                self.assertIsNotNone(frame_result.estimated_motion)

            # Estimates will be none until we get a successful estimate, or after it has lost
            if not has_been_found or has_been_lost:
                self.assertIsNone(frame_result.estimated_pose)
            else:
                self.assertIsNotNone(frame_result.estimated_pose)
        self.assertTrue(has_been_found)

    def test_is_consistent_with_fixed_seed(self):
        # Actually run the system using mocked images
        num_frames = 20
        max_time = 50
        speed = 0.1
        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            width=640, height=480, num_stars=150,
            length=max_time * speed, speed=speed,
            close_ratio=0.6, min_size=10, max_size=100
        )

        subject = LibVisOMonoSystem(motion_threshold=1000)
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL, seed=0)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result1 = subject.finish_trial()

        subject.start_trial(ImageSequenceType.SEQUENTIAL, seed=0)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result2 = subject.finish_trial()

        has_any_estimate = False
        self.assertEqual(len(result1.results), len(result2.results))
        for frame_result_1, frame_result_2 in zip(result1.results, result2.results):
            self.assertEqual(frame_result_1.timestamp, frame_result_2.timestamp)
            self.assertEqual(frame_result_1.tracking_state, frame_result_2.tracking_state)
            if frame_result_1.estimated_motion is None or frame_result_2.estimated_motion is None:
                self.assertEqual(frame_result_1.estimated_motion, frame_result_2.estimated_motion)
            else:
                has_any_estimate = True
                motion1 = frame_result_1.estimated_motion
                motion2 = frame_result_2.estimated_motion

                loc_diff = motion1.location - motion2.location
                self.assertNPClose(loc_diff, np.zeros(3), rtol=0, atol=1e-14)
                quat_diff = motion1.rotation_quat(True) - motion2.rotation_quat(True)
                self.assertNPClose(quat_diff, np.zeros(4), rtol=0, atol=1e-14)
        self.assertTrue(has_any_estimate)

    def test_is_different_with_changed_seed(self):
        # Actually run the system using mocked images
        num_frames = 50
        max_time = 50
        speed = 0.1
        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            width=640, height=480, num_stars=150,
            length=max_time * speed, speed=speed,
            close_ratio=0.6, min_size=10, max_size=100
        )

        subject = LibVisOMonoSystem(motion_threshold=1000)
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)
        subject.set_stereo_offset(image_builder.get_stereo_offset())

        subject.start_trial(ImageSequenceType.SEQUENTIAL, seed=0)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result1 = subject.finish_trial()

        subject.start_trial(ImageSequenceType.SEQUENTIAL, seed=2)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result2 = subject.finish_trial()

        self.assertEqual(len(result1.results), len(result2.results))
        different_tracking = 0
        loc_diff = np.zeros(3)
        quat_diff = np.zeros(4)
        for frame_result_1, frame_result_2 in zip(result1.results, result2.results):
            self.assertEqual(frame_result_1.timestamp, frame_result_2.timestamp)
            if frame_result_1.tracking_state != frame_result_2.tracking_state:
                different_tracking += 1
            elif frame_result_1.estimated_motion is not None and frame_result_2.estimated_motion is not None:
                motion1 = frame_result_1.estimated_motion
                motion2 = frame_result_2.estimated_motion

                loc_diff += np.abs(motion1.location - motion2.location)
                quat_diff += np.abs(motion1.rotation_quat(True) - motion2.rotation_quat(True))
        if different_tracking <= 0:
            # If the tracking is the same, make sure the estimates are at least different
            self.assertNotNPClose(loc_diff, np.zeros(3), rtol=0, atol=1e-10)
            self.assertNotNPClose(quat_diff, np.zeros(4), rtol=0, atol=1e-10)
