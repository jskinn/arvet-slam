# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
import transforms3d as tf3d
import arvet.util.transform as tf
from arvet.util.test_helpers import ExtendedTestCase
import arvet_slam.systems.visual_odometry.libviso2 as viso


class TestMakeRelativePose(ExtendedTestCase):

    def test_returns_transform_object(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = viso.make_relative_pose(frame_delta)
        self.assertIsInstance(pose, tf.Transform)

    def test_rearranges_location_coordinates(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPEqual((13.2, -10, 22.4), pose.location)

    def test_changes_rotation_each_axis(self):
        frame_delta = np.array([[1, 0, 0, 10],
                                [0, 1, 0, -22.4],
                                [0, 0, 1, 13.2],
                                [0, 0, 0, 1]])
        # Roll, rotation around z-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 0, 1), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((np.pi / 6, 0, 0), pose.euler)

        # Pitch, rotation around x-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((1, 0, 0), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((0, -np.pi / 6, 0), pose.euler)

        # Yaw, rotation around negative y-axis for libviso2
        frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((0, 1, 0), np.pi / 6, True)
        pose = viso.make_relative_pose(frame_delta)
        self.assertNPClose((0, 0, -np.pi / 6), pose.euler)

    def test_combined(self):
        frame_delta = np.identity(4)
        for _ in range(10):
            loc = np.random.uniform(-1000, 1000, 3)
            rot_axis = np.random.uniform(-1, 1, 3)
            rot_angle = np.random.uniform(-np.pi, np.pi)
            frame_delta[0:3, 3] = -loc[1], -loc[2], loc[0]
            frame_delta[0:3, 0:3] = tf3d.axangles.axangle2mat((-rot_axis[1], -rot_axis[2], rot_axis[0]),
                                                              rot_angle, False)
            pose = viso.make_relative_pose(frame_delta)
            self.assertNPEqual(loc, pose.location)
            self.assertNPClose(tf3d.quaternions.axangle2quat(rot_axis, rot_angle, False), pose.rotation_quat(True))


class TestLibVisOStereo(unittest.TestCase):

    def test_get_properties_is_overridden_by_settings(self):
        settings = {
            'seed': 9989,
            'in_width': 860,
            'in_height': 500,
            'in_fx': 401.3,
            'in_fy': 399.8,
            'in_cx': 450.1,
            'in_cy': 335.2,
            'base': 22.3,
            'matcher_nms_n': 4,
            'matcher_nms_tau': 55,
            'matcher_match_binsize': 103,
            'matcher_match_radius': 5,
            'matcher_match_disp_tolerance': 198,
            'matcher_outlier_disp_tolerance': 22,
            'matcher_outlier_flow_tolerance': 34,
            'matcher_multi_stage': True,
            'matcher_half_resolution': False,
            'bucketing_max_features': 223,
            'bucketing_bucket_width': 85,
            'bucketing_bucket_height': 34
        }
        subject = viso.LibVisOStereoSystem(
            matcher_nms_n=23.2,
            matcher_nms_tau=16,
            matcher_match_binsize=98,
            matcher_match_radius=6,
            matcher_match_disp_tolerance=53,
            matcher_outlier_disp_tolerance=63,
            matcher_outlier_flow_tolerance=98,
            matcher_multi_stage=False,
            matcher_half_resolution=True,
            matcher_refinement=viso.MatcherRefinement.PIXEL,
            bucketing_max_features=332,
            bucketing_bucket_width=87,
            bucketing_bucket_height=43,
            ransac_iters=3004,
            inlier_threshold=4.33,
            reweighting=True
        )
        properties = subject.get_properties(settings=settings)
        for column in settings.keys():
            self.assertEqual(settings[column], properties[column])

    def test_get_properties_reads_from_object_or_is_nan_when_not_in_settings(self):
        subject = viso.LibVisOStereoSystem(
            matcher_nms_n=23,
            matcher_nms_tau=16,
            matcher_match_binsize=98,
            matcher_match_radius=6,
            matcher_match_disp_tolerance=53,
            matcher_outlier_disp_tolerance=63,
            matcher_outlier_flow_tolerance=98,
            matcher_multi_stage=False,
            matcher_half_resolution=True,
            matcher_refinement=viso.MatcherRefinement.PIXEL,
            bucketing_max_features=332,
            bucketing_bucket_width=87,
            bucketing_bucket_height=43,
            ransac_iters=3004,
            inlier_threshold=4.33,
            reweighting=True
        )
        properties = subject.get_properties()
        for key, value in {
            'matcher_nms_n': 23,
            'matcher_nms_tau': 16,
            'matcher_match_binsize': 98,
            'matcher_match_radius': 6,
            'matcher_match_disp_tolerance': 53,
            'matcher_outlier_disp_tolerance': 63,
            'matcher_outlier_flow_tolerance': 98,
            'matcher_multi_stage': False,
            'matcher_half_resolution': True,
            'matcher_refinement': viso.MatcherRefinement.PIXEL,
            'bucketing_max_features': 332,
            'bucketing_bucket_width': 87,
            'bucketing_bucket_height': 43,
            'ransac_iters': 3004,
            'inlier_threshold': 4.33,
            'reweighting': True
        }.items():
            self.assertEqual(value, properties[key])
        for column in [
            'in_width',
            'in_height',
            'in_fx',
            'in_fy',
            'in_cx',
            'in_cy',
            'base',
            'seed'
        ]:
            self.assertTrue(np.isnan(properties[column]))

    def test_get_properties_only_returns_the_requested_properties(self):
        settings = {
            'seed': 42,
            'in_width': 860,
            'in_height': 500,
            'in_fx': 401.3,
            'in_fy': 399.8,
            'in_cx': 450.1,
            'in_cy': 335.2,
            'base': 22.3
        }
        other_properties = {
            'matcher_nms_n': 23,
            'matcher_nms_tau': 16,
            'matcher_match_binsize': 98,
            'matcher_match_radius': 6,
            'matcher_match_disp_tolerance': 53,
            'matcher_outlier_disp_tolerance': 63,
            'matcher_outlier_flow_tolerance': 98,
            'matcher_multi_stage': False,
            'matcher_half_resolution': True,
            'matcher_refinement': viso.MatcherRefinement.PIXEL,
            'bucketing_max_features': 332,
            'bucketing_bucket_width': 87,
            'bucketing_bucket_height': 43,
            'ransac_iters': 3004,
            'inlier_threshold': 4.33,
            'reweighting': True
        }
        subject = viso.LibVisOStereoSystem(
            matcher_nms_n=23.2,
            matcher_nms_tau=16,
            matcher_match_binsize=98,
            matcher_match_radius=6,
            matcher_match_disp_tolerance=53,
            matcher_outlier_disp_tolerance=63,
            matcher_outlier_flow_tolerance=98,
            matcher_multi_stage=False,
            matcher_half_resolution=True,
            matcher_refinement=viso.MatcherRefinement.PIXEL,
            bucketing_max_features=332,
            bucketing_bucket_width=87,
            bucketing_bucket_height=43,
            ransac_iters=3004,
            inlier_threshold=4.33,
            reweighting=True
        )
        columns = list(subject.get_columns())
        np.random.shuffle(columns)
        columns1 = {column for idx, column in enumerate(columns) if idx % 2 == 0}
        columns2 = set(columns) - columns1

        properties = subject.get_properties(columns1, settings=settings)
        for column in columns1:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            elif column in other_properties:
                self.assertEqual(other_properties[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns2:
            self.assertNotIn(column, properties)

        properties = subject.get_properties(columns2, settings=settings)
        for column in columns2:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            elif column in other_properties:
                self.assertEqual(other_properties[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns1:
            self.assertNotIn(column, properties)


class TestLibVisOMono(unittest.TestCase):

    def test_get_properties_is_overridden_by_settings(self):
        settings = {
            'seed': 9989,
            'in_width': 860,
            'in_height': 500,
            'in_fx': 401.3,
            'in_fy': 399.8,
            'in_cx': 450.1,
            'in_cy': 335.2,
            'matcher_nms_n': 4,
            'matcher_nms_tau': 55,
            'matcher_match_binsize': 103,
            'matcher_match_radius': 5,
            'matcher_match_disp_tolerance': 198,
            'matcher_outlier_disp_tolerance': 22,
            'matcher_outlier_flow_tolerance': 34,
            'matcher_multi_stage': True,
            'matcher_half_resolution': False,
            'bucketing_max_features': 223,
            'bucketing_bucket_width': 85,
            'bucketing_bucket_height': 34
        }
        subject = viso.LibVisOMonoSystem(
            matcher_nms_n=23.2,
            matcher_nms_tau=16,
            matcher_match_binsize=98,
            matcher_match_radius=6,
            matcher_match_disp_tolerance=53,
            matcher_outlier_disp_tolerance=63,
            matcher_outlier_flow_tolerance=98,
            matcher_multi_stage=False,
            matcher_half_resolution=True,
            matcher_refinement=viso.MatcherRefinement.PIXEL,
            bucketing_max_features=332,
            bucketing_bucket_width=87,
            bucketing_bucket_height=43,
            height=1.223,
            pitch=np.pi / 32,
            ransac_iters=3004,
            inlier_threshold=4.33,
            motion_threshold=332
        )
        properties = subject.get_properties(settings=settings)
        for column in settings.keys():
            self.assertEqual(settings[column], properties[column])

    def test_get_properties_reads_from_object_or_is_nan_when_not_in_settings(self):
        subject = viso.LibVisOMonoSystem(
            matcher_nms_n=23,
            matcher_nms_tau=16,
            matcher_match_binsize=98,
            matcher_match_radius=6,
            matcher_match_disp_tolerance=53,
            matcher_outlier_disp_tolerance=63,
            matcher_outlier_flow_tolerance=98,
            matcher_multi_stage=False,
            matcher_half_resolution=True,
            matcher_refinement=viso.MatcherRefinement.PIXEL,
            bucketing_max_features=332,
            bucketing_bucket_width=87,
            bucketing_bucket_height=43,
            height=1.223,
            pitch=np.pi / 32,
            ransac_iters=3004,
            inlier_threshold=4.33,
            motion_threshold=332
        )
        properties = subject.get_properties()
        for key, value in {
            'matcher_nms_n': 23,
            'matcher_nms_tau': 16,
            'matcher_match_binsize': 98,
            'matcher_match_radius': 6,
            'matcher_match_disp_tolerance': 53,
            'matcher_outlier_disp_tolerance': 63,
            'matcher_outlier_flow_tolerance': 98,
            'matcher_multi_stage': False,
            'matcher_half_resolution': True,
            'matcher_refinement': viso.MatcherRefinement.PIXEL,
            'bucketing_max_features': 332,
            'bucketing_bucket_width': 87,
            'bucketing_bucket_height': 43,
            'height': 1.223,
            'pitch': np.pi / 32,
            'ransac_iters': 3004,
            'inlier_threshold': 4.33,
            'motion_threshold': 332
        }.items():
            self.assertEqual(value, properties[key])
        for column in [
            'in_width',
            'in_height',
            'in_fx',
            'in_fy',
            'in_cx',
            'in_cy',
            'seed'
        ]:
            self.assertTrue(np.isnan(properties[column]))

    def test_get_properties_only_returns_the_requested_properties(self):
        settings = {
            'seed': 42,
            'in_width': 860,
            'in_height': 500,
            'in_fx': 401.3,
            'in_fy': 399.8,
            'in_cx': 450.1,
            'in_cy': 335.2,
            'base': 22.3
        }
        other_properties = {
            'matcher_nms_n': 23,
            'matcher_nms_tau': 16,
            'matcher_match_binsize': 98,
            'matcher_match_radius': 6,
            'matcher_match_disp_tolerance': 53,
            'matcher_outlier_disp_tolerance': 63,
            'matcher_outlier_flow_tolerance': 98,
            'matcher_multi_stage': False,
            'matcher_half_resolution': True,
            'matcher_refinement': viso.MatcherRefinement.PIXEL,
            'bucketing_max_features': 332,
            'bucketing_bucket_width': 87,
            'bucketing_bucket_height': 43,
            'height': 1.223,
            'pitch': np.pi / 32,
            'ransac_iters': 3004,
            'inlier_threshold': 4.33,
            'motion_threshold': 332
        }
        subject = viso.LibVisOMonoSystem(
            matcher_nms_n=23.2,
            matcher_nms_tau=16,
            matcher_match_binsize=98,
            matcher_match_radius=6,
            matcher_match_disp_tolerance=53,
            matcher_outlier_disp_tolerance=63,
            matcher_outlier_flow_tolerance=98,
            matcher_multi_stage=False,
            matcher_half_resolution=True,
            matcher_refinement=viso.MatcherRefinement.PIXEL,
            bucketing_max_features=332,
            bucketing_bucket_width=87,
            bucketing_bucket_height=43,
            height=1.223,
            pitch=np.pi / 32,
            ransac_iters=3004,
            inlier_threshold=4.33,
            motion_threshold=332
        )
        columns = list(subject.get_columns())
        np.random.shuffle(columns)
        columns1 = {column for idx, column in enumerate(columns) if idx % 2 == 0}
        columns2 = set(columns) - columns1

        properties = subject.get_properties(columns1, settings=settings)
        for column in columns1:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            elif column in other_properties:
                self.assertEqual(other_properties[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns2:
            self.assertNotIn(column, properties)

        properties = subject.get_properties(columns2, settings=settings)
        for column in columns2:
            self.assertIn(column, properties)
            if column in settings:
                self.assertEqual(settings[column], properties[column])
            elif column in other_properties:
                self.assertEqual(other_properties[column], properties[column])
            else:
                self.assertTrue(np.isnan(properties[column]))
        for column in columns1:
            self.assertNotIn(column, properties)
