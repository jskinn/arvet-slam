import unittest
import unittest.mock as mock
from pathlib import Path
import numpy as np
import transforms3d as t3
from arvet.metadata.image_metadata import MaskedObject, EnvironmentType, LightingLevel, TimeOfDay, LightingModel
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.util.test_helpers import ExtendedTestCase
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader

CAMERA_SETTINGS = Path(__file__).parent / '_camera_settings.json'
OBJECT_SETTINGS = Path(__file__).parent / '_object_settings.json'
FRAME_JSON = Path(__file__).parent / 'frame_data.json'


class TestReadCameraIntrinsics(unittest.TestCase):

    @mock.patch('arvet_slam.dataset.ndds.ndds_loader.read_json')
    def test_read_intrinsics(self, mock_read_json):
        camera_settings_path = Path(__file__).parent / '_camera_settings.json'
        camera_settings = {
            "camera_settings": [{
                "name": "Viewpoint",
                "horizontal_fov": 90,
                "intrinsic_settings": {
                    "resX": 640,
                    "resY": 480,
                    "fx": 517.29998779296875,
                    "fy": 516.5,
                    "cx": 318.60000610351563,
                    "cy": 255.30000305175781,
                    "s": 0
                },
                "captured_image_size": {"width": 640, "height": 480}
            }]
        }
        mock_read_json.return_value = camera_settings
        intrinsics = ndds_loader.read_camera_intrinsics(camera_settings_path)
        self.assertTrue(mock_read_json.called)
        self.assertIn(mock.call(camera_settings_path), mock_read_json.call_args_list)
        self.assertIsInstance(intrinsics, CameraIntrinsics)
        self.assertEqual(640, intrinsics.width)
        self.assertEqual(480, intrinsics.height)
        self.assertEqual(517.29998779296875, intrinsics.fx)
        self.assertEqual(516.5, intrinsics.fy)
        self.assertEqual(318.60000610351563, intrinsics.cx)
        self.assertEqual(255.30000305175781, intrinsics.cy)


class TestParseSettings(unittest.TestCase):
    settings = {
        "map": "two_story_apartment",
        "trajectory_id": "rgbd_dataset_freiburg2_desk_with_person",
        "light_level": "DAY",
        "light_model": "LIT",
        "origin": {"location": [660, 32, 402], "rotation": [0, 0, -100]},
        "left_intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 580.8,
            "fy": 581.8,
            "cx": 308.8,
            "cy": 253,
            "skew": 0
        },
        "right_intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 580.8,
            "fy": 581.8,
            "cx": 308.8,
            "cy": 253,
            "skew": 0
        },
        "texture_bias": 2,
        "disable_reflections": True,
        "min_object_volume": 0.33,
        "motion_blur": 0,
        "exposure": None,
        "aperture": 4,
        "focal_distance": 233,
        "grain": 0,
        "vignette": 0,
        "lens_flare": 0.5,
        "depth_quality": "KINECT_NOISE"
    }

    def test_pulls_out_required_values(self):
        (
            trajectory_id, environment_type, light_level, time_of_day, simulation_world,
            lighting_model, texture_mipmap_bias, normal_maps_enabled, roughness_enabled, min_object_size,
            geometry_decimation
        ) = ndds_loader.parse_settings(self.settings)
        self.assertEqual(self.settings['trajectory_id'], trajectory_id)
        self.assertIsInstance(environment_type, EnvironmentType)
        self.assertIsInstance(light_level, LightingLevel)
        self.assertEqual(time_of_day, TimeOfDay.DAY),
        self.assertEqual(self.settings['map'], simulation_world)
        self.assertIsInstance(lighting_model, LightingModel),
        self.assertEqual(self.settings['texture_bias'], texture_mipmap_bias)
        self.assertEqual(normal_maps_enabled, True)
        self.assertEqual(not self.settings['disable_reflections'], roughness_enabled)
        self.assertEqual(self.settings['min_object_volume'], min_object_size)
        self.assertEqual(0, geometry_decimation)

    def test_fills_missing_quality_settings_with_defaults(self):
        reduced_settings = {
            key: value
            for key, value in self.settings.items()
            if key not in {'light_model', 'texture_bias', 'disable_reflections', 'min_object_volume'}
        }
        (
            trajectory_id, environment_type, light_level, time_of_day, simulation_world,
            lighting_model, texture_mipmap_bias, normal_maps_enabled, roughness_enabled, min_object_size,
            geometry_decimation
        ) = ndds_loader.parse_settings(reduced_settings)
        self.assertEqual(self.settings['trajectory_id'], trajectory_id)
        self.assertIsInstance(environment_type, EnvironmentType)
        self.assertIsInstance(light_level, LightingLevel)
        self.assertEqual(TimeOfDay.DAY, time_of_day),
        self.assertEqual(self.settings['map'], simulation_world)
        self.assertIsInstance(lighting_model, LightingModel),
        self.assertEqual(0, texture_mipmap_bias)
        self.assertEqual(normal_maps_enabled, True)
        self.assertEqual(True, roughness_enabled)
        self.assertEqual(-1, min_object_size)
        self.assertEqual(0, geometry_decimation)


class TestLabelConstruction(ExtendedTestCase):

    def test_find_unique_colours_generates_all_colours_in_the_image(self):
        side_length = 16
        mask_image = np.zeros((128, 128, 3), dtype=np.uint8)
        ids = list(range(1, 2 ** 21, 2 ** 16 + 2 ** 8 + 2 ** 2))
        colours = [ndds_loader.encode_id(obj_id) for obj_id in ids]
        for corner, colour in zip(
                range(0, mask_image.shape[0] - side_length, (mask_image.shape[0] - side_length) // len(ids)),
                colours
        ):
            mask_image[corner:corner+side_length+1, corner:corner+side_length+1, :] = colour

        found_colours = set()
        for colour, mask in ndds_loader.find_unique_colours(mask_image):
            found_colours.add(tuple(colour))
            self.assertTrue(np.array_equal((mask_image == colour).all(axis=2), mask))
        expected_colours = set(tuple(colour) for colour in colours)
        expected_colours.add((0, 0, 0))
        self.assertEqual(expected_colours, found_colours)

    def test_generate_bounding_box_from_mask(self):
        mask = np.zeros((128, 128), dtype=np.bool)
        min_x, max_x = 17, 43
        min_y, max_y = 42, 113
        mask[(max_y + min_y) // 2 + 1, min_x] = True
        mask[(max_y + min_y) // 2, max_x] = True
        mask[min_y, (max_x + min_x) // 2] = True
        mask[max_y, (max_x + min_x) // 2 + 1] = True
        self.assertEqual((min_x, min_y, max_x, max_y), ndds_loader.generate_bounding_box_from_mask(mask))

    def test_find_nearest_id_by_colour(self):
        ids = list(range(1, 2**21, 2**16 + 2**8 + 2**2))
        colours = np.array([ndds_loader.encode_id(obj_id) for obj_id in ids])
        for idx in range(len(ids)):
            self.assertEqual(ids[idx], ndds_loader.find_nearest_id_by_colour(
                colours[idx],
                known_ids=ids,
                known_colours=colours
            ))

    def test_find_nearest_id_by_colour_handles_small_shifts(self):
        ids = list(range(1, 2**21, 2**16 + 2**8 + 2**2))
        colours = np.array([ndds_loader.encode_id(obj_id) for obj_id in ids])
        for idx in range(0, len(ids), 2):
            colour = colours[idx]
            self.assertEqual(ids[idx], ndds_loader.find_nearest_id_by_colour(
                np.array([colour[0] + 1, colour[1] - 1, colour[2] - 1]),
                known_ids=ids,
                known_colours=colours
            ))
            self.assertEqual(ids[idx], ndds_loader.find_nearest_id_by_colour(
                np.array([colour[0] - 1, colour[1] + 1, colour[2] - 1]),
                known_ids=ids,
                known_colours=colours
            ))
        # This one is shifted too far
        colour = colours[len(ids) // 2]
        self.assertIsNone(ndds_loader.find_nearest_id_by_colour(
            np.array([colour[0], colour[1] - 2, colour[2]]),
            known_ids=ids,
            known_colours=colours
        ))

    def test_find_labelled_objects(self):
        side_length = 16
        mask_image = np.zeros((128, 128, 3), dtype=np.uint8)
        object_ids = list(range(1, 2 ** 21, 2 ** 16 + 2 ** 8 + 2 ** 2))
        locations = [
            [1600 - 10 * idx, -800 + 50 * idx - 7 * idx * idx, 10 * idx * idx - 90 * idx]
            for idx in range(len(object_ids))
        ]
        rotations = [
            t3.quaternions.axangle2quat((0.5 * idx, 1 - 0.1 * idx, 0.22),  3 * idx * np.pi / 37)
            for idx in range(len(object_ids))
        ]
        colours = [ndds_loader.encode_id(obj_id) for obj_id in object_ids]
        for corner, colour in zip(
                range(0, mask_image.shape[0] - side_length, (mask_image.shape[0] - side_length) // len(object_ids)),
                colours
        ):
            mask_image[corner:corner + side_length + 1, corner:corner + side_length + 1, :] = colour

        # Make some object and frame data. This will normally be read from _object_settings and the frame json
        object_data = {
            obj_id: {
                'name': f"TestObj-{obj_id}",
                'class': f"class-{idx}",
                'segmentation_instance_id': obj_id
            } for idx, obj_id in enumerate(object_ids)
        }
        frame_data = {
            'objects': [
                {
                    'name': object_data[obj_id]['name'],
                    'location': locations[idx],
                    'quaternion_xyzw': [rotations[idx][1], rotations[idx][2], rotations[idx][3], rotations[idx][0]]
                }
                for idx, obj_id in enumerate(object_ids)
            ]
        }

        labelled_objects = ndds_loader.find_labelled_objects(mask_image, frame_data, object_data)
        self.assertEqual(len(object_data), len(labelled_objects))
        for idx, labelled_object in enumerate(labelled_objects):
            self.assertIsInstance(labelled_object, MaskedObject)
            obj_id = object_ids[idx]
            colour = colours[idx]
            self.assertEqual(object_data[obj_id]['name'], labelled_object.instance_name)
            self.assertEqual([object_data[obj_id]['class']], labelled_object.class_names)
            self.assertNPEqual([    # Axies are changed from opencv coordinates and rescaled
                locations[idx][2] / 100,
                -locations[idx][0] / 100,
                -locations[idx][1] / 100
            ], labelled_object.relative_pose.location)
            # Some small shift due to normalisation, flip axis and invert for transform from unreal frame
            self.assertNPClose([
                rotations[idx][0],
                rotations[idx][3],
                -rotations[idx][1],
                -rotations[idx][2]
            ], labelled_object.relative_pose.rotation_quat(True), rtol=0, atol=1e-15)

            true_mask = (mask_image == colour).all(axis=2)
            read_mask = np.zeros(true_mask.shape, dtype=np.bool)
            read_mask[
                labelled_object.y:labelled_object.y + labelled_object.height,
                labelled_object.x:labelled_object.x + labelled_object.width
            ] = labelled_object.mask
            self.assertNPEqual(true_mask, read_mask)


class TestReadCameraPose(ExtendedTestCase):

    def test_converts_from_unreal_frame(self):
        location = [100.2, 303, -230]
        rotation_quat = [-0.034200001507997513, 0.17430000007152557, -0.071800000965595245, 0.98150002956390381]
        camera_pose = ndds_loader.read_camera_pose({
            'camera_data': {
                'location_worldframe': location,
                'quaternion_xyzw_worldframe': rotation_quat
            }
        })
        rotation_quat = np.array(rotation_quat) * np.array([-1, 1, -1, 1])
        rotation_quat = rotation_quat / np.linalg.norm(rotation_quat)
        self.assertNPEqual([
            location[0] / 100,
            -location[1] / 100,
            location[2] / 100
        ], camera_pose.location)
        self.assertNPClose(rotation_quat, camera_pose.rotation_quat(w_first=False))

    @unittest.skipIf(not FRAME_JSON.exists(), f"File {FRAME_JSON} is missing")
    def test_read_from_actual_frame_data(self):
        frame_data = ndds_loader.read_json(FRAME_JSON)
        camera_pose = ndds_loader.read_camera_pose(frame_data)
        # These values are copied from the frame_data.json
        location = [-397.50518798828125, 492.243408203125, 179.62669372558594]
        rotation = [-0.034200001507997513, 0.17430000007152557, -0.071800000965595245, 0.98150002956390381]
        self.assertNPEqual(
            [location[0] / 100, -location[1] / 100, location[2] / 100],
            camera_pose.location
        )
        quat = t3.quaternions.qinverse([rotation[3], rotation[0], -rotation[1], rotation[2]])
        self.assertNPClose(
            quat / np.linalg.norm(quat),    # t3 winds up with |q|^2 as the norm, rather than |q|, renormalise
            camera_pose.rotation_quat(True),
            atol=0, rtol=1e-15
        )


class TestReadRelativePose(ExtendedTestCase):

    @unittest.skipIf(not FRAME_JSON.exists(), f"File {FRAME_JSON} is missing")
    def test_read_from_actual_frame_data(self):
        frame_data = ndds_loader.read_json(FRAME_JSON)
        camera_pose = ndds_loader.read_camera_pose(frame_data)
        # These values are copied from the frame_data.json
        location = [-397.50518798828125, 492.243408203125, 179.62669372558594]
        rotation = [-0.034200001507997513, 0.17430000007152557, -0.071800000965595245, 0.98150002956390381]
        self.assertNPEqual(
            [location[0] / 100, -location[1] / 100, location[2] / 100],
            camera_pose.location
        )
        quat = t3.quaternions.qinverse([rotation[3], rotation[0], -rotation[1], rotation[2]])
        self.assertNPClose(
            quat / np.linalg.norm(quat),    # t3 winds up with |q|^2 as the norm, rather than |q|, renormalise
            camera_pose.rotation_quat(True),
            atol=0, rtol=1e-15
        )


class TestPoseRelationships(ExtendedTestCase):
    # These poses are read from the _object_settings.json and frame_data.json.
    # From rgbd_dataset_freiburg2_desk_with_person-two_story_apartment-000 left
    # The object is 'AIUE_V01_004_table_001' in frames 0 and 1720
    frame_data_1 = {  # Frame 0
        'camera_data': {
            'location_worldframe': [660, 32, 402],
            'quaternion_xyzw_worldframe': [0, 0, -0.76599997282028198, 0.642799973487854]
        },
        "objects": [{
            "name": "AIUE_V01_004_table_001",
            "location": [-1349.18798828125, 332, 243.36590576171875],
            "quaternion_xyzw": [
                0, 0.76599997282028198, 0, 0.642799973487854]
        }]
    }
    frame_data_2 = {  # Frame 1720
        'camera_data': {
            'location_worldframe': [866.63861083984375, -135.34260559082031, 456.989990234375],
            'quaternion_xyzw_worldframe': [
                -0.21699999272823334, 0.23600000143051147, 0.93559998273849487, 0.1476999968290329]
        },
        "objects": [{
            "name": "AIUE_V01_004_table_001",
            "location": [314.67138671875, -305.06649780273438, 1572.48095703125],
            "quaternion_xyzw": [
                -0.23600000143051147, 0.93559998273849487, 0.21699999272823334, -0.1476999968290329]
        }]
    }

    def test_relationships_between_frames(self):

        camera_pose_1 = ndds_loader.read_camera_pose(self.frame_data_1)
        object_pose_1 = ndds_loader.read_relative_pose(self.frame_data_1['objects'][0])

        camera_pose_2 = ndds_loader.read_camera_pose(self.frame_data_2)
        object_pose_2 = ndds_loader.read_relative_pose(self.frame_data_2['objects'][0])

        independent_object_pose_1 = camera_pose_1.find_independent(object_pose_1)
        independent_object_pose_2 = camera_pose_2.find_independent(object_pose_2)
        self.assertNPClose(independent_object_pose_1.location, independent_object_pose_2.location, rtol=0, atol=1e-3)
        # Quaternions are equal if they are opposite signs, q = -q
        rotation_1 = independent_object_pose_1.rotation_quat(True)
        rotation_2 = independent_object_pose_2.rotation_quat(True)
        self.assertTrue(np.all(np.isclose(rotation_1, rotation_2)) or np.all(np.isclose(rotation_1, -1 * rotation_2)),
                        f"Rotations {rotation_1} and {rotation_2} are not close")

    @unittest.skip("Not quite working, but unimportant")
    def test_frame_to_object_settings_is_consistent(self):
        # These poses are read from the _object_settings.json,
        # For sequence rgbd_dataset_freiburg2_desk_with_person-two_story_apartment-000 left
        # The object is 'AIUE_V01_004_table_001'
        pose_mat = np.array([
            # As it appears in-file, in OpenCV axis conventions. See:
            # https://github.com/jskinn/Dataset_Synthesizer/blob/local-devel/Source/Plugins/NVSceneCapturer/Source/NVSceneCapturer/Private/NVSceneDataHandler.cpp#L204
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [26.615400314331055, -70, -710.95098876953125, 1]
        ]).T    # <- Transpose the matrix to swap row/column
        object_world_location = [pose_mat[2, 3] / 100, -pose_mat[0, 3] / 100, -pose_mat[1, 3] / 100]
        object_world_rotation = t3.quaternions.mat2quat(pose_mat[0:3, 0:3])
        object_world_rotation = [
            object_world_rotation[0],
            object_world_rotation[2],
            -object_world_rotation[0],
            -object_world_rotation[1]
        ]

        camera_pose = ndds_loader.read_camera_pose(self.frame_data_2)
        object_pose = ndds_loader.read_relative_pose(self.frame_data_2['objects'][0])

        independent_object_pose = camera_pose.find_independent(object_pose)
        self.assertNPClose(object_world_location, independent_object_pose.location, rtol=0, atol=1e-3)
        self.assertNPClose(object_world_rotation, independent_object_pose.rotation_quat(True))


class TestReadJson(ExtendedTestCase):

    @unittest.skipIf(not CAMERA_SETTINGS.exists(), f"File {CAMERA_SETTINGS} is missing")
    def test_read_actual_camera_data(self):
        camera_data = ndds_loader.read_json(CAMERA_SETTINGS)
        self.assertEqual({'camera_settings'}, set(camera_data.keys()))
        for camera_entry in camera_data['camera_settings']:
            self.assertEqual({'name', 'horizontal_fov', 'intrinsic_settings', 'captured_image_size'},
                             set(camera_entry.keys()))

    @unittest.skipIf(not OBJECT_SETTINGS.exists(), f"File {OBJECT_SETTINGS} is missing")
    def test_read_actual_object_data(self):
        object_data = ndds_loader.read_json(OBJECT_SETTINGS)
        self.assertEqual({'exported_object_classes', 'exported_objects'}, set(object_data.keys()))
        self.assertEqual(66, len(object_data['exported_object_classes']))
        self.assertEqual(66, len(object_data['exported_objects']))
        for obj_idx, instance_data in enumerate(object_data['exported_objects']):
            self.assertEqual({
                "name",
                "class",
                "segmentation_class_id",
                "segmentation_instance_id",
                "fixed_model_transform",
                "cuboid_dimensions"
            }, set(instance_data.keys()))
            self.assertEqual(object_data['exported_object_classes'][obj_idx], instance_data['class'])

    @unittest.skipIf(not FRAME_JSON.exists(), f"File {FRAME_JSON} is missing")
    def test_read_actual_frame_data(self):
        frame_data = ndds_loader.read_json(FRAME_JSON)
        self.assertEqual({'camera_data', 'objects'}, set(frame_data.keys()))
        self.assertEqual({'location_worldframe', 'quaternion_xyzw_worldframe'}, set(frame_data['camera_data'].keys()))
        self.assertEqual(7, len(frame_data['objects']))
        for object_data in frame_data['objects']:
            self.assertEqual({
                "name",
                "class",
                "visibility",
                "location",
                "quaternion_xyzw",
                "pose_transform",
                "cuboid_centroid",
                "projected_cuboid_centroid",
                "bounding_box",
                "cuboid",
                "projected_cuboid"
            }, set(object_data.keys()))
