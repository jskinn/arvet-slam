import unittest
from pathlib import Path
import numpy as np
import transforms3d as t3
from arvet.util.transform import Transform
from arvet.util.test_helpers import ExtendedTestCase
from arvet.metadata.image_metadata import MaskedObject
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader


CAMERA_SETTINGS = Path(__file__).parent / '_camera_settings.json'
OBJECT_SETTINGS = Path(__file__).parent / '_object_settings.json'
FRAME_JSON = Path(__file__).parent / 'frame_data.json'


class TestLabelConstruction(unittest.TestCase):

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
        ids = list(range(1, 2 ** 21, 2 ** 16 + 2 ** 8 + 2 ** 2))
        colours = [ndds_loader.encode_id(obj_id) for obj_id in ids]
        for corner, colour in zip(
                range(0, mask_image.shape[0] - side_length, (mask_image.shape[0] - side_length) // len(ids)),
                colours
        ):
            mask_image[corner:corner + side_length + 1, corner:corner + side_length + 1, :] = colour
        # Make some object and frame data. This will normally be read from _object_settings and the frame json
        object_data = {
            obj_id: {
                'name': f"TestObj-{obj_id}",
                'class': f"class-{idx}",
                'segmentation_instance_id': obj_id
            } for idx, obj_id in enumerate(ids)
        }
        object_data[0] = {
            'name': 'background',
            'class': 'bg',
            'segmentation_instance_id': 0
        }
        frame_data = {
            'objects': [
                {
                    'name': obj_data['name'],
                    'location': [1600 - 10 * idx],
                    'quaternion_xyzw': [
                        0.15469999611377716,
                        0.072200000286102295,
                        0.015900000929832458,
                        0.98519998788833618
                    ]
                }
                for idx, obj_data in enumerate(object_data)
            ]
        }

        labelled_objects = ndds_loader.find_labelled_objects(mask_image, frame_data, object_data)
        self.assertEqual(len(object_data), len(labelled_objects))
        for idx, labelled_object in enumerate(labelled_objects):
            obj_id = ids[idx]
            colour = colours[idx]
            true_mask = (mask_image == colour).all(axis=2)
            # TODO: Finish this test
            self.assertIsInstance(MaskedObject, labelled_object)


class TestReadCameraPose(ExtendedTestCase):

    @unittest.skipIf(not FRAME_JSON.exists(), f"File {FRAME_JSON} is missing")
    def test_frame_data(self):
        frame_data = ndds_loader.read_json(FRAME_JSON)
        camera_pose = ndds_loader.read_camera_pose(frame_data)
        self.assertNPEqual((-3.9750518798828125, -4.92243408203125, 1.7962669372558594), camera_pose.location)


class TestMakeCameraPose(ExtendedTestCase):
    # TODO: These tests are not working, I think the exporter is doing something to the axes
    # See
    # https://github.com/jskinn/Dataset_Synthesizer/blob/local-devel/Source/Plugins/NVSceneCapturer/Source/NVSceneCapturer/Private/NVSceneFeatureExtractor_DataExport.cpp#L143

    def test_make_camera_pose_makes_transform(self):
        tx, ty, tz = -751.05047607421875, 247.639404296875, 43.617000579833984
        qx, qy, qz, qw = 0.17219999432563782, 0.073399998247623444, -0.032900001853704453, 0.98180001974105835
        pose = ndds_loader.make_camera_pose(tx, ty, tz, qw, qx, qy, qz)
        self.assertIsInstance(pose, Transform)
        self.assertNPEqual((tx / 100, -ty / 100, tz / 100), pose.location)

        expected_rot = t3.quaternions.qinverse((qw, qx, qy, qz))
        expected_rot = np.array([expected_rot[0], expected_rot[1], -expected_rot[2], expected_rot[3]])
        expected_rot = expected_rot / np.linalg.norm(expected_rot)
        rotation = pose.rotation_quat(True)
        self.assertNPClose(expected_rot, rotation, rtol=0, atol=1e-8)

    def test_make_camera_pose_preserves_relationships_between_frames(self):
        # These poses are read from the _object_settings.json and frame_data.json.
        # The object is 'AIUE_V01_Sofa' in frames 663 and 2294
        camera_pose_1 = ndds_loader.make_camera_pose(
            tx=-410.0596923828125, ty=498.23709106445313, tz=196.84210205078125,
            qx=-0.019700000062584877, qy=0.080499999225139618, qz=-0.011400000192224979, qw=0.99650001525878906
        )
        object_pose_1 = ndds_loader.make_camera_pose(
            tx=-1027.91796875, ty=249.625, tz=-54.341701507568359,
            qx=0.080499999225139618, qy=0.011400000192224979, qz=-0.019700000062584877, qw=0.99650001525878906
        )

        camera_pose_2 = ndds_loader.make_camera_pose(
            tx=-397.59298706054688, ty=492.22360229492188, tz=180.04620361328125,
            qx=-0.032900001853704453, qy=0.17219999432563782, qz=-0.073399998247623444, qw=0.98180001974105835
        )
        object_pose_2 = ndds_loader.make_camera_pose(
            tx=-1016.236572265625, ty=251.15089416503906, tz=101.87190246582031,
            qx=0.17219999432563782, qy=0.073399998247623444, qz=-0.032900001853704453, qw=0.98180001974105835
        )

        independent_object_pose_1 = object_pose_1.find_independent(camera_pose_1)
        independent_object_pose_2 = object_pose_2.find_independent(camera_pose_2)
        self.assertNPClose(independent_object_pose_1.location, independent_object_pose_2.location)
        self.assertNPClose(independent_object_pose_1.rotation_quat(True),
                           independent_object_pose_2.rotation_quat(True))

    def test_make_camera_pose_preserves_relationships(self):
        # These poses are read from the _object_settings.json and frame_data.json.
        # The object is 'AIUE_V01_Sofa'
        rot = t3.quaternions.mat2quat(np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ]))
        object_world_pose = ndds_loader.make_camera_pose(
            tx=-536.55902099609375, ty=-1.6353000402450562, tz=-523.95599365234375,
            qw=rot[0], qx=rot[1], qy=rot[2], qz=rot[3]
        )

        camera_pose = ndds_loader.make_camera_pose(
            tx=-397.50518798828125, ty=492.243408203125, tz=179.62669372558594,
            qx=-0.034200001507997513, qy=0.17430000007152557, qz=-0.071800000965595245, qw=0.98150002956390381
        )
        object_pose = ndds_loader.make_camera_pose(
            tx=-1015.677001953125, ty=254.06309509277344, tz=99.785499572753906,
            qx=0.17430000007152557, qy=0.071800000965595245, qz=-0.034200001507997513, qw=0.98150002956390381
        )

        independent_object_pose = camera_pose.find_independent(object_pose)
        camera_independent_of_object = object_pose.find_independent(camera_pose)
        object_relative_to_camera = camera_pose.find_relative(object_pose)
        camera_relative_to_object = object_pose.find_relative(camera_pose)

        camera_relative_to_world_object = object_world_pose.find_relative(camera_pose)

        self.assertNPClose(object_world_pose.location, independent_object_pose.location)
        self.assertNPClose(object_world_pose.rotation_quat(True), independent_object_pose.rotation_quat(True))
