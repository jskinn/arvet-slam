import unittest
from pathlib import Path
import numpy as np
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader


CAMERA_SETTINGS = Path(__file__).parent / '_camera_settings.json'
OBJECT_SETTINGS = Path(__file__).parent / '_object_settings.json'
FRAME_JSON = Path(__file__).parent / 'frame_data.json'


class JsonTestCase(unittest.TestCase):

    def assertJsonEqualWithNaN(self, expected, actual, allow_nan_equality=True, _recursive_key: str = ''):
        """
        Assert that two lumps of json are equal, recursively.
        Expects only dicts, lists, and integral types.
        NaN values _are_ considered equal here, because we care more about the structure than values.
        so {'foo': NaN} should equal itself, as should {'bar: [NaN, NaN, NaN]}
        :param expected:
        :param actual:
        :param allow_nan_equality: Consider NaN values as equal. Default true.
        :param _recursive_key: Pass indexes between the recursions so they can appear in messages. Don't set,
        :return:
        """
        if isinstance(expected, dict) and isinstance(actual, dict):
            self.assertEqual(set(expected.keys()), set(actual.keys()),
                             f"Dicts at {_recursive_key} did not have the same keys")
            for key in expected.keys():
                # Recurse to the nested structure
                self.assertJsonEqualWithNaN(
                    expected=expected[key],
                    actual=actual[key],
                    allow_nan_equality=bool(allow_nan_equality),
                    _recursive_key=_recursive_key + '.' + key
                )
        elif isinstance(expected, list) and isinstance(actual, list):
            self.assertEqual(len(expected), len(actual),
                             f"Lists at {_recursive_key} do not have the same length")
            for idx, (expected_elem, actual_elem) in enumerate(zip(expected, actual)):
                # Recurse to the list elements
                self.assertJsonEqualWithNaN(
                    expected=expected_elem,
                    actual=actual_elem,
                    allow_nan_equality=bool(allow_nan_equality),
                    _recursive_key=_recursive_key + '.' + str(idx)
                )
        elif allow_nan_equality and isinstance(expected, float) and np.isnan(expected):
            self.assertTrue(np.isnan(actual), f"Expected {_recursive_key} to be NaN, but was {actual}")
        else:
            self.assertEqual(expected, actual, f"Expected {_recursive_key} to be '{expected}', but it was '{actual}'")


class TestReadJson(JsonTestCase):

    @unittest.skipIf(not FRAME_JSON.exists(), f"File {FRAME_JSON} is missing")
    def test_read_frame_json(self):
        data = ndds_loader.read_json(FRAME_JSON)
        self.assertJsonEqualWithNaN({
            "camera_data": {
                "location_worldframe": [-397.50518798828125, 492.243408203125, 179.62669372558594],
                "quaternion_xyzw_worldframe": [-0.034200001507997513, 0.17430000007152557,
                                               -0.071800000965595245, 0.98150002956390381]
            },
            "objects": [
                {
                    "name": "AIUE_V01_table3",
                    "class": "coffee table",
                    "visibility": 0,
                    "location": [-750.34210205078125, 249.82000732421875, 42.260601043701172],
                    "quaternion_xyzw": [0.17430000007152557, 0.071800000965595245,
                                        -0.034200001507997513, 0.98150002956390381],
                    "pose_transform": [
                        [0.12890000641345978, -0.34709998965263367, 0.92890000343322754, 0],
                        [0.98739999532699585, -0.042199999094009399, -0.15279999375343323, 0],
                        [-0.092200003564357758, -0.93690001964569092, -0.33730000257492065, 0],
                        [-750.34210205078125, 249.82000732421875, 42.260601043701172, 1]
                    ],
                    "cuboid_centroid": [-751.20819091796875, 241.02330017089844, 39.093799591064453],
                    "projected_cuboid_centroid": [-np.nan, -np.nan],
                    "bounding_box":
                        {
                            "top_left": [-np.nan, -np.nan],
                            "bottom_right": [-np.nan, -np.nan]
                        },
                    "cuboid": [
                        [-705.6259765625, 196.08619689941406, 123.78919982910156],
                        [-772.6859130859375, 198.952392578125, 134.16830444335938],
                        [-770.964599609375, 216.43519592285156, 140.46200561523438],
                        [-703.90460205078125, 213.56900024414063, 130.08299255371094],
                        [-731.451904296875, 265.61138916015625, -62.274200439453125],
                        [-798.51177978515625, 268.47760009765625, -51.895099639892578],
                        [-796.79052734375, 285.96038818359375, -45.601398468017578],
                        [-729.7305908203125, 283.09420776367188, -55.980400085449219]
                    ],
                    "projected_cuboid": [
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan]
                    ]
                },
                {
                    "name": "AIUE_V01_Sofa",
                    "class": "couch",
                    "visibility": 0,
                    "location": [-1015.677001953125, 254.06309509277344, 99.785499572753906],
                    "quaternion_xyzw": [0.17430000007152557, 0.071800000965595245, -0.034200001507997513,
                                        0.98150002956390381],
                    "pose_transform": [
                        [0.12890000641345978, -0.34709998965263367, 0.92890000343322754, 0],
                        [0.98739999532699585, -0.042199999094009399, -0.15279999375343323, 0],
                        [-0.092200003564357758, -0.93690001964569092, -0.33730000257492065, 0],
                        [-1015.677001953125, 254.06309509277344, 99.785499572753906, 1]
                    ],
                    "cuboid_centroid": [-1019.3566284179688, 223.6376953125, 88.541297912597656],
                    "projected_cuboid_centroid": [-np.nan, -np.nan],
                    "bounding_box":
                        {
                            "top_left": [-np.nan, -np.nan],
                            "bottom_right": [-np.nan, -np.nan]
                        },
                    "cuboid": [
                        [-893.11102294921875, 151.43299865722656, 152.67999267578125],
                        [-1125.9185791015625, 161.38330078125, 188.71220397949219],
                        [-1119.5777587890625, 225.78300476074219, 211.89599609375],
                        [-886.7703857421875, 215.83270263671875, 175.86380004882813],
                        [-919.1353759765625, 221.49240112304688, -34.813301086425781],
                        [-1151.9429931640625, 231.44279479980469, 1.2188999652862549],
                        [-1145.6021728515625, 295.84249877929688, 24.402599334716797],
                        [-912.7947998046875, 285.89208984375, -11.629500389099121]
                    ],
                    "projected_cuboid": [
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan]
                    ]
                },
                {
                    "name": "AIUE_V01_Sofa2",
                    "class": "couch",
                    "visibility": 0,
                    "location": [-548.47662353515625, 227.8988037109375, 50.108898162841797],
                    "quaternion_xyzw": [0.17430000007152557, 0.071800000965595245, -0.034200001507997513,
                                        0.98150002956390381],
                    "pose_transform": [
                        [0.12890000641345978, -0.34709998965263367, 0.92890000343322754, 0],
                        [0.98739999532699585, -0.042199999094009399, -0.15279999375343323, 0],
                        [-0.092200003564357758, -0.93690001964569092, -0.33730000257492065, 0],
                        [-548.47662353515625, 227.89889526367188, 50.108898162841797, 1]
                    ],
                    "cuboid_centroid": [-552.08880615234375, 190.51710510253906, 36.460700988769531],
                    "projected_cuboid_centroid": [-np.nan, -np.nan],
                    "bounding_box":
                        {
                            "top_left": [-np.nan, -np.nan],
                            "bottom_right": [-np.nan, -np.nan]
                        },
                    "cuboid": [
                        [-472.9569091796875, 104.77359771728516, 134.80589294433594],
                        [-604.71112060546875, 110.40489959716797, 155.19790649414063],
                        [-597.35699462890625, 185.09719848632813, 182.08689880371094],
                        [-465.60281372070313, 179.46589660644531, 161.69500732421875],
                        [-506.82049560546875, 195.93699645996094, -109.16560363769531],
                        [-638.5748291015625, 201.56820678710938, -88.773597717285156],
                        [-631.220703125, 276.260498046875, -61.884601593017578],
                        [-499.46649169921875, 270.62930297851563, -82.276496887207031]
                    ],
                    "projected_cuboid": [
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan]
                    ]
                },
                {
                    "name": "AIUE_V01_Sofa3",
                    "class": "couch",
                    "visibility": 0,
                    "location": [95.201797485351563, 200.65220642089844, -50.383701324462891],
                    "quaternion_xyzw": [0.17430000007152557, 0.071800000965595245, -0.034200001507997513,
                                        0.98150002956390381],
                    "pose_transform": [
                        [0.12890000641345978, -0.34709998965263367, 0.92890000343322754, 0],
                        [0.98739999532699585, -0.042199999094009399, -0.15279999375343323, 0],
                        [-0.092200003564357758, -0.93690001964569092, -0.33730000257492065, 0],
                        [95.201797485351563, 200.65220642089844, -50.383800506591797, 1]
                    ],
                    "cuboid_centroid": [91.2041015625, 160.04890441894531, -65.000801086425781],
                    "projected_cuboid_centroid": [-np.nan, -np.nan],
                    "bounding_box":
                        {
                            "top_left": [-np.nan, -np.nan],
                            "bottom_right": [-np.nan, -np.nan]
                        },
                    "cuboid": [
                        [274.96630859375, 40.236099243164063, 88.962799072265625],
                        [-46.844600677490234, 53.990501403808594, 138.77029418945313],
                        [-38.84320068359375, 135.25729370117188, 168.02619934082031],
                        [282.96759033203125, 121.50289916992188, 118.21869659423828],
                        [221.25129699707031, 184.84060668945313, -298.0277099609375],
                        [-100.55960083007813, 198.59489440917969, -248.22030639648438],
                        [-92.558197021484375, 279.8616943359375, -218.96440124511719],
                        [229.25270080566406, 266.10739135742188, -268.77191162109375]
                    ],
                    "projected_cuboid": [
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan]
                    ]
                },
                {
                    "name": "AIUE_V01_Flower",
                    "class": "potted plant",
                    "visibility": 0,
                    "location": [-1111.0482177734375, 304.19241333007813, -8.0936002731323242],
                    "quaternion_xyzw": [0.17430000007152557, 0.071800000965595245, -0.034200001507997513,
                                        0.98150002956390381],
                    "pose_transform": [
                        [0.12890000641345978, -0.34709998965263367, 0.92890000343322754, 0],
                        [0.98739999532699585, -0.042199999094009399, -0.15279999375343323, 0],
                        [-0.092200003564357758, -0.93690001964569092, -0.33730000257492065, 0],
                        [-1111.0482177734375, 304.19241333007813, -8.0937004089355469, 1]
                    ],
                    "cuboid_centroid": [-1116.1099853515625, 214.83219909667969, -63.813899993896484],
                    "projected_cuboid_centroid": [-np.nan, -np.nan],
                    "bounding_box":
                        {
                            "top_left": [-np.nan, -np.nan],
                            "bottom_right": [-np.nan, -np.nan]
                        },
                    "cuboid": [
                        [-1043.8553466796875, 91.213699340820313, -45.428699493408203],
                        [-1189.4859619140625, 97.438102722167969, -22.88909912109375],
                        [-1170.4931640625, 290.33990478515625, 46.555099487304688],
                        [-1024.862548828125, 284.1156005859375, 24.015499114990234],
                        [-1061.726806640625, 139.32449340820313, -174.18290710449219],
                        [-1207.357421875, 145.54890441894531, -151.64329528808594],
                        [-1188.3643798828125, 338.45071411132813, -82.1990966796875],
                        [-1042.7337646484375, 332.22640991210938, -104.73870086669922]
                    ],
                    "projected_cuboid": [
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan]
                    ]
                },
                {
                    "name": "AIUE_V01_001_Book_8",
                    "class": "book",
                    "visibility": 0,
                    "location": [-764.8955078125, 248.502197265625, -18.26930046081543],
                    "quaternion_xyzw": [-0.91979998350143433, 0.1046999990940094, -0.34970000386238098,
                                        0.14350000023841858],
                    "pose_transform": [
                        [0.67350000143051147, 0.19079999625682831, -0.71420001983642578, 0],
                        [0.73339998722076416, -0.29300001263618469, 0.61339998245239258, 0],
                        [0.092200003564357758, 0.93690001964569092, 0.33730000257492065, 0],
                        [-764.8955078125, 248.502197265625, -18.26930046081543, 1]
                    ],
                    "cuboid_centroid": [-764.8955078125, 248.502197265625, -18.26930046081543],
                    "projected_cuboid_centroid": [-np.nan, -np.nan],
                    "bounding_box":
                        {
                            "top_left": [-np.nan, -np.nan],
                            "bottom_right": [-np.nan, -np.nan]
                        },
                    "cuboid": [
                        [-747.6123046875, 247.63229370117188, -15.666299819946289],
                        [-768.9219970703125, 256.14590454101563, -33.487598419189453],
                        [-769.22760009765625, 253.0408935546875, -34.605400085449219],
                        [-747.9180908203125, 244.52720642089844, -16.784099578857422],
                        [-760.5634765625, 243.9635009765625, -1.9329999685287476],
                        [-781.87298583984375, 252.47720336914063, -19.754400253295898],
                        [-782.1787109375, 249.37210083007813, -20.872200012207031],
                        [-760.86907958984375, 240.85850524902344, -3.050800085067749]
                    ],
                    "projected_cuboid": [
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan]
                    ]
                },
                {
                    "name": "AIUE_V01_001_Book_21",
                    "class": "book",
                    "visibility": 0,
                    "location": [-764.4630126953125, 250.64190673828125, -15.77910041809082],
                    "quaternion_xyzw": [-0.94679999351501465, 0.091799996793270111, -0.26820001006126404,
                                        0.15209999680519104],
                    "pose_transform": [
                        [0.53589999675750732, 0.23880000412464142, -0.80980002880096436, 0],
                        [0.83920001983642578, -0.25540000200271606, 0.47999998927116394, 0],
                        [0.092200003564357758, 0.93690001964569092, 0.33730000257492065, 0],
                        [-764.4630126953125, 250.64190673828125, -15.77910041809082, 1]
                    ],
                    "cuboid_centroid": [-764.4630126953125, 250.64190673828125, -15.77910041809082],
                    "projected_cuboid_centroid": [-np.nan, -np.nan],
                    "bounding_box":
                        {
                            "top_left": [-np.nan, -np.nan],
                            "bottom_right": [-np.nan, -np.nan]
                        },
                    "cuboid": [
                        [-750.0919189453125, 251.24130249023438, -17.542900085449219],
                        [-768.65728759765625, 256.89208984375, -28.162300109863281],
                        [-768.89581298828125, 254.47050476074219, -29.034000396728516],
                        [-750.330322265625, 248.8197021484375, -18.414699554443359],
                        [-760.03021240234375, 246.81329345703125, -2.5243000984191895],
                        [-778.5958251953125, 252.46400451660156, -13.143699645996094],
                        [-778.834228515625, 250.04240417480469, -14.015500068664551],
                        [-760.2686767578125, 244.3916015625, -3.3959999084472656]
                    ],
                    "projected_cuboid": [
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan],
                        [-np.nan, -np.nan]
                    ]
                }
            ]
        }, data, allow_nan_equality=True)

    @unittest.skipIf(not CAMERA_SETTINGS.exists(), f"File {CAMERA_SETTINGS} is missing")
    def test_read_camera_settings(self):
        data = ndds_loader.read_json(CAMERA_SETTINGS)
        self.assertJsonEqualWithNaN({
            "camera_settings": [
                {
                    "name": "Viewpoint",
                    "horizontal_fov": 90,
                    "intrinsic_settings":
                        {
                            "resX": 640,
                            "resY": 480,
                            "fx": 517.29998779296875,
                            "fy": 516.5,
                            "cx": 318.60000610351563,
                            "cy": 255.30000305175781,
                            "s": 0
                        },
                    "captured_image_size":
                        {
                            "width": 640,
                            "height": 480
                        }
                }
            ]
        }, data, allow_nan_equality=True)

    @unittest.skipIf(not OBJECT_SETTINGS.exists(), f"File {OBJECT_SETTINGS} is missing")
    def test_read_object_settings(self):
        data = ndds_loader.read_json(OBJECT_SETTINGS)
        self.assertJsonEqualWithNaN({
            "exported_object_classes": [
                "oven",
                "chair",
                "chair",
                "chair",
                "chair",
                "chair",
                "desk",
                "coffee table",
                "coffee table",
                "couch",
                "couch",
                "chair",
                "couch",
                "dining table",
                "coffee table",
                "chair",
                "chair",
                "chair",
                "chair",
                "chair",
                "chair",
                "chair",
                "chair",
                "potted plant",
                "potted plant",
                "potted plant",
                "potted plant",
                "potted plant",
                "potted plant",
                "kettle",
                "potted plant",
                "vase",
                "bottle",
                "vase",
                "bottle",
                "bottle",
                "bottle",
                "vase",
                "potted plant",
                "potted plant",
                "potted plant",
                "refrigerator",
                "book",
                "book",
                "book",
                "book",
                "book",
                "book",
                "book",
                "book",
                "book",
                "bowl",
                "bowl",
                "bowl",
                "bowl",
                "sink",
                "sink",
                "tap",
                "spoon",
                "spoon",
                "spoon",
                "spoon",
                "fork",
                "fork",
                "knife",
                "knife"
            ],
            "exported_objects": [
                {
                    "name": "AIUE_V01_oven",
                    "class": "oven",
                    "segmentation_class_id": 168,
                    "segmentation_instance_id": 1534440,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [48.944400787353516, -107.25900268554688, 264.17999267578125, 1]
                    ],
                    "cuboid_dimensions": [58.115699768066406, 80.851997375488281, 15.931699752807617]
                },
                {
                    "name": "AIUE_V01_chair",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 537054,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-78.758399963378906, -0.076300002634525299, 518.54400634765625, 1]
                    ],
                    "cuboid_dimensions": [48.703201293945313, 85.503196716308594, 51.627700805664063]
                },
                {
                    "name": "AIUE_V01_chair3",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 626563,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-142.24029541015625, -0.076300002634525299, 535.69219970703125, 1]
                    ],
                    "cuboid_dimensions": [48.703201293945313, 85.503196716308594, 51.627700805664063]
                },
                {
                    "name": "AIUE_V01_chair2_5",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 613776,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-209.37969970703125, -0.076300002634525299, 550.33599853515625, 1]
                    ],
                    "cuboid_dimensions": [48.703201293945313, 85.503196716308594, 51.627700805664063]
                },
                {
                    "name": "AIUE_V01_chair4",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 639350,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-279.49951171875, -0.076300002634525299, 565.848876953125, 1]
                    ],
                    "cuboid_dimensions": [48.703201293945313, 85.503196716308594, 51.627700805664063]
                },
                {
                    "name": "AIUE_V01_chair6",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 664924,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-336.6785888671875, -0.076300002634525299, 585.130615234375, 1]
                    ],
                    "cuboid_dimensions": [48.703201293945313, 85.503196716308594, 51.627700805664063]
                },
                {
                    "name": "AIUE_V01_table",
                    "class": "desk",
                    "segmentation_class_id": 98,
                    "segmentation_instance_id": 1981985,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-216.14999389648438, 0.34150001406669617, -161.343994140625, 1]
                    ],
                    "cuboid_dimensions": [142.90109252929688, 76.092697143554688, 78.671798706054688]
                },
                {
                    "name": "AIUE_V01_table2",
                    "class": "coffee table",
                    "segmentation_class_id": 70,
                    "segmentation_instance_id": 1994772,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-619.697021484375, 1.5123000144958496, -420.09201049804688, 1]
                    ],
                    "cuboid_dimensions": [92.659599304199219, 43.593700408935547, 100.90899658203125]
                },
                {
                    "name": "AIUE_V01_table3",
                    "class": "coffee table",
                    "segmentation_class_id": 70,
                    "segmentation_instance_id": 2007559,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-265.6099853515625, -0.53680002689361572, -541.7080078125, 1]
                    ],
                    "cuboid_dimensions": [67.918899536132813, 18.660699844360352, 200.30059814453125]
                },
                {
                    "name": "AIUE_V01_Sofa",
                    "class": "couch",
                    "segmentation_class_id": 84,
                    "segmentation_instance_id": 1943624,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-536.55902099609375, -1.6353000402450562, -523.95599365234375, 1]
                    ],
                    "cuboid_dimensions": [235.78950500488281, 68.738700866699219, 201.83990478515625]
                },
                {
                    "name": "AIUE_V01_Sofa2",
                    "class": "couch",
                    "segmentation_class_id": 84,
                    "segmentation_instance_id": 1956411,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-66.571601867675781, 0.19349999725818634, -500.781005859375, 1]
                    ],
                    "cuboid_dimensions": [133.44180297851563, 79.724800109863281, 262.63958740234375]
                },
                {
                    "name": "AIUE_V01_Chair5",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 652137,
                    "fixed_model_transform": [
                        [-0.669700026512146, 0, 0.79809999465942383, 0],
                        [-0.92170000076293945, 0, -0.77340000867843628, 0],
                        [0, -0.9869999885559082, 0, 0],
                        [-365.74099731445313, 0.78430002927780151, -204.49400329589844, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Sofa3",
                    "class": "couch",
                    "segmentation_class_id": 84,
                    "segmentation_instance_id": 1969198,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [585.4730224609375, 0.14830000698566437, -501.67999267578125, 1]
                    ],
                    "cuboid_dimensions": [325.93280029296875, 86.742301940917969, 416.60281372070313]
                },
                {
                    "name": "AIUE_V01_Table4_18",
                    "class": "dining table",
                    "segmentation_class_id": 112,
                    "segmentation_instance_id": 2020346,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [557.6619873046875, -0.47589999437332153, 229.75799560546875, 1]
                    ],
                    "cuboid_dimensions": [249.44990539550781, 75.939201354980469, 303.12020874023438]
                },
                {
                    "name": "AIUE_V01_table5",
                    "class": "coffee table",
                    "segmentation_class_id": 70,
                    "segmentation_instance_id": 2033133,
                    "fixed_model_transform": [
                        [0.015599999576807022, 0, 0.99989998340606689, 0],
                        [-0.99989998340606689, 0, 0.015599999576807022, 0],
                        [0, -1, 0, 0],
                        [515.656982421875, 1.3286000490188599, -529.72802734375, 1]
                    ],
                    "cuboid_dimensions": [92.659599304199219, 43.593700408935547, 100.90899658203125]
                },
                {
                    "name": "AIUE_V01_Chair7",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 677711,
                    "fixed_model_transform": [
                        [-0.76599997282028198, 0, 0.642799973487854, 0],
                        [-0.642799973487854, 0, -0.76599997282028198, 0],
                        [0, -1, 0, 0],
                        [434.875, -0.082199998199939728, 171.21600341796875, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Chair8",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 690498,
                    "fixed_model_transform": [
                        [-0.76599997282028198, 0, 0.642799973487854, 0],
                        [-0.642799973487854, 0, -0.76599997282028198, 0],
                        [0, -1, 0, 0],
                        [434.875, -0.082199998199939728, 275.34820556640625, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Chair9",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 703285,
                    "fixed_model_transform": [
                        [-0.642799973487854, 0, -0.76599997282028198, 0],
                        [0.76599997282028198, 0, -0.642799973487854, 0],
                        [0, -1, 0, 0],
                        [603.56719970703125, -0.082199998199939728, 85.312103271484375, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Chair10",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 549841,
                    "fixed_model_transform": [
                        [-0.642799973487854, 0, -0.76599997282028198, 0],
                        [0.76599997282028198, 0, -0.642799973487854, 0],
                        [0, -1, 0, 0],
                        [499.43490600585938, -0.082199998199939728, 85.311996459960938, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Chair11",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 562628,
                    "fixed_model_transform": [
                        [0.76599997282028198, 0, -0.642799973487854, 0],
                        [0.642799973487854, 0, 0.76599997282028198, 0],
                        [0, -1, 0, 0],
                        [681.91448974609375, -0.082199998199939728, 308.3385009765625, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Chair12",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 575415,
                    "fixed_model_transform": [
                        [0.76599997282028198, 0, -0.642799973487854, 0],
                        [0.642799973487854, 0, 0.76599997282028198, 0],
                        [0, -1, 0, 0],
                        [681.91461181640625, -0.082199998199939728, 204.20620727539063, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Chair13",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 588202,
                    "fixed_model_transform": [
                        [0.642799973487854, 0, 0.76599997282028198, 0],
                        [-0.76599997282028198, 0, 0.642799973487854, 0],
                        [0, -1, 0, 0],
                        [521.13677978515625, -0.082199998199939728, 369.86270141601563, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Chair14",
                    "class": "chair",
                    "segmentation_class_id": 56,
                    "segmentation_instance_id": 600989,
                    "fixed_model_transform": [
                        [0.642799973487854, 0, 0.76599997282028198, 0],
                        [-0.76599997282028198, 0, 0.642799973487854, 0],
                        [0, -1, 0, 0],
                        [625.26910400390625, -0.082199998199939728, 369.86279296875, 1]
                    ],
                    "cuboid_dimensions": [109.85160064697266, 96.405899047851563, 113.51280212402344]
                },
                {
                    "name": "AIUE_V01_Flower",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 933451,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-616.35400390625, 0.14759999513626099, -653.864013671875, 1]
                    ],
                    "cuboid_dimensions": [147.49589538574219, 205.8988037109375, 138.60609436035156]
                },
                {
                    "name": "AIUE_V01_Flower2_34",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 959025,
                    "fixed_model_transform": [
                        [0, 0, 0.76980000734329224, 0],
                        [-0.76980000734329224, 0, 0, 0],
                        [0, -0.76980000734329224, 0, 0],
                        [545.3585205078125, -75.951896667480469, 200.66639709472656, 1]
                    ],
                    "cuboid_dimensions": [147.49589538574219, 205.8988037109375, 138.60609436035156]
                },
                {
                    "name": "AIUE_V01_Flower3",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 971812,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [516.02899169921875, -75.853797912597656, 246.30900573730469, 1]
                    ],
                    "cuboid_dimensions": [23.538700103759766, 36.908401489257813, 28.649200439453125]
                },
                {
                    "name": "AIUE_V01_Flower4",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 984599,
                    "fixed_model_transform": [
                        [0.98479998111724854, 0, -0.17360000312328339, 0],
                        [0.17360000312328339, 0, 0.98479998111724854, 0],
                        [0, -1, 0, 0],
                        [524.672607421875, -75.853797912597656, 187.36430358886719, 1]
                    ],
                    "cuboid_dimensions": [23.538700103759766, 36.908401489257813, 28.649200439453125]
                },
                {
                    "name": "AIUE_V01_Flower5",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 997386,
                    "fixed_model_transform": [
                        [-1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, -1, 0, 0],
                        [560.2022705078125, -75.853797912597656, 166.74690246582031, 1]
                    ],
                    "cuboid_dimensions": [23.538700103759766, 36.908401489257813, 28.649200439453125]
                },
                {
                    "name": "AIUE_V01_Flower6",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 1010173,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [518.1309814453125, -42.771400451660156, -525.7340087890625, 1]
                    ],
                    "cuboid_dimensions": [70.950103759765625, 140.41419982910156, 65.533302307128906]
                },
                {
                    "name": "AIUE_V01_Teapot",
                    "class": "kettle",
                    "segmentation_class_id": 140,
                    "segmentation_instance_id": 2045920,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-359.77301025390625, -85.64990234375, 499.92999267578125, 1]
                    ],
                    "cuboid_dimensions": [22.084600448608398, 22.953899383544922, 25.161899566650391]
                },
                {
                    "name": "AIUE_V01_Flower7",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 1022960,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-381.9949951171875, -85.513397216796875, 561.45501708984375, 1]
                    ],
                    "cuboid_dimensions": [38.642101287841797, 59.544200897216797, 34.511798858642578]
                },
                {
                    "name": "AIUE_V01_pot",
                    "class": "vase",
                    "segmentation_class_id": 252,
                    "segmentation_instance_id": 1777393,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-295.5, -158.23699951171875, -129.92300415039063, 1]
                    ],
                    "cuboid_dimensions": [12.715999603271484, 21.851600646972656, 12.715900421142578]
                },
                {
                    "name": "AIUE_V01_pot2",
                    "class": "bottle",
                    "segmentation_class_id": 28,
                    "segmentation_instance_id": 1790180,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-279.11099243164063, -158.25900268554688, -131.11099243164063, 1]
                    ],
                    "cuboid_dimensions": [8.5855998992919922, 29.856500625610352, 8.6705999374389648]
                },
                {
                    "name": "AIUE_V01_pot4",
                    "class": "vase",
                    "segmentation_class_id": 252,
                    "segmentation_instance_id": 1815754,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-249.93400573730469, -158.20199584960938, -130.58200073242188, 1]
                    ],
                    "cuboid_dimensions": [9.4914999008178711, 22.244800567626953, 9.4926996231079102]
                },
                {
                    "name": "AIUE_V01_pot5",
                    "class": "bottle",
                    "segmentation_class_id": 28,
                    "segmentation_instance_id": 1828541,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-155.59330749511719, -158.20199584960938, -130.58200073242188, 1]
                    ],
                    "cuboid_dimensions": [9.4914999008178711, 22.244800567626953, 9.4926996231079102]
                },
                {
                    "name": "AIUE_V01_pot6",
                    "class": "bottle",
                    "segmentation_class_id": 28,
                    "segmentation_instance_id": 1841328,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-315.52261352539063, -158.20199584960938, -130.58189392089844, 1]
                    ],
                    "cuboid_dimensions": [9.4914999008178711, 22.244800567626953, 9.4926996231079102]
                },
                {
                    "name": "AIUE_V01_pot8",
                    "class": "bottle",
                    "segmentation_class_id": 28,
                    "segmentation_instance_id": 1866902,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-228.58360290527344, -158.25900268554688, -131.11099243164063, 1]
                    ],
                    "cuboid_dimensions": [8.5855998992919922, 29.856500625610352, 8.6705999374389648]
                },
                {
                    "name": "AIUE_V01_pot9",
                    "class": "vase",
                    "segmentation_class_id": 252,
                    "segmentation_instance_id": 1879689,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-202.65980529785156, -158.23699951171875, -129.92300415039063, 1]
                    ],
                    "cuboid_dimensions": [12.715999603271484, 21.851600646972656, 12.715900421142578]
                },
                {
                    "name": "AIUE_V01_Flower8_10",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 1035747,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-622.6455078125, -41.737998962402344, -432.069091796875, 1]
                    ],
                    "cuboid_dimensions": [23.538700103759766, 36.908401489257813, 28.649200439453125]
                },
                {
                    "name": "AIUE_V01_Flower9_5",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 1048534,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-259.76699829101563, -19.256500244140625, -540.07049560546875, 1]
                    ],
                    "cuboid_dimensions": [23.538700103759766, 36.908401489257813, 28.649200439453125]
                },
                {
                    "name": "AIUE_V01_Flower10",
                    "class": "potted plant",
                    "segmentation_class_id": 182,
                    "segmentation_instance_id": 946238,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-265.135986328125, -74.931503295898438, -169.218994140625, 1]
                    ],
                    "cuboid_dimensions": [42.481601715087891, 68.133598327636719, 37.940799713134766]
                },
                {
                    "name": "AIUE_V01_fridge",
                    "class": "refrigerator",
                    "segmentation_class_id": 196,
                    "segmentation_instance_id": 1061321,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [150.56199645996094, 1.0253000259399414, 539.57708740234375, 1]
                    ],
                    "cuboid_dimensions": [99.310096740722656, 180, 35.42449951171875]
                },
                {
                    "name": "AIUE_V01_001_Book_3",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 51148,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-613.62579345703125, -43.637199401855469, -399.76870727539063, 1]
                    ],
                    "cuboid_dimensions": [18.785999298095703, 3.7781000137329102, 19.040000915527344]
                },
                {
                    "name": "AIUE_V01_001_Book_6",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 76722,
                    "fixed_model_transform": [
                        [0.5, 0, 0.86599999666213989, 0],
                        [-0.86599999666213989, 0, 0.5, 0],
                        [0, -1, 0, 0],
                        [-615.69757080078125, -46.932598114013672, -401.09359741210938, 1]
                    ],
                    "cuboid_dimensions": [22.116300582885742, 2.5848000049591064, 18.538799285888672]
                },
                {
                    "name": "AIUE_V01_001_Book_4",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 63935,
                    "fixed_model_transform": [
                        [0.5, 0, 0.86599999666213989, 0],
                        [0.86599999666213989, 0, -0.5, 0],
                        [0, 1, 0, 0],
                        [-264.01458740234375, -21.058399200439453, -492.15240478515625, 1]
                    ],
                    "cuboid_dimensions": [22.131599426269531, 3.3814001083374023, 19.101299285888672]
                },
                {
                    "name": "AIUE_V01_001_Book_9",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 115083,
                    "fixed_model_transform": [
                        [0.76599997282028198, 0, 0.642799973487854, 0],
                        [-0.642799973487854, 0, 0.76599997282028198, 0],
                        [0, -1, 0, 0],
                        [-265.11199951171875, -24.475200653076172, -499.81069946289063, 1]
                    ],
                    "cuboid_dimensions": [29.054800033569336, 3.3143000602722168, 19.229999542236328]
                },
                {
                    "name": "AIUE_V01_001_Book_12",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 12787,
                    "fixed_model_transform": [
                        [0.642799973487854, 0, 0.76599997282028198, 0],
                        [-0.76599997282028198, 0, 0.642799973487854, 0],
                        [0, -1, 0, 0],
                        [-265.15579223632813, -27.359600067138672, -492.3931884765625, 1]
                    ],
                    "cuboid_dimensions": [22.121999740600586, 2.5848000049591064, 18.545600891113281]
                },
                {
                    "name": "AIUE_V01_001_Book_16",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 25574,
                    "fixed_model_transform": [
                        [0.86599999666213989, 0, 0.5, 0],
                        [-0.5, 0, 0.86599999666213989, 0],
                        [0, -1, 0, 0],
                        [-264.85598754882813, -30.390800476074219, -488.72109985351563, 1]
                    ],
                    "cuboid_dimensions": [15.736900329589844, 3.3459999561309814, 13.582200050354004]
                },
                {
                    "name": "AIUE_V01_001_Book_7",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 89509,
                    "fixed_model_transform": [
                        [0.5, 0, -0.86599999666213989, 0],
                        [0.86599999666213989, 0, 0.5, 0],
                        [0, -1, 0, 0],
                        [-269.57638549804688, -26.945899963378906, -597.968994140625, 1]
                    ],
                    "cuboid_dimensions": [22.131599426269531, 3.3814001083374023, 19.101299285888672]
                },
                {
                    "name": "AIUE_V01_001_Book_8",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 102296,
                    "fixed_model_transform": [
                        [0.76599997282028198, 0, -0.642799973487854, 0],
                        [-0.642799973487854, 0, -0.76599997282028198, 0],
                        [0, 1, 0, 0],
                        [-270.67379760742188, -23.528999328613281, -599.3544921875, 1]
                    ],
                    "cuboid_dimensions": [29.054800033569336, 3.3143000602722168, 19.229999542236328]
                },
                {
                    "name": "AIUE_V01_001_Book_21",
                    "class": "book",
                    "segmentation_class_id": 14,
                    "segmentation_instance_id": 38361,
                    "fixed_model_transform": [
                        [0.642799973487854, 0, -0.76599997282028198, 0],
                        [-0.76599997282028198, 0, -0.642799973487854, 0],
                        [0, 1, 0, 0],
                        [-270.71759033203125, -20.644699096679688, -597.7283935546875, 1]
                    ],
                    "cuboid_dimensions": [22.121999740600586, 2.5848000049591064, 18.545600891113281]
                },
                {
                    "name": "AIUE_V01_001_Bowl_24",
                    "class": "bowl",
                    "segmentation_class_id": 42,
                    "segmentation_instance_id": 140657,
                    "fixed_model_transform": [
                        [0.17360000312328339, 0, 0.98479998111724854, 0],
                        [-0.98479998111724854, 0, 0.17360000312328339, 0],
                        [0, -1, 0, 0],
                        [-353.6864013671875, -85.572898864746094, 308.59149169921875, 1]
                    ],
                    "cuboid_dimensions": [20.323400497436523, 10.355299949645996, 18.074199676513672]
                },
                {
                    "name": "AIUE_V01_001_Bowl_2",
                    "class": "bowl",
                    "segmentation_class_id": 42,
                    "segmentation_instance_id": 127870,
                    "fixed_model_transform": [
                        [0.17360000312328339, 0, 0.98479998111724854, 0],
                        [-0.98479998111724854, 0, 0.17360000312328339, 0],
                        [0, -1, 0, 0],
                        [-353.6864013671875, -89.402099609375, 308.59149169921875, 1]
                    ],
                    "cuboid_dimensions": [20.323400497436523, 10.355299949645996, 18.074199676513672]
                },
                {
                    "name": "AIUE_V01_001_Bowl_3",
                    "class": "bowl",
                    "segmentation_class_id": 42,
                    "segmentation_instance_id": 153444,
                    "fixed_model_transform": [
                        [0.17360000312328339, 0, 0.98479998111724854, 0],
                        [-0.96979999542236328, -0.17360000312328339, 0.17100000381469727, 0],
                        [0.17100000381469727, -0.98479998111724854, -0.03020000085234642, 0],
                        [-355.57479858398438, -93.173301696777344, 308.95849609375, 1]
                    ],
                    "cuboid_dimensions": [20.323400497436523, 10.355299949645996, 18.074199676513672]
                },
                {
                    "name": "AIUE_V01_001_Bowl_4",
                    "class": "bowl",
                    "segmentation_class_id": 42,
                    "segmentation_instance_id": 166231,
                    "fixed_model_transform": [
                        [0.17360000312328339, 0, 0.98479998111724854, 0],
                        [-0.96979999542236328, 0.17360000312328339, 0.17100000381469727, 0],
                        [-0.17100000381469727, -0.98479998111724854, 0.03020000085234642, 0],
                        [-351.8096923828125, -97.209396362304688, 308.29470825195313, 1]
                    ],
                    "cuboid_dimensions": [20.323400497436523, 10.355299949645996, 18.074199676513672]
                },
                {
                    "name": "AIUE_V01_oven_cupboard_sink_left",
                    "class": "sink",
                    "segmentation_class_id": 210,
                    "segmentation_instance_id": 1560014,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-229.66499328613281, 0.23149999976158142, 304.01400756835938, 1]
                    ],
                    "cuboid_dimensions": [67.382896423339844, 15.414999961853027, 59.839099884033203]
                },
                {
                    "name": "AIUE_V01_oven_cupboard_sink_right",
                    "class": "sink",
                    "segmentation_class_id": 210,
                    "segmentation_instance_id": 1572801,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-229.66499328613281, 0.23149999976158142, 304.01400756835938, 1]
                    ],
                    "cuboid_dimensions": [67.382896423339844, 15.414999961853027, 59.839099884033203]
                },
                {
                    "name": "AIUE_V01_oven_cupboard_tap",
                    "class": "tap",
                    "segmentation_class_id": 238,
                    "segmentation_instance_id": 1585588,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-229.66499328613281, 0.23149999976158142, 304.01400756835938, 1]
                    ],
                    "cuboid_dimensions": [14.937999725341797, 43.783000946044922, 24.362699508666992]
                },
                {
                    "name": "AIUE_V01_001_wooden_spoon_12",
                    "class": "spoon",
                    "segmentation_class_id": 224,
                    "segmentation_instance_id": 396397,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-0.99970000982284546, 0.024399999529123306, 0, 0],
                        [-0.024399999529123306, -0.99970000982284546, 0, 0],
                        [0.000699999975040555, -2.9714000225067139, -8.2753000259399414, 1]
                    ],
                    "cuboid_dimensions": [0.87309998273849487, 29.206699371337891, 5.5132999420166016]
                },
                {
                    "name": "AIUE_V01_001_wooden_spoon_21",
                    "class": "spoon",
                    "segmentation_class_id": 224,
                    "segmentation_instance_id": 409184,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [-0.27489998936653137, -3.0362999439239502, 8.4706001281738281, 1]
                    ],
                    "cuboid_dimensions": [3.8475000858306885, 29.034099578857422, 6.750999927520752]
                },
                {
                    "name": "AIUE_V01_001_spoon_30",
                    "class": "spoon",
                    "segmentation_class_id": 224,
                    "segmentation_instance_id": 358036,
                    "fixed_model_transform": [
                        [0, 0, 1, 0],
                        [-0.14519999921321869, -0.98940002918243408, 0, 0],
                        [0.98940002918243408, -0.14519999921321869, 0, 0],
                        [-6.6897001266479492, -8.4783000946044922, 1.3817000389099121, 1]
                    ],
                    "cuboid_dimensions": [19.404600143432617, 1.1683000326156616, 4.659599781036377]
                },
                {
                    "name": "AIUE_V01_001_spoon_2",
                    "class": "spoon",
                    "segmentation_class_id": 224,
                    "segmentation_instance_id": 345249,
                    "fixed_model_transform": [
                        [0.63980001211166382, 0, -0.7685999870300293, 0],
                        [0.11159999668598175, -0.98940002918243408, 0.092900000512599945, 0],
                        [-0.76039999723434448, -0.14519999921321869, -0.63300001621246338, 0],
                        [-2.9521999359130859, -8.4783000946044922, 2.8239998817443848, 1]
                    ],
                    "cuboid_dimensions": [19.404600143432617, 1.1683000326156616, 4.659599781036377]
                },
                {
                    "name": "AIUE_V01_001_fork_34",
                    "class": "fork",
                    "segmentation_class_id": 126,
                    "segmentation_instance_id": 255740,
                    "fixed_model_transform": [
                        [-0.28519999980926514, 0, 0.95850002765655518, 0],
                        [-0.12430000305175781, -0.99159997701644897, -0.037000000476837158, 0],
                        [0.95039999485015869, -0.12970000505447388, 0.28279998898506165, 0],
                        [-4.3256001472473145, -11.27079963684082, -0.05469999834895134, 1]
                    ],
                    "cuboid_dimensions": [17.989900588989258, 1.0145000219345093, 2.1354000568389893]
                },
                {
                    "name": "AIUE_V01_001_fork_2",
                    "class": "fork",
                    "segmentation_class_id": 126,
                    "segmentation_instance_id": 242953,
                    "fixed_model_transform": [
                        [0.22220000624656677, -0.081699997186660767, 0.9715999960899353, 0],
                        [0.12680000066757202, -0.98559999465942383, -0.11190000176429749, 0],
                        [0.96670001745223999, 0.14800000190734863, -0.2085999995470047, 0],
                        [-5.7059001922607422, -11.27079963684082, -1.4129999876022339, 1]
                    ],
                    "cuboid_dimensions": [17.989900588989258, 1.0145000219345093, 2.1354000568389893]
                },
                {
                    "name": "AIUE_V01_001_knife_38",
                    "class": "knife",
                    "segmentation_class_id": 154,
                    "segmentation_instance_id": 281314,
                    "fixed_model_transform": [
                        [0, 0.24750000238418579, 0.96890002489089966, 0],
                        [-0.2800000011920929, -0.93010002374649048, 0.23759999871253967, 0],
                        [0.95999997854232788, -0.27129998803138733, 0.069300003349781036, 0],
                        [-3.5390000343322754, -8.5652999877929688, 1.6004999876022339, 1]
                    ],
                    "cuboid_dimensions": [19.609500885009766, 1.0178999900817871, 1.9016000032424927]
                },
                {
                    "name": "AIUE_V01_001_knife_2",
                    "class": "knife",
                    "segmentation_class_id": 154,
                    "segmentation_instance_id": 268527,
                    "fixed_model_transform": [
                        [0.032099999487400055, 0.24740000069141388, -0.96840000152587891, 0],
                        [0.28780001401901245, -0.93010002374649048, -0.2281000018119812, 0],
                        [-0.95719999074935913, -0.27129998803138733, -0.10109999775886536, 0],
                        [-5.6908001899719238, -8.5652999877929688, 0.75449997186660767, 1]
                    ],
                    "cuboid_dimensions": [19.609500885009766, 1.0178999900817871, 1.9016000032424927]
                }
            ]
        }, data, allow_nan_equality=True)
