import unittest
import unittest.mock as mock
from pathlib import Path
import numpy as np
import xxhash
import cv2
from json import load as json_load
import arvet.util.image_utils as image_utils
from arvet.metadata.camera_intrinsics import CameraIntrinsics
import arvet_slam.dataset.euroc.euroc_loader as euroc_loader
import arvet_slam.dataset.euroc.euroc_validator as euroc_validator


def load_dataset_location():
    conf_json = Path(__file__).parent / 'euroc_location.json'
    if conf_json.is_file():
        with conf_json.open('r') as fp:
            son = json_load(fp)
            return Path(son['location']), str(son['sequence'])
    return None, None


DATASET_ROOT, SEQUENCE = load_dataset_location()


class TestEuRoCValidator(unittest.TestCase):

    @mock.patch('import arvet_slam.dataset.euroc.euroc_validator.image_utils')
    @mock.patch('import arvet_slam.dataset.euroc.euroc_validator.euroc_loader')
    def test_valid_image_is_valid(self, mock_loader, mock_utils):
        pass
