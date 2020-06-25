import unittest
import os.path
import logging
from json import load as json_load
from pathlib import Path
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.core.image_collection import ImageCollection
from arvet.core.image import StereoImage
import arvet_slam.dataset.euroc.euroc_loader as euroc_loader


def load_dataset_location():
    conf_json = Path(__file__).parent / 'euroc_location.json'
    if conf_json.is_file():
        with conf_json.open('r') as fp:
            son = json_load(fp)
            return Path(son['location']), str(son['sequence'])
    return None, None


dataset_root, sequence = load_dataset_location()


class TestEuRoCLoaderIntegration(unittest.TestCase):

    @unittest.skipIf(
        dataset_root is None or not (dataset_root / sequence).exists(),
        "Could not find the EuRoC dataset {0}, cannot run integration test".format(sequence)
    )
    def test_load_configured_sequence(self):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file, allow_write=True)
        im_manager.set_image_manager(image_manager)
        logging.disable(logging.CRITICAL)

        # count the number of images we expect to import
        cam0 = dataset_root / sequence / 'mav0' / 'cam0' / 'data'
        cam1 = dataset_root / sequence / 'mav0' / 'cam1' / 'data'
        num_images = sum(1 for file in cam0.iterdir()
                         if file.is_file() and file.suffix == '.png' and (cam1 / file.name).exists())

        # Make sure there is nothing in the database
        ImageCollection.objects.all().delete()
        StereoImage.objects.all().delete()

        result = euroc_loader.import_dataset(
            dataset_root / sequence,
            sequence
        )
        self.assertIsInstance(result, ImageCollection)
        self.assertIsNotNone(result.pk)
        self.assertEqual(1, ImageCollection.objects.all().count())
        self.assertEqual(num_images, StereoImage.objects.all().count())   # Make sure we got all the images

        # Make sure we got the depth and position data
        for timestamp, image in result:
            self.assertIsNotNone(image.camera_pose)
            self.assertIsNotNone(image.right_camera_pose)

        # Clean up after ourselves by dropping the collections for the models
        ImageCollection._mongometa.collection.drop()
        StereoImage._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        logging.disable(logging.NOTSET)
