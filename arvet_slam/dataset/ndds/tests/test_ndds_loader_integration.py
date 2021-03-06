import unittest
import logging
import shutil
from json import load as json_load
from pathlib import Path
import arvet.database.tests.database_connection as dbconn
from arvet.core.image_collection import ImageCollection
from arvet.core.image import StereoImage
import arvet_slam.dataset.ndds.ndds_loader as ndds_loader
from arvet_slam.dataset.ndds.depth_noise import DepthNoiseQuality


def load_dataset_location():
    conf_json = Path(__file__).parent / 'ndds_location.json'
    if conf_json.is_file():
        with conf_json.open('r') as fp:
            son = json_load(fp)
            return Path(son['location']), str(son['sequence']), str(son['zip_sequence'])
    return None, None, None


DATASET_ROOT, SEQUENCE, ZIPPED_SEQUENCE = load_dataset_location()


class TestNDDSLoaderIntegration(unittest.TestCase):

    @unittest.skipIf(
        DATASET_ROOT is None or not (DATASET_ROOT / SEQUENCE).exists(),
        f"Could not find the NDDS dataset at {DATASET_ROOT / SEQUENCE}, cannot run integration test"
    )
    def test_load_configured_sequence(self):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager(mock=False)
        logging.disable(logging.CRITICAL)

        # count the number of images we expect to import
        left_folder = DATASET_ROOT / SEQUENCE / 'left'
        right_folder = DATASET_ROOT / SEQUENCE / 'right'
        num_images = sum(
            1 for file in left_folder.iterdir()
            if file.is_file()
            and file.suffix == '.png'
            and '.' not in file.stem
            and (right_folder / file.name).exists()
        )

        # Make sure there is nothing in the database
        ImageCollection.objects.all().delete()
        StereoImage.objects.all().delete()

        result = ndds_loader.import_dataset(
            DATASET_ROOT,
            SEQUENCE,
            DepthNoiseQuality.KINECT_NOISE.name
        )
        self.assertIsInstance(result, ImageCollection)
        self.assertIsNotNone(result.pk)
        self.assertTrue(result.is_depth_available)
        self.assertTrue(result.is_stereo_available)

        self.assertEqual(1, ImageCollection.objects.all().count())
        self.assertEqual(num_images, StereoImage.objects.all().count())   # Make sure we got all the images

        # Make sure we got the position data
        for timestamp, image in result:
            self.assertIsNotNone(image.camera_pose)
            self.assertIsNotNone(image.right_camera_pose)
            self.assertIsNotNone(image.depth)

        # Clean up after ourselves by dropping the collections for the models
        ImageCollection._mongometa.collection.drop()
        StereoImage._mongometa.collection.drop()
        dbconn.tear_down_image_manager()
        logging.disable(logging.NOTSET)

    @unittest.skipIf(
        DATASET_ROOT is None
        or not ZIPPED_SEQUENCE.endswith('.tar.gz')  # Must actually be a compressed file, not a directory
        or not (DATASET_ROOT / ZIPPED_SEQUENCE).is_file(),
        f"Could not find compressed NDDS dataset at {DATASET_ROOT / ZIPPED_SEQUENCE}, cannot run integration test"
    )
    def test_load_zipped_sequence(self):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager(mock=False)
        logging.disable(logging.CRITICAL)

        # Make sure there is nothing in the database
        ImageCollection.objects.all().delete()
        StereoImage.objects.all().delete()

        # Make sure the un-tarred folder does not exist
        sequence_name = ZIPPED_SEQUENCE.split('.')[0]
        extracted_folder = DATASET_ROOT / sequence_name
        if extracted_folder.exists():
            shutil.rmtree(extracted_folder)

        result = ndds_loader.import_dataset(
            DATASET_ROOT,
            sequence_name,
            DepthNoiseQuality.KINECT_NOISE.name
        )
        self.assertIsInstance(result, ImageCollection)
        self.assertIsNotNone(result.pk)
        self.assertTrue(result.is_depth_available)
        self.assertTrue(result.is_stereo_available)

        self.assertEqual(1, ImageCollection.objects.all().count())
        self.assertGreater(StereoImage.objects.all().count(), 0)   # Make sure we got some number of images

        # Make sure we got position data and depth for all frames
        for timestamp, image in result:
            self.assertIsNotNone(image.camera_pose)
            self.assertIsNotNone(image.right_camera_pose)
            self.assertIsNotNone(image.depth)

        # Make sure the extracted folder is cleaned up
        self.assertFalse(extracted_folder.exists())

        # Clean up after ourselves by dropping the collections for the models
        ImageCollection._mongometa.collection.drop()
        StereoImage._mongometa.collection.drop()
        dbconn.tear_down_image_manager()
        logging.disable(logging.NOTSET)
