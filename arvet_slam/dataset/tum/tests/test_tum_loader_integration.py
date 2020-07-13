import unittest
import logging
import shutil
from json import load as json_load
from pathlib import Path
import arvet.database.tests.database_connection as dbconn
from arvet.core.image_collection import ImageCollection
from arvet.core.image import Image
import arvet_slam.dataset.tum.tum_loader as tum_loader


def load_dataset_location():
    conf_json = Path(__file__).parent / 'tum_location.json'
    if conf_json.is_file():
        with conf_json.open('r') as fp:
            son = json_load(fp)
            return Path(son['location']), str(son['sequence'])
    return None, None


dataset_root, sequence = load_dataset_location()


class TestTUMLoaderIntegration(unittest.TestCase):

    @unittest.skipIf(
        dataset_root is None or not (dataset_root / sequence).exists(),
        "Could not find the TUM dataset to load, cannot run integration test"
    )
    def test_load_configured_sequence(self):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager(mock=False)
        logging.disable(logging.CRITICAL)

        # Make sure there is nothing in the database
        ImageCollection.objects.all().delete()
        Image.objects.all().delete()

        # count the number of images we expect to import
        rgb_images = dataset_root / sequence / 'rgb'
        depth_images = dataset_root / sequence / 'depth'
        num_images = min(
            sum(1 for file in rgb_images.iterdir() if file.is_file() and file.suffix == '.png'),
            sum(1 for file in depth_images.iterdir() if file.is_file() and file.suffix == '.png')
        )

        result = tum_loader.import_dataset(
            dataset_root / sequence,
            sequence
        )
        self.assertIsInstance(result, ImageCollection)
        self.assertIsNotNone(result.pk)
        self.assertEqual(1, ImageCollection.objects.all().count())
        # Make sure we got all the images (there are 756 RGB images but only 755 depth maps)
        self.assertEqual(num_images, Image.objects.all().count())

        # Make sure we got the depth and position data
        for timestamp, image in result:
            self.assertIsNotNone(image.depth)
            self.assertIsNotNone(image.camera_pose)

        # Clean up after ourselves by dropping the collections for the models
        ImageCollection._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()
        logging.disable(logging.NOTSET)

    @unittest.skipIf(
        dataset_root is None or not (dataset_root / 'rgbd_dataset_freiburg1_360').exists(),
        "Could not find the TUM dataset to load, cannot run integration test"
    )
    def test_load_rgbd_dataset_freiburg1_360(self):
        dbconn.connect_to_test_db()
        dbconn.setup_image_manager(mock=False)
        logging.disable(logging.CRITICAL)

        # Make sure there is nothing in the database
        ImageCollection.objects.all().delete()
        Image.objects.all().delete()

        result = tum_loader.import_dataset(
            dataset_root / 'rgbd_dataset_freiburg1_360',
            'rgbd_dataset_freiburg1_360'
        )
        self.assertIsInstance(result, ImageCollection)
        self.assertIsNotNone(result.pk)
        self.assertEqual(1, ImageCollection.objects.all().count())
        # Make sure we got all the images (there are 756 RGB images but only 755 depth maps)
        self.assertEqual(755, Image.objects.all().count())

        # Make sure we got the depth and position data
        for timestamp, image in result:
            self.assertIsNotNone(image.depth)
            self.assertIsNotNone(image.camera_pose)

        # Clean up after ourselves by dropping the collections for the models
        ImageCollection._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()
        logging.disable(logging.NOTSET)

    @unittest.skipIf(
        dataset_root is None or not (dataset_root / 'rgbd_dataset_freiburg1_desk.tgz').exists(),
        "Could not find the TUM dataset to load, cannot run integration test"
    )
    def test_load_rgbd_dataset_freiburg1_desk_from_tarball(self):
        # Ensure the uncompressed dataset doesn't exist, so we can
        if (dataset_root / 'rgbd_dataset_freiburg1_desk').is_dir():
            shutil.rmtree(dataset_root / 'rgbd_dataset_freiburg1_desk')

        dbconn.connect_to_test_db()
        dbconn.setup_image_manager(mock=False)
        logging.disable(logging.CRITICAL)

        # Make sure there is nothing in the database
        ImageCollection.objects.all().delete()
        Image.objects.all().delete()

        result = tum_loader.import_dataset(
            dataset_root / 'rgbd_dataset_freiburg1_desk.tgz',
            'rgbd_dataset_freiburg1_desk'
        )
        self.assertIsInstance(result, ImageCollection)
        self.assertIsNotNone(result.pk)
        self.assertEqual(1, ImageCollection.objects.all().count())
        # Make sure we got all the images (there are 756 RGB images but only 755 depth maps)
        self.assertEqual(595, Image.objects.all().count())

        # Make sure we got the depth and position data
        for timestamp, image in result:
            self.assertIsNotNone(image.depth)
            self.assertIsNotNone(image.camera_pose)

        # Make sure the loader cleaned up after itself by removing the extracted data
        self.assertFalse((dataset_root / 'rgbd_dataset_freiburg1_desk').exists())

        # Clean up after ourselves by dropping the collections for the models
        ImageCollection._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        dbconn.tear_down_image_manager()
        logging.disable(logging.NOTSET)
