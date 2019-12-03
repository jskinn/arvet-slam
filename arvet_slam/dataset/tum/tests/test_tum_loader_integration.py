import unittest
import os.path
import logging
import shutil
from pathlib import Path
import arvet.database.tests.database_connection as dbconn
import arvet.database.image_manager as im_manager
from arvet.core.image_collection import ImageCollection
from arvet.core.image import Image
import arvet_slam.dataset.tum.tum_loader as tum_loader

# The hard-coded path to where some of the TUM dataset is stored, for testing.
tum_dataset_path = Path('/media/john/Disk4/phd_data/datasets/TUM-rgbd')


class TestTUMLoaderIntegration(unittest.TestCase):

    @unittest.skipIf(
        not (tum_dataset_path / 'rgbd_dataset_freiburg1_360').exists(),
        "Could not find the TUM dataset to load, cannot run integration test"
    )
    def test_load_rgbd_dataset_freiburg1_360(self):
        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)
        logging.disable(logging.CRITICAL)

        # Make sure there is nothing in the database
        ImageCollection.objects.all().delete()
        Image.objects.all().delete()

        result = tum_loader.import_dataset(
            tum_dataset_path / 'rgbd_dataset_freiburg1_360',
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
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        logging.disable(logging.NOTSET)

    @unittest.skipIf(
        not (tum_dataset_path / 'rgbd_dataset_freiburg1_desk.tgz').exists(),
        "Could not find the TUM dataset to load, cannot run integration test"
    )
    def test_load_rgbd_dataset_freiburg1_desk_from_tarball(self):
        # Ensure the uncompressed dataset doesn't exist, so we can
        if (tum_dataset_path / 'rgbd_dataset_freiburg1_desk').is_dir():
            shutil.rmtree(tum_dataset_path / 'rgbd_dataset_freiburg1_desk')

        dbconn.connect_to_test_db()
        image_manager = im_manager.DefaultImageManager(dbconn.image_file)
        im_manager.set_image_manager(image_manager)
        logging.disable(logging.CRITICAL)

        # Make sure there is nothing in the database
        ImageCollection.objects.all().delete()
        Image.objects.all().delete()

        result = tum_loader.import_dataset(
            tum_dataset_path / 'rgbd_dataset_freiburg1_desk.tgz',
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
        self.assertFalse((tum_dataset_path / 'rgbd_dataset_freiburg1_desk').exists())

        # Clean up after ourselves by dropping the collections for the models
        ImageCollection._mongometa.collection.drop()
        Image._mongometa.collection.drop()
        if os.path.isfile(dbconn.image_file):
            os.remove(dbconn.image_file)
        logging.disable(logging.NOTSET)
