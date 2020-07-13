import unittest
import unittest.mock as mock
import shutil
from bson import ObjectId
from pathlib import Path
from pymodm.context_managers import no_auto_dereference
from orbslam2 import VocabularyBuilder
from arvet.config.path_manager import PathManager
import arvet.database.tests.database_connection as dbconn
from arvet.core.sequence_type import ImageSequenceType
from arvet.core.image_collection import ImageCollection
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult
from arvet_slam.systems.slam.orbslam2 import OrbSlam2, SensorMode, VOCABULARY_FOLDER
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode


class TestORBSLAM2BuildVocabulary(unittest.TestCase):
    temp_folder = Path(__file__).parent / 'temp-orbslam2-vocabs'
    path_manager = None
    image_collection = None

    max_time = 50
    num_frames = 100
    speed = 0.1

    @classmethod
    def setUpClass(cls):
        dbconn.setup_image_manager()
        cls.temp_folder.mkdir(parents=True, exist_ok=True)
        cls.path_manager = PathManager([Path(__file__).parent], cls.temp_folder)

        image_builder = DemoImageBuilder(
            mode=ImageMode.STEREO, stereo_offset=0.15,
            width=320, height=240, num_stars=500,
            length=cls.max_time * cls.speed, speed=cls.speed,
            min_size=4, max_size=50
        )

        # Make an image source from the image builder
        images = []
        for time in range(cls.num_frames):
            image = image_builder.create_frame(time)
            images.append(image)
        cls.image_collection = ImageCollection(
            images=images,
            timestamps=list(range(len(images))),
            sequence_type=ImageSequenceType.SEQUENTIAL
        )

    @classmethod
    def tearDownClass(cls):
        if cls.temp_folder.exists():
            shutil.rmtree(cls.temp_folder)
        vocab_folder = Path(__file__).parent / VOCABULARY_FOLDER
        if vocab_folder.exists():
            shutil.rmtree(vocab_folder)
        dbconn.tear_down_image_manager()

    def test_can_build_a_vocab_file(self):
        subject = OrbSlam2(
            mode=SensorMode.STEREO,
            vocabulary_branching_factor=6,
            vocabulary_depth=3,
            vocabulary_seed=158627,
            orb_num_features=125,
            orb_scale_factor=2,
            orb_num_levels=2,
            orb_ini_threshold_fast=12,
            orb_min_threshold_fast=7
        )
        subject.pk = ObjectId()     # object needs to be "saved" to generate a vocab filename
        self.assertEqual('', subject.vocabulary_file)
        subject.build_vocabulary([self.image_collection], self.path_manager.get_output_dir())
        self.assertNotEqual('', subject.vocabulary_file)
        vocab_path = self.path_manager.find_file(subject.vocabulary_file)
        self.assertTrue(vocab_path.exists())
        vocab_path.unlink()

    @mock.patch('arvet_slam.systems.slam.orbslam2.VocabularyBuilder', autospec=type(VocabularyBuilder))
    def test_passes_orb_settings_to_vocabulary(self, mock_vocabulary_builder_class):
        mock_builder = mock.create_autospec(spec=VocabularyBuilder, spec_set=True)
        mock_vocabulary_builder_class.return_value = mock_builder

        branching_factor = 12
        vocab_depth = 2
        vocab_seed = 1618673921

        num_features = 12253
        scale_factor = 1.23415
        num_levels = 7
        ini_threshold = 15
        min_threshold = 22
        subject = OrbSlam2(
            mode=SensorMode.STEREO,
            vocabulary_branching_factor=branching_factor,
            vocabulary_depth=vocab_depth,
            vocabulary_seed=vocab_seed,
            orb_num_features=num_features,
            orb_scale_factor=scale_factor,
            orb_num_levels=num_levels,
            orb_ini_threshold_fast=ini_threshold,
            orb_min_threshold_fast=min_threshold
        )
        subject.pk = ObjectId()
        subject.build_vocabulary([self.image_collection], self.path_manager.get_output_dir())

        self.assertTrue(mock_vocabulary_builder_class.called)
        self.assertEqual(mock.call(
            num_features, scale_factor, num_levels, 31, 0, 2, 1, 31, min(min_threshold, ini_threshold)
        ), mock_vocabulary_builder_class.call_args)

        vocab_path = self.path_manager.get_output_dir() / subject.vocabulary_file
        self.assertTrue(mock_builder.add_image.called)
        self.assertTrue(mock_builder.build_vocabulary.called)
        self.assertEqual(mock.call(
            str(vocab_path),
            branchingFactor=branching_factor,
            numLevels=vocab_depth,
            seed=vocab_seed
        ), mock_builder.build_vocabulary.call_args)

    def test_building_allows_orbslam_to_run(self):
        subject = OrbSlam2(
            mode=SensorMode.STEREO,
            vocabulary_branching_factor=6,
            vocabulary_depth=4,
            vocabulary_seed=158627,
            orb_num_features=1000,
            orb_scale_factor=1.2,
            orb_num_levels=7,
            orb_ini_threshold_fast=12,
            orb_min_threshold_fast=7
        )
        subject.pk = ObjectId()  # object needs to be "saved" to generate a vocab filename
        subject.build_vocabulary([self.image_collection], self.path_manager.get_output_dir())

        subject.resolve_paths(self.path_manager)
        subject.set_camera_intrinsics(self.image_collection.camera_intrinsics, self.max_time / self.num_frames)
        subject.set_stereo_offset(self.image_collection.stereo_offset)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for timestamp, image in self.image_collection:
            subject.process_image(image, timestamp)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertTrue(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(len(self.image_collection), len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(self.image_collection.timestamps[idx], frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)


@unittest.skip
class TestORBSLAM2VocabularyBuilder(unittest.TestCase):
    temp_folder = Path(__file__).parent / 'temp-test-orbslam2'
    vocab_file = Path(__file__).parent / 'ORBvoc-test.txt'

    def tearDown(self) -> None:
        if self.vocab_file.exists():
            self.vocab_file.unlink()

    @classmethod
    def setUpClass(cls):
        cls.temp_folder.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder)

    def test_produces_vocab_usable_by_orbslam_by_default(self):
        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, self.vocab_file, 10, 6)
        result = run_orbslam_with_vocab(self.vocab_file, self.temp_folder, seed=1000, num_frames=25)

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertTrue(result.success)
        self.assertEqual(25, len(result.results))
        self.assertTrue(any(frame_result.tracking_state is TrackingState.OK for frame_result in result.results))

    def test_can_have_narrow_branching_factor(self):
        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, self.vocab_file, branching_factor=2, depth=6)
        result = run_orbslam_with_vocab(self.vocab_file, self.temp_folder, seed=1000, num_frames=25)

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertTrue(result.success)
        self.assertEqual(25, len(result.results))
        self.assertTrue(any(frame_result.tracking_state is TrackingState.OK for frame_result in result.results))

    def test_can_have_shallow_tree(self):
        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, self.vocab_file, 10, 3)
        result = run_orbslam_with_vocab(self.vocab_file, self.temp_folder, seed=1000, num_frames=25)

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertTrue(result.success)
        self.assertEqual(25, len(result.results))
        self.assertTrue(any(frame_result.tracking_state is TrackingState.OK for frame_result in result.results))

    def test_can_have_narrow_tree(self):
        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, self.vocab_file, 2, 8, num_variants=100)
        result = run_orbslam_with_vocab(self.vocab_file, self.temp_folder, seed=1000, num_frames=25)

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertTrue(result.success)
        self.assertEqual(25, len(result.results))
        self.assertTrue(any(frame_result.tracking_state is TrackingState.OK for frame_result in result.results))

    def test_can_have_wide_tree(self):
        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, self.vocab_file, 20, 2, num_variants=100)
        result = run_orbslam_with_vocab(self.vocab_file, self.temp_folder, seed=1000, num_frames=25)

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertTrue(result.success)
        self.assertEqual(25, len(result.results))
        self.assertTrue(any(frame_result.tracking_state is TrackingState.OK for frame_result in result.results))

    def test_can_have_small_tree(self):
        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, self.vocab_file, 3, 3)
        result = run_orbslam_with_vocab(self.vocab_file, self.temp_folder, seed=1000, num_frames=25)

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertTrue(result.success)
        self.assertEqual(25, len(result.results))
        self.assertTrue(any(frame_result.tracking_state is TrackingState.OK for frame_result in result.results))

    def test_can_have_large_tree(self):
        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, self.vocab_file, 8, 7, num_variants=100)
        result = run_orbslam_with_vocab(self.vocab_file, self.temp_folder, seed=1000, num_frames=25)

        self.assertIsInstance(result, SLAMTrialResult)
        self.assertTrue(result.success)
        self.assertEqual(25, len(result.results))
        self.assertTrue(any(frame_result.tracking_state is TrackingState.OK for frame_result in result.results))

    def test_can_control_seed(self):
        temp_dir = Path(__file__).parent
        vocab_file_1 = temp_dir / 'seeded_vocab_1.txt'
        vocab_file_2 = temp_dir / 'seeded_vocab_2.txt'
        vocab_file_3 = temp_dir / 'seeded_vocab_3.txt'

        # Building the vocab changes the stored set of descriptors. So we need to re-feed it images each time
        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, vocab_file_1, 3, 4, num_variants=10, seed=0)

        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, vocab_file_2, 3, 4, num_variants=10, seed=0)

        vocab_builder = VocabularyBuilder()
        create_vocab(vocab_builder, vocab_file_3, 3, 4, num_variants=10, seed=22)

        # Read the data from the files
        with vocab_file_1.open('rb') as fp:
            file_contents_1 = fp.read()
        vocab_file_1.unlink()

        with vocab_file_2.open('rb') as fp:
            file_contents_2 = fp.read()
        vocab_file_2.unlink()

        with vocab_file_3.open('rb') as fp:
            file_contents_3 = fp.read()
        vocab_file_3.unlink()

        self.assertEqual(file_contents_1, file_contents_2)
        self.assertNotEqual(file_contents_1, file_contents_3)
        self.assertNotEqual(file_contents_2, file_contents_3)


def create_vocab(vocab_builder: VocabularyBuilder, vocab_path: Path,
                 branching_factor: int, depth: int, num_variants: int = 10, seed=100):
    """
    Tiny script to create a vocabulary from the demo image builder
    This gives me a vocab designed to handle the synthetic images I throw at it while testing.
    :return:
    """
    total_time = 10  # seconds
    num_frames = 20  # Total frames to pull
    speed = 3.0      # Units / second
    for img_seed in range(num_variants):
        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR, seed=img_seed,
            length=total_time * speed
        )
        for idx in range(num_frames):
            time = total_time * idx / num_frames
            image = image_builder.create_frame(time)
            vocab_builder.add_image(image.pixels)
    vocab_builder.build_vocabulary(str(vocab_path), branching_factor, depth, seed)


def run_orbslam_with_vocab(vocab_path, temp_folder, seed=1000, num_frames=25):
    # Actually run the system using mocked images
    max_time = 50
    speed = 0.1
    path_manager = PathManager([Path(__file__).parent], temp_folder)
    image_builder = DemoImageBuilder(
        seed=seed,
        mode=ImageMode.STEREO, stereo_offset=0.15,
        width=320, height=240, num_stars=500,
        length=max_time * speed, speed=speed,
        min_size=4, max_size=50
    )
    subject = OrbSlam2(
        vocabulary_file=vocab_path,
        mode=SensorMode.STEREO,
        orb_ini_threshold_fast=12,
        orb_min_threshold_fast=7
    )
    subject.resolve_paths(path_manager)
    subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)
    subject.set_stereo_offset(image_builder.get_stereo_offset())

    subject.start_trial(ImageSequenceType.SEQUENTIAL)
    for idx in range(num_frames):
        time = max_time * idx / num_frames
        image = image_builder.create_frame(time)
        subject.process_image(image, time)
    return subject.finish_trial()
