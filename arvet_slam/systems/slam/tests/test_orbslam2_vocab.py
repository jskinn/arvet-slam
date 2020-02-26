import unittest
import shutil
from pathlib import Path
from orbslam2 import VocabularyBuilder
from arvet.config.path_manager import PathManager
from arvet.core.sequence_type import ImageSequenceType
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult
from arvet_slam.systems.slam.orbslam2 import OrbSlam2, SensorMode
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode


class TestORBSLAM2Vocab(unittest.TestCase):
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
