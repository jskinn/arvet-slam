import unittest
from pathlib import Path
import shutil

from arvet.config.path_manager import PathManager
from arvet.core.sequence_type import ImageSequenceType
from arvet_slam.systems.slam.orbslam2 import OrbSlam2, SensorMode
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode
from arvet_slam.systems.slam.tests.create_vocabulary import create_vocab


@unittest.skip("Not running profiling")
class TestRunORBSLAMProfile(unittest.TestCase):
    num_frames = 1000
    max_time = 50
    speed = 0.1
    temp_folder = 'temp-test-orbslam2'
    vocab_path = Path(__file__).parent / 'ORBvoc-synth.txt'

    @classmethod
    def setUpClass(cls):
        (Path(__file__).parent / cls.temp_folder).mkdir(exist_ok=True)
        if not cls.vocab_path.exists():  # If there is no vocab file, make one
            print("Creating vocab file, this may take a while...")
            create_vocab(cls.vocab_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder)

    def test_profile_mono(self, ):
        import cProfile as profile

        stats_file = "orbslam_mono.prof"

        path_manager = PathManager([Path(__file__).parent], self.temp_folder)
        system = OrbSlam2(
            vocabulary_file=self.vocab_path,
            mode=SensorMode.MONOCULAR,
            orb_num_features=1000,
            orb_num_levels=8,
            orb_scale_factor=1.2,
            orb_ini_threshold_fast=7,
            orb_min_threshold_fast=12
        )

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            width=640, height=480, num_stars=150,
            length=self.max_time * self.speed, speed=self.speed,
            close_ratio=0.6, min_size=10, max_size=100
        )

        profile.runctx("run_orbslam(system, image_builder, path_manager, self.num_frames, self.max_time)",
                       locals=locals(), globals=globals(), filename=stats_file)

    def test_profile_rgbd(self, ):
        import cProfile as profile

        stats_file = "orbslam_rgbd.prof"

        path_manager = PathManager([Path(__file__).parent], self.temp_folder)
        system = OrbSlam2(
            vocabulary_file=self.vocab_path,
            mode=SensorMode.RGBD,
            orb_ini_threshold_fast=12,
            orb_min_threshold_fast=7
        )

        image_builder = DemoImageBuilder(
            mode=ImageMode.RGBD, stereo_offset=0.15,
            width=320, height=240, num_stars=500,
            length=self.max_time * self.speed, speed=self.speed,
            min_size=4, max_size=50
        )

        profile.runctx("run_orbslam(system, image_builder, path_manager, self.num_frames, self.max_time)",
                       locals=locals(), globals=globals(), filename=stats_file)

    def test_profile_stereo(self, ):
        import cProfile as profile

        stats_file = "orbslam_stereo.prof"

        path_manager = PathManager([Path(__file__).parent], self.temp_folder)
        system = OrbSlam2(
            vocabulary_file=self.vocab_path,
            mode=SensorMode.STEREO,
            orb_ini_threshold_fast=12,
            orb_min_threshold_fast=7
        )

        image_builder = DemoImageBuilder(
            mode=ImageMode.STEREO, stereo_offset=0.15,
            width=320, height=240, num_stars=500,
            length=self.max_time * self.speed, speed=self.speed,
            min_size=4, max_size=50
        )

        profile.runctx("run_orbslam(system, image_builder, path_manager, self.num_frames, self.max_time)",
                       locals=locals(), globals=globals(), filename=stats_file)


def run_orbslam(system, image_builder, path_manager, num_frames, max_time):
    system.resolve_paths(path_manager)
    system.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)
    system.set_stereo_offset(image_builder.get_stereo_offset())
    system.start_trial(ImageSequenceType.SEQUENTIAL)
    for idx in range(num_frames):
        time = max_time * idx / num_frames
        image = image_builder.create_frame(time)
        system.process_image(image, time)
    system.finish_trial()
