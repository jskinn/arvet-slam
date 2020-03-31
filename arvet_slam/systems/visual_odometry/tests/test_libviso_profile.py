import unittest
from arvet.core.sequence_type import ImageSequenceType
from arvet_slam.systems.visual_odometry.libviso2 import LibVisOStereoSystem, LibVisOMonoSystem
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode


@unittest.skip("Not running profiling")
class TestRunLibVisOProfile(unittest.TestCase):
    num_frames = 1000
    max_time = 50
    speed = 0.1

    def test_profile_mono(self, ):
        import cProfile as profile

        stats_file = "libviso_mono.prof"

        system = LibVisOMonoSystem()

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR, stereo_offset=0.15,
            width=640, height=480, num_stars=150,
            length=self.max_time * self.speed, speed=self.speed,
            close_ratio=0.6, min_size=10, max_size=100
        )

        profile.runctx("run_libviso(system, image_builder, self.num_frames, self.max_time, 0)",
                       locals=locals(), globals=globals(), filename=stats_file)

    def test_profile_stereo(self, ):
        import cProfile as profile

        stats_file = "libviso_stereo.prof"

        system = LibVisOStereoSystem()

        image_builder = DemoImageBuilder(
            mode=ImageMode.STEREO, stereo_offset=0.15,
            width=640, height=480, num_stars=150,
            length=self.max_time * self.speed, speed=self.speed,
            close_ratio=0.6, min_size=10, max_size=100
        )

        profile.runctx("run_libviso(system, image_builder, self.num_frames, self.max_time, 0)",
                       locals=locals(), globals=globals(), filename=stats_file)


def run_libviso(system, image_builder, num_frames, max_time, seed=0):
    system.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)
    system.set_stereo_offset(image_builder.get_stereo_offset())
    system.start_trial(ImageSequenceType.SEQUENTIAL, seed=seed)
    for idx in range(num_frames):
        time = max_time * idx / num_frames
        image = image_builder.create_frame(time)
        system.process_image(image, time)
    system.finish_trial()
