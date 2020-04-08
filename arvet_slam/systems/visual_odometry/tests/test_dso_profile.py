import unittest
from arvet.core.sequence_type import ImageSequenceType
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode
from arvet_slam.systems.visual_odometry.direct_sparse_odometry import DSO, RectificationMode


@unittest.skip("Not running profiling")
class TestRunDSOMProfile(unittest.TestCase):
    num_frames = 1000
    max_time = 50
    speed = 0.1

    def test_profile(self, ):
        import cProfile as profile

        stats_file = "dso.prof"

        system = DSO(
            rectification_mode=RectificationMode.NONE,
            # These should be irrelevant
            rectification_intrinsics=CameraIntrinsics(
                width=320,
                height=240,
                fx=160,
                fy=160,
                cx=160,
                cy=120
            )
        )

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=self.max_time * self.speed, speed=self.speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        profile.runctx("run_dso(system, image_builder, self.num_frames, self.max_time)",
                       locals=locals(), globals=globals(), filename=stats_file)


def run_dso(system, image_builder, num_frames, max_time):
    system.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)
    system.start_trial(ImageSequenceType.SEQUENTIAL)
    for idx in range(num_frames):
        time = max_time * idx / num_frames
        image = image_builder.create_frame(time)
        system.process_image(image, time)
    system.finish_trial()
