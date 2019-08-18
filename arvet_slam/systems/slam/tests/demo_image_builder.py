import numpy as np
from arvet.util.transform import Transform
import arvet.metadata.image_metadata as imeta
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.image import Image, StereoImage
from arvet_slam.systems.slam.orbslam2 import SensorMode


class DemoImageBuilder:

    def __init__(
            self, mode: SensorMode, seed: int = 0, width: int = 320, height: int = 240,
            focal_length: float = None, stereo_offset: float = 0.15,
            num_stars: int = 300, close_ratio: float = 0.5, min_size: float = 4.0, max_size: float = 50.0,
            speed: float = 5.0, length: float = 60, corridor_width: float = 2.0
    ):
        self.mode = mode
        self.width = width
        self.height = height
        self.speed = speed
        self.focal_length = max(focal_length, 1.0) if focal_length is not None else width / 2
        self.stereo_offset = stereo_offset
        random = np.random.RandomState(seed=seed)

        # z values for stars beyond the end of the motion
        z_values = sorted((
            random.uniform(length + corridor_width, 10 * (length + corridor_width))
            for _ in range(int((1 - close_ratio) * num_stars))
        ), reverse=True)
        # z values for stars within the range of the motion, evenly distributed
        z_values = z_values + list(np.arange(length, 0, -1 * length / (num_stars - len(z_values))))

        # Create some random rectangular sprites whose midpoints are in frame from the first frame
        lim_x = width / (2 * self.focal_length)
        lim_y = height / (2 * self.focal_length)
        self.stars = [{
            'pos': (
                random.uniform(-lim_x * z_value, lim_x * z_value),
                random.uniform(-lim_y * z_value, lim_y * z_value),
                z_value
            ),
            'colour': random.randint(20, 256)
        } for idx, z_value in enumerate(z_values) if z_value > 0]

        # Having placed the stars, choose a size such that they are visible but not too close to the camera
        for star in self.stars:
            x, y, z = star['pos']
            if z > length:
                # Distant, don't need to worry about corridor
                width_at_nearest = (z - length) / self.focal_length
                star['width'] = random.uniform(min_size * width_at_nearest, max_size * width_at_nearest)
                star['height'] = random.uniform(min_size * width_at_nearest, max_size * width_at_nearest)
            else:
                # closest z is when z = 2 * x * f / W or z = 2 * y * f / H
                # min_dim = min_size * z / f
                # cancel f from both equations
                closest_z = 2 * abs(x) / width
                star['width'] = random.uniform(min_size * closest_z, max_size * closest_z)
                closest_z = 2 * abs(y) / height
                star['height'] = random.uniform(min_size * closest_z, max_size * closest_z)

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=self.width,
            height=self.height,
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.width / 2,
            cy=self.height / 2
        )

    def get_stereo_offset(self) -> Transform:
        return Transform([0, -1 * self.stereo_offset, 0])

    def create_frame(self, time: float) -> Image:
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        depth = None
        if self.mode is SensorMode.RGBD:
            depth = (1000 + 2 * len(self.stars)) * np.ones((self.height, self.width), dtype=np.float16)
        f = self.focal_length
        cx = frame.shape[1] / 2
        cy = frame.shape[0] / 2

        for star in self.stars:
            x, y, z = star['pos']
            z -= self.speed * time
            if z <= 0:
                break   # Stars are sorted by z value, so once they're past the camera, stop.

            left = int(np.round(f * ((x - star['width'] / 2) / z) + cx))
            right = int(np.round(f * ((x + star['width'] / 2) / z) + cx))

            top = int(np.round(f * ((y - star['height'] / 2) / z) + cy))
            bottom = int(np.round(f * ((y + star['height'] / 2) / z) + cy))

            left = max(0, min(frame.shape[1], left))
            right = max(0, min(frame.shape[1], right))
            top = max(0, min(frame.shape[0], top))
            bottom = max(0, min(frame.shape[0], bottom))

            frame[top:bottom, left:right] = star['colour']
            if depth is not None:
                depth[top:bottom, left:right] = z

        metadata = imeta.make_metadata(
            pixels=frame,
            depth=depth,
            source_type=imeta.ImageSourceType.SYNTHETIC,
            camera_pose=Transform(
                location=[time * self.speed, 0, 0],
                rotation=[0, 0, 0, 1]
            ),
            intrinsics=CameraIntrinsics(
                width=frame.shape[1],
                height=frame.shape[0],
                fx=f,
                fy=f,
                cx=cx,
                cy=cy
            )
        )

        # If we're building a stereo image, make the right image
        if self.mode is SensorMode.STEREO:
            right_frame = np.zeros((self.height, self.width), dtype=np.uint8)
            for star in self.stars:
                x, y, z = star['pos']
                x -= self.stereo_offset
                z -= self.speed * time
                if z <= 0:
                    break

                left = int(np.round(f * ((x - star['width'] / 2) / z) + cx))
                right = int(np.round(f * ((x + star['width'] / 2) / z) + cx))

                top = int(np.round(f * ((y - star['height'] / 2) / z) + cy))
                bottom = int(np.round(f * ((y + star['height'] / 2) / z) + cy))

                left = max(0, min(frame.shape[1], left))
                right = max(0, min(frame.shape[1], right))
                top = max(0, min(frame.shape[0], top))
                bottom = max(0, min(frame.shape[0], bottom))

                right_frame[top:bottom, left:right] = star['colour']
            right_metadata = imeta.make_right_metadata(
                pixels=right_frame,
                left_metadata=metadata,
                source_type=imeta.ImageSourceType.SYNTHETIC,
                camera_pose=Transform(
                    location=[time * self.speed, -1 * self.stereo_offset, 0],
                    rotation=[0, 0, 0, 1]
                ),
                intrinsics=CameraIntrinsics(
                    width=frame.shape[1],
                    height=frame.shape[0],
                    fx=f,
                    fy=f,
                    cx=cx,
                    cy=cy
                )
            )
            return StereoImage(
                pixels=frame,
                metadata=metadata,
                right_pixels=right_frame,
                right_metadata=right_metadata
            )

        return Image(
            pixels=frame,
            depth=depth,
            metadata=metadata
        )
