from enum import Enum
import numpy as np
from arvet.util.transform import Transform
import arvet.metadata.image_metadata as imeta
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet.core.image import Image, StereoImage


class ImageMode(Enum):
    MONOCULAR = 0
    STEREO = 1
    RGBD = 2


class DemoImageBuilder:

    def __init__(
            self, mode: ImageMode = ImageMode.MONOCULAR, seed: int = 0, width: int = 320, height: int = 240,
            focal_length: float = None, stereo_offset: float = 0.15,
            num_stars: int = 300, close_ratio: float = 0.5, min_size: float = 4.0, max_size: float = 50.0,
            speed: float = 5.0, length: float = 60, corridor_width: float = 2.0, colour: bool = False
    ):
        self.mode = mode
        self.width = width
        self.height = height
        self.speed = speed
        self.focal_length = max(focal_length, 1.0) if focal_length is not None else width / 2
        self.stereo_offset = stereo_offset
        self.colour = bool(colour)
        random = np.random.default_rng(seed=seed)

        # Z values for stars in two groups
        # The first are those that can only be seen during part of the sequence, and the other can be seen for all of it
        # By similar triangles, the depth at which a point is visible for the entire sequence is
        # (0.5 * width) / focal_length = (0.5 * length) / z
        # z = (length / width) * focal_length
        close_z = self.focal_length * length / self.width
        z_values = sorted((
            close_z + random.exponential(2 * close_z, size=int((1 - close_ratio) * num_stars))
        ), reverse=True)
        # z values for stars within the range of the motion, evenly distributed
        z_values = z_values + list(np.arange(close_z, corridor_width, -1 * close_z / (num_stars - len(z_values))))

        # Create some random rectangular sprites whose midpoints are in frame from the first frame
        lim_x = width / (2 * self.focal_length)
        lim_y = height / (2 * self.focal_length)
        self.stars = [{
            'pos': (
                random.uniform(-lim_x * z_value, lim_x * z_value + length),
                random.uniform(-lim_y * z_value, lim_y * z_value),
                z_value
            ),
            'colour': random.integers(20, 256, size=3 if colour else 1)
        } for idx, z_value in enumerate(z_values) if z_value > 0]

        # Having placed the stars, choose a size such that they are visible but not too close to the camera
        for star in self.stars:
            x, y, z = star['pos']
            # Since we're not moving forward, we need only ensure the star is visible based on its depth
            ratio = z / self.focal_length
            star['width'] = random.uniform(min_size * ratio, max_size * ratio)
            star['height'] = random.uniform(min_size * ratio, max_size * ratio)

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
        img_shape = (self.height, self.width, 3) if self.colour else (self.height, self.width)
        frame = np.zeros(img_shape, dtype=np.uint8)
        depth = None
        if self.mode is ImageMode.RGBD:
            depth = (1000 + 2 * len(self.stars)) * np.ones((self.height, self.width), dtype=np.float64)
        f = self.focal_length
        cx = frame.shape[1] / 2
        cy = frame.shape[0] / 2

        for star in self.stars:
            x, y, z = star['pos']
            x -= self.speed * time
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
        if self.mode is ImageMode.STEREO:
            right_frame = np.zeros(img_shape, dtype=np.uint8)
            for star in self.stars:
                x, y, z = star['pos']
                x -= self.stereo_offset + self.speed * time
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
                image_group='test',
                metadata=metadata,
                right_pixels=right_frame,
                right_metadata=right_metadata
            )

        if depth is not None:
            return Image(pixels=frame, image_group='test', depth=depth, metadata=metadata)
        return Image(pixels=frame, image_group='test', metadata=metadata)

    def visualise_sequence(self, max_time: float, frame_interval: float = 1):
        import matplotlib.pyplot as plt
        from matplotlib.animation import ArtistAnimation

        fig = plt.figure()

        images = []
        for time in np.linspace(0, max_time, int(max_time / frame_interval)):
            image = self.create_frame(time)
            ax_img = plt.imshow(image.pixels, cmap='gray', animated=True)
            images.append([ax_img])

        ani = ArtistAnimation(fig, images, interval=frame_interval * 1000, blit=True, repeat_delay=2 * max_time)
        plt.show()

