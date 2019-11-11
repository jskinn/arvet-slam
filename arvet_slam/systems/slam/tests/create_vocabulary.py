from orbslam2 import VocabularyBuilder
from arvet_slam.systems.slam.tests.demo_image_builder import DemoImageBuilder, ImageMode


def create_vocab(vocab_path="ORBvoc-synth.txt"):
    """
    Tiny script to create a vocabulary from the demo image builder
    This gives me a vocab designed to handle the synthetic images I throw at it while testing.
    :return:
    """
    total_time = 10  # seconds
    num_frames = 50
    image_builder = DemoImageBuilder(mode=ImageMode.MONOCULAR)
    vocab_builder = VocabularyBuilder()
    for idx in range(num_frames):
        time = total_time * idx / num_frames
        image = image_builder.create_frame(time)
        vocab_builder.add_image(image.pixels)
    vocab_builder.build_vocabulary(str(vocab_path))


if __name__ == '__main__':
    # Make this module runnable
    create_vocab()
