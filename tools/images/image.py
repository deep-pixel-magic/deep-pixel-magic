import tensorflow as tf


def load_png(file, batched=True):
    """Loads a PNG image file into a tensorflow tensor.

    Args:
        file: The file path.
        batched: Whether or not to add a batch dimension.

    Returns:
        The image tensor.
    """

    file = tf.io.read_file(file)
    image = tf.image.decode_png(file, channels=3)
    image = tf.cast(image, tf.float32)

    if batched:
        image = tf.expand_dims(image, axis=0)

    return image


def load_jpg(file, batched=True):
    """Loads a JPG image file into a tensorflow tensor.

    Args:
        file: The file path.
        batched: Whether or not to add a batch dimension.

    Returns:
        The image tensor.
    """

    file = tf.io.read_file(file)
    image = tf.image.decode_jpeg(file, channels=3)
    image = tf.cast(image, tf.float32)

    if batched:
        image = tf.expand_dims(image, axis=0)

    return image
