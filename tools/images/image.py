import tensorflow as tf


def load_png(file, batched=True):
    file = tf.io.read_file(file)
    image = tf.image.decode_png(file, channels=3)

    if batched:
        image = tf.expand_dims(image, axis=0)

    return image
