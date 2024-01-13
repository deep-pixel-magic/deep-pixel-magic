import tensorflow as tf


def load_png(file, batched=True):
    file = tf.io.read_file(file)
    image = tf.image.decode_png(file, channels=3)
    image = tf.cast(image, tf.float32)

    if batched:
        image = tf.expand_dims(image, axis=0)

    return image

def normalize_img(img, mean):
    return (img - mean) / 127.5

def denormalize_img(img, mean):
    return (img * 127.5) + mean
