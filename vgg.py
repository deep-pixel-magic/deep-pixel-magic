import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.keras.applications.vgg19 import preprocess_input

from models.vgg.vgg import VggBuilder


def main():

    data_dir_low_res = "./.cache/data/DIV2K_valid_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_valid_HR"

    image_files_low_res = sorted(tf.io.gfile.glob(data_dir_low_res + "/*.png"))
    image_files_high_res = sorted(tf.io.gfile.glob(
        data_dir_high_res + "/*.png"))

    data_set_low_res = tf.data.Dataset.from_tensor_slices(image_files_low_res)
    data_set_low_res = data_set_low_res.map(tf.io.read_file)
    data_set_low_res = data_set_low_res.map(lambda x: tf.image.decode_png(
        x, channels=3), num_parallel_calls=AUTOTUNE)

    data_set_high_res = tf.data.Dataset.from_tensor_slices(
        image_files_high_res)
    data_set_high_res = data_set_high_res.map(tf.io.read_file)
    data_set_high_res = data_set_high_res.map(lambda x: tf.image.decode_png(
        x, channels=3), num_parallel_calls=AUTOTUNE)

    data_set_training = tf.data.Dataset.zip(
        data_set_low_res, data_set_high_res)

    data_set_training = data_set_training.batch(1)

    iterator_training = data_set_training.as_numpy_iterator()
    element_training = list(iterator_training)[9]

    vgg = VggBuilder(layer='block4_conv2').build(
        input_shape=(None, None, 3))

    vgg_in = preprocess_input(element_training[1])
    prediction = vgg(vgg_in)

    plt.figure()
    plt.imshow(prediction[0, :, :, 0])
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
