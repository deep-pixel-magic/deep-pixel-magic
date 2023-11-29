import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError

from models.edsr.edsr import EdsrNetwork
from models.edsr.edsr_trainer import EdsrTrainer


def main():
    data_dir_low_res = "./.data/div2k/DIV2K_valid_LR_bicubic/X4"
    data_dir_high_res = "./.data/div2k/DIV2K_valid_HR"

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
    element_training = iterator_training.next()

    # f, plots = plt.subplots(1, 2)
    # plots[0].imshow(element_training[0])
    # plots[1].imshow(element_training[1])
    # plt.show()

    model = EdsrNetwork().build(scale=4, num_filters=32, num_residual_blocks=8)

    latest = tf.train.latest_checkpoint('./colab')
    model.load_weights(latest)

    prediction = model.predict(element_training[0])
    prediced_img = tf.squeeze(prediction)
    # prediced_img = tf.cast(prediced_img, tf.int32)

    img = Image.open(
        "./.data/div2k/DIV2K_valid_LR_bicubic/X4/0801x4.png").convert("RGB")

    new_width = img.size[0] * 4
    new_height = img.size[1] * 4

    img = img.resize((new_width, new_height), Image.BICUBIC)

    figure, plots = plt.subplots(1, 3, sharex=True, sharey=True)
    plots[0].imshow(img)
    plots[1].imshow(prediced_img)
    plots[2].imshow(tf.squeeze(element_training[1]))
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
