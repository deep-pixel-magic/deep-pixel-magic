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

    sample_img = 2
    element_training = list(iterator_training)[sample_img]

    # f, plots = plt.subplots(1, 2)
    # plots[0].imshow(element_training[0])
    # plots[1].imshow(element_training[1])
    # plt.show()

    model = EdsrNetwork().build(scale=4, num_filters=64, num_residual_blocks=16)

    latest = tf.train.latest_checkpoint('./.cache/models/edsr')
    model.load_weights(latest)

    prediction = model.predict(element_training[0])

    predicted_img = tf.squeeze(prediction)
    # predicted_img = tf.clip_by_value(predicted_img, 0, 255)
    # predicted_img = tf.round(predicted_img)
    predicted_img = tf.cast(predicted_img, tf.uint8)

    img_to_save = Image.fromarray(predicted_img.numpy())
    img_to_save.save("prediction.png", "PNG")

    img = Image.open(
        f"./.cache/data/DIV2K_valid_LR_bicubic/X4/08{sample_img + 1:02}x4.png").convert("RGB")

    img.save("original.png", "PNG")

    new_width = img.size[0] * 4
    new_height = img.size[1] * 4

    img = img.resize((new_width, new_height), Image.BICUBIC)

    figure, plots = plt.subplots(1, 3, sharex=True, sharey=True)
    plots[0].imshow(img)
    plots[1].imshow(predicted_img)
    plots[2].imshow(tf.squeeze(element_training[1]))
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
