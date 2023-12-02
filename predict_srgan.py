import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError

from models.srgan.srgan import SrganNetwork

from tools.datasets.div2k.tensorflow import TensorflowDataset, TensorflowPreprocessor


def main():
    data_dir_low_res = "./.cache/data/DIV2K_valid_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_valid_HR"

    bundle = TensorflowDataset(data_dir_low_res, data_dir_high_res)
    dataset = TensorflowPreprocessor(bundle).preprocess(
        batch_size=16, crop_size=96, scale=4)

    iterator_training = dataset.as_numpy_iterator()

    element = iterator_training.next()

    figure, plots = plt.subplots(1, 2, sharex=True, sharey=True)
    plots[0].imshow(element[0][0])
    plots[1].imshow(element[1][0])
    plt.show()

    # sample_img = 2
    # element_training = list(iterator_training)[sample_img]

    # f, plots = plt.subplots(1, 2)
    # plots[0].imshow(element_training[0])
    # plots[1].imshow(element_training[1])
    # plt.show()

    # model = SrganNetwork().build()

    # latest = tf.train.latest_checkpoint('./.cache/models/srgan')
    # model.load_weights(latest)

    # prediction = model.predict(element_training[0])

    # predicted_img = tf.squeeze(prediction)
    # predicted_img = tf.cast(predicted_img, tf.uint8)

    # img_to_save = Image.fromarray(predicted_img.numpy())
    # img_to_save.save("prediction.png", "PNG")

    # img = Image.open(
    #     f"./.cache/data/DIV2K_valid_LR_bicubic/X4/08{sample_img + 1:02}x4.png").convert("RGB")

    # img.save("original.png", "PNG")

    # new_width = img.size[0] * 4
    # new_height = img.size[1] * 4

    # img = img.resize((new_width, new_height), Image.BICUBIC)

    # figure, plots = plt.subplots(1, 3, sharex=True, sharey=True)
    # plots[0].imshow(img)
    # plots[1].imshow(predicted_img)
    # plots[2].imshow(tf.squeeze(element_training[1]))
    # plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
