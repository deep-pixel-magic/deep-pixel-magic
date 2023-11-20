import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError

from models.edsr.edsr import EdsrNetwork
from models.edsr.edsr_trainer import EdsrTrainer


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(
        shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(
        shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def main():
    out_dir = 'out/models/'
    out_file = os.path.join(out_dir, 'model.weights')

    data_dir_low_res = "./.data/div2k/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.data/div2k/DIV2K_train_HR"

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

    # Map low resolution images to matching high resolution images.
    data_set_training = tf.data.Dataset.zip(
        data_set_low_res, data_set_high_res)

    data_set_training = data_set_training.map(lambda lr, hr: random_crop(
        lr, hr, scale=4), num_parallel_calls=AUTOTUNE)

    # iterator_low_res = data_set_low_res.as_numpy_iterator()
    # iterator_high_res = data_set_high_res.as_numpy_iterator()
    # iterator_training = data_set_training.as_numpy_iterator()

    # element_low = iterator_low_res.next()
    # element_high = iterator_high_res.next()
    # element_training = iterator_training.next()

    # f, plots = plt.subplots(2, 2)
    # plots[0, 0].imshow(element_low)
    # plots[0, 1].imshow(element_high)
    # plots[1, 0].imshow(element_training[0])
    # plots[1, 1].imshow(element_training[1])
    # plt.show()

    data_set_training = data_set_training.batch(16)
    data_set_training = data_set_training.repeat()
    data_set_training = data_set_training.prefetch(buffer_size=AUTOTUNE)

    model = EdsrNetwork().build(scale=4, num_filters=32, num_residual_blocks=8)

    # optimizer = optimizers.Adam(learning_rate=PiecewiseConstantDecay(
    #     boundaries=[200000], values=[1e-4, 5e-5]))
    # model.compile(optimizer=optimizer, loss='mean_absolute_error')
    # model.fit(data_set_training, epochs=300, steps_per_epoch=800)

    EdsrTrainer(model=model, loss=MeanAbsoluteError()).train(
        data_set_training, epochs=300, steps=800)

    model.save_weights(out_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
