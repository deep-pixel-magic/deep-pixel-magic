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


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=4):
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


def random_flip(low_res_img, high_res_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (low_res_img, high_res_img),
                   lambda: (tf.image.flip_left_right(low_res_img),
                            tf.image.flip_left_right(high_res_img)))


def random_rotate(low_res_img, high_res_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(low_res_img, rn), tf.image.rot90(high_res_img, rn)


def main():
    out_dir = './.cache/models/edsr/'
    out_file = os.path.join(out_dir, 'model.pkg')

    data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_train_HR"

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

    # data_set_training.cache('./.cache/data/cached/dataset.cache')

    data_set_training = data_set_training.map(lambda lr, hr: random_crop(
        lr, hr, scale=4), num_parallel_calls=AUTOTUNE)

    data_set_training = data_set_training.map(
        lambda lr, hr: random_flip(lr, hr), num_parallel_calls=AUTOTUNE)
    data_set_training = data_set_training.map(
        lambda lr, hr: random_rotate(lr, hr), num_parallel_calls=AUTOTUNE)

    batch_size = 16

    initial_data_set_cardinality = data_set_training.cardinality().numpy()
    batched_data_set_cardinality = initial_data_set_cardinality // batch_size

    data_set_training = data_set_training.batch(batch_size)
    data_set_training = data_set_training.repeat()
    data_set_training = data_set_training.prefetch(buffer_size=AUTOTUNE)

    model = EdsrNetwork().build(scale=4, num_filters=64, num_residual_blocks=16)

    trainer = EdsrTrainer(model=model, learning_rate=PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]))

    trainer.train(data_set_training, epochs=11,
                  steps=batched_data_set_cardinality)

    model.save_weights(out_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
