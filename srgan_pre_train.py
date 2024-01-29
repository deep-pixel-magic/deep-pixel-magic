import os
import sys

import tensorflow as tf

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from models.srgan.data_processing import normalize_input_lr, normalize_input_hr
from models.srgan.srgan_pre_trainer import SrganPreTrainer
from models.srgan.srgan_network import SrganNetwork
from tools.datasets.div2k.tensorflow import TensorflowImageDataset, TensorflowImageDatasetBundle, TensorflowImagePreprocessor


def main():
    data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_train_HR"

    batch_size = 16
    crop_size = 96

    dataset_lr = TensorflowImageDataset(
        data_dir_low_res, normalizer=lambda x: normalize_input_lr(x))
    dataset_hr = TensorflowImageDataset(
        data_dir_high_res, normalizer=lambda x: normalize_input_hr(x))

    bundle = TensorflowImageDatasetBundle(dataset_lr, dataset_hr)

    dataset = TensorflowImagePreprocessor(bundle).preprocess(
        batch_size=batch_size, crop_size=crop_size, scale=4)

    initial_data_set_cardinality = bundle.num()
    batched_data_set_cardinality = initial_data_set_cardinality // batch_size

    num_steps_per_epoch = batched_data_set_cardinality
    num_pre_train_epochs = 1000

    with tf.device('/GPU:0'):

        generator = SrganNetwork().build(num_filters=64, num_residual_blocks=16)

        decay_boundaries = [
            (num_pre_train_epochs // 2) * num_steps_per_epoch,
        ]

        decay_values = [
            1e-4,
            1e-5,
        ]

        generator_lr = PiecewiseConstantDecay(
            boundaries=decay_boundaries, values=decay_values)

        pre_trainer = SrganPreTrainer(
            generator=generator, learning_rate=generator_lr)

        pre_trainer.train(dataset, epochs=num_pre_train_epochs,
                          steps=batched_data_set_cardinality)

        os.makedirs('./.cache/models/srgan', exist_ok=True)
        generator.save_weights(
            './.cache/models/srgan/generator_pre_trained.weights.h5', overwrite=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
