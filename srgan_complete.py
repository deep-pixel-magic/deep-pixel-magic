import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from models.srgan.srgan_network import SrganNetwork
from models.srgan.discriminator_network import SrganDiscriminatorNetwork
from models.srgan.srgan_trainer import SrganTrainer
from models.srgan.srgan_pre_trainer import SrganPreTrainer
from models.srgan.data_processing import normalize_input_lr, normalize_input_hr

from tools.datasets.div2k.tensorflow import TensorflowImageDataset, TensorflowImageDatasetBundle, TensorflowImagePreprocessor


def main():
    with tf.device('/GPU:0'):

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
            batch_size=batch_size,
            crop_size=crop_size,
            scale=4,
            shuffle_buffer_size=batch_size,
            cache=False)

        initial_data_set_cardinality = bundle.num()
        batched_data_set_cardinality = initial_data_set_cardinality // batch_size

        num_steps_per_epoch = batched_data_set_cardinality
        num_pre_train_epochs = 1000
        num_fine_tune_epochs = 1000

        generator = SrganNetwork().build(num_filters=64, num_residual_blocks=16)

        # Pre-train the model using pixel-wise loss.

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

        # Fine-tune the model using perceptual and adversarial loss.

        discriminator = SrganDiscriminatorNetwork(img_res=crop_size).build()

        performed_steps = num_pre_train_epochs * num_steps_per_epoch

        decay_boundaries = [
            performed_steps + (num_fine_tune_epochs // 2) *
            num_steps_per_epoch,
        ]

        decay_values = [
            1e-4,
            1e-5,
        ]

        generator_lr = PiecewiseConstantDecay(
            boundaries=decay_boundaries, values=decay_values)

        discriminator_lr = PiecewiseConstantDecay(
            boundaries=[(num_fine_tune_epochs // 2) * num_steps_per_epoch], values=decay_values)

        trainer = SrganTrainer(
            generator=generator,
            discriminator=discriminator,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr)

        trainer.train(dataset, epochs=num_pre_train_epochs +
                      num_fine_tune_epochs, steps=batched_data_set_cardinality)

        os.makedirs('./.cache/models/srgan', exist_ok=True)
        discriminator.save_weights(
            './.cache/models/srgan/discriminator.weights.h5', overwrite=True)
        generator.save_weights(
            './.cache/models/srgan/generator.weights.h5', overwrite=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
