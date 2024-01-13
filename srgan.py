import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from models.srgan.srgan_network import SrganNetwork
from models.srgan.discriminator_network import SrganDiscriminatorNetwork
from models.srgan.srgan_trainer import SrganTrainer
from models.srgan.srgan_pre_trainer import SrganPreTrainer

from tools.datasets.div2k.tensorflow import TensorflowImageDataset, TensorflowImageDatasetBundle, TensorflowImagePreprocessor


def main():
    data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_train_HR"

    rgb_mean = np.array([0.4488, 0.4371, 0.4040]) * 255

    batch_size = 16
    crop_size = 96

    dataset_lr = TensorflowImageDataset(data_dir_low_res, normalizer=lambda x: (x - rgb_mean) / 127.5)
    dataset_hr = TensorflowImageDataset(data_dir_high_res, normalizer=lambda x: (x - rgb_mean) / 127.5)

    bundle = TensorflowImageDatasetBundle(dataset_lr, dataset_hr)

    dataset = TensorflowImagePreprocessor(bundle).preprocess(
        batch_size=batch_size, crop_size=crop_size, scale=4)

    initial_data_set_cardinality = bundle.num()
    batched_data_set_cardinality = initial_data_set_cardinality // batch_size

    num_steps_per_epoch = batched_data_set_cardinality
    num_pre_train_epochs = 200
    num_fine_tune_epochs = 200


    with tf.device('/GPU:1'):

        generator = SrganNetwork().build(num_filters=64, num_residual_blocks=16,
                                        use_batch_normalization=True)

        # Pre-train the model using pixel-wise loss.

        decay_boundaries = [
            100 * num_steps_per_epoch,  # first 100 epochs
        ]

        decay_values = [
            1e-4,  # first 100 epochs
            1e-5,  # next 100 epochs
        ]

        generator_lr = PiecewiseConstantDecay(
            boundaries=decay_boundaries, values=decay_values)

        pre_trainer = SrganPreTrainer(
            generator=generator, learning_rate=generator_lr)

        pre_trainer.train(dataset, epochs=num_pre_train_epochs,
                        steps=batched_data_set_cardinality)

        # Fine-tune the model using perceptual and adversarial loss.

        discriminator = SrganDiscriminatorNetwork(img_res=crop_size).build()

        performed_steps = num_pre_train_epochs * num_steps_per_epoch

        decay_boundaries = [
            performed_steps + 100 * num_steps_per_epoch,  # first 100 epochs
        ]

        decay_values = [
            1e-4,  # first 100 epochs
            1e-5,  # next 100 epochs
        ]

        generator_lr = PiecewiseConstantDecay(
            boundaries=decay_boundaries, values=decay_values)
        
        discriminator_lr = PiecewiseConstantDecay(
            boundaries=[100 * num_steps_per_epoch], values=decay_values)

        trainer = SrganTrainer(
            generator=generator, 
            discriminator=discriminator, 
            generator_lr=generator_lr, 
            discriminator_lr=discriminator_lr)

        trainer.train(dataset, epochs=num_pre_train_epochs +
                    num_fine_tune_epochs, steps=batched_data_set_cardinality)

        os.makedirs('./.cache/models/srgan', exist_ok=True)
        discriminator.save_weights('./.cache/models/srgan/discriminator.weights.h5', overwrite=True)
        generator.save_weights('./.cache/models/srgan/generator.weights.h5', overwrite=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
