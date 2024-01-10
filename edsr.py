import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError

from models.edsr.edsr_network import EdsrNetwork
from models.edsr.edsr_trainer import EdsrNetworkTrainer

from tools.datasets.div2k.tensorflow import TensorflowDataset, TensorflowPreprocessor


def main():
    out_dir = './.cache/models/edsr/'
    out_file = os.path.join(out_dir, 'model.pkg')

    data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_train_HR"

    batch_size = 16

    bundle = TensorflowDataset(data_dir_low_res, data_dir_high_res)
    dataset = TensorflowPreprocessor(bundle).preprocess(
        batch_size=batch_size, crop_size=128, scale=4)

    initial_data_set_cardinality = bundle.num()
    batched_data_set_cardinality = initial_data_set_cardinality // batch_size

    num_steps_per_epoch = batched_data_set_cardinality
    num_pre_train_epochs = 30
    num_fine_tune_epochs = 30

    model = EdsrNetwork().build(scale=4, num_filters=64,
                                num_residual_blocks=16, residual_block_scaling=0.1)

    # Pre-train the model using pixel-wise loss.

    decay_boundaries = [
        10 * num_steps_per_epoch,  # first 10 epochs
        15 * num_steps_per_epoch,  # next 5 epochs
        20 * num_steps_per_epoch,  # next 5 epochs
    ]

    decay_values = [
        1e-4,  # first 10 epochs
        5e-5,  # next 5 epochs
        2.5e-5,  # next 5 epochs
        1.25e-5,  # remaining epochs
    ]

    learning_rate = PiecewiseConstantDecay(
        boundaries=decay_boundaries, values=decay_values)

    trainer = EdsrNetworkTrainer(
        model=model, learning_rate=learning_rate, use_content_loss=False)

    trainer.train(dataset, epochs=num_pre_train_epochs,
                  steps=batched_data_set_cardinality)

    # Fine-tune the model using perceptual loss.

    performed_steps = num_pre_train_epochs * num_steps_per_epoch

    decay_boundaries = [
        performed_steps + 10 * num_steps_per_epoch,  # first 10 epochs
        performed_steps + 20 * num_steps_per_epoch,  # next 10 epochs
        performed_steps + 30 * num_steps_per_epoch,  # next 10 epochs
    ]

    decay_values = [
        1e-6,  # first 10 epochs
        5e-7,  # next 10 epochs
        2.5e-7,  # next 10 epochs
        1.25e-7,  # remaining epochs
    ]

    learning_rate = PiecewiseConstantDecay(
        boundaries=decay_boundaries, values=decay_values)

    trainer = EdsrNetworkTrainer(
        model=model, learning_rate=learning_rate, use_content_loss=True)

    trainer.train(dataset, epochs=num_pre_train_epochs + num_fine_tune_epochs,
                  steps=batched_data_set_cardinality)

    model.save_weights(out_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
