import os
import sys
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError

from models.edsr.edsr_network import EdsrNetwork
from models.edsr.edsr_trainer import EdsrNetworkTrainer

from tools.datasets.div2k.tensorflow import TensorflowImageDataset, TensorflowImagePreprocessor


def main():
    # logical_devices = tf.config.list_logical_devices('GPU')
    # selected_logical_devices = logical_devices[1:3]

    # print('using logical devices:')
    # for device in selected_logical_devices:
    #     print('  ', device)

    # strategy = tf.distribute.MirroredStrategy(devices=selected_logical_devices)

    out_dir = './.cache/models/edsr/'
    out_file = os.path.join(out_dir, 'model.pkg')

    data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_train_HR"

    batch_size = 16

    bundle = TensorflowImageDataset(data_dir_low_res, data_dir_high_res)
    dataset = TensorflowImagePreprocessor(bundle).preprocess(
        batch_size=batch_size, crop_size=128, scale=4)

    initial_data_set_cardinality = bundle.num()
    batched_data_set_cardinality = initial_data_set_cardinality // batch_size

    num_steps_per_epoch = batched_data_set_cardinality
    num_pre_train_epochs = 50
    num_fine_tune_epochs = 50

    with tf.device('/GPU:1'):
        model = EdsrNetwork().build(scale=4, num_filters=256,
                                    num_residual_blocks=32, residual_block_scaling=0.1)

        # Pre-train the model using pixel-wise loss.

        decay_boundaries = [
            20 * num_steps_per_epoch,
            30 * num_steps_per_epoch,
            40 * num_steps_per_epoch,
        ]

        decay_values = [
            1e-4,  # first 20 epochs
            5e-5,  # next 10 epochs
            2.5e-5,  # next 10 epochs
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
            performed_steps + 20 * num_steps_per_epoch,
            performed_steps + 30 * num_steps_per_epoch,
            performed_steps + 40 * num_steps_per_epoch,
        ]

        decay_values = [
            1e-4,  # first 10 epochs
            5e-5,  # next 10 epochs
            2.5e-5,  # next 10 epochs
            1.25e-5,  # remaining epochs
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
