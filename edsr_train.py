import os
import sys
import tensorflow as tf

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from models.edsr.edsr_network import EdsrNetwork
from models.edsr.edsr_trainer import EdsrNetworkTrainer
from models.edsr.data_processing import normalize_input

from tools.datasets.div2k.tensorflow import TensorflowImageDataset, TensorflowImageDatasetBundle, TensorflowImagePreprocessor


def main():
    with tf.device('/GPU:0'):

        data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
        data_dir_high_res = "./.cache/data/DIV2K_train_HR"

        batch_size = 16
        crop_size = 96

        dataset_lr = TensorflowImageDataset(
            data_dir_low_res, normalizer=lambda x: normalize_input(x))
        dataset_hr = TensorflowImageDataset(
            data_dir_high_res, normalizer=lambda x: normalize_input(x))

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
        num_epochs = 1000

        model = EdsrNetwork().build(scale=4, num_filters=64,
                                    num_residual_blocks=16, residual_block_scaling=0.1)

        decay_boundaries = [
            (num_epochs // 2) * num_steps_per_epoch,
        ]

        decay_values = [
            1e-4,
            1e-5,
        ]

        learning_rate = PiecewiseConstantDecay(
            boundaries=decay_boundaries, values=decay_values)

        trainer = EdsrNetworkTrainer(
            model=model, learning_rate=learning_rate)

        trainer.train(dataset, epochs=num_epochs,
                      steps=batched_data_set_cardinality)

        os.makedirs('./.cache/models/edsr', exist_ok=True)
        model.save_weights(
            './.cache/models/edsr/edsr.weights.h5', overwrite=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
