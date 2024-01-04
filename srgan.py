import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError

from models.srgan.srgan import SrganNetwork
from models.srgan.discriminator import SrganDiscriminatorNetwork
from models.srgan.srgan_trainer import SrganTrainer

from tools.datasets.div2k.tensorflow import TensorflowDataset, TensorflowPreprocessor


def main():
    out_dir = './.cache/models/srgan/'
    out_file = os.path.join(out_dir, 'model.pkg')

    data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_train_HR"

    batch_size = 8

    bundle = TensorflowDataset(data_dir_low_res, data_dir_high_res)
    dataset = TensorflowPreprocessor(bundle).preprocess(
        batch_size=batch_size, crop_size=96, scale=4)

    initial_data_set_cardinality = bundle.num()
    batched_data_set_cardinality = initial_data_set_cardinality // batch_size

    generator = SrganNetwork().build()
    discriminator = SrganDiscriminatorNetwork().build()

    trainer = SrganTrainer(generator=generator, discriminator=discriminator, learning_rate=PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]))

    trainer.train(dataset, epochs=1,
                  steps=batched_data_set_cardinality)

    generator.save_weights(out_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
