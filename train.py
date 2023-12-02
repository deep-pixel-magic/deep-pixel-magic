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

from tools.datasets.div2k.preprocessor import Preprocessor
from tools.datasets.div2k.tensorflow import TensorflowDataset


def main():
    out_dir = './.cache/models/edsr/'
    out_file = os.path.join(out_dir, 'model.pkg')

    data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_train_HR"

    batch_size = 16

    bundle = TensorflowDataset(data_dir_low_res, data_dir_high_res)
    dataset = Preprocessor(bundle).preprocess(
        batch_size=batch_size, crop_size=96, scale=4)

    initial_data_set_cardinality = bundle.num()
    batched_data_set_cardinality = initial_data_set_cardinality // batch_size

    model = EdsrNetwork().build(scale=4, num_filters=64, num_residual_blocks=16)

    trainer = EdsrTrainer(model=model, learning_rate=PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]))

    trainer.train(dataset, epochs=11,
                  steps=batched_data_set_cardinality)

    model.save_weights(out_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
