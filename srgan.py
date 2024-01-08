import os
import sys

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from models.srgan.srgan_network import SrganNetwork
from models.srgan.discriminator_network import SrganDiscriminatorNetwork
from models.srgan.srgan_trainer import SrganTrainer
from models.srgan.srgan_pre_trainer import SrganPreTrainer

from tools.datasets.div2k.tensorflow import TensorflowDataset, TensorflowPreprocessor


def main():
    out_dir = './.cache/models/srgan/'
    out_file = os.path.join(out_dir, 'model.pkg')

    data_dir_low_res = "./.cache/data/DIV2K_train_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_train_HR"

    batch_size = 16

    bundle = TensorflowDataset(data_dir_low_res, data_dir_high_res)
    dataset = TensorflowPreprocessor(bundle).preprocess(
        batch_size=batch_size, crop_size=128, scale=4)

    initial_data_set_cardinality = bundle.num()
    batched_data_set_cardinality = initial_data_set_cardinality // batch_size

    generator = SrganNetwork().build()
    # discriminator = SrganDiscriminatorNetwork().build()

    learning_rate = PiecewiseConstantDecay(
        boundaries=[2000, 3000, 4000], values=[1e-4, 5e-5, 2.5e-5, 1.25e-5])

    pre_trainer = SrganPreTrainer(
        generator=generator, learning_rate=learning_rate)

    pre_trainer.train(dataset, epochs=80, steps=batched_data_set_cardinality)

    # learning_rate = PiecewiseConstantDecay(
    #     boundaries=[5000, 6000, 7000, 8000], values=[1e-4, 5e-5, 2.5e-5, 1.25e-5, 0.625e-5])

    # trainer = SrganTrainer(
    #     generator=generator, discriminator=discriminator, learning_rate=learning_rate)

    # trainer.train(dataset, epochs=84, steps=batched_data_set_cardinality)

    generator.save_weights(out_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
