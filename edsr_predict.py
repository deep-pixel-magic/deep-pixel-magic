import sys

import tensorflow as tf
from PIL import Image

from models.edsr.edsr_network import EdsrNetwork
from models.edsr.data_processing import normalize_input, postprocess_output

from tools.datasets.div2k import div2k
from tools.datasets.div2k.image import Div2kImage

from tools.images.image import load_png


def main():
    img_id = 869

    img_lr = load_png(
        f'./.cache/data/DIV2K_valid_LR_bicubic/X4/0{img_id}x4.png')
    img_hr = load_png(
        f'./.cache/data/DIV2K_valid_HR/0{img_id}.png', batched=False)

    img_lr_norm = normalize_input(img_lr)

    model = EdsrNetwork().build(scale=4, num_filters=64,
                                num_residual_blocks=16, residual_block_scaling=0.1, trainable=False)
    model.load_weights('./.cache/models/edsr/edsr.weights.h5')

    prediction = model.predict(img_lr_norm)

    predicted_img = tf.squeeze(prediction)
    predicted_img = postprocess_output(predicted_img)

    Image.fromarray(predicted_img.numpy()).save("edsr.prediction.png", "PNG")
    Image.fromarray(tf.cast(img_hr, tf.uint8).numpy()
                    ).save("edsr.original.png", "PNG")

    img = Div2kImage('./.cache/data/',
                     dataset_info=div2k.Info(subset='valid', resolution='LR', sampling='bicubic', scale='X4'))

    img.load(img_id)
    img.scale(4)
    img.save("edsr.upsampled.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
