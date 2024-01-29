import sys
import tensorflow as tf

from PIL import Image

from models.srgan.srgan_network import SrganNetwork
from models.srgan.data_processing import normalize_input_lr, normalize_input_hr, postprocess_output

from tools.datasets.div2k import div2k
from tools.datasets.div2k.image import Div2kImage

from tools.images.image import load_png


def main():
    img_id = 869

    img_lr = load_png(
        f'./.cache/data/DIV2K_valid_LR_bicubic/X4/0{img_id}x4.png')
    img_hr = load_png(
        f'./.cache/data/DIV2K_valid_HR/0{img_id}.png', batched=False)

    img_lr_norm = normalize_input_lr(img_lr)

    generator = SrganNetwork().build(
        num_filters=64, num_residual_blocks=16, trainable=False)
    generator.load_weights('./.cache/models/srgan/generator.weights.h5')

    prediction = generator.predict(img_lr_norm)

    predicted_img = tf.squeeze(prediction)
    predicted_img = postprocess_output(predicted_img)

    Image.fromarray(predicted_img.numpy()).save("srgan.prediction.png", "PNG")
    Image.fromarray(tf.cast(img_hr, tf.uint8).numpy()
                    ).save("srgan.original.png", "PNG")

    img = Div2kImage('./.cache/data/',
                     dataset_info=div2k.Info(subset='valid', resolution='LR', sampling='bicubic', scale='X4'))

    img.load(img_id)
    img.scale(4)
    img.save('srgan.upsampled.png')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)