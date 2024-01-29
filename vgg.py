import os
import sys
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input

from models.vgg.vgg import VggBuilder
from tools.images.image import load_png


def main():
    img_id = 805

    img_hr = load_png(
        f'./.cache/data/DIV2K_valid_HR/0{img_id}.png', batched=True)

    vgg_layers = ['block5_conv4']
    vgg = VggBuilder(layers=vgg_layers).build(input_shape=(None, None, 3))

    vgg_in = preprocess_input(img_hr)
    prediction = vgg(vgg_in)

    prediction = tf.squeeze(prediction)
    prediction = tf.round(prediction)
    prediction = tf.cast(prediction, tf.uint8)
    prediction = tf.clip_by_value(prediction, 0, 255)

    tf.print(prediction)

    os.makedirs("./.cache/predictions/vgg", exist_ok=True)
    for channel in range(512):
        Image.fromarray(prediction[:, :, channel].numpy()).save(
            f"./.cache/predictions/vgg/vgg.prediction.{channel}.png", "PNG")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
