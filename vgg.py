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
    Image.fromarray(prediction.numpy()).save("vgg.prediction.png", "PNG")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
