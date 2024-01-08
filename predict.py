import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from models.edsr.edsr_network import EdsrNetwork

from tools.datasets.div2k import div2k
from tools.datasets.div2k.tensorflow import TensorflowDataset, TensorflowPreprocessor
from tools.datasets.div2k.image import Div2kImage

from tools.images.image import load_png


def main():
    data_dir_low_res = "./.cache/data/DIV2K_valid_LR_bicubic/X4"
    data_dir_high_res = "./.cache/data/DIV2K_valid_HR"

    # dataset = TensorflowDataset(data_dir_low_res, data_dir_high_res).batched(1)

    # iterator_training = dataset.as_numpy_iterator()
    # element = iterator_training.next()
    # element = iterator_training.next()
    # element = iterator_training.next()
    # element = iterator_training.next()
    # element = iterator_training.next()

    img_lr = load_png('./.cache/data/DIV2K_valid_LR_bicubic/X4/0805x4.png')
    img_hr = load_png('./.cache/data/DIV2K_valid_HR/0805.png')

    model = EdsrNetwork().build(scale=4, num_filters=64, num_residual_blocks=16)

    latest = tf.train.latest_checkpoint('./.cache/models/edsr')
    model.load_weights(latest)

    prediction = model.predict(img_lr)

    predicted_img = prediction[0]
    predicted_img = tf.round(predicted_img)
    predicted_img = tf.clip_by_value(predicted_img, 0, 255)
    predicted_img = tf.cast(predicted_img, tf.uint8)

    img_to_save = Image.fromarray(predicted_img.numpy())
    img_to_save.save("prediction.png", "PNG")

    img = Div2kImage('./.cache/data/',
                     dataset_info=div2k.Info(subset='valid', resolution='LR', sampling='bicubic', scale='X4'))

    img.load(805)
    img.scale(4)

    figure, plots = plt.subplots(1, 3, sharex=True, sharey=True)
    plots[0].imshow(img.get())
    plots[0].title.set_text('Upsampled')

    plots[1].imshow(predicted_img)
    plots[1].title.set_text('Prediction')

    plots[2].imshow(tf.squeeze(img_hr))
    plots[2].title.set_text('Original')

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("interrupted")

        sys.exit(130)
