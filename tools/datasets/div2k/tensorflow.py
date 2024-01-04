import tensorflow as tf
from tensorflow.data import AUTOTUNE


class TensorflowDataset:

    def __init__(self, low_res_dir, high_res_dir):
        """Constructor.

        Args:
            directory: The directory containing the images.
        """

        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir

        self.dataset = self.__load()

    def batched(self, batch_size=16):
        """Returns a batched version of the dataset."""

        return self.dataset.batch(batch_size)

    def rgb_mean(self):
        """Returns the mean of the RGB values of the images."""

        sum_red = 0.0
        sum_green = 0.0
        sum_blue = 0.0

        samples = 0

        for _, hr_img in self.dataset:

            image = tf.image.convert_image_dtype(hr_img, tf.float32)

            sum_red += tf.reduce_sum(image[:, :, 0])
            sum_green += tf.reduce_sum(image[:, :, 1])
            sum_blue += tf.reduce_sum(image[:, :, 2])

            samples += tf.cast(tf.shape(image)
                               [0] * tf.shape(image)[1], tf.float32)

        sum_red /= samples
        sum_green /= samples
        sum_blue /= samples

        return sum_red.numpy(), sum_green.numpy(), sum_blue.numpy()

    def num(self):
        return self.dataset.cardinality().numpy()

    def __load(self):
        """Preprocesses the images."""

        low_res_dataset = self.__dataset_from(self.low_res_dir)
        high_res_dataset = self.__dataset_from(self.high_res_dir)

        return tf.data.Dataset.zip(low_res_dataset, high_res_dataset)

    def __dataset_from(self, directory):
        """Creates a dataset from the images in the specified directory.

        Args:
            directory: The directory containing the images.

        Returns:
            The dataset.
        """

        glob = tf.io.gfile.glob(directory + "/*.png")
        img_files = sorted(glob)

        data_set = tf.data.Dataset.from_tensor_slices(img_files)
        data_set = data_set.map(tf.io.read_file)
        data_set = data_set.map(lambda x: tf.image.decode_png(
            x, channels=3), num_parallel_calls=AUTOTUNE)

        return data_set


class TensorflowPreprocessor:

    def __init__(self, dataset: TensorflowDataset):
        """Constructor.

        Args:
            dataset: The dataset.
        """

        self.dataset = dataset

    def preprocess(self, batch_size=16, crop_size=96, scale=4):
        """Preprocesses the images."""

        dataset = self.dataset.dataset

        crop_operator = RandomCropOperator(crop_size, scale)
        flip_operator = RandomFlipOperator()
        rotate_operator = RandomRotateOperator()

        dataset = dataset.map(crop_operator, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(flip_operator, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(rotate_operator, num_parallel_calls=AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset


class RandomCropOperator:

    def __init__(self, hr_crop_size=96, scale=4):
        self.hr_crop_size = hr_crop_size
        self.scale = scale

    def __call__(self, lr_img, hr_img):
        scale = self.scale
        hr_crop_size = self.hr_crop_size

        lr_crop_size = hr_crop_size // scale
        lr_img_shape = tf.shape(lr_img)[:2]

        lr_w = tf.random.uniform(
            shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
        lr_h = tf.random.uniform(
            shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

        hr_w = lr_w * scale
        hr_h = lr_h * scale

        lr_img_cropped = lr_img[lr_h:lr_h +
                                lr_crop_size, lr_w:lr_w + lr_crop_size]
        hr_img_cropped = hr_img[hr_h:hr_h +
                                hr_crop_size, hr_w:hr_w + hr_crop_size]

        return lr_img_cropped, hr_img_cropped


class RandomFlipOperator:

    def __init__(self):
        pass

    def __call__(self, low_res_img, high_res_img):
        rn = tf.random.uniform(shape=(), maxval=1)
        return tf.cond(rn < 0.5,
                       lambda: (low_res_img, high_res_img),
                       lambda: (tf.image.flip_left_right(low_res_img),
                                tf.image.flip_left_right(high_res_img)))


class RandomRotateOperator:

    def __init__(self):
        pass

    def __call__(self, low_res_img, high_res_img):
        rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        return tf.image.rot90(low_res_img, rn), tf.image.rot90(high_res_img, rn)


class NormalizationOperator:

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, low_res_img, high_res_img):
        return (low_res_img - self.mean) / 127.5, (high_res_img - self.mean) / 127.5
