import tensorflow as tf
from tensorflow.data import AUTOTUNE


class TensorflowImageDataset:
    """A dataset of images stored in a directory.

    The images are assumed to be PNG files.
    """

    def __init__(self, directory, normalizer=None):
        """Constructor.

        Args:
            directory: The directory containing the images.
            normalizer: An optional function that normalizes the images.
        """

        self.__dataset = self.__dataset_from(directory, normalizer)

    def batched(self, batch_size=16):
        """Returns a batched version of the dataset.

        Args:
            batch_size: The batch size.

        Returns:
            The batched dataset.
        """

        return self.__dataset.batch(batch_size)

    def rgb_mean(self):
        """Calculates the mean RGB values of the images in the dataset.

        Returns:
            The mean RGB values.
        """

        sum_red = 0.0
        sum_green = 0.0
        sum_blue = 0.0

        samples = 0

        for _, img in self.__dataset:

            image = tf.image.convert_image_dtype(img, tf.float32)

            sum_red += tf.reduce_sum(image[:, :, 0])
            sum_green += tf.reduce_sum(image[:, :, 1])
            sum_blue += tf.reduce_sum(image[:, :, 2])

            samples += tf.cast(tf.shape(image)
                               [0] * tf.shape(image)[1], tf.float32)

        sum_red /= samples
        sum_green /= samples
        sum_blue /= samples

        return sum_red.numpy(), sum_green.numpy(), sum_blue.numpy()

    def dataset(self):
        """Returns the internal tensorflow dataset.

        Returns:
            The tensorflow dataset.
        """

        return self.__dataset

    def num(self):
        """Returns the cardinality of the dataset.

        Returns:
            The cardinality of the dataset.
        """

        return self.__dataset.cardinality().numpy()

    def __dataset_from(self, directory, normalizer=None):
        """Creates a dataset from the images in the specified directory.

        Args:
            directory: The directory containing the images.
            normalizer: An optional function that normalizes the images.

        Returns:
            The tensorflow dataset.
        """

        glob = tf.io.gfile.glob(directory + "/*.png")
        files = sorted(glob)

        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(lambda x: tf.io.read_file(x),
                              num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(lambda x: tf.image.decode_png(
            x, channels=3), num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(lambda x: tf.cast(
            x, tf.float32), num_parallel_calls=AUTOTUNE)

        if normalizer is not None:
            dataset = dataset.map(normalizer, num_parallel_calls=AUTOTUNE)

        return dataset


class TensorflowImageDatasetBundle:
    """A bundle of two datasets of images stored in directories.

    One dataset contains the low resolution images and the other dataset contains the high resolution images.
    """

    def __init__(self, dataset_lr: TensorflowImageDataset, dataset_hr: TensorflowImageDataset):
        """Constructor.

        Args:
            dataset_lr: The low resolution dataset.
            dataset_hr: The high resolution dataset.
        """

        self.__dataset_lr = dataset_lr
        self.__dataset_hr = dataset_hr

        if dataset_lr.num() != dataset_hr.num():
            raise ValueError(
                'The number of samples in the low resolution and high resolution datasets must be the same.')

        self.__dataset = tf.data.Dataset.zip(
            (dataset_lr.dataset(), dataset_hr.dataset()))

    def dataset(self):
        """Returns the internal tensorflow dataset.

        Returns:
            The tensorflow dataset.
        """

        return self.__dataset

    def num(self):
        """Returns the cardinality of the dataset.

        Returns:
            The cardinality of the dataset.
        """

        return self.__dataset_lr.num()


class TensorflowImagePreprocessor:
    """Preprocesses a dataset of images."""

    def __init__(self, dataset: TensorflowImageDatasetBundle):
        """Constructor.

        Args:
            dataset: The dataset to preprocess.
        """

        self.dataset = dataset

    def preprocess(self, batch_size=16, crop_size=96, scale=4, shuffle_buffer_size=16, cache=False):
        """Preprocesses the images.

        Args:
            batch_size: The batch size.
            crop_size: The crop size.
            scale: The scale factor.
            shuffle_buffer_size: The shuffle buffer size.
            cache: Whether to cache the dataset.

        Returns:
            The preprocessed dataset.
        """

        dataset = self.dataset.dataset()

        crop_operator = RandomCropOperator(crop_size, scale)
        flip_operator = RandomFlipOperator()
        rotate_operator = RandomRotateOperator()

        if cache:
            dataset = dataset.cache()

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

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
