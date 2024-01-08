import numpy as np
import tensorflow as tf

from tensorflow.python.layers.utils import normalize_tuple


class IcnrInitializer:
    """Represents a kernel initializer for transpose convolutions used for sub pixel convolution upsampling."""

    def __init__(self, initializer=tf.keras.initializers.GlorotUniform(), scale=1):
        """ICNR initializer for checkerboard artifact free transpose convolution.

        Code adapted from https://github.com/kostyaev/ICNR
        Discussed at https://github.com/Lasagne/Lasagne/issues/862
        Original paper: https://arxiv.org/pdf/1707.02937.pdf

        Args:
            initializer: The initializer used for the kernels (glorot uniform, etc.).
            scale: The scale factor of the sub pixel convolution.
        """

        self.scale = normalize_tuple(scale, 2, "scale")
        self.initializer = initializer

    def __call__(self, shape, dtype):
        """Creates the kernel tensor.

        Args:
            shape: The shape of the kernel tensor.
            dtype: The data type of the kernel tensor.

        Returns:
            The kernel tensor.
        """

        if self.scale == 1:
            return self.initializer(shape)

        size = shape[:2]
        new_shape = np.array(shape)
        new_shape[:2] //= self.scale

        x = self.initializer(new_shape, dtype)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize(x, size=size, method="nearest")
        x = tf.transpose(x, perm=[1, 2, 0, 3])

        # if self.scale == 1:
        #     return self.initializer(shape)

        # shape = list(shape)
        # new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]

        # x = self.initializer(new_shape, dtype)

        # x = tf.transpose(x, perm=[2, 0, 1, 3])
        # x = tf.image.resize(
        #     x, size=(shape[0] * self.scale, shape[1] * self.scale), method="nearest")
        # x = tf.nn.space_to_depth(x, block_size=self.scale)
        # x = tf.transpose(x, perm=[1, 2, 0, 3])

        return x
