import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Input, PReLU, Lambda


class SrganNetwork:
    """Represents a SRGAN network for image super resolution.

    The model is based on the paper "Image Super-Resolution Using a Generative Adversarial Network".
    """

    def __init__(self):
        """Constructor."""

        pass

    def build(self, num_filters=64, num_residual_blocks=16):
        """Builds the actual SRGAN model.

        Args:
            num_filters: The number of filters.
            num_residual_blocks: The number of residual blocks.
        """

        shape = (None, None, 3)

        x_in = Input(shape=shape)
        x = Lambda(self.__normalize())(x_in)

        x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        for _ in range(num_residual_blocks):
            x = self.__residual_block(x, num_filters)

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x_1, x])

        x = self.__upsample(x, num_filters * 4)
        x = self.__upsample(x, num_filters * 4)

        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        x = Lambda(self.__denormalize())(x)

        return Model(x_in, x)

    def __residual_block(self, x_in, num_filters, momentum=0.8):
        """Creates a residual block.

        Args:
            x_in: The input layer.
            num_filters: The number of filters.
            momentum: The momentum for the batch normalization layers.

        Returns:
            The residual block layer.
        """

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = BatchNormalization(momentum=momentum)(x)
        x = PReLU(shared_axes=[1, 2])(x)

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Add()([x_in, x])

        return x

    def __upsample(self, x_in, num_filters):
        """Upsamples the input using sub-pixel convolution.

        Args:
            x_in: The input layer.
            num_filters: The number of filters.

        Returns:
            The upsample layer.
        """

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = Lambda(self.__pixel_shuffle(scale=2))(x)
        x = PReLU(shared_axes=[1, 2])(x)

        return x

    def __pixel_shuffle(self, scale):
        """Creates a lambda using tensorflow's internal depth_to_space method.

        Args:
            scale: The upsampling scale.

        Returns:
            A lambda that performs the depth_to_space operation.
        """

        return lambda x: tf.nn.depth_to_space(x, scale)

    def __normalize(self):
        """Normalizes the input.

        Assumes an input interval of [0, 255].
        """

        return lambda x: x / 255.0

    def __denormalize(self):
        """Denormalizes the input.

        Assumes a normalized input interval of [0, 1].
        """

        return lambda x: x * 255
