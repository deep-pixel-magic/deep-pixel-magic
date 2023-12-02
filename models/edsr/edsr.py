import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Conv2D, Input, PReLU, Lambda, Conv2DTranspose, Rescaling
from tensorflow.keras.initializers import GlorotUniform

from models.common.icnr import IcnrInitializer


class EdsrNetwork:
    """Represents an EDSR network for image super resolution.

    The network is based on the paper "Enhanced Deep Residual Networks for Single Image Super-Resolution".
    """

    def __init__(self):
        """Constructor."""

    def build(self, scale, num_filters=64, num_residual_blocks=8, residual_block_scaling=None):
        """Builds the actual EDSR model.

        Args:
            scale: The upsampling scale.
            num_filters: The number of filters.
            num_residual_blocks: The number of residual blocks.
            residual_block_scaling: The scaling factor for the residual blocks.
        """

        shape = (None, None, 3)

        x_in = Input(shape=shape)
        # layer_stack = Lambda(self.__normalize())(input_layer)
        x = Rescaling(
            scale=1.0 / 255.0, offset=0.0)(x_in)

        x = x_res = Conv2D(
            num_filters, 3, padding='same')(x)

        for _ in range(num_residual_blocks):
            x_res = self.__residual_block(
                x_res, num_filters, residual_block_scaling)

        x_res = Conv2D(
            num_filters, 3, padding='same')(x_res)
        x = Add()([x, x_res])

        x = self.__upsample_block(x, num_filters, scale)
        x = Conv2D(3, 3, padding='same')(x)

        # layer_stack = Lambda(self.__denormalize())(layer_stack)
        x = Rescaling(scale=255.0, offset=0.0)(x)

        return Model(x_in, x, name="edsr")

    def __residual_block(self, x_in, filters, scaling):
        """Creates a residual block.

        Args:
            input_layer: The input layer.
            filters: The number of filters.
            scaling: The scaling factor.
        """

        x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
        x = Conv2D(filters, 3, padding='same')(x)

        if scaling:
            x = Lambda(lambda x: x * scaling)(x)

        x = Add()([x_in, x])
        return x

    def __upsample_block(self, x_in, num_filters, scale):
        """Creates an upsampling block.

        Args:
            layer_stack: The input layer.
            scale: The upsampling scale.
            num_filters: The number of filters.
        """

        if scale == 0:
            raise ValueError('scale must be greater than 0')

        if (scale & (scale - 1)) != 0:
            raise ValueError('scale must be a power of 2')

        upsample_instances = int(math.log2(scale))

        for _ in range(upsample_instances):
            x_in = self.__upsample_instance(x_in, num_filters, 2)

        return x_in

    def __upsample_instance(self, x_in, num_filters, factor):
        """Creates a single upsampling instance.

        An upsampling instance consists of a convolutional layer and a pixel shuffle layer.

        Args:
            x: The input layer.
            factor: The upsampling factor.
        """

        kernel_initializer = IcnrInitializer(
            tf.keras.initializers.GlorotUniform(), scale=factor)

        x_in = Conv2DTranspose(
            filters=num_filters * (factor ** 2),
            kernel_size=3,
            kernel_initializer=kernel_initializer,
            padding='same')(x_in)

        x_in = Lambda(self.__pixel_shuffle(scale=factor))(x_in)

        return x_in

    def __pixel_shuffle(self, scale):
        """Creates a pixel shuffle layer.

        Args:
            scale: The upsampling scale.
        """

        return lambda x: tf.nn.depth_to_space(input=x, block_size=scale)

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
