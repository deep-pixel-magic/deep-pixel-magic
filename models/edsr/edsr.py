import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models


class EdsrNetwork:
    """Represents an EDSR network for image super resolution.

    The network is based on the paper "Enhanced Deep Residual Networks for Single Image Super-Resolution"
    """

    def __init__(self):
        """Constructor."""

        pass

    def build(self, scale, num_filters=64, num_residual_blocks=8, residual_block_scaling=None):
        """Builds the actual EDSR model.

        Args:
            scale: The upsampling scale.
            num_filters: The number of filters.
            num_residual_blocks: The number of residual blocks.
            residual_block_scaling: The scaling factor for the residual blocks.
        """

        in_shape = (None, None, 3)

        input_layer = layers.Input(shape=in_shape)

        # layer_stack = layers.Lambda(self.__normalize)(input_layer)
        layer_stack = input_layer

        layer_stack = residual_stack = layers.Conv2D(
            num_filters, 3, padding='same')(layer_stack)

        for _ in range(num_residual_blocks):
            residual_stack = self.__residual_block(
                residual_stack, num_filters, residual_block_scaling)

        residual_stack = layers.Conv2D(
            num_filters, 3, padding='same')(residual_stack)
        layer_stack = layers.Add()([layer_stack, residual_stack])

        layer_stack = self.__upsample_block(layer_stack, num_filters, scale)
        layer_stack = layers.Conv2D(3, 3, padding='same')(layer_stack)

        # layer_stack = layers.Lambda(self.__denormalize)(layer_stack)

        return models.Model(input_layer, layer_stack, name="edsr")

    def __residual_block(self, input_layer, filters, scaling):
        """Creates a residual block.

        Args:
            input_layer: The input layer.
            filters: The number of filters.
            scaling: The scaling factor.
        """

        layer_stack = layers.Conv2D(
            filters, 3, padding='same', activation='relu')(input_layer)
        layer_stack = layers.Conv2D(filters, 3, padding='same')(layer_stack)

        if scaling:
            layer_stack = layers.Lambda(lambda x: x * scaling)(layer_stack)

        layer_stack = layers.Add()([input_layer, layer_stack])
        return layer_stack

    def __upsample_block(self, layer_stack, num_filters, scale):
        """Creates an upsampling block.

        Args:
            layer_stack: The input layer.
            scale: The upsampling scale.
            num_filters: The number of filters.
        """

        def __upsample_instance(x, factor):
            """Creates a single upsampling instance.

            An upsampling instance consists of a convolutional layer and a pixel shuffle layer.

            Args:
                x: The input layer.
                factor: The upsampling factor.
            """

            x = layers.Conv2D(num_filters * (factor ** 2),
                              3, padding='same')(x)
            return layers.Lambda(self.__pixel_shuffle(scale=factor))(x)

        if scale == 2:
            layer_stack = __upsample_instance(
                layer_stack, 2)
        elif scale == 3:
            layer_stack = __upsample_instance(
                layer_stack, 3)
        elif scale == 4:
            layer_stack = __upsample_instance(
                layer_stack, 2)
            layer_stack = __upsample_instance(
                layer_stack, 2)

        return layer_stack

    def __pixel_shuffle(self, scale):
        """Creates a pixel shuffle layer.

        Args:
            scale: The upsampling scale.
        """

        return lambda x: tf.nn.depth_to_space(input=x, block_size=scale)

    def __normalize(self, x):
        """Normalizes the input."""

        return tf.cast(x, tf.float32) / 255.0

    def __denormalize(self, x):
        """Denormalizes the input."""

        return x * 255
