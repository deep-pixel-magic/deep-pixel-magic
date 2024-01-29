import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Conv2DTranspose, Input, PReLU, Lambda

from models.common.kernels import IcnrInitializer


class SrganNetwork:
    """Represents a SRGAN network for image super resolution.

    The model is based on the paper "Image Super-Resolution Using a Generative Adversarial Network".
    """

    def __init__(self):
        """Constructor."""

    def build(self, num_filters=64, num_residual_blocks=16, trainable=True):
        """Builds the actual SRGAN model.

        Args:
            num_filters: The number of filters.
            num_residual_blocks: The number of residual blocks.
        """

        shape = (None, None, 3)

        x_in = Input(shape=shape)
        x = Conv2D(num_filters, kernel_size=9, padding='same')(x_in)
        x = x_pre_res = PReLU(shared_axes=[1, 2])(x)

        for _ in range(num_residual_blocks):
            x = self.__residual_block(x, num_filters, trainable=trainable)

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(trainable=trainable)(x)
        x = Add()([x_pre_res, x])

        x = self.__upsample_pixel_shuffle(x, num_filters * 4, factor=2)
        x = self.__upsample_pixel_shuffle(x, num_filters * 4, factor=2)

        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)

        model = Model(x_in, x)
        model.trainable = trainable

        return model

    def __residual_block(self, x_in, num_filters, momentum=0.8, trainable=True):
        """Creates a residual block.

        Args:
            x_in: The input layer.
            num_filters: The number of filters.
            momentum: The momentum for the batch normalization layers.

        Returns:
            The residual block layer.
        """

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = BatchNormalization(momentum=momentum, trainable=trainable)(x)
        x = PReLU(shared_axes=[1, 2])(x)

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=momentum, trainable=trainable)(x)
        x = Add()([x_in, x])

        return x

    def __upsample_deconvolution(self, x_in, num_filters, factor=2):
        """Upsamples the input using sub-pixel convolution.

        Args:
            x_in: The input layer.
            num_filters: The number of filters.

        Returns:
            The upsample layer.
        """

        kernel_initializer = IcnrInitializer(
            tf.keras.initializers.GlorotUniform(),
            scale=factor)

        x = Conv2DTranspose(
            num_filters,
            kernel_size=factor,
            strides=factor,
            padding='same',
            kernel_initializer=kernel_initializer)(x_in)

        x = PReLU(shared_axes=[1, 2])(x)

        return x

    def __upsample_pixel_shuffle(self, x_in, num_filters, factor):
        """Upsamples the input using sub-pixel convolution.

        Args:
            x_in: The input layer.
            num_filters: The number of filters.

        Returns:
            The upsample layer.
        """

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = Lambda(self.__pixel_shuffle(scale=factor))(x)
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
