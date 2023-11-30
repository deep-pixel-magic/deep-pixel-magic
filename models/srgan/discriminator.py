from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, Lambda


class SrganDiscriminator:
    """Represents the SRGAN discriminator network."""

    def __init__(self, resolution_in=96):
        """Constructor."""

        self.resolution_in = resolution_in

    def build(self, num_filters=64):
        """Builds the discriminator network.

        Args:
            num_filters: The number of filters.

        Returns:
            The discriminator model.
        """

        shape = (self.resolution_in, self.resolution_in, 3)

        x_in = Input(shape=shape)
        x = Lambda(self.__normalize())(x_in)

        x = self.__discriminator_block(
            x, num_filters, batch_normalization=False)
        x = self.__discriminator_block(x, num_filters, strides=2)

        x = self.__discriminator_block(x, num_filters * 2)
        x = self.__discriminator_block(x, num_filters * 2, strides=2)

        x = self.__discriminator_block(x, num_filters * 4)
        x = self.__discriminator_block(x, num_filters * 4, strides=2)

        x = self.__discriminator_block(x, num_filters * 8)
        x = self.__discriminator_block(x, num_filters * 8, strides=2)

        x = Flatten()(x)

        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(x_in, x)

    def __discriminator_block(self, x_in, num_filters, strides=1, batch_normalization=True, momentum=0.8):
        """Creates a discriminator block.

        Args:
            x_in: The input layer.
            num_filters: The number of filters.
            strides: The strides.
            batch_normalization: Whether to use batch normalization.
            momentum: The momentum for the batch normalization layers.

        Returns:
            A discriminator layer.
        """

        x = Conv2D(num_filters, kernel_size=3,
                   strides=strides, padding='same')(x_in)

        if batch_normalization:
            x = BatchNormalization(momentum=momentum)(x)

        return LeakyReLU(alpha=0.2)(x)

    def __normalize(self):
        """Normalizes the input.

        Assumes an input interval of [0, 255].
        """

        return lambda x: x / 127.5 - 1
