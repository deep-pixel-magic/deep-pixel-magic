from tensorflow.python.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19


class VggBuilder:
    """A helper class for building a VGG model."""

    def __init__(self, vgg_output_layer=5):
        """Constructor.

        Args:
            vgg_output_layer: The output layer to use.
        """

        self.vgg_output_layer = vgg_output_layer

    def build(self, input_shape):
        """Creates an instance of the pretrained keraas VGG model.

        Args:
            input_shape: The input shape of the model.

        Returns:
            The VGG model.
        """

        vgg = VGG19(include_top=False, input_shape=input_shape)
        output_shape = vgg.layers[self.vgg_output_layer].output

        return Model(vgg.input, output_shape)
