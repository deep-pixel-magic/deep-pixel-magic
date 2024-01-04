from tensorflow.python.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19


class VggBuilder:
    """A helper class for building a VGG based model.

    The pretrained VGG model can be used for style transfer or perceptual loss.
    """

    def __init__(self, layers):
        """Constructor.

        Args:
            layer: The output layer of the pretrained VGG19 network to use.
        """

        self.layers = layers

    def build(self, input_shape):
        """Creates an instance of the pretrained keraas VGG model.

        Args:
            input_shape: The input shape of the model.

        Returns:
            The VGG model.
        """

        vgg = VGG19(include_top=False, weights="imagenet",
                    input_shape=input_shape)
        vgg.trainable = False

        outputs = [vgg.get_layer(layer).output for layer in self.layers]

        model = Model(vgg.input, outputs)
        return model
