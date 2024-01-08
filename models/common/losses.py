import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.applications.vgg19 import preprocess_input


compute_mean_squared_error = tf.keras.losses.MeanSquaredError()
compute_binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=False)


@tf.function
def compute_content_loss(high_res_img, super_res_img, vgg_model):
    """Calculates the content loss of the super resolution image using the keras VGG model.

    Args:
        high_res_img: The high resolution image.
        super_res_img: The generated super resolution image.

    Returns:
        The content loss.
    """

    high_res_copy = tf.identity(high_res_img)
    super_res_copy = tf.identity(super_res_img)

    high_res_img_pp = preprocess_input(high_res_copy)
    super_res_img_pp = preprocess_input(super_res_copy)

    high_res_features = vgg_model(high_res_img_pp)
    super_res_features = vgg_model(super_res_img_pp)

    return compute_perceptual_loss(high_res_features, super_res_features)


@tf.function
def compute_perceptual_loss(high_res_features, super_res_features):
    """Calculates the perceptual loss of the super resolution image.

    Args:
        high_res_features: The high resolution features.
        super_res_features: The generated super resolution features.

    Returns:
        The perceptual loss.
    """

    loss = 0
    vgg_layers_weights = [0.1, 0.1, 1, 1, 1]

    for hr_features, sr_features, weight in zip(high_res_features, super_res_features, vgg_layers_weights):
        loss += tf.reduce_mean(tf.math.abs(hr_features -
                                           sr_features)) * weight

    return loss


@tf.function
def compute_perceptual_loss_euclidean(high_res_features, super_res_features):
    """Calculates the perceptual loss of the super resolution image.

    Args:
        high_res_features: The high resolution features.
        super_res_features: The generated super resolution features.

    Returns:
        The perceptual loss.
    """

    loss = 0
    vgg_layers_weights = [0.1, 0.1, 1, 1, 1]

    for hr_features, sr_features, weight in zip(high_res_features, super_res_features, vgg_layers_weights):
        loss += K.mean(K.square(hr_features - sr_features)) * weight

    return loss


@tf.function
def compute_pixel_loss(high_res_img, super_res_img):
    """Calculates the pixel loss of the super resolution image.

    Args:
        high_res_img: The high resolution image.
        super_res_img: The generated super resolution image.

    Returns:
        The pixel loss.
    """

    return compute_mean_squared_error(high_res_img, super_res_img)


@tf.function
def compute_discriminator_loss(high_res_img, super_res_img):

    high_res_loss = compute_binary_cross_entropy(
        tf.ones_like(high_res_img), high_res_img)
    super_res_loss = compute_binary_cross_entropy(
        tf.zeros_like(super_res_img), super_res_img)

    return high_res_loss + super_res_loss


@tf.function
def compute_generator_loss(super_res_img):
    """Computes the generator loss.

    The generator's loss quantifies how well it was able to trick the discriminator. 
    Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1).

    Args:
        super_res_img: The generated super resolution image.

    Returns:
        The generator loss.
    """

    return compute_binary_cross_entropy(tf.ones_like(super_res_img), super_res_img)
