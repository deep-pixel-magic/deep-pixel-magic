import tensorflow as tf

compute_mean_squared_error = tf.keras.losses.MeanSquaredError()
compute_binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=False)


@tf.function
def compute_perceptual_loss(high_res_img, super_res_img, vgg_model, vgg_layer_weights, feature_scale=1 / 12.75):
    """Calculates the content loss of the super resolution image using the keras VGG model.

    Args:
        high_res_img: The high resolution image.
        super_res_img: The generated super resolution image.

    Returns:
        The content loss.
    """

    high_res_copy = tf.identity(high_res_img)
    super_res_copy = tf.identity(super_res_img)

    high_res_img_pp = tf.reverse(high_res_copy, axis=[-1])
    super_res_img_pp = tf.reverse(super_res_copy, axis=[-1])

    # high_res_img_pp = preprocess_input(high_res_copy)
    # super_res_img_pp = preprocess_input(super_res_copy)

    high_res_features = vgg_model(high_res_img_pp, training=False)
    super_res_features = vgg_model(super_res_img_pp, training=False)

    if isinstance(high_res_features, list):
        return compute_euclidean_distance_weighted(high_res_features, super_res_features, vgg_layer_weights, feature_scale)

    return compute_euclidean_distance(high_res_features, super_res_features, feature_scale)


@tf.function
def compute_euclidean_distance(high_res_features, super_res_features, feature_scale):
    """Calculates the weighted euclidean distance between the high resolution features and the super resolution features.

    Args:
        high_res_features: The high resolution features.
        super_res_features: The generated super resolution features.
        vgg_layer_weights: The weights of the VGG layers.
        feature_scale: The feature scale.

    Returns:
        The average euclidean distance across all layers.
    """

    scaled_high_res_features = high_res_features * feature_scale
    scaled_super_res_features = super_res_features * feature_scale

    loss = tf.norm(scaled_high_res_features -
                   scaled_super_res_features, ord='euclidean')
    return loss


@tf.function
def compute_euclidean_distance_weighted(high_res_features, super_res_features, vgg_layer_weights, feature_scale):
    """Calculates the weighted euclidean distance between the high resolution features and the super resolution features.

    Args:
        high_res_features: The high resolution features.
        super_res_features: The generated super resolution features.
        vgg_layer_weights: The weights of the VGG layers.
        feature_scale: The feature scale.

    Returns:
        The average euclidean distance across all layers.
    """

    loss = 0

    for hr_features, sr_features, weight in zip(high_res_features, super_res_features, vgg_layer_weights):
        scaled_hr_features = hr_features * feature_scale
        scaled_sr_features = sr_features * feature_scale

        euclidean_distance = tf.norm(
            scaled_hr_features - scaled_sr_features, ord='euclidean')

        loss += euclidean_distance * weight

    return loss / len(vgg_layer_weights)


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
        tf.ones_like(high_res_img),
        high_res_img)

    super_res_loss = compute_binary_cross_entropy(
        tf.zeros_like(super_res_img),
        super_res_img)

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
