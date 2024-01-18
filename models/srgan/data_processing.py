import tensorflow as tf

def normalize_input_lr(img):
    """Normalizes a low resolution input image.
    
    Description:
        This method expects an RGB image in the range of [0, 255].
        The network expects input images to be in the range of [0, 1] (RGB space).
        
    Args:
        img: The input image.

    Returns:
        The normalized image.
    """

    return img / 255

def normalize_input_hr(img):
    """Normalizes a high resolution input image.
    
    Description:
        This method expects an RGB image in the range of [0, 255].
        The network expects input images to be in the range of [-1, 1] (RGB space).
        
    Args:
        img: The input image.

    Returns:
        The normalized image.
    """

    return (img / 127.5) - 1.0

def denormalize_output(img):
    """Denormalizes the output image.
    
    Description:
        The network outputs images in the range of [-1, 1]. This method
        converts them back to [0, 255] (RGB space).

    Args:
        img: The input image.

    Returns:
        The denormalized image.
    """

    return (img + 1.0) * 127.5

def postprocess_output(img):
    """Postprocesses the output image.

    Description:
        This method rounds the pixel values to the nearest integer and clips
        them to the range of [0, 255]. Also, it converts the pixel data type
        to uint8.

    Args:
        img: The output image.

    Returns:
        The postprocessed image.
    """

    output = denormalize_output(img)

    output = tf.round(output)
    output = tf.clip_by_value(output, 0, 255)
    output = tf.cast(output, tf.uint8)

    return output