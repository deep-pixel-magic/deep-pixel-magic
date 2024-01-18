import tensorflow as tf


@tf.function
def compute_psnr(high_res_img, super_res_img):
    """Calculates the PSNR (Peak Signal-to-Noise Ratio) of a super resolution image compared to the original high resolution image.

    Description:
        The PSNR is calculated on a batch of images.
        This method expects the images to be in the range of [0, 255].

    Args:
        high_res_img: The high resolution image.
        super_res_img: The generated super resolution image.

    Returns:
        The PSNR.
    """

    batched_psnr = tf.image.psnr(high_res_img, super_res_img, max_val=255.0)
    return tf.reduce_mean(batched_psnr)


@tf.function
def compute_ssim(high_res_img, super_res_img):
    """Calculates the SSIM (Structural Similarity) of a super resolution image compared to the original high resolution image.

    Description:
        The SSIM is calculated on a batch of images.
        This method expects the images to be in the range of [0, 255].

    Args:
        high_res_img: The high resolution image.
        super_res_img: The generated super resolution image.

    Returns:
        The SSIM.
    """

    batched_ssim = tf.image.ssim(high_res_img, super_res_img, max_val=255.0, filter_size=11,
                                 filter_sigma=1.5, k1=0.01, k2=0.03)
    return tf.reduce_mean(batched_ssim)
