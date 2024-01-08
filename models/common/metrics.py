import tensorflow as tf


@tf.function
def compute_psnr(high_res_img, super_res_img):
    """Calculates the PSNR (Peak Signal-to-Noise Ratio) of a super resolution image compared to the original high resolution image.

    Args:
        high_res_img: The high resolution image.
        super_res_img: The generated super resolution image.

    Returns:
        The PSNR.
    """

    raw_psnr = tf.image.psnr(high_res_img, super_res_img, max_val=255)
    psnr = tf.reduce_mean(raw_psnr)

    return psnr


@tf.function
def compute_ssim(high_res_img, super_res_img):
    """Calculates the SSIM (Structural Similarity) of a super resolution image compared to the original high resolution image.

    Args:
        high_res_img: The high resolution image.
        super_res_img: The generated super resolution image.

    Returns:
        The SSIM.
    """

    raw_ssim = tf.image.ssim(high_res_img, super_res_img, max_val=255.0, filter_size=11,
                             filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim = tf.reduce_mean(raw_ssim)

    return ssim
