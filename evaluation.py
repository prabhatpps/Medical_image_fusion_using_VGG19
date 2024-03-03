import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt


def psnr(img1, img2):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: PSNR value.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(img1)
    return 10 * log10((max_val ** 2) / mse)


def entropy_value(img):
    """
    Computes the entropy value of an image.

    Parameters:
        img (np.ndarray): Input image.

    Returns:
        float: Entropy value.
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    hist = hist / np.sum(hist)
    entropy_val = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy_val


def mutual_information(img1, img2):
    """
    Computes the mutual information between two images.

    Parameters:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: Mutual information value.
    """
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256)
    pxy = hist_2d / float(np.sum(hist_2d))
    joint_entropy = -np.sum(pxy * np.log2(pxy + np.finfo(float).eps))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    entropy_x = -np.sum(px * np.log2(px + np.finfo(float).eps))
    entropy_y = -np.sum(py * np.log2(py + np.finfo(float).eps))
    mutual_info_val = entropy_x + entropy_y - joint_entropy
    return mutual_info_val


def mse(img1, img2):
    """
    Computes the Mean Squared Error (MSE) between two images.

    Parameters:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: MSE value.
    """
    return np.mean((img1 - img2) ** 2)


def rmse(img1, img2):
    """
    Computes the Root Mean Squared Error (RMSE) between two images.

    Parameters:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: RMSE value.
    """
    return sqrt(mse(img1, img2))


def evaluate_metrics(image1, image2, fused_image):
    """
    Evaluates various image fusion metrics.

    Parameters:
        image1 (np.ndarray): First input image.
        image2 (np.ndarray): Second input image.
        fused_image (np.ndarray): Fused image.

    Returns:
        tuple: Tuple containing PSNR, SSIM, entropy, mutual information, MSE, and RMSE values.
    """
    psnr_value = psnr(image1, fused_image)
    ssim_value = ssim(image1, fused_image, data_range=fused_image.max() - fused_image.min())
    en_value = entropy_value(fused_image)
    mi_value = mutual_information(image1, fused_image)
    mse_value = mse(image1, fused_image)
    rmse_value = rmse(image1, fused_image)

    return psnr_value, ssim_value, en_value, mi_value, mse_value, rmse_value
