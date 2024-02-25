import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(img1)
    return 10 * log10((max_val ** 2) / mse)

def entropy_value(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    hist = hist / np.sum(hist)
    entropy_val = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy_val

def mutual_information(img1, img2):
    # Compute histograms
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256)

    # Compute joint entropy
    pxy = hist_2d / float(np.sum(hist_2d))
    joint_entropy = -np.sum(pxy * np.log2(pxy + np.finfo(float).eps))

    # Compute individual entropies
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    entropy_x = -np.sum(px * np.log2(px + np.finfo(float).eps))
    entropy_y = -np.sum(py * np.log2(py + np.finfo(float).eps))

    # Compute mutual information
    mutual_info_val = entropy_x + entropy_y - joint_entropy
    return mutual_info_val

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def rmse(img1, img2):
    return sqrt(mse(img1, img2))

def evaluate_metrics(image1, image2, fused_image):
    psnr_value = psnr(image1, fused_image)
    ssim_value = ssim(image1, fused_image, data_range=fused_image.max() - fused_image.min())
    en_value = entropy_value(fused_image)
    mi_value = mutual_information(image1, fused_image)
    mse_value = mse(image1, fused_image)
    rmse_value = rmse(image1, fused_image)

    return psnr_value, ssim_value, en_value, mi_value, mse_value, rmse_value
