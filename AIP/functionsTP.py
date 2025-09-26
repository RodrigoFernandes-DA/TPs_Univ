import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import scipy.ndimage as ndimage


def bilateral_filter_scratch(src, d, sigma_color, sigma_space):
    radius = d // 2
    h, w = src.shape
    src = src.astype(np.float32)

    # Create Gaussian spatial kernel
    x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))

    dst = np.zeros_like(src)

    for i in range(h):
        for j in range(w):
            # Define the neighborhood boundaries
            i_min = max(i - radius, 0)
            i_max = min(i + radius + 1, h)
            j_min = max(j - radius, 0)
            j_max = min(j + radius + 1, w)

            # Extract the local region
            region = src[i_min:i_max, j_min:j_max]

            # Compute the Gaussian range kernel
            intensity_diff = region - src[i, j]
            range_kernel = np.exp(-(intensity_diff**2) / (2 * sigma_color**2))

            # Combine the spatial and range kernels
            combined_kernel = spatial_kernel[(i_min - i + radius):(i_max - i + radius),
                                             (j_min - j + radius):(j_max - j + radius)] * range_kernel

            # Normalize weights
            norm_factor = np.sum(combined_kernel)
            if norm_factor == 0:
                norm_factor = 1

            # Compute the filtered pixel value
            dst[i, j] = np.sum(region * combined_kernel) / norm_factor

    return dst.astype(src.dtype)


def isotropic_diffusion(image, num_iterations=100, eta=0.1):
    """
    Apply isotropic diffusion (heat equation) to an image.
    
    Parameters:
        image (numpy.ndarray): Grayscale image as a 2D array.
        num_iterations (int): Number of diffusion iterations.
        eta (float): Time step or learning rate.
    
    Returns:
        numpy.ndarray: Diffused image after specified iterations.
    """
    u = image.astype(np.float32)
    u = u / u.max()  # Normalize to [0, 1]

    for _ in range(num_iterations):
        u += eta * ndimage.laplace(u)

    return u


import numpy as np

def anisotropic_diffusion(image, num_iterations=15, eta=0.25, K=15):
    """
    Perona-Malik Anisotropic Diffusion (2D)
    
    Parameters:
        image (np.ndarray): Input grayscale image.
        num_iterations (int): Number of iterations to run.
        eta (float): Time step (should be ≤ 0.25 for stability).
        K (float): Edge threshold parameter (controls edge sensitivity).
    
    Returns:
        np.ndarray: The diffused image.
    """
    u = image.astype(np.float32)
    u = u / u.max()  # Normalize to [0, 1]
    
    for _ in range(num_iterations):
        # Compute gradients in 4 directions
        uN = np.roll(u, -1, axis=0)
        uS = np.roll(u, 1, axis=0)
        uE = np.roll(u, -1, axis=1)
        uW = np.roll(u, 1, axis=1)
        
        # Differences (gradients)
        dN = uN - u
        dS = uS - u
        dE = uE - u
        dW = uW - u
        
        # Diffusion coefficients
        cN = np.exp(-(dN / K)**2)
        cS = np.exp(-(dS / K)**2)
        cE = np.exp(-(dE / K)**2)
        cW = np.exp(-(dW / K)**2)
        
        # Update rule
        u += eta * (cN * dN + cS * dS + cE * dE + cW * dW)
    
    return u
