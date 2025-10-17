import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import scipy.ndimage as ndimage
# import skimage.filter as filt


######## PART 1 - IMAGE DENOISING ###############

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


######## PART 2 - IMAGE SEGMENTATION ###############


def create_A(a, b, N):
    """
    a: float
    alpha parameter

    b: float
    beta parameter

    N: int
    N is the number of points sampled on the snake curve: (x(p_i), y(p_i)), i=0,...,N-1
    """
    row = np.r_[
        -2*a - 6*b, 
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N,N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A

def create_external_edge_force_gradients_from_img( img, sigma=30. ):
    """
    Given an image, returns 2 functions, fx & fy, that compute
    the gradient of the external edge force in the x and y directions.

    img: ndarray
        The image.
    """
    # Gaussian smoothing.
    # smoothed = filt.gaussian_filter( (img-img.min()) / (img.max()-img.min()), sigma )
    smoothed = cv2.GaussianBlur((img-img.min()) / (img.max()-img.min()), (5,5), sigma=10.0)
    # Gradient of the image in x and y directions.
    giy, gix = np.gradient( smoothed )
    # Gradient magnitude of the image.
    gmi = (gix**2 + giy**2)**(0.5)
    # Normalize. This is crucial (empirical observation).
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())

    # Gradient of gradient magnitude of the image in x and y directions.
    ggmiy, ggmix = np.gradient( gmi )

    def fx(x, y):
        """
        Return external edge force in the x direction.

        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ggmix[ (y.round().astype(int), x.round().astype(int)) ]

    def fy(x, y):
        """
        Return external edge force in the y direction.

        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ggmiy[ (y.round().astype(int), x.round().astype(int)) ]

    return fx, fy

def iterate_snake(x, y, a, b, fx, fy, gamma=0.1, n_iters=10, return_all=True):
    """
    x: ndarray
        intial x coordinates of the snake

    y: ndarray
        initial y coordinates of the snake

    a: float
        alpha parameter

    b: float
        beta parameter

    fx: callable
        partial derivative of first coordinate of external energy function. This is the first element of the gradient of the external energy.

    fy: callable
        see fx.

    gamma: float
        step size of the iteration
    
    n_iters: int
        number of times to iterate the snake

    return_all: bool
        if True, a list of (x,y) coords are returned corresponding to each iteration.
        if False, the (x,y) coords of the last iteration are returned.
    """
    A = create_A(a,b,x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x,y))
        y_ = np.dot(B, y + gamma*fy(x,y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append( (x_.copy(),y_.copy()) )

    if return_all:
        return snakes
    else:
        return (x,y)
    

if __name__ == "__main__":
    image_path = 'Image2.jpg'
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(L_image, cv2.COLOR_BGR2GRAY)

    smoothed = cv2.GaussianBlur((image-image.min()) / (image.max()-image.min()), (5,5), sigma=10.0)

    plt.imshow(smoothed, cmap='gray')
    plt.title('Original Image')

