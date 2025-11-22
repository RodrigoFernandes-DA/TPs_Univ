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


"""
Greedy Deformable Contour (Snakes) implementation in Python.

Usage:
  - Provide image_path and initial_points (Nx2 array).
  - Call greedy_snake(...)
  - The function returns final_points and some diagnostics and plots the result.

This implementation depends on:
  numpy, scipy, opencv (cv2), matplotlib

If you don't have an initial contour, set init_from='circle' to auto-generate one.
"""


def image_external_energy(img_gray, w_line=0.0, w_edge=1.0, sigma=1.0):
    """
    Compute external energy:
       E_ext = w_line * I  - w_edge * |grad( G_sigma * I )|
    Returns normalized energy (lower is more attractive).
    """
    img = img_gray.astype(np.float32)
    # optional smoothing
    smooth = ndimage.gaussian_filter(img, sigma=sigma)
    # line: intensity itself
    E_line = smooth
    # edge: gradient magnitude
    gy, gx = np.gradient(smooth)
    E_edge = np.hypot(gx, gy)
    E = w_line * E_line - w_edge * E_edge
    # normalize to [0,1]
    E = (E - E.min()) / (E.max() - E.min() + 1e-12)
    return E


def curvature_at_points(pts):
    """
    Discrete curvature magnitude (squared second derivative)
    pts: (N,2)
    returns: curv (N,)
    """
    N = len(pts)
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    # second difference vector
    d2 = prev - 2 * pts + nxt
    curv = np.sqrt((d2 ** 2).sum(axis=1))
    return curv


def greedy_snake(img, init_pts,
                 alpha=0.1,  # elasticity (continuity)
                 beta=0.4,   # global stiffness (will allow per-point beta_j)
                 gamma=1.0,  # weight of external energy
                 w_line=0.0,
                 w_edge=1.0,
                 sigma=1.0,
                 M=7,       # search window size (MxM)
                 max_iter=250,
                 move_fraction_tol=0.01,  # stop when fraction moved < this
                 corner_thresh_rel=0.6,   # relative threshold for corner detection
                 shrink_M_if_stable=True,
                 min_M=3,
                 verbose=True):
    """
    Greedy snake algorithm.

    img: grayscale image (H,W) array
    init_pts: (N,2) float array of initial snake points (x,y)
    returns: final_pts, history
    """

    img_gray = img.astype(np.float32)
    H, W = img_gray.shape

    E_ext = image_external_energy(img_gray, w_line=w_line, w_edge=w_edge, sigma=sigma)
    # For fast lookup:
    ext_interp = E_ext  # we'll index nearest-pixel positions (neighborhood small)

    pts = init_pts.copy().astype(np.float32)
    N = len(pts)

    # initial per-point beta_j
    beta_j = np.ones(N, dtype=np.float32) * beta

    # Precompute neighbor offsets for the MxM grid centered at 0
    half = M // 2
    offsets = [(dx, dy) for dy in range(-half, half + 1) for dx in range(-half, half + 1)]

    history = {'moved_fraction': [], 'M': [], 'curv_max': []}

    for it in range(max_iter):
        moved = np.zeros(N, dtype=bool)
        # compute curvature and detect corners: local maxima in curvature
        curv = curvature_at_points(pts)
        # local maxima check
        is_local_max = (curv > np.roll(curv, 1)) & (curv > np.roll(curv, -1))
        if curv.max() > 0:
            corner_thresh = corner_thresh_rel * curv.max()
        else:
            corner_thresh = 0.0
        corner_mask = is_local_max & (curv >= corner_thresh)
        # set beta_j = 0 at corners (so stiffness doesn't smooth corners away)
        beta_j = np.where(corner_mask, 0.0, beta)

        # For each point i, search neighborhood
        for i in range(N):
            pi = pts[i]
            pprev = pts[(i - 1) % N]
            pnext = pts[(i + 1) % N]

            best_E = 1e12
            best_p = pi.copy()

            # If M is small, create offsets accordingly
            for dx, dy in offsets:
                candidate = np.array([pi[0] + dx, pi[1] + dy], dtype=np.float32)
                x, y = candidate
                # check boundaries (keep inside image bounds)
                if x < 0 or x >= W or y < 0 or y >= H:
                    continue

                # Internal energy: continuity (elasticity)
                E_cont = np.sum((candidate - pprev) ** 2)

                # Curvature energy: squared second difference magnitude
                E_curv = np.sum((pprev - 2 * candidate + pnext) ** 2)

                # External energy: image energy at nearest pixel
                ix = int(round(x))
                iy = int(round(y))
                # guard
                ix = min(max(ix, 0), W - 1)
                iy = min(max(iy, 0), H - 1)
                E_image = ext_interp[iy, ix]

                # Total energy
                E = alpha * E_cont + beta_j[i] * E_curv + gamma * E_image

                if E < best_E:
                    best_E = E
                    best_p = candidate

            # Move point if improved
            if np.linalg.norm(best_p - pi) >= 1e-3:
                pts[i] = best_p
                moved[i] = True

        frac_moved = moved.mean()
        history['moved_fraction'].append(frac_moved)
        history['M'].append(M)
        history['curv_max'].append(curv.max())

        if verbose:
            print(f"iter {it+1:03d}: moved_frac={frac_moved:.4f}, M={M}, curv_max={curv.max():.4f}")

        # optionally shrink M if stable
        if shrink_M_if_stable and frac_moved < 0.02 and M > min_M:
            # reduce search window to refine
            M = max(min_M, M - 2)
            half = M // 2
            offsets = [(dx, dy) for dy in range(-half, half + 1) for dx in range(-half, half + 1)]
            if verbose:
                print(f"  -> shrinking neighborhood to M={M}")

        # stopping criterion
        if frac_moved <= move_fraction_tol:
            if verbose:
                print(f"Converged at iteration {it+1}. moved_frac={frac_moved:.4f}")
            break

    return pts, history, E_ext


def plot_snake(img, init_pts, final_pts, E_ext=None, title="Snake result"):
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    # initial contour
    init = np.vstack([init_pts, init_pts[0]])
    final = np.vstack([final_pts, final_pts[0]])
    plt.plot(init[:, 0], init[:, 1], '--', linewidth=1.5, label='initial')
    plt.plot(final[:, 0], final[:, 1], '-', linewidth=2.0, label='final')
    plt.scatter(final_pts[:, 0], final_pts[:, 1], s=10)
    plt.legend()
    plt.title(title)
    plt.axis('off')
    plt.show()

    if E_ext is not None:
        plt.figure(figsize=(6, 5))
        plt.title("External energy (normalized)")
        plt.imshow(E_ext, cmap='viridis')
        plt.colorbar()
        plt.axis('off')
        plt.show()


