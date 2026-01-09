import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import scipy.ndimage as ndimage
from scipy.spatial.distance import directed_hausdorff


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

def image_external_energy(img_gray, w_line=0.0, w_edge=1.0, sigma=1.0):
    """
    Compute external energy where edges have LOWEST values.
    This is CRITICAL: The snake moves toward LOWEST energy points.
    """
    img = img_gray.astype(np.float32)
    
    # Smooth the image
    smooth = ndimage.gaussian_filter(img, sigma=sigma)
    
    # Line term: intensity itself (optional)
    E_line = smooth
    
    # Edge term: gradient magnitude (edges have HIGH gradient)
    gy, gx = np.gradient(smooth)
    E_edge = np.hypot(gx, gy)
    
    # Combine: we want edges to be LOW energy valleys
    # So subtract edge magnitude (edges = valleys)
    E = w_line * E_line - w_edge * E_edge
    
    # Normalize to [0, 1]
    E_min, E_max = E.min(), E.max()
    if E_max - E_min > 1e-12:
        E_normalized = (E - E_min) / (E_max - E_min)
    else:
        E_normalized = np.zeros_like(E)
    
    return E_normalized

def curvature_at_points(pts):
    """
    Compute curvature magnitude.
    """
    N = len(pts)
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    d2 = prev - 2 * pts + nxt
    curv = np.sqrt((d2 ** 2).sum(axis=1))
    return curv

def greedy_snake(img, init_pts,
                 alpha=0.1,   # continuity weight (keep points evenly spaced)
                 beta=0.4,    # curvature weight (smoothness)
                 gamma=1.0,   # external energy weight - CRITICAL
                 w_line=0.0,
                 w_edge=1.0,
                 sigma=1.0,
                 M=9,        # search window size
                 max_iter=200,
                 move_fraction_tol=0.01,
                 verbose=True):
    """
    Simplified and corrected greedy snake algorithm.
    Key fixes:
    1. Proper energy normalization
    2. Correct continuity energy computation
    3. Fixed point update logic
    """
    
    H, W = img.shape[:2]
    
    # Compute external energy map (edges = low values)
    E_ext = image_external_energy(img, w_line=w_line, w_edge=w_edge, sigma=sigma)
    
    # Initialize snake
    pts = init_pts.copy().astype(np.float32)
    N = len(pts)
    
    # Store average distance between points (for continuity)
    avg_dist = np.mean([np.linalg.norm(pts[i] - pts[(i+1)%N]) 
                       for i in range(N)])
    
    # Precompute neighborhood offsets
    half = M // 2
    offsets = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            offsets.append((dx, dy))
    
    # Sort by distance from center for better performance
    offsets.sort(key=lambda x: abs(x[0]) + abs(x[1]))
    
    history = {'moved_fraction': [], 'energy': []}
    
    for iteration in range(max_iter):
        moved = np.zeros(N, dtype=bool)
        total_energy = 0
        
        # Make a copy of current points
        new_pts = pts.copy()
        
        for i in range(N):
            current_pt = pts[i]
            prev_pt = pts[(i-1) % N]
            next_pt = pts[(i+1) % N]
            
            best_energy = float('inf')
            best_candidate = current_pt
            
            # Search in neighborhood
            for dx, dy in offsets:
                candidate = np.array([current_pt[0] + dx, current_pt[1] + dy])
                
                # Boundary check
                if (candidate[0] < 0 or candidate[0] >= W or 
                    candidate[1] < 0 or candidate[1] >= H):
                    continue
                
                # === COMPUTE ENERGIES ===
                
                # 1. Continuity energy: encourage equal spacing
                dist_to_prev = np.linalg.norm(candidate - prev_pt)
                E_cont = (dist_to_prev - avg_dist) ** 2
                
                # 2. Curvature energy: discourage sharp bends
                # Second derivative approximation
                curvature_vec = prev_pt - 2*candidate + next_pt
                E_curv = np.sum(curvature_vec ** 2)
                
                # 3. External energy: sample from energy map
                # Use bilinear interpolation for smoother sampling
                x, y = candidate
                x0, y0 = int(np.floor(x)), int(np.floor(y))
                x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
                
                # Bilinear interpolation weights
                wx = x - x0
                wy = y - y0
                
                # Get 4 neighboring energy values
                E00 = E_ext[y0, x0]
                E01 = E_ext[y1, x0]
                E10 = E_ext[y0, x1]
                E11 = E_ext[y1, x1]
                
                # Interpolated external energy
                E_img = (1-wx)*(1-wy)*E00 + wx*(1-wy)*E10 + (1-wx)*wy*E01 + wx*wy*E11
                
                # 4. Balloon force (OUTWARD pressure) - CRITICAL to prevent shrinking
                # Compute outward normal
                tangent = next_pt - prev_pt
                normal = np.array([-tangent[1], tangent[0]])
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                
                # Outward pressure energy (negative = outward movement)
                balloon_weight = 0.1  # Small outward push
                E_balloon = -balloon_weight * np.dot(normal, candidate - current_pt)
                
                # Total energy (weighted sum)
                E_total = (alpha * E_cont + 
                          beta * E_curv + 
                          gamma * E_img + 
                          E_balloon)
                
                if E_total < best_energy:
                    best_energy = E_total
                    best_candidate = candidate
            
            # Update point if we found a better position
            if not np.array_equal(best_candidate, current_pt):
                new_pts[i] = best_candidate
                moved[i] = True
                total_energy += best_energy
        
        # Update average distance for next iteration
        if np.any(moved):
            avg_dist = np.mean([np.linalg.norm(new_pts[i] - new_pts[(i+1)%N]) 
                              for i in range(N)])
        
        # Calculate moved fraction
        moved_fraction = np.mean(moved)
        history['moved_fraction'].append(moved_fraction)
        history['energy'].append(total_energy / N if N > 0 else 0)
        
        # Update points
        pts = new_pts.copy()
        
        if verbose and iteration % 20 == 0:
            print(f"Iter {iteration:3d}: Moved {moved_fraction:.3f}, "
                  f"Avg energy {history['energy'][-1]:.4f}")
        
        # Check convergence
        if moved_fraction < move_fraction_tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
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


def dice_score(mask1, mask2):
    """Compute Dice similarity coefficient between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    return (2. * intersection) / (mask1.sum() + mask2.sum())

def create_mask_from_contour(contour, shape):
    """Create a binary mask from a contour"""
    mask = np.zeros(shape, dtype=np.uint8)
    # Convert contour points to integer coordinates
    contour_int = contour.astype(np.int32)
    cv2.fillPoly(mask, [contour_int], 255)
    return mask

def hausdorff_distance(contour1, contour2):
    """Compute Hausdorff distance between two contours"""
    return max(directed_hausdorff(contour1, contour2)[0],
               directed_hausdorff(contour2, contour1)[0])