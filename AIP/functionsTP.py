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


def isotropic_diffusion(image, num_iterations=100, kappa=0.1):
    u = image.astype(np.float32)
    u = u / u.max()  # Normalize to [0, 1]

    for _ in range(num_iterations):
        u += kappa * ndimage.laplace(u)

    return u


def anisotropic_diffusion(image, num_iterations=15, kappa=0.25, K=15):
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
        u += kappa * (cN * dN + cS * dS + cE * dE + cW * dW)
    
    return u


######## PART 2 - IMAGE SEGMENTATION ###############


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


def compute_image_energy(image):
    """Compute edge map for image energy - multiple methods"""
    # Method 1: Gradient magnitude (original)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Method 2: Canny edges (often better for snakes)
    edges = cv2.Canny(image, 50, 150)
    
    # Method 3: Gradient of Gaussian
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    sobel_x_blur = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_blur = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_blur = np.sqrt(sobel_x_blur**2 + sobel_y_blur**2)
    
    # Combine methods
    edge_map = np.zeros_like(grad_mag, dtype=np.float32)
    
    # Normalize each component
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    if edges.max() > 0:
        edges_normalized = edges.astype(np.float32) / edges.max()
    else:
        edges_normalized = np.zeros_like(edges, dtype=np.float32)
    if grad_mag_blur.max() > 0:
        grad_mag_blur = grad_mag_blur / grad_mag_blur.max()
    
    # Combine: Canny edges are most reliable
    edge_map = 0.7 * edges_normalized + 0.2 * grad_mag_blur + 0.1 * grad_mag
    
    # Invert so edges have low energy
    edge_energy = 1 - edge_map
    
    return edge_energy, edge_map


def compute_normal_vectors(points):
    """Compute outward-pointing normal vectors for each point"""
    n = len(points)
    normals = np.zeros((n, 2))
    
    for i in range(n):
        # Get previous and next points
        prev = points[(i-1) % n]
        next_pt = points[(i+1) % n]
        
        # Compute tangent vector (average of forward and backward differences)
        tangent = np.array([next_pt[0] - prev[0], next_pt[1] - prev[1]])
        tangent_len = np.linalg.norm(tangent)
        if tangent_len > 0:
            tangent = tangent / tangent_len
        
        # Normal is perpendicular to tangent (rotate 90 degrees)
        normals[i] = np.array([-tangent[1], tangent[0]])
    
    return normals


def compute_curvature(points):
    """Compute curvature at each point of the snake"""
    n = len(points)
    curvatures = np.zeros(n)
    
    for i in range(n):
        # Get previous, current, and next points
        prev = points[(i-2) % n]
        curr = points[i]
        next_pt = points[(i+2) % n]
        
        # Compute first derivatives using central differences
        dx1 = curr[0] - prev[0]
        dy1 = curr[1] - prev[1]
        dx2 = next_pt[0] - curr[0]
        dy2 = next_pt[1] - curr[1]
        
        # Compute second derivatives
        ddx = dx2 - dx1
        ddy = dy2 - dy1
        
        # Compute curvature magnitude
        denom = (dx1**2 + dy1**2)**1.5
        if denom > 0:
            curvatures[i] = np.abs(dx1*ddy - dy1*ddx) / denom
        else:
            curvatures[i] = 0
    
    return curvatures


def find_local_maxima(curvatures, window=5):
    """Find local maxima in curvature"""
    n = len(curvatures)
    is_maxima = np.zeros(n, dtype=bool)
    
    for i in range(n):
        # Check neighborhood
        neighborhood = []
        for j in range(-window//2, window//2 + 1):
            idx = (i + j) % n
            neighborhood.append(curvatures[idx])
        
        # Check if current point is maximum in neighborhood
        if curvatures[i] == max(neighborhood) and curvatures[i] > 0:
            is_maxima[i] = True
    
    return is_maxima


def greedy_snake(image, initial_points, M=7, alpha=0.5, beta=0.5, gamma=1.0, 
                 kappa=0.3, curvature_threshold=0.1, max_iterations=200, 
                 min_movement_ratio=0.01, adaptive_weights=False):
    
    # Precompute image energy
    edge_energy, edge_map = compute_image_energy(image)
    
    # Initialize snake points
    points = initial_points.copy().astype(np.float32)
    n_points = len(points)
    
    # Compute initial average distance between points
    d = np.mean([np.linalg.norm(points[(i+1)%n_points] - points[i]) 
                 for i in range(n_points)])
    
    # Storage for iteration history
    history = [points.copy()]
    
    # Search offset coordinates for MxM neighborhood
    offset = M // 2
    search_offsets = []
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            search_offsets.append((dx, dy))
    
    # For adaptive weights
    iteration_weights = np.ones(max_iterations)
    if adaptive_weights:
        # Gradually reduce internal forces, keep image force
        iteration_weights = np.linspace(1.0, 0.5, max_iterations)
    
    for iteration in range(max_iterations):
        new_points = points.copy()
        moved_count = 0
        
        # Compute normal vectors for balloon force
        normals = compute_normal_vectors(points)
        
        # Adaptive weight for this iteration
        w = iteration_weights[iteration] if iteration < len(iteration_weights) else 0.5
        
        # Step 1: For each point, search neighborhood for minimum energy
        for i in range(n_points):
            # Get neighboring points indices
            prev_idx = (i-1) % n_points
            next_idx = (i+1) % n_points
            
            prev_point = points[prev_idx]
            next_point = points[next_idx]
            curr_point = points[i]
            
            # Store best position and energy
            best_energy = float('inf')
            best_position = curr_point.copy()
            
            # Search in MxM neighborhood
            for dx, dy in search_offsets:
                candidate = np.array([curr_point[0] + dx, curr_point[1] + dy])
                
                # Ensure candidate is within image bounds
                if (0 <= candidate[0] < image.shape[1] and 
                    0 <= candidate[1] < image.shape[0]):
                    
                    # 1. Continuity energy
                    dist_to_prev = np.linalg.norm(candidate - prev_point)
                    dist_to_next = np.linalg.norm(candidate - next_point)
                    
                    # Target distance adapts based on current size
                    continuity_energy = w * alpha * ((dist_to_prev - d)**2 + (dist_to_next - d)**2)
                    
                    # 2. Curvature energy
                    # Get points for curvature calculation (skip immediate neighbors for stability)
                    prev_for_curv = points[(i-2) % n_points]
                    next_for_curv = points[(i+2) % n_points]
                    
                    dx1 = candidate[0] - prev_for_curv[0]
                    dy1 = candidate[1] - prev_for_curv[1]
                    dx2 = next_for_curv[0] - candidate[0]
                    dy2 = next_for_curv[1] - candidate[1]
                    
                    # Second derivatives
                    ddx = dx2 - dx1
                    ddy = dy2 - dy1
                    
                    # Curvature magnitude
                    denom = (dx1**2 + dy1**2)**1.5
                    if denom > 0:
                        curvature = np.abs(dx1*ddy - dy1*ddx) / denom
                        curvature_energy = w * beta * curvature**2
                    else:
                        curvature_energy = 0
                    
                    # 3. Image energy (edge strength)
                    x_int = int(candidate[0])
                    y_int = int(candidate[1])
                    if 0 <= x_int < image.shape[1] and 0 <= y_int < image.shape[0]:
                        # Use gamma directly (not weighted by w)
                        image_energy_val = gamma * edge_energy[y_int, x_int]
                    else:
                        image_energy_val = float('inf')
                    
                    # 4. Balloon force (inflation/deflation)
                    # kappa > 0 expands, kappa < 0 contracts
                    normal = normals[i]
                    balloon_force = -kappa  # Negative because we want to minimize energy
                    
                    # 5. Gradient of image energy (additional pull toward edges)
                    if x_int > 0 and x_int < image.shape[1]-1 and y_int > 0 and y_int < image.shape[0]-1:
                        # Compute gradient of edge map
                        grad_x = edge_map[y_int, x_int+1] - edge_map[y_int, x_int-1]
                        grad_y = edge_map[y_int+1, x_int] - edge_map[y_int-1, x_int]
                        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                        gradient_force = -0.2 * gradient_magnitude  # Pull toward edges
                    else:
                        gradient_force = 0
                    
                    # Total energy (image energy is not weighted by w)
                    total_energy = (continuity_energy + curvature_energy + 
                                   image_energy_val + balloon_force + gradient_force)
                    
                    if total_energy < best_energy:
                        best_energy = total_energy
                        best_position = candidate
            
            # Update point if it moved
            if not np.array_equal(best_position, curr_point):
                moved_count += 1
                new_points[i] = best_position
        
        # Step 2: Compute curvature and detect corners
        curvatures = compute_curvature(new_points)
        local_maxima = find_local_maxima(curvatures, window=5)
        
        # Mark corners
        corners = local_maxima & (curvatures > curvature_threshold)
        
        # Step 3: Update average distance d
        # Adaptive distance: if snake should shrink, reduce target distance
        distances = []
        for i in range(n_points):
            if not corners[i]:
                next_i = (i+1) % n_points
                if not corners[next_i]:
                    dist = np.linalg.norm(new_points[next_i] - new_points[i])
                    distances.append(dist)
        
        if distances:
            current_avg_dist = np.mean(distances)
            # If kappa < 0 (contracting), gradually reduce target distance
            if kappa < -0.1:
                d = 0.95 * d + 0.05 * current_avg_dist
            else:
                d = current_avg_dist
        
        # Update points
        points = new_points.copy()
        history.append(points.copy())
        
        # Check stopping criterion
        movement_ratio = moved_count / n_points
        if movement_ratio < min_movement_ratio and iteration > 20:
            break
    
    return points, history, iteration + 1


def initialize_snake_from_image(image_shape, center=None, radius=None, n_points=80):
    """Initialize a circular snake around the image center or given center"""
    if center is None:
        center = (image_shape[1] // 2, image_shape[0] // 2)
    
    if radius is None:
        radius = min(image_shape) // 2.4  # Start smaller
    
    # Create circular contour
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    points = np.zeros((n_points, 2), dtype=np.float32)
    
    for i, angle in enumerate(angles):
        points[i, 0] = center[0] + radius * np.cos(angle)
        points[i, 1] = center[1] + radius * np.sin(angle)
    
    return points


def test_parameter_combinations(img_gray, initial_points, expected_mask):
    """Test different parameter combinations and return the best result"""
    
    # Define parameter combinations to test
    # Focus on combinations that help SHRINK the snake
    alphas = [0.1, 0.2, 0.3]  # Continuity weight (lower for more flexibility)
    betas = [0.1, 0.2, 0.3]   # Curvature weight (lower for more flexibility)
    gammas = [1.5, 2.0]       # Image energy weight (higher to pull toward edges)
    kappas = [-0.3, -0.5]     # Balloon force (NEGATIVE to contract)
    
    # Create combinations (3*3*2*2 = 36, we'll take 12 diverse ones)
    param_combinations = []
    for alpha in alphas[:2]:  # Take first 2 alphas
        for beta in betas[:2]:  # Take first 2 betas
            for gamma in gammas:
                for kappa in kappas:
                    param_combinations.append((alpha, beta, gamma, kappa))
                    if len(param_combinations) >= 12:
                        break
            if len(param_combinations) >= 12:
                break
        if len(param_combinations) >= 12:
            break
    
    # Ensure we have exactly 12
    param_combinations = param_combinations[:12]
    
    best_dice = -1
    best_params = None
    best_result = None
    results = []
    
    print("Testing 12 parameter combinations (focused on contraction):")
    print("-" * 70)
    print(f"{'Alpha':<8} {'Beta':<8} {'Gamma':<8} {'Kappa':<8} {'Dice':<8} {'Iterations':<12}")
    print("-" * 70)
    
    for idx, (alpha, beta, gamma, kappa) in enumerate(param_combinations):
        # Run snake algorithm with current parameters
        final_points, history, iterations = greedy_snake(
            img_gray, 
            initial_points,
            M=11,  # Larger neighborhood
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            kappa=kappa,  # Negative to contract
            curvature_threshold=0.05,
            max_iterations=200,
            min_movement_ratio=0.005,  # More sensitive stopping
            adaptive_weights=True
        )
        
        # Create mask from final snake contour
        snake_mask = create_mask_from_contour(final_points, img_gray.shape)
        
        # Compute Dice score
        dice = dice_score(snake_mask > 0, expected_mask > 0)
        
        # Store results
        results.append({
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'kappa': kappa,
            'dice': dice,
            'iterations': iterations,
            'final_points': final_points,
            'snake_mask': snake_mask,
            'history': history
        })
        
        # Print current result
        print(f"{alpha:<8.2f} {beta:<8.2f} {gamma:<8.2f} {kappa:<8.2f} "
              f"{dice:<8.4f} {iterations:<12}")
        
        # Update best result
        if dice > best_dice:
            best_dice = dice
            best_params = (alpha, beta, gamma, kappa)
            best_result = results[-1]
    
    print("-" * 70)
    print(f"Best parameters: alpha={best_params[0]:.2f}, beta={best_params[1]:.2f}, "
          f"gamma={best_params[2]:.2f}, kappa={best_params[3]:.2f}")
    print(f"Best Dice score: {best_dice:.4f}")
    
    return best_result, results


def plot_best_result(img_gray, best_result, expected_mask, initial_points):
    """Create comprehensive plot for the best result"""
    
    final_points = best_result['final_points']
    snake_mask = best_result['snake_mask']
    alpha = best_result['alpha']
    beta = best_result['beta']
    gamma = best_result['gamma']
    kappa = best_result['kappa']
    dice = best_result['dice']
    iterations = best_result['iterations']
    history = best_result['history']
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Original image with initial and final snake
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img_gray, cmap='gray')
    ax1.plot(initial_points[:, 0], initial_points[:, 1], 'c-', linewidth=1, alpha=0.7, label='Initial')
    ax1.plot(final_points[:, 0], final_points[:, 1], 'r-', linewidth=2, label='Final')
    ax1.set_title('Snake Evolution')
    ax1.legend(loc='upper right')
    ax1.axis('off')
    
    # 2. Edge map
    _, edge_map = compute_image_energy(img_gray)
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(edge_map, cmap='hot')
    ax2.plot(final_points[:, 0], final_points[:, 1], 'w-', linewidth=1.5, alpha=0.8)
    ax2.set_title('Edge Map with Final Snake')
    ax2.axis('off')
    
    # 3. Snake mask
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(snake_mask, cmap='gray')
    ax3.set_title('Snake Mask')
    ax3.axis('off')
    
    # 4. Expected contour
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(expected_mask, cmap='gray')
    ax4.set_title('Expected Contour')
    ax4.axis('off')
    
    # 5. Overlay comparison
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    # Draw expected contour in green
    contours_expected, _ = cv2.findContours(expected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_expected:
        cv2.drawContours(overlay, contours_expected, -1, (0, 255, 0), 2)
    # Draw snake contour in red
    snake_contour = final_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [snake_contour], True, (255, 0, 0), 2)
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(overlay)
    ax5.set_title(f'Comparison\nDice Score: {dice:.4f}')
    ax5.axis('off')
    
    # 6. Parameter summary and evolution
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate area change
    initial_mask = create_mask_from_contour(initial_points, img_gray.shape)
    initial_area = np.sum(initial_mask > 0)
    final_area = np.sum(snake_mask > 0)
    expected_area = np.sum(expected_mask > 0)
    
    param_text = f"PARAMETERS:\n"
    param_text += f"α (continuity): {alpha:.2f}\n"
    param_text += f"β (curvature): {beta:.2f}\n"
    param_text += f"γ (image): {gamma:.2f}\n"
    param_text += f"κ (balloon): {kappa:.2f}\n\n"
    
    param_text += f"PERFORMANCE:\n"
    param_text += f"Dice Score: {dice:.4f}\n"
    param_text += f"Iterations: {iterations}\n"
    param_text += f"Points: {len(final_points)}\n\n"
    
    param_text += f"AREA ANALYSIS:\n"
    param_text += f"Initial: {initial_area:.0f} px\n"
    param_text += f"Final: {final_area:.0f} px\n"
    param_text += f"Expected: {expected_area:.0f} px\n"
    param_text += f"Shrinkage: {100*(initial_area-final_area)/initial_area:.1f}%"
    
    ax6.text(0.1, 0.5, param_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle(f'Active Contour (Snakes) - Best Result\n'
                 f'κ={kappa:.2f} (negative = contraction)', fontsize=14)
    plt.tight_layout()
    
    # Additional figure: Evolution of the snake
    fig2, axs = plt.subplots(1, 4, figsize=(16, 4))
    iteration_indices = [0, iterations//3, 2*iterations//3, iterations-1]
    
    for idx, ax in enumerate(axs):
        iter_idx = min(iteration_indices[idx], len(history)-1)
        points_at_iter = history[iter_idx]
        
        ax.imshow(img_gray, cmap='gray')
        ax.plot(points_at_iter[:, 0], points_at_iter[:, 1], 'r-', linewidth=2)
        ax.plot(points_at_iter[:, 0], points_at_iter[:, 1], 'go', markersize=3)
        ax.set_title(f'Iteration {iter_idx}')
        ax.axis('off')
    
    plt.suptitle('Snake Evolution Over Iterations', fontsize=14)
    plt.tight_layout()
    
    return fig, fig2





######## PART 3 - IMAGE SEGMENTATION ###############
    
def generate_bounding_boxes(segments):
    boxes = []
    segment_ids = np.unique(segments)
    
    for seg_id in segment_ids:
        mask = (segments == seg_id)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            area = (ymax - ymin) * (xmax - xmin)
            aspect_ratio = (xmax - xmin) / max(ymax - ymin, 1)
            
            if area > 300 and 0.2 < aspect_ratio < 5:
                boxes.append([xmin, ymin, xmax, ymax, area])
    
    return boxes

def non_max_suppression(boxes, overlap_thresh=0.5):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    
    picked = []
    
    idxs = np.argsort(scores)[::-1]
    
    while len(idxs) > 0:
        i = idxs[0]
        picked.append(i)
        
        if len(idxs) == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersection = w * h
        area1 = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1)
        area2 = (x2[idxs[1:]] - x1[idxs[1:]] + 1) * (y2[idxs[1:]] - y1[idxs[1:]] + 1)
        
        iou = intersection / (area1 + area2 - intersection)
        
        keep_idxs = np.where(iou < overlap_thresh)[0]
        idxs = idxs[keep_idxs + 1]
    
    return boxes[picked]


