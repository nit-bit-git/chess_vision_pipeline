import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from .utils.utils import _line_intersection, _cluster_lines, _order_points

# =========================================================================
# STRATEGY 1: Fine Contour Approximation (Full Board Proportions Enforced)
# =========================================================================
def _strategy1_contour(
    img: np.ndarray,
    gray: np.ndarray,
    img_area: float,
    debug: bool = False
) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Canny with auto thresholds
    median = np.median(blurred)
    edges = cv2.Canny(blurred, 0.66 * median, 1.33 * median)
    
    # Dilate to close gaps in the board boundary
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    if debug:
        plt.figure(figsize=(6, 6))
        plt.imshow(dilated, cmap='gray')
        plt.title("Dilated Edges Input")
        plt.axis("off")
        plt.show()
    
    # RETR_TREE or RETR_LIST is safer here than RETR_EXTERNAL 
    # because broken outer lines turn internal grid-squares into valid candidates
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    board_corners = None

    # =========================================================================
    # STEP 1: COMPUTE STATISTICAL BASELINES (The Anchor)
    # =========================================================================
    contour_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter out tiny specks and things taking up almost the whole screen
        if (img_area * 0.0005) < area < (img_area * 0.2):
            contour_areas.append(area)
            
    if not contour_areas:
        raise Exception("Image contains zero structured grid components.")
        
    median_tile_area = np.median(contour_areas)
    min_full_board_area = 35.0 * median_tile_area  # Lower bound for skewed angles
    max_full_board_area = 80.0 * median_tile_area  # Upper bound restricting tables



    for cnt in contours[:25]:  # Look deeper into the hierarchy (top 20)
        area = cv2.contourArea(cnt)
        if area < (img.shape[0] * img.shape[1] * 0.1): 
            continue  # Ignore tiny noise artifacts
            
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        
        # FIX #1: Run approximation directly on the HULL, not the raw cnt
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            raw_corners = approx.reshape(4, 2).astype(np.float32)
            print(f" raw_corners: {raw_corners}")
            board_corners = _order_points(raw_corners)
            break 

    if board_corners is None:
        if debug: print("Strategy 1 Failed. Reverting to Homography Perspective Projection...")
            
    return board_corners


def _visualize_corners(
    img: np.ndarray,
    corners: np.ndarray,
    title: str = "Board Corners"
) -> None:
    """Visualize detected corners."""
    overlay = img.copy()
    
    # Draw outline
    cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 255, 0), 4)
    
    # Draw corner markers
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    labels = ["TL", "TR", "BR", "BL"]
    
    for i, (pt, col, lbl) in enumerate(zip(corners, colors, labels)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(overlay, (x, y), 15, col, -1)
        cv2.putText(overlay, lbl, (x + 20, y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2)
    
    # Display
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# =========================================================================
# STRATEGY 2: Center-Out Predictive Homography (Hallucinates Perfect 8x8 from Core Perspective)
# =========================================================================

def strategy2_grayscale_hough_grid(img: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    Strategy 2: "Center-Out Predictive Homography"
    Uses directional morphology to isolate the pristine inner grid, computes the 
    perspective from the clear center, and mathematically projects 4 squares outward 
    to hallucinate the perfect 8x8 boundaries.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    cx, cy = w / 2, h / 2
    
    if debug:
        print("\n" + "="*70)
        print("STRATEGY 2: Center-Out Predictive Homography")
        print("="*70)

    # ========== Phase 1: Contrast & Edge Detection ==========
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    enhanced = clahe.apply(gray)
    
    base = min(gray.shape)
    radius = max(5, int(base * 0.01)) | 1
    filtered = cv2.bilateralFilter(enhanced, radius, 75, 75)
    
    v = np.median(filtered)
    sigma = 0.33
    lower = int(max(0, v * (1.0 - sigma)))
    upper = int(min(255, v * (1.0 + sigma)))
    edges = cv2.Canny(filtered, lower, upper)

    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(enhanced, cmap='gray')
        axes[0].set_title("Phase 1.1: CLAHE Enhanced", fontweight="bold")
        axes[1].imshow(filtered, cmap='gray')
        axes[1].set_title("Phase 1.2: Bilateral Filter", fontweight="bold")
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title(f"Phase 1.3: Canny Edges ({lower}-{upper})", fontweight="bold")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.show()

    # ========== Phase 2: Directional Morphology (NO EROSION) ==========
    bridge_length = 5  
    line_length = max(15, min(h, w) // 80) 
    stretch_length = line_length * 4       

    # Vertical Pass
    kernel_v_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_length))
    edges_v = cv2.dilate(edges, kernel_v_bridge, iterations=1)
    kernel_v_filter = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length))
    edges_v = cv2.morphologyEx(edges_v, cv2.MORPH_OPEN, kernel_v_filter)
    kernel_v_stretch = cv2.getStructuringElement(cv2.MORPH_RECT, (1, stretch_length))
    edges_v = cv2.dilate(edges_v, kernel_v_stretch, iterations=1)

    # Horizontal Pass
    kernel_h_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_length, 1))
    edges_h = cv2.dilate(edges, kernel_h_bridge, iterations=1)
    kernel_h_filter = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1))
    edges_h = cv2.morphologyEx(edges_h, cv2.MORPH_OPEN, kernel_h_filter)
    kernel_h_stretch = cv2.getStructuringElement(cv2.MORPH_RECT, (stretch_length, 1))
    edges_h = cv2.dilate(edges_h, kernel_h_stretch, iterations=1)

    # Combine (Notice there is no erosion to destroy the 1-pixel lines!)
    edges_cleaned = cv2.bitwise_or(edges_v, edges_h)

    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(edges_v, cmap='gray')
        axes[0].set_title("Phase 2.1: Vertical Pass", fontweight="bold")
        axes[1].imshow(edges_h, cmap='gray')
        axes[1].set_title("Phase 2.2: Horizontal Pass", fontweight="bold")
        axes[2].imshow(edges_cleaned, cmap='gray')
        axes[2].set_title("Phase 2.3: Combined 1D Edges", fontweight="bold")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.show()

    # ========== Phase 3: Independent Directional Hough Lines ==========
    threshold = max(30, int(0.08 * max(h, w)))
    
    # We run Hough separately on V and H maps. Zero chance of cross-contamination.
    v_lines_raw = cv2.HoughLines(edges_v, 1, np.pi / 180, threshold)
    h_lines_raw = cv2.HoughLines(edges_h, 1, np.pi / 180, threshold)

    
    
    if v_lines_raw is None or h_lines_raw is None:
        if debug: print("[Step 1] FAILED: Missing directional lines.")
        return None

    # Cluster duplicate lines within ~4% of image width
    tolerance = min(h, w) * 0.04 
    v_lines = _cluster_lines(v_lines_raw, True, cx, cy, tolerance)
    h_lines = _cluster_lines(h_lines_raw, False, cx, cy, tolerance)

    if debug:
        print(f"[Step 2] Found {len(v_lines)} pristine V-lines and {len(h_lines)} pristine H-lines.")
        
        dbg_lines = img.copy()
        
        # Helper to draw infinite lines from (rho, theta)
        def draw_polar_line(img_tgt, rho, theta, color):
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 4000 * (-b)), int(y0 + 4000 * a))
            pt2 = (int(x0 - 4000 * (-b)), int(y0 - 4000 * a))
            cv2.line(img_tgt, pt1, pt2, color, 2)

        # Draw H lines (Green) and V lines (Red)
        for line in h_lines: draw_polar_line(dbg_lines, line[0], line[1], (0, 255, 0))
        for line in v_lines: draw_polar_line(dbg_lines, line[0], line[1], (0, 0, 255))
            
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(dbg_lines, cv2.COLOR_BGR2RGB))
        plt.title(f"Phase 3: Clustered Hough Lines ({len(h_lines)}H, {len(v_lines)}V)", fontweight="bold")
        plt.axis("off")
        plt.show()

    # ========== Phase 4: Identify The Clean Center Core ==========
    # We need a pristine window of lines. 5 lines = 4 squares (16 inner points). 
    # max_possible = min(len(v_lines), len(h_lines))
    # window_size = max(3, min(max_possible, 9))  # Use whatever is available
    window_size = 5 if len(v_lines) >= 5 and len(h_lines) >= 5 else 4
    if len(v_lines) < window_size or len(h_lines) < window_size:
        if debug: print("[Step 3] FAILED: Not enough grid lines to build perspective.")
        return None

    def get_center_window(lines, target_intercept, window_size):
        """Finds the contiguous block of lines mathematically closest to the center axis."""
        min_dist = float('inf')
        best_window = None
        for i in range(len(lines) - window_size + 1):
            window = lines[i:i+window_size]
            center_idx = window_size // 2
            # Intercept is assumed to be stored at index 2 from your cluster_lines output
            dist = abs(window[center_idx][2] - target_intercept)
            if dist < min_dist:
                min_dist = dist
                best_window = window
        return best_window

    v_core = get_center_window(v_lines, cx, window_size)
    h_core = get_center_window(h_lines, cy, window_size)

    # Accumulate the 16 (or 25) central intersections
    src_flat = []
    dst_img = []
    for u, v_line in enumerate(v_core):
        for v, h_line in enumerate(h_core):
            pt = _line_intersection((v_line[0], v_line[1]), (h_line[0], h_line[1]))
            if pt:
                src_flat.append([u, v])
                dst_img.append(pt)
                
    src_flat = np.array(src_flat, dtype=np.float32)
    dst_img = np.array(dst_img, dtype=np.float32)

    # Compute perfect perspective from the inner core
    H, _ = cv2.findHomography(src_flat, dst_img, cv2.RANSAC)
    if H is None:
        return None

    # ========== Phase 5: Predictive Topology (Snap to Center & Extrapolate) ==========
    # 1. Map the true image center (w/2, h/2) down to our flat coordinate space
    H_inv = np.linalg.inv(H)
    center_img = np.array([[[cx, cy]]], dtype=np.float32)
    center_flat = cv2.perspectiveTransform(center_img, H_inv)[0][0]
    u_c, v_c = center_flat[0], center_flat[1]
    
    # 2. Since the mathematical center of an 8x8 board is an intersection, 
    # we round our continuous center coordinates to the nearest exact integer grid node.
    u_bc = round(u_c)
    v_bc = round(v_c)
    
    # 3. An 8x8 board goes exactly 4 squares outward from the center node in every direction!
    corners_flat = np.array([
        [[u_bc - 4.0, v_bc - 4.0]],  # TL
        [[u_bc + 4.0, v_bc - 4.0]],  # TR
        [[u_bc + 4.0, v_bc + 4.0]],  # BR
        [[u_bc - 4.0, v_bc + 4.0]]   # BL
    ], dtype=np.float32)
    
    # 4. Project those mathematically perfect boundaries BACK up into the 3D camera view
    board_corners = cv2.perspectiveTransform(corners_flat, H)[:, 0, :]

    if debug:
        print("[Step 5] SUCCESS: Hallucinated 8x8 boundaries from core homography.")
        
        # Draw the visual proof
        overlay = img.copy()
        
        # 1. Draw the Inner Core (Cyan)
        for pt in dst_img:
            cv2.circle(overlay, tuple(pt.astype(int)), 6, (0, 255, 255), -1)
            
        # 2. Draw the True Mathematical Center (White Crosshair)
        center_node = cv2.perspectiveTransform(np.array([[[u_bc, v_bc]]], dtype=np.float32), H)[0][0]
        c_pt = tuple(center_node.astype(int))
        cv2.drawMarker(overlay, c_pt, (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
            
        # 3. Draw the Extrapolated 8x8 Corners (Green)
        cv2.polylines(overlay, [board_corners.astype(np.int32)], True, (0, 255, 0), 4)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        for i, (pt, col, lbl) in enumerate(zip(board_corners, colors, labels)):
            cv2.circle(overlay, tuple(pt.astype(int)), 12, col, -1)
            cv2.putText(overlay, lbl, tuple((pt + 15).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                        
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Phase 4 & 5: Core Perspective & Predicted 8x8 Boundaries", fontweight="bold")
        plt.axis("off")
        plt.show()

    return board_corners

# def extrapolate_to_n_lines(lines: List[Tuple[float, float]], n: int = 9, board_size: int = 800) -> List[Tuple[float, float]]:
#     """
#     Optimized Extrapolator: Calculates exactly how many lines are missing 
#     in a gap and interpolates them evenly.
#     """
#     if not lines:
#         return []
    
#     # Sort lines spatially by rho
#     lines = sorted(lines, key=lambda x: x[0])
    
#     # Calculate median tile size ONCE based on the actual detected lines
#     gaps = [lines[i+1][0] - lines[i][0] for i in range(len(lines)-1)]
#     median_gap = np.median(gaps) if gaps else (board_size / (n - 1))
#     if median_gap < 20:  # Safety check against degenerate gaps
#         median_gap = board_size / (n - 1)
        
#     while len(lines) < n:
#         inserted = False
        
#         # 1. Look for missing lines INSIDE the board
#         for i in range(len(lines) - 1):
#             gap_size = lines[i+1][0] - lines[i][0]
            
#             if gap_size > 1.5 * median_gap:
#                 # Math: How many lines are missing inside this gap?
#                 # A gap of 2.0x median means 1 missing line. 3.0x means 2 missing lines.
#                 missing_count = int(round(gap_size / median_gap)) - 1
                
#                 # Safety: Don't insert more lines than we actually need to reach N=9
#                 missing_count = min(missing_count, n - len(lines))
                
#                 if missing_count > 0:
#                     # Calculate exact spacing for the missing lines
#                     step_rho = gap_size / (missing_count + 1)
#                     step_theta = (lines[i+1][1] - lines[i][1]) / (missing_count + 1)
                    
#                     # Insert all missing lines evenly spaced into the gap
#                     for j in range(1, missing_count + 1):
#                         new_rho = lines[i][0] + j * step_rho
#                         new_theta = lines[i][1] + j * step_theta
#                         lines.insert(i + j, (new_rho, new_theta))
                        
#                     inserted = True
#                     break # Break to avoid list-mutation bugs, restart the while loop
        
#         # 2. If the inside is completely solid, pad the OUTSIDE edges
#         if not inserted:
#             # Check which side is furthest from the board boundary
#             dist_to_0 = lines[0][0]
#             dist_to_end = board_size - lines[-1][0]
            
#             if dist_to_0 > dist_to_end:
#                 # Prepend a line to the top/left
#                 new_rho = lines[0][0] - median_gap
#                 new_theta = lines[0][1]
#                 lines.insert(0, (new_rho, new_theta))
#             else:
#                 # Append a line to the bottom/right
#                 new_rho = lines[-1][0] + median_gap
#                 new_theta = lines[-1][1]
#                 lines.append((new_rho, new_theta))
                
#     # Final sanity trim just in case we overshot N
#     return lines[:n]


def robustly_extrapolate_lines(lines: List[Tuple[float, float]], n: int = 9, board_size: int = 800) -> List[Tuple[float, float]]:
    """
    Center-Out Ideal Grid Fitting (Rigid Phase Consensus).
    
    This solves the "pieces messing up outer edges" bug by treating the grid 
    as a single locked mathematical object. It finds the consensus angle and spacing,
    checking from the center of the board outwards to ignore local edge noise entirely.
    """
    if not lines:
        return []
        
    center_val = board_size / 2.0
    
    # Sort raw lines by distance to center to prioritize the clean inner board
    lines_center_out_raw = sorted(lines, key=lambda x: abs(x[0] - center_val))
    
    # =========================================================================
    # PHASE 1: FORCE PERFECT PARALLELISM (Center-Outwards)
    # =========================================================================
    # Take up to the 5 most central lines to determine the true angle.
    # This prevents edge pieces from skewing the median inclination.
    core_lines_for_angle = lines_center_out_raw[:min(5, len(lines_center_out_raw))]
    
    # Find the true median angle, accounting for OpenCV's 180-degree wrap-around
    thetas = [t - np.pi if t > np.pi/2 else t for _, t in core_lines_for_angle]
    median_theta = float(np.median(thetas))
    if median_theta < 0:
        median_theta += np.pi
        
    # Overwrite all lines with the center-derived median angle
    parallel_lines = []
    for rho, theta in lines:
        if abs(theta - median_theta) > np.pi / 2:
            parallel_lines.append((-rho, median_theta))
        else:
            parallel_lines.append((rho, median_theta))
            
    # =========================================================================
    # PHASE 2: GLOBAL IDEAL GRID FITTING (Rigid Phase Consensus)
    # =========================================================================
    
    # Order the now-parallel lines checking from center to outwards
    lines_center_out = sorted(parallel_lines, key=lambda x: abs(x[0] - center_val))
    
    if len(lines_center_out) > 1:
        # Calculate the gap using ONLY the safely unoccluded inner lines
        # Take up to the 5 lines closest to the center, sort them spatially to find gaps
        safe_core_lines = sorted(lines_center_out[:min(5, len(lines_center_out))], key=lambda x: x[0])
        raw_gaps = np.diff([l[0] for l in safe_core_lines])
        
        rough_median = np.median(raw_gaps)
        if rough_median < 10: 
            rough_median = board_size / (n - 1)
            
        # Recover the true tile width by dividing gaps that span missing lines
        unit_gaps = [g / max(1, round(g / rough_median)) for g in raw_gaps]
        median_gap = float(np.median(unit_gaps))
    else:
        median_gap = board_size / (n - 1)

    # Safety clamp: Ensure the gap isn't tiny noise or larger than half the board
    if median_gap < 20 or median_gap > board_size / 2:  
        median_gap = board_size / (n - 1)

    # 1. Find the Anchor Line (The most reliable line, usually near the center)
    primary_anchor_rho = lines_center_out[0][0]

    # 2. Refine the Anchor via Consensus of lines, checking from center outwards
    # Every detected line casts a "vote" for exactly where the anchor should be
    consensus_anchors = []
    for rho, _ in lines_center_out:
        steps_from_anchor = round((rho - primary_anchor_rho) / median_gap)
        implied_anchor = rho - steps_from_anchor * median_gap
        consensus_anchors.append(implied_anchor)
        
    best_anchor = float(np.median(consensus_anchors))

    # 3. Project the Infinite Grid to best fit the board bounding box
    # We want the grid's center line to align as closely to the board center as possible
    grid_mid_index = (n - 1) / 2.0
    
    # Where does the mathematical grid have a line closest to the image `center`?
    steps_to_center = round((center_val - best_anchor) / median_gap)
    center_line_rho = best_anchor + steps_to_center * median_gap
    
    # We assign this center_line_rho to the grid's middle index to lock the boundaries
    first_rho = center_line_rho - grid_mid_index * median_gap

    # Generate exactly N perfect, rigid lines
    ideal_lines = []
    for i in range(n):
        ideal_lines.append((first_rho + i * median_gap, median_theta))
        
    return ideal_lines
# def robustly_extrapolate_lines(lines: List[Tuple[float, float]], n: int = 9, board_size: int = 800) -> List[Tuple[float, float]]:
#     """
#     Global Ideal Grid Fitting (Rigid Phase Consensus).
    
#     This solves the "pieces messing up outer edges" bug by treating the grid 
#     as a single locked mathematical object. It finds the consensus angle and spacing,
#     then rigidly replaces all lines, ignoring local edge noise entirely.
#     """
#     if not lines:
#         return []
        
#     # =========================================================================
#     # PHASE 1: FORCE PERFECT PARALLELISM
#     # =========================================================================
#     # Find the true median angle, accounting for OpenCV's 180-degree wrap-around
#     thetas = [t - np.pi if t > np.pi/2 else t for _, t in lines]
#     median_theta = float(np.median(thetas))
#     if median_theta < 0:
#         median_theta += np.pi
        
#     # Overwrite all lines with the median angle
#     parallel_lines = []
#     for rho, theta in lines:
#         if abs(theta - median_theta) > np.pi / 2:
#             parallel_lines.append((-rho, median_theta))
#         else:
#             parallel_lines.append((rho, median_theta))
            
#     # Sort strictly by physical distance (rho)
#     lines = sorted(parallel_lines, key=lambda x: x[0])
    
#     # =========================================================================
#     # PHASE 2: GLOBAL IDEAL GRID FITTING (Rigid Phase Consensus)
#     # =========================================================================
#     if len(lines) > 1:
#         raw_gaps = np.diff([l[0] for l in lines])
#         rough_median = np.median(raw_gaps)
#         if rough_median < 10: 
#             rough_median = board_size / (n - 1)
            
#         # Recover the true tile width by dividing gaps that span missing lines
#         unit_gaps = [g / max(1, round(g / rough_median)) for g in raw_gaps]
#         median_gap = float(np.median(unit_gaps))
#     else:
#         median_gap = board_size / (n - 1)

#     # Safety clamp
#     if median_gap < 20 or median_gap > board_size / 2:  
#         median_gap = board_size / (n - 1)

#     # 1. Find the Anchor Line (The most reliable line, usually near the center)
#     center = board_size / 2
#     lines_by_dist = sorted(lines, key=lambda x: abs(x[0] - center))
#     primary_anchor_rho = lines_by_dist[0][0]

#     # 2. Refine the Anchor via Consensus of all lines
#     # Every detected line casts a "vote" for exactly where the anchor should be
#     consensus_anchors = []
#     for rho, _ in lines:
#         steps_from_anchor = round((rho - primary_anchor_rho) / median_gap)
#         implied_anchor = rho - steps_from_anchor * median_gap
#         consensus_anchors.append(implied_anchor)
        
#     best_anchor = float(np.median(consensus_anchors))

#     # 3. Project the Infinite Grid to best fit the board bounding box
#     # We want the grid's center line to align as closely to the board center as possible
#     grid_mid_index = (n - 1) / 2.0
    
#     # Where does the mathematical grid have a line closest to the image `center`?
#     steps_to_center = round((center - best_anchor) / median_gap)
#     center_line_rho = best_anchor + steps_to_center * median_gap
    
#     # We assign this center_line_rho to the grid's middle index to lock the boundaries
#     first_rho = center_line_rho - grid_mid_index * median_gap

#     # Generate exactly N perfect, rigid lines
#     ideal_lines = []
#     for i in range(n):
#         ideal_lines.append((first_rho + i * median_gap, median_theta))
        
#     return ideal_lines
