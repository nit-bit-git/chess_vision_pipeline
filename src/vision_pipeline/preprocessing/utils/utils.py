import cv2
import numpy as np
from typing import Optional, Tuple, List

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    xmy = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],        # TL
        pts[np.argmax(xmy)],      # TR
        pts[np.argmax(s)],        # BR
        pts[np.argmin(xmy)],      # BL
    ], dtype=np.float32)


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    pts = np.asarray(pts, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))




def _line_intersection(line1: Tuple[float, float], line2: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Finds exact intersection of two polar lines (rho, theta)."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    det = (c1 * s2) - (s1 * c2)
    if abs(det) < 1e-6:
        return None
    x = float(((s2 * rho1) - (s1 * rho2)) / det)
    y = float(((c1 * rho2) - (c2 * rho1)) / det)
    return (x, y)


def _normalize_line(rho: float, theta: float) -> Tuple[float, float]:
    """Normalizes angle to avoid 180-degree wrap-around splits."""
    if theta > np.pi * 0.8:  # Wrap around near 180 degrees
        theta -= np.pi
        rho = -rho
    elif theta < -np.pi * 0.2:
        theta += np.pi
        rho = -rho
    return rho, theta

def _cluster_lines(lines_raw: List, is_vertical: bool, cx: float, cy: float, tolerance: float) -> List:
    """Clusters duplicate Hough lines based on their spatial intercept.ie. image space clustering."""
    if lines_raw is None: 
        return []
        
    lines = []
    for line in lines_raw:
        rho, theta = _normalize_line(line[0][0], line[0][1])
        c, s = np.cos(theta), np.sin(theta)
        
        # Calculate where the line intersects the center axis
        if is_vertical:
            intercept = (rho - cy * s) / (c if abs(c) > 1e-5 else 1e-5)
        else:
            intercept = (rho - cx * c) / (s if abs(s) > 1e-5 else 1e-5)
            
        lines.append((intercept, rho, theta))
        
    # Sort left-to-right (verticals) or top-to-bottom (horizontals)
    lines.sort(key=lambda x: x[0])
    
    clusters = []
    curr = [lines[0]]
    for item in lines[1:]:
        if item[0] - curr[-1][0] < tolerance:
            curr.append(item)
        else:
            clusters.append(curr)
            curr = [item]
    clusters.append(curr)
    
    # Average the grouped lines
    merged = []
    for cluster in clusters:
        avg_rho = np.mean([x[1] for x in cluster])
        avg_theta = np.mean([x[2] for x in cluster])
        if avg_theta < 0:
            avg_theta += np.pi
            avg_rho = -avg_rho
        # Tuple: (rho, theta, spatial_intercept)
        merged.append((avg_rho, avg_theta, np.mean([x[0] for x in cluster])))
        
    return merged

def _merge_lines(lines, rho_tol=30, theta_tol=np.pi/10):
    """Merges lines with similar rho and theta values. ie. hough space duplicates."""
    if not lines:
        return []
    
    # Sort lines by rho (distance from origin)
    lines = sorted(lines, key=lambda x: x[0])
    groups = []
    current_group = [lines[0]]
    
    for rho, theta in lines[1:]:
        prev_rho, prev_theta = current_group[-1]
        
        # If the line is close in both distance and angle, group it
        if abs(rho - prev_rho) < rho_tol and abs(theta - prev_theta) < theta_tol:
            current_group.append((rho, theta))
        else:
            # Average the group to create a single representative line
            groups.append(np.mean(current_group, axis=0))
            current_group = [(rho, theta)]
            
    groups.append(np.mean(current_group, axis=0))
    return groups

def filter_lines_by_spacing(lines, tolerance=0.45):
    """
    Removes lines that don't fit the expected uniform spacing.
    More aggressive than before — rejects anything outside tolerance*median_gap.
    """
    if len(lines) < 3:
        return lines
    lines = sorted(lines, key=lambda x: x[0])
    gaps = [lines[i+1][0] - lines[i][0] for i in range(len(lines)-1)]
    median_gap = np.median(gaps)
    
    valid = [lines[0]]
    for i in range(1, len(lines)):
        gap = lines[i][0] - valid[-1][0]  # gap from last ACCEPTED line
        if tolerance * median_gap < gap < (2 - tolerance) * median_gap:
            valid.append(lines[i])
    return valid

def _draw_label(img, text, cx, cy, font_scale=0.38, thickness=1):
    """
    Draw a chess-coordinate label with a dark filled background box
    so it is legible on both light and dark squares.
 
    Rendering order:
      1. Semi-opaque dark rectangle  (background)
      2. Slightly offset black text  (shadow / pseudo-bold)
      3. White text on top           (foreground)
    """
    font     = cv2.FONT_HERSHEY_SIMPLEX
    pad      = 2   # pixels of padding around the text inside the box
 
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
 
    # Box coords centred on (cx, cy)
    x0 = cx - tw // 2 - pad
    y0 = cy - th // 2 - pad
    x1 = cx + tw // 2 + pad
    y1 = cy + th // 2 + pad + baseline
 
    # 1. Dark filled box (drawn directly — no alpha blend needed for debug)
    cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), cv2.FILLED)
 
    # Text anchor (bottom-left of text bounding box)
    tx = cx - tw // 2
    ty = cy + th // 2
 
    # 2. Shadow pass — dark grey, 1 px offset, thicker stroke → fake bold
    cv2.putText(img, text, (tx + 1, ty + 1), font,
                font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
 
    # 3. Foreground — bright yellow-white
    cv2.putText(img, text, (tx, ty), font,
                font_scale, (240, 230, 60), thickness, cv2.LINE_AA)
 
# ─────────────────────────────────────────────────────────────────────────────
#  CHESS COORDINATE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def get_intersection(line1, line2):
    """Solves the linear equations of two lines to find their (x, y) intersection."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    
    # Solve Ax = b
    try:
        x0, y0 = np.linalg.solve(A, b)
        return int(np.round(x0.item())), int(np.round(y0.item()))
    except np.linalg.LinAlgError:
        # Lines are parallel
        return None

# def detect_board_orientation(
#     intersections: list,
#     warped_img: np.ndarray,
# ):
#     """
#     Identify which grid corner is a1 by sampling square brightness.
 
#     # On a correctly colored chessboard, exactly two opposite corners are dark
#     # and two are light. We estimate the corner square brightness, identify the
#     # two darkest corners, and use their positions to infer board orientation.
 
#     Returns
#     -------
#     a1_row, a1_col  : grid indices of the a1 square's top-left intersection
#     flip_rows       : True → rank increases as row index decreases
#     flip_cols       : True → file increases as col index decreases
#     brightness      : 8×8 float array of median square brightness (for debug)
#     """
#     H, W = warped_img.shape[:2]
#     gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) \
#            if warped_img.ndim == 3 else warped_img
 
#     # Sample median brightness of each square via a filled polygon mask
#     brightness = np.zeros((8, 8), dtype=float)
#     ek = np.ones((5, 5), np.uint8)
#     for r in range(8):
#         for c in range(8):
#             poly = np.array(
#                 [intersections[r][c], intersections[r][c+1],
#                  intersections[r+1][c+1], intersections[r+1][c]],
#                 dtype=np.int32,
#             )
#             mask = np.zeros((H, W), dtype=np.uint8)
#             cv2.fillPoly(mask, [poly], 255)
#             mask = cv2.erode(mask, ek, iterations=1)   # avoid edge lines
#             vals = gray[mask > 0]
#             brightness[r, c] = float(np.median(vals)) if len(vals) else 128.0
 
#     # Find the two darkest corner squares
#     corner_map = {'TL': (0,0), 'TR': (0,7), 'BL': (7,0), 'BR': (7,7)}
#     corner_b   = {name: brightness[r, c] for name, (r, c) in corner_map.items()}
#     sorted_c   = sorted(corner_b.items(), key=lambda x: x[1])
#     dark1, dark2 = sorted_c[0][0], sorted_c[1][0]
 
#     # The two dark corners must be diagonal — if not, fall back to darkest single
#     if {dark1, dark2} in ({'TL','BR'}, {'TR','BL'}):
#         a1_candidate = dark1          # darker of the two diagonal dark corners
#     else:
#         a1_candidate = sorted_c[0][0]  ## need better conditioning for determining a1 as poor lighting may break
 
#     # Orientation lookup: corner name → (a1_row, a1_col, flip_rows, flip_cols)
#     orientation_map = {
#         'BL': (7, 0, False, False),   # normal white-side view
#         'BR': (7, 7, False, True),    # left-right mirror
#         'TL': (0, 0, True,  False),   # black-side view
#         'TR': (0, 7, True,  True),    # rotated 180°
#     }
#     a1_row, a1_col, flip_rows, flip_cols = orientation_map[a1_candidate]
#     return a1_row, a1_col, flip_rows, flip_cols, brightness
    
def detect_board_orientation(
    intersections: list,
    pieces_found: bool,
    warped_img: np.ndarray,
):
    """
    Identify which grid corner is a1 by analyzing aggregate structural brightness.
    
    Rule 1: a1 is ALWAYS a Dark square.
    Rule 2: a1 is ALWAYS on the White Player's side (Rank 1).
    """
    H, W = warped_img.shape[:2]
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) \
           if warped_img.ndim == 3 else warped_img

    # 1. Sample median brightness of each square
    brightness = np.zeros((8, 8), dtype=float)
    ek = np.ones((5, 5), np.uint8)
    for r in range(8):
        for c in range(8):
            poly = np.array(
                [intersections[r][c], intersections[r][c+1],
                 intersections[r+1][c+1], intersections[r+1][c]],
                dtype=np.int32,
            )
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            mask = cv2.erode(mask, ek, iterations=1)   # avoid edge lines
            vals = gray[mask > 0]
            brightness[r, c] = float(np.median(vals)) if len(vals) else 128.0

    if not pieces_found:
        """
        Empty board: corner brightness is unreliable (glare, lighting artifacts).
        Instead, use the checkerboard pattern + standard assumption:
        White is ALWAYS at the bottom in standard chess view.
        a1 is ALWAYS a dark square.
        """
        
        # Analyze the full 8×8 pattern (not just corners)
        pattern_0_bright = []  # Squares where (r+c) is even
        pattern_1_bright = []  # Squares where (r+c) is odd
        
        ek = np.ones((5, 5), np.uint8)
        for r in range(8):
            for c in range(8):
                poly = np.array([
                    intersections[r][c], intersections[r][c+1],
                    intersections[r+1][c+1], intersections[r+1][c]
                ], dtype=np.int32)
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                mask = cv2.erode(mask, ek, iterations=1)
                vals = gray[mask > 0]
                square_brightness = float(np.median(vals)) if len(vals) else 128.0
                
                if (r + c) % 2 == 0:
                    pattern_0_bright.append(square_brightness)
                else:
                    pattern_1_bright.append(square_brightness)
        
        # Which pattern is dark?
        pattern_0_median = np.median(pattern_0_bright)
        pattern_1_median = np.median(pattern_1_bright)
        is_pattern_0_dark = pattern_0_median < pattern_1_median
        
        # Standard assumption: White at bottom
        # a1 must be on white's side AND be dark
        # Bottom-right (BR): row=7, col=7 → (7+7)=14 (even) → Pattern 0
        # Bottom-left (BL):  row=7, col=0 → (7+0)=7  (odd)  → Pattern 1
        
        if is_pattern_0_dark:
            a1_candidate = 'BR'  # Pattern 0 is dark, BR is Pattern 0
        else:
            a1_candidate = 'BL'  # Pattern 1 is dark, BL is Pattern 1
        
        
        print(f"[Empty Board] Pattern 0 brightness: {pattern_0_median:.1f}")
        print(f"[Empty Board] Pattern 1 brightness: {pattern_1_median:.1f}")
        print(f"[Empty Board] Dark pattern: {0 if is_pattern_0_dark else 1}")
        print(f"[Empty Board] a1 at: {a1_candidate}")
    else:
        print("Pieces found, using piece positions to determine orientation.")
        # =========================================================================
        # STEP 1: Determine the Dark Pattern (Ignores pieces by using 32 squares)
        # =========================================================================
        # Pattern 0: TL, BR, etc. (r+c is even)
        # Pattern 1: TR, BL, etc. (r+c is odd)
        pattern_0_vals = [brightness[r, c] for r in range(8) for c in range(8) if (r + c) % 2 == 0]
        pattern_1_vals = [brightness[r, c] for r in range(8) for c in range(8) if (r + c) % 2 != 0]
        
        # Whichever pattern has the lower median is physically the dark squares
        is_pattern_0_dark = np.median(pattern_0_vals) < np.median(pattern_1_vals)

        # =========================================================================
        # STEP 2: Determine White's Location (Top vs Bottom)
        # =========================================================================
        # White pieces are brighter than Black pieces. Compare mean brightness of ranks.
        top_brightness = np.mean(brightness[0:2, :])
        bottom_brightness = np.mean(brightness[6:8, :])
        
        white_is_bottom = bottom_brightness > top_brightness

        # =========================================================================
        # STEP 3: Assign A1 (The Dark corner on White's side)
        # =========================================================================
        if white_is_bottom:
            # Check the two bottom corners: BL (7,0) and BR (7,7)
            # BL is Pattern 1 (7+0=7). BR is Pattern 0 (7+7=14).
            if is_pattern_0_dark:
                a1_candidate = 'BR' 
            else:
                a1_candidate = 'BL' 
        else:
            # White is at the Top
            # Check the two top corners: TL (0,0) and TR (0,7)
            # TL is Pattern 0 (0+0=0). TR is Pattern 1 (0+7=7).
            if is_pattern_0_dark:
                a1_candidate = 'TL' 
            else:
                a1_candidate = 'TR' 

    # Orientation lookup: corner name → (a1_row, a1_col, flip_rows, flip_cols)
    orientation_map = {
        'BL': (7, 0, False, False),   # a1 at bottom-left: rank decreases upward
        'BR': (7, 7, False, True),    # a1 at bottom-right: rank decreases upward, file decreases rightward
        'TL': (0, 0, True,  False),   # a1 at top-left: rank increases downward
        'TR': (0, 7, True,  True),    # a1 at top-right: rank increases downward, file decreases rightward
    }
    print(f"Detected a1 corner: {a1_candidate}, {orientation_map[a1_candidate]}")
    a1_row, a1_col, flip_rows, flip_cols = orientation_map[a1_candidate]
    return a1_row, a1_col, flip_rows, flip_cols, brightness    

def label_chess_squares(
    intersections: list,
    pieces_found: bool,
    warped_img:    np.ndarray,
):
    """
    Assign a chess coordinate string (e.g. 'a1', 'h8') to every one of the
    64 squares.
 
    Parameters
    ----------
    intersections : 9×9 list of (x, y) points in the *warped* image.
    warped_img    : the perspective-corrected board image (BGR or gray).
 
    Returns
    -------
    labels : list of 64 tuples  (grid_row, grid_col, chess_name)
             grid_row / grid_col are 0-indexed square positions in the
             warped image (row 0 = top of image, col 0 = left).
             chess_name is a string like 'a1' … 'h8'.
    """
    a1_row, a1_col, flip_rows, flip_cols, brightness = detect_board_orientation(intersections,pieces_found, warped_img)
 
    labels = []
    for r in range(8):
        for c in range(8):
            # ── Map grid position → chess rank / file ─────────────────────
            # rank_idx ∈ [0,7]:  0 = rank 1 … 7 = rank 8
            # file_idx ∈ [0,7]:  0 = file a … 7 = file h
            if flip_rows:
                rank_idx = r - a1_row          # increases upward in grid
            else:
                rank_idx = a1_row - r          # increases downward in grid
 
            if flip_cols:
                file_idx = a1_col - c
            else:
                file_idx = c - a1_col
 
            # Clamp to valid range (handles edge cases from imperfect detection)
            rank_idx = max(0, min(7, rank_idx))
            file_idx = max(0, min(7, file_idx))
 
            rank = rank_idx + 1                     # 1 … 8
            file = chr(ord('a') + file_idx)         # 'a' … 'h'
            chess_name = f"{file}{rank}"
 
            labels.append((r, c, chess_name))
 
    return labels
