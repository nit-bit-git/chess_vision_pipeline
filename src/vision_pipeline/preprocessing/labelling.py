
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils.utils import get_intersection, label_chess_squares, _draw_label


def get_square_centers_in_original(
    merged_h,
    merged_v,
    M_warp,
    img_orig,
    warped_img,
    pieces_found: bool,
    debug=True,
):
    """
    Compute the 64 square centers in original-image coordinates and assign
    chess coordinate labels.
 
    Parameters
    ----------
    merged_h    : list of 9 (rho, theta) or (x1,y1,x2,y2) horizontal lines.
    merged_v    : list of 9 (rho, theta) or (x1,y1,x2,y2) vertical lines.
    M_warp      : 3×3 perspective transform (original → warped).
    img_orig    : original BGR image.
    warped_img  : perspective-corrected BGR image.
    debug       : draw annotated overlays when True.
 
    Returns
    -------
    centers_orig  : np.ndarray, shape (64, 2), float32 — center of each square
                    in original-image pixel coordinates.
    intersections : 9×9 list[list[(x, y)]] — grid corner points in warped space.
    chess_labels  : list[str], length 64 — chess coordinate for each center,
                    index-aligned with centers_orig.
                    e.g. chess_labels[0] == 'a8', centers_orig[0] == its pixel.
    """
    assert len(merged_h) == 9, f"Expected 9 H lines, got {len(merged_h)}"
    assert len(merged_v) == 9, f"Expected 9 V lines, got {len(merged_v)}"
 
    # ── 1. Compute all 81 grid intersections ─────────────────────────────────
    intersections = []
    for h in merged_h:
        row_pts = []
        for v in merged_v:
            pt = get_intersection(h, v)
            if pt is None:
                pt = (0.0, 0.0)   # parallel fallback — should not occur
            row_pts.append(pt)
        intersections.append(row_pts)
 
    # Sort rows top→bottom, columns left→right
    intersections.sort(key=lambda row: np.mean([p[1] for p in row]))
    for i, row in enumerate(intersections):
        intersections[i] = sorted(row, key=lambda p: p[0])
 
    print(f"intersections[0][0] = {intersections[0][0]} (should be top-left)")
    print(f"intersections[0][8] = {intersections[0][8]} (should be top-right)")
    print(f"intersections[8][0] = {intersections[8][0]} (should be bottom-left)")
    print(f"intersections[8][8] = {intersections[8][8]} (should be bottom-right)")
    
    assert len(intersections) == 9
    assert all(len(row) == 9 for row in intersections)
 
    # ── 2. Get orientation-aware chess labels ─────────────────────────────────
    # label_chess_squares returns [(grid_row, grid_col, chess_name), ...]
    # ordered row-major (r=0..7, c=0..7)
    label_tuples = label_chess_squares(intersections, pieces_found, warped_img)
 
    # ── 3. Compute 64 square centers (warped coords) and store labels ─────────
    square_centers_warped = []
    chess_labels          = []          # ← stored, index-aligned with centers
 
    for r, c, chess_name in label_tuples:
        tl = intersections[r][c]
        tr = intersections[r][c + 1]
        bl = intersections[r + 1][c]
        br = intersections[r + 1][c + 1]
        cx = (tl[0] + tr[0] + bl[0] + br[0]) / 4.0
        cy = (tl[1] + tr[1] + bl[1] + br[1]) / 4.0
        square_centers_warped.append([cx, cy])
        chess_labels.append(chess_name)          # ← stored here
 
    centers_w    = np.array(square_centers_warped, dtype=np.float32).reshape(-1, 1, 2)
 
    # ── 4. Inverse-warp centers back to original image coordinates ────────────
    M_inv        = np.linalg.inv(M_warp)
    centers_orig = cv2.perspectiveTransform(centers_w, M_inv).reshape(-1, 2)

    square_center_map = {
        chess_labels[i]: (float(centers_orig[i, 0]), float(centers_orig[i, 1]))
        for i in range(len(chess_labels))
    }
    # ── 5. Debug overlay ──────────────────────────────────────────────────────
    if debug:
        overlay = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB).copy()
 
        for i, pt in enumerate(centers_orig):
            cx, cy    = int(pt[0]), int(pt[1])
            label     = chess_labels[i]
 
            # Dot colour: teal for dark squares, green for light squares
            is_dark   = (ord(label[0]) - ord('a') + int(label[1])) % 2 == 0
            dot_color = (0, 200, 220) if is_dark else (50, 230, 100)
 
            cv2.circle(overlay, (cx, cy), 5, dot_color, -1)
            _draw_label(overlay, label, cx, cy, font_scale=0.8, thickness=1)
 
        # Highlight a1 with a larger red ring so orientation is easy to verify
        if 'a1' in chess_labels:
            idx    = chess_labels.index('a1')
            cx, cy = int(centers_orig[idx][0]), int(centers_orig[idx][1])
            cv2.circle(overlay, (cx, cy), 12, (220, 30, 30), 2)
 
        plt.figure(figsize=(11, 8))
        plt.imshow(overlay)
        plt.title("Step 4: 64 Square Centers with Chess Labels  (a1 = red ring)",
                  fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
 
    return centers_orig, intersections, chess_labels, square_center_map