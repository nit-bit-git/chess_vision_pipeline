import cv2
import numpy as np
import matplotlib.pyplot as plt
from .transforms import extrapolate_to_n_lines
from .utils.utils import _merge_lines, filter_lines_by_spacing

def warp_board(img, corners, board_size=800, debug=True):
    """
    Warps the detected board quadrilateral into a perfect square.
    corners: [top-left, top-right, bottom-right, bottom-left]
    """
    dst = np.array([
        [0, 0],
        [board_size - 1, 0],
        [board_size - 1, board_size - 1],
        [0, board_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (board_size, board_size))

    if debug:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title("Step 2: Warped Board")
        plt.axis("off")
        plt.show()

    return warped, M

def detect_grid_lines_on_warped(warped, board_size=800, debug=True):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.Canny(closed, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        raise Exception("No lines found on warped board")

    h_lines, v_lines = [], []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        if angle < 10 or angle > 170:
            # Normalize: always positive rho
            if rho < 0:
                rho = -rho
                theta = theta + np.pi
            v_lines.append((rho, theta))
        elif 80 < angle < 100:
            h_lines.append((rho, theta))

    # ── Cluster with different tolerances for H vs V ───────────────────
    # Verticals need larger rho_tol because piece edges create nearby duplicates
    merged_h = _merge_lines(h_lines, rho_tol=20)
    merged_v = _merge_lines(v_lines, rho_tol=40)  # wider for verticals

    print(f"  After clustering: {len(merged_h)}H, {len(merged_v)}V")

    # ── Spacing filter: removes rogue lines that break uniform spacing ──
    # Apply BEFORE extrapolation so we don't enforce bad lines
    merged_h = filter_lines_by_spacing(merged_h, tolerance=0.45)
    merged_v = filter_lines_by_spacing(merged_v, tolerance=0.45)

    print(f"  After spacing filter: {len(merged_h)}H, {len(merged_v)}V")

    # ── Projection fallback if too few lines remain ────────────────────
    def projection_fallback(gray_img, axis, existing, min_expected=7):
        if len(existing) >= min_expected:
            return existing
        print(f"  Running projection fallback (axis={axis})...")
        try:
            from scipy.signal import find_peaks
        except ImportError:
            return existing

        edge = cv2.Canny(cv2.GaussianBlur(gray_img, (3, 3), 0), 30, 100)
        projection = np.sum(edge, axis=axis).astype(float)
        projection /= projection.max()
        peaks, _ = find_peaks(
            projection,
            height=0.2,
            distance=gray_img.shape[1 - axis] // 12
        )
        if axis == 1:  # horizontal lines
            result = [(float(p), np.pi / 2) for p in peaks]
        else:          # vertical lines
            result = [(float(p), 0.001) for p in peaks]  # tiny non-zero theta avoids singularity
        print(f"  Projection found {len(result)} lines")
        return result if len(result) >= len(existing) else existing

    merged_h = projection_fallback(gray, axis=1, existing=merged_h)
    merged_v = projection_fallback(gray, axis=0, existing=merged_v)

    # ── Enforce exactly 9 lines via extrapolation ──────────────────────
    merged_h = extrapolate_to_n_lines(merged_h, n=9)
    merged_v = extrapolate_to_n_lines(merged_v, n=9)

    print(f"  Final: {len(merged_h)}H, {len(merged_v)}V")

    if debug:
        overlay = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        h, w = overlay.shape[:2]

        def draw_line(img, rho, theta, color, thickness=2):
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
            pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

        for hl in merged_h:
            draw_line(overlay, hl[0], hl[1], (0, 255, 0))
        for vl in merged_v:
            draw_line(overlay, vl[0], vl[1], (255, 0, 0))

        # Also mark rho positions on axes for easy debugging
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(overlay)
        axes[0].set_title(f"Step 3: Grid Lines ({len(merged_h)}H, {len(merged_v)}V)")
        axes[0].axis("off")

        # Rho distribution plot — helps diagnose clustering issues
        all_v_rhos = sorted([l[0] for l in merged_v])
        all_h_rhos = sorted([l[0] for l in merged_h])
        axes[1].barh(range(len(all_v_rhos)), all_v_rhos, color='red', alpha=0.7, label='V rhos')
        axes[1].barh([x + 0.4 for x in range(len(all_h_rhos))], all_h_rhos, color='green', alpha=0.7, label='H rhos')
        axes[1].set_title("Rho values (should be ~evenly spaced)")
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    return merged_h, merged_v
