"""
Chessboard Corner Detection - Dual Strategy Approach

Strategy 1: Fine contour approximation (when board boundary is clear)
Strategy 2: HSV Hue → Homography → Grid Detection (fallback when pieces obscure boundary)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .transforms import _strategy1_contour, strategy2_grayscale_hough_grid, _visualize_corners, extrapolate_to_n_lines
from .utils.utils import _order_points, _merge_lines, filter_lines_by_spacing


def find_board_corners(img: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Detect 4 corners of chessboard using dual-strategy approach.
    
    Strategy 1: Fine contour approximation on dilated edges
    ├─ Works when: board boundary is clear and isolated
    ├─ Speed: ~30ms
    └─ Returns first successful match
    
    Strategy 2: HSV Hue → Homography → Grid Detection
    ├─ Works when: pieces obscure outer boundary, only grid visible
    ├─ Speed: ~100ms  
    └─ Fallback when Strategy 1 fails
    
    Parameters
    ----------
    img : BGR image
    debug : show step-by-step visualization
    
    Returns
    -------
    corners : (4, 2) array [TL, TR, BR, BL]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    
    # =========================================================================
    # STRATEGY 1: Fine Contour Approximation (Fast, Deterministic)
    # =========================================================================
    if debug:
        print("[Main] Strategy 1: Fine contour approximation...")
    
    try:
        corners = _strategy1_contour(img, gray, img_area, debug=debug)
        if corners is not None:
            if debug:
                print("[Main] ✓ Strategy 1 SUCCEEDED")
                _visualize_corners(img, corners, "Strategy 1: Contour Approximation")
            return corners
    except Exception as e:
        if debug:
            print(f"[Main] Strategy 1 failed: {e}")
    
    
    # =========================================================================
    # STRATEGY 2: Grayscale Canny + Hough Grid Detection (Robust Fallback)
    # =========================================================================
    if debug:
        print("[Main] Strategy 2: Grayscale Canny + Hough grid detection...")
    
    try:
        corners = strategy2_grayscale_hough_grid(img, debug=debug)
        if corners is not None:
            corners = _order_points(corners)
            if debug:
                print("[Main] ✓ Strategy 2 SUCCEEDED")
                _visualize_corners(img, corners, "Strategy 2: Grayscale Canny + Hough Grid")
            return corners
    except Exception as e:
        if debug:
            print(f"[Main] Strategy 2 failed: {e}")
    
    # =========================================================================
    # FAILURE: Both strategies failed
    # =========================================================================
    raise RuntimeError(
        "Board corner detection failed with both strategies.\n"
        "Possible causes:\n"
        "  - Image too dark or washed out\n"
        "  - Board at extreme angle (>75°)\n"
        "  - Board partially obscured\n"
        "  - Non-standard board colors\n"
        "Try: better lighting, closer image, frontal angle"
    )

