import numpy as np

from pipeline_state import PipelineState
from vision_pipeline.preprocessing.utils.utils import calculate_tilt_ratio

from .detect_corners import find_board_corners
from .perspective_correction import warp_board, detect_grid_lines_on_warped
from .labelling import get_square_centers_in_original
def preprocess_image(image: np.ndarray, pieces_found: bool, state: PipelineState) -> tuple:
    """
    Wrapper function: detect corners and return the detected corners.
    """
    corners = None
    try:
        corners = find_board_corners(image, debug=False)
        print(f"✓ Detection successful!")
        print(f"  TL: {corners[0]}")
        print(f"  TR: {corners[1]}")
        print(f"  BR: {corners[2]}")
        print(f"  BL: {corners[3]}")
    except RuntimeError as e:
            print(f"✗ Detection failed: {e}")

    tilt_ratio = calculate_tilt_ratio(corners)
    state.tilt_ratio = tilt_ratio
    warped, M = warp_board(image, corners, board_size= 800, debug=False)            # Step 2
    merged_h, merged_v = detect_grid_lines_on_warped(warped, debug=False)  # Step 3
    centers, intersections, labels, sq_map, overlay = get_square_centers_in_original(merged_h, merged_v, M, image, warped, pieces_found=pieces_found, debug=True)  # Step 4

    return  centers, intersections, labels, sq_map, overlay, M

    
