# """
# Visualize chess moves on the board image.
# Shows source square in red, destination square in blue.
# """

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def get_grid_from_center(
    center: Tuple[float, float],
    intersections: List[List[Tuple[float, float]]],
    debug: bool = False
) -> Tuple[int, int]:
    """
    Find grid indices (r, c) from a center point by finding the closest square.
    
    Works with ANY board orientation (no assumptions about layout).
    
    Parameters
    ----------
    center : (cx, cy) tuple
        Center point coordinates
    intersections : 9×9 grid of corner points
    debug : print details
    
    Returns
    -------
    (r, c) : grid indices (0-7)
    """
    
    cx, cy = center
    min_distance = float('inf')
    best_r, best_c = 0, 0
    
    # Try all 64 squares
    for r in range(8):
        for c in range(8):
            # Get this square's 4 corners
            tl = intersections[r][c]
            tr = intersections[r][c + 1]
            bl = intersections[r + 1][c]
            br = intersections[r + 1][c + 1]
            
            # Calculate this square's center
            sq_cx = (tl[0] + tr[0] + bl[0] + br[0]) / 4.0
            sq_cy = (tl[1] + tr[1] + bl[1] + br[1]) / 4.0
            
            # Distance from given center to this square's center
            distance = np.sqrt((cx - sq_cx)**2 + (cy - sq_cy)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_r, best_c = r, c
    
    if debug:
        print(f"[Grid Lookup] Center ({cx:.1f}, {cy:.1f}) → Grid ({best_r}, {best_c}), dist={min_distance:.1f}")
    
    return best_r, best_c


def get_grid_from_square_name(
    square_name: str,
    sq_map: dict,
    intersections: List[List[Tuple[float, float]]],
    debug: bool = False
) -> Tuple[int, int]:
    """
    Get grid indices from a chess square name (e.g., 'e4').
    
    Parameters
    ----------
    square_name : str (e.g., 'e4', 'a1')
    sq_map : dict mapping square names to centers
    intersections : 9×9 grid
    debug : print details
    
    Returns
    -------
    (r, c) : grid indices
    """
    
    if square_name not in sq_map:
        print(f"[Error] Square '{square_name}' not in map")
        return None
    
    center = sq_map[square_name]
    r, c = get_grid_from_center(center, intersections, debug=debug)
    
    return r, c

# def show_move_on_board(
#     move: str,
#     sq_map: dict,
#     image: np.ndarray,
#     intersections: List[List[Tuple[float, float]]],
#     M_warp: np.ndarray,
#     alpha: float = 0.4,
#     debug: bool = True
# ) -> np.ndarray:
#     """
#     Visualize move with intersections transformed to original image space.
#     """
    
#     move = move.replace(' ', '').strip()
#     if len(move) < 4:
#         print(f"[Error] Invalid move format: {move}")
#         return image
    
#     src_square = move[0:2]
#     dst_square = move[2:4]
    
#     if src_square not in sq_map or dst_square not in sq_map:
#         print(f"[Error] Square not in map")
#         return image
    
#     if debug:
#         print(f"\n[Move Visualization]")
#         print(f"  Move: {src_square} → {dst_square}")
#         print(f"  M_warp shape: {M_warp.shape}")
    
#     # ===== Compute inverse transformation =====
#     M_inv = np.linalg.inv(M_warp)
    
#     if debug:
#         print(f"  M_inv shape: {M_inv.shape}")
    
#     # ===== Transform intersections from warped → original space =====
#     intersections_orig = []
#     for row in intersections:
#         row_orig = []
#         for pt in row:
#             # Reshape point correctly for perspectiveTransform
#             # Input must be shape (1, 1, 2)
#             pt_array = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
            
#             try:
#                 pt_transformed = cv2.perspectiveTransform(pt_array, M_inv)
#                 pt_orig = tuple(pt_transformed[0][0].astype(int))
#             except cv2.error as e:
#                 if debug:
#                     print(f"[Warning] Transform failed for {pt}: {e}")
#                 pt_orig = tuple(np.array(pt).astype(int))
            
#             row_orig.append(pt_orig)
#         intersections_orig.append(row_orig)
    
#     if debug:
#         print(f"  Transformed intersections to original space")
    
#     overlay = image.copy()
    
#     # Get grid indices from square names
#     src_r, src_c = get_grid_from_square_name(src_square, sq_map, intersections, debug=False)
#     dst_r, dst_c = get_grid_from_square_name(dst_square, sq_map, intersections, debug=False)
    
#     if debug:
#         print(f"  Source grid: ({src_r}, {src_c})")
#         print(f"  Dest grid: ({dst_r}, {dst_c})")
    
#     # Draw source square (RED) using ORIGINAL space intersections
#     if 0 <= src_r <= 7 and 0 <= src_c <= 7:
#         tl = intersections_orig[src_r][src_c]
#         tr = intersections_orig[src_r][src_c + 1]
#         br = intersections_orig[src_r + 1][src_c + 1]
#         bl = intersections_orig[src_r + 1][src_c]
#         pts = np.array([tl, tr, br, bl], dtype=np.int32)
#         cv2.fillPoly(overlay, [pts], (0, 0, 255))  # RED
    
#     # Draw destination square (BLUE) using ORIGINAL space intersections
#     if 0 <= dst_r <= 7 and 0 <= dst_c <= 7:
#         tl = intersections_orig[dst_r][dst_c]
#         tr = intersections_orig[dst_r][dst_c + 1]
#         br = intersections_orig[dst_r + 1][dst_c + 1]
#         bl = intersections_orig[dst_r + 1][dst_c]
#         pts = np.array([tl, tr, br, bl], dtype=np.int32)
#         cv2.fillPoly(overlay, [pts], (255, 0, 0))  # BLUE
    
#     # Blend
#     result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
#     # Draw arrow using sq_map centers (already in original space)
#     src_center = np.array(sq_map[src_square])
#     dst_center = np.array(sq_map[dst_square])
#     src_pt = tuple(src_center.astype(int))
#     dst_pt = tuple(dst_center.astype(int))
#     cv2.arrowedLine(result, src_pt, dst_pt, (0, 255, 255), 4, tipLength=0.3)
    
#     if debug:
#         print(f"  ✓ Move visualization complete")
    
#     return result

# def show_move_on_board_display(
#     move: str,
#     sq_map: dict,
#     image: np.ndarray,
#     intersections: List[List[Tuple[float, float]]],
#     alpha: float = 0.4,
#     M_warp: np.ndarray = None,
# ) -> None:
#     """
#     Display the move visualization in matplotlib.
#     """
#     result1 = show_move_on_board(move, sq_map, image, intersections, M_warp, alpha, debug=True)
   
#     # plt.figure(figsize=(12, 10))
#     # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#     # plt.title(f"Move: {move} (Red=Source, Blue=Destination)")
#     # plt.axis("off")
#     # plt.tight_layout()
#     # plt.show()
#     M_inv = np.linalg.inv(M_warp)
#     result2 = show_move_on_board(move, sq_map, image, intersections, M_inv, alpha, debug=False)
    
#     # Display both and see which one looks correct
#     fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
#     axes[0].imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
#     axes[0].set_title(f"Move: {move} (Using M)")
#     axes[0].axis("off")
    
#     axes[1].imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
#     axes[1].set_title(f"Move: {move} (Using M_inv)")
#     axes[1].axis("off")
    
#     plt.tight_layout()
#     plt.show()
def show_move_on_board(
    move: str,
    sq_map: dict,
    image: np.ndarray,
    intersections: List[List[Tuple[float, float]]],
    alpha: float = 0.4,
    debug: bool = True
) -> np.ndarray:
    """
    Visualize move using sq_map centers directly.
    No perspective transform needed!
    """
    
    move = move.replace(' ', '').strip()
    if len(move) < 4:
        print(f"[Error] Invalid move format: {move}")
        return image
    
    src_square = move[0:2]
    dst_square = move[2:4]
    
    if src_square not in sq_map or dst_square not in sq_map:
        print(f"[Error] Square not in map")
        return image
    
    if debug:
        print(f"\n[Move Visualization]")
        print(f"  Move: {src_square} → {dst_square}")
    
    overlay = image.copy()
    
    # Get grid indices (just for drawing order, not transformation)
    src_r, src_c = get_grid_from_square_name(src_square, sq_map, intersections, debug=False)
    dst_r, dst_c = get_grid_from_square_name(dst_square, sq_map, intersections, debug=False)
    
    # Use intersections to get corners, but in WARPED space for reference
    # Then use sq_map for actual drawing
    
    # Draw source square (RED) - circles around sq_map center
    src_center = np.array(sq_map[src_square])
    cv2.circle(overlay, tuple(src_center.astype(int)), 35, (0, 0, 255), -1)  # RED
    
    # Draw destination square (BLUE) - circles around sq_map center  
    dst_center = np.array(sq_map[dst_square])
    cv2.circle(overlay, tuple(dst_center.astype(int)), 35, (255, 0, 0), -1)  # BLUE
    
    # Blend
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Draw arrow
    src_pt = tuple(src_center.astype(int))
    dst_pt = tuple(dst_center.astype(int))
    cv2.arrowedLine(result, src_pt, dst_pt, (0, 255, 255), 4, tipLength=0.3)
    
    if debug:
        print(f"  ✓ Move visualization complete")
    
    return result


def show_move_on_board_display(
    move: str,
    sq_map: dict,
    image: np.ndarray,
    intersections: List[List[Tuple[float, float]]],
    alpha: float = 0.4,
) -> None:
    """Display the move visualization in matplotlib."""
    result = show_move_on_board(move, sq_map, image, intersections, alpha, debug=True)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Move: {move} (Red=Source, Blue=Destination)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()