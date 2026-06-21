# """
# Main chessboard corner detection entry point.

# Key improvements:
#   • Multi-candidate ranking: doesn't short-circuit on first success
#   • Cross-validation: validates candidates across all stages
#   • Proper fallback: if Stage N returns weak candidates, Stage N+1 still runs
#   • Configurable stage weighting
#   • Clean debug visualization (separate from core logic)
# """

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Tuple, Optional, List, Dict, Any

# from .transforms import (

#     _stage1_contour,
#     _stage2_color,
#     _stage3_hough,
#     _stage4_features,
#     snap_all_corners,
#     _stage1_contour_from_edges,
#     order_points,
#     is_good_quad,
#     polygon_area,
# )
# def score_quad_candidate(quad: np.ndarray, img_shape: Tuple[int, int]) -> float:
#     """
#     Evaluates how likely a 4-point polygon is to be a real chessboard.
#     Returns a score between 0.0 (terrible) and 1.0 (perfect match).
#     """
#     img_h, img_w = img_shape[:2]
#     img_area = img_h * img_w
    
#     # Format shapes correctly
#     pts = quad.reshape(4, 2).astype(np.float32)
    
#     # 1. Size Verification (Too small means noise, too big means background frame)
#     contour_area = cv2.contourArea(pts)
#     area_ratio = contour_area / img_area
#     if area_ratio < 0.05 or area_ratio > 0.95:
#         return 0.0  # Discard immediately
        
#     # 2. Extent/Solidity Check (Is it filled like a bounding box?)
#     # Get minimum area bounding rectangle (handles rotation)
#     rect = cv2.minAreaRect(pts)
#     rect_w, rect_h = rect[1]
#     rect_area = rect_w * rect_h
    
#     if rect_area == 0: return 0.0
#     extent = contour_area / rect_area # Perfect rectangles are close to 1.0
    
#     # 3. Aspect Ratio Check (Is it uniform or highly stretched?)
#     if rect_h == 0 or rect_w == 0: return 0.0
#     aspect_ratio = min(rect_w, rect_h) / max(rect_w, rect_h) 
#     # Boards look like a square (1.0). Even with perspective distortion, rarely < 0.4
#     if aspect_ratio < 0.35:
#         return 0.0
        
#     # 4. Angle Orthogonality Check (Are the corners close to 90 degrees?)
#     # Calculate inner angles via vector dot products
#     def get_angle(p1, p2, p3):
#         v1 = p1 - p2
#         v2 = p3 - p2
#         cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
#         return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

#     angles = [
#         get_angle(pts[3], pts[0], pts[1]), # Corner 0
#         get_angle(pts[0], pts[1], pts[2]), # Corner 1
#         get_angle(pts[1], pts[2], pts[3]), # Corner 2
#         get_angle(pts[2], pts[3], pts[0])  # Corner 3
#     ]
    
#     # Penalize if any corner angle is highly acute or obtuse (e.g., < 45 or > 135)
#     angle_penalty = 1.0
#     for angle in angles:
#         if angle < 45.0 or angle > 135.0:
#             angle_penalty *= 0.2 # Heavily penalize non-orthogonal shapes
            
#     # Composite scoring weight distribution
#     # We favor a balance of size, rect-completeness, square shape, and sharp angles
#     final_score = area_ratio * extent * aspect_ratio * angle_penalty
    
#     return float(final_score)

# def find_board_corners(
#     img: np.ndarray,
#     debug: bool = False
# ) -> np.ndarray:
#     """
#     Simple Stage 1 with merged HSV Range + Hue OTSU preprocessing.
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_shape = img.shape
#     h, w = img_shape[:2]
#     base = min(h, w)
    
#     # ========== STAGE 1: CONTOUR DETECTION (with pre-computed edges) ==========
#     stage1_candidates = _stage1_contour(gray, img_shape)
#     candidates = []
#     for item in stage1_candidates:
#         quad_pts = item["quad"]
#         geom_score = score_quad_candidate(quad_pts, img_shape)  # Already a structured array from order_points()
#         combined_score = geom_score * item["score"]
#         print(f"Stage 1 candidate: item_score={item['score']:.4f}, geom_score={geom_score:.4f}, combined_score={combined_score:.4f}")
#         if geom_score > 0:  # Validates aspect ratio, convexity, and orthogonality
#             candidates.append({
#                 "quad": quad_pts, 
#                 "score": combined_score
#             })
                
#     if not candidates:
#         raise RuntimeError("No valid quadrilateral structures found that match a board profile.")
         
#     # 3. Pick candidate with the highest combined profile
#     best_candidate = max(candidates, key=lambda c: c["score"])
#     best_quad = best_candidate["quad"].astype(np.float32)
    
#     print(f"Selected quad with geometric confidence score: {best_candidate['score']:.4f}")
    
#     # Refine with Harris snapping
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     snapped = snap_all_corners(best_quad, blurred)
#     if is_good_quad(snapped, img_shape):
#         corners = order_points(snapped)
#     else:
#         corners = best_quad
    
#     if debug:
#         _visualize_corners(img, corners, stages_used="contour")
    
#     return corners

# def _draw_corners_on_image(
#     img: np.ndarray,
#     corners: np.ndarray,
#     title: str = "",
#     highlight: bool = False
# ) -> np.ndarray:
#     """
#     Draw corners on image with clear markers and labels.
    
#     Parameters
#     ----------
#     img : BGR image
#     corners : (4, 2) corner points
#     title : subplot title
#     highlight : if True, use larger markers (for final winner)
    
#     Returns
#     -------
#     overlay : image with drawn corners
#     """
#     overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
#     h, w = img.shape[:2]
    
#     # Corner markers
#     colors = [
#         (255, 100, 100),   # TL - red
#         (100, 255, 100),   # TR - green
#         (100, 100, 255),   # BR - blue
#         (255, 255, 100),   # BL - yellow
#     ]
#     labels = ["TL", "TR", "BR", "BL"]
    
#     # Marker size scales with image
#     if highlight:
#         r_dot = max(12, int(0.015 * min(h, w)))  # Larger for final winner
#         line_width = 3
#     else:
#         r_dot = max(8, int(0.010 * min(h, w)))   # Smaller for stage results
#         line_width = 2
    
#     # Draw corner points
#     for i, (pt, col, lbl) in enumerate(zip(corners, colors, labels)):
#         x, y = int(pt[0]), int(pt[1])
#         cv2.circle(overlay, (x, y), r_dot, col, -1)
        
#         # Draw corner label with background
#         font_scale = max(0.5, r_dot / 12)
#         text_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        
#         # Text background
#         text_x = x + r_dot + 5
#         text_y = y - 5
#         cv2.rectangle(overlay, 
#                      (text_x - 2, text_y - text_size[1] - 2),
#                      (text_x + text_size[0] + 2, text_y + 2),
#                      col, -1)
        
#         # Text label
#         cv2.putText(overlay, lbl, (text_x, text_y),
#                    cv2.FONT_HERSHEY_SIMPLEX,
#                    font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    
#     # Draw board outline
#     corner_ints = corners.astype(np.int32)
#     cv2.polylines(overlay, [corner_ints], True, (0, 255, 0), line_width)
    
#     return overlay


# def _visualize_corners_detailed(
#     img: np.ndarray,
#     best_corners: np.ndarray,
#     stages_used: str,
#     all_stage_results: Dict[str, List[np.ndarray]]
# ) -> None:
#     """
#     Display detailed per-stage corner detection results with multi-subplot visualization.
    
#     Shows:
#       - Top row: All stage results
#       - Bottom row: Final winner (largest)
    
#     Parameters
#     ----------
#     img : Input BGR image
#     best_corners : Final selected corners (4, 2)
#     stages_used : String of which stages ran (e.g., "contour → goodfeatures")
#     all_stage_results : Dict mapping stage_name -> list of corner arrays
#     """
#     stage_names = ["Stage 1: Contour", "Stage 2: Color", "Stage 3: Hough", "Stage 4: Features"]
#     stage_keys = ["contour", "color_hsv", "hough_lines", "goodfeatures"]
    
#     # Determine number of subplots (5 total: 4 stages + 1 final)
#     fig, axes = plt.subplots(2, 3, figsize=(16, 10))
#     axes = axes.flatten()
    
#     # Plot each stage's results
#     for idx, (stage_name, stage_key) in enumerate(zip(stage_names, stage_keys)):
#         ax = axes[idx]
#         overlay = img.copy()
        
#         if all_stage_results[stage_key]:
#             # Draw all candidates from this stage (top 3)
#             candidates = all_stage_results[stage_key][:3]
            
#             for i, corners in enumerate(candidates):
#                 corner_ints = corners.astype(np.int32)
                
#                 # Draw outline with varying opacity/thickness
#                 alpha = 0.3 + (0.5 * (1 - i / max(len(candidates), 1)))
#                 thickness = 2 if i == 0 else 1
                
#                 overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
#                 cv2.polylines(overlay, [corner_ints], True, (0, 255, 0), thickness)
#                 overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
#             # Draw best candidate for this stage more prominently
#             best_for_stage = candidates[0]
#             overlay = _draw_corners_on_image(overlay, best_for_stage, highlight=False)
            
#             ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#             ax.set_title(f"{stage_name}\n({len(candidates)} candidates)", fontsize=11, fontweight="bold")
#         else:
#             # No results from this stage
#             ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             ax.set_title(f"{stage_name}\n(no results)", fontsize=11, color="red", fontweight="bold")
        
#         ax.axis("off")
    
#     # Final winner (larger, prominent)
#     ax_final = axes[4]
#     overlay_final = _draw_corners_on_image(img, best_corners, highlight=True)
#     ax_final.imshow(cv2.cvtColor(overlay_final, cv2.COLOR_BGR2RGB))
#     ax_final.set_title("🏆 FINAL WINNER\n(Highest Score)", 
#                       fontsize=13, fontweight="bold", color="darkgreen",
#                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
#     ax_final.axis("off")
    
#     # Hide unused subplot
#     axes[-1].axis("off")
    
#     # Overall title
#     fig.suptitle(f"Chessboard Corner Detection - Per-Stage Results\n[Stages Run: {stages_used}]",
#                 fontsize=14, fontweight="bold", y=0.98)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()


# def _visualize_corners(
#     img: np.ndarray,
#     corners: np.ndarray,
#     stages_used: str
# ) -> None:
#     """
#     Simple single-image visualization (backward compatible).
#     Use _visualize_corners_detailed for detailed per-stage view.
#     """
#     overlay = _draw_corners_on_image(img, corners, highlight=True)
    
#     fig, ax = plt.subplots(1, 1, figsize=(12, 10))
#     ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#     ax.set_title(f"Chessboard Corners  [stages: {stages_used}]", 
#                 fontsize=14, fontweight="bold")
#     ax.axis("off")
#     plt.tight_layout()
#     plt.show()


