

    # import cv2
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from typing import Tuple, Optional, List, Dict, Any


    # # ─────────────────────────────────────────────────────────────────────────────
    # #  GEOMETRY UTILITIES
    # # ─────────────────────────────────────────────────────────────────────────────

    # def order_points(pts: np.ndarray) -> np.ndarray:
    #     """
    #     Order 4 points as [TL, TR, BR, BL].
        
    #     Uses the sum/difference trick with correct sign convention:
    #       TL → smallest (x+y)
    #       BR → largest  (x+y)
    #       TR → largest  (x-y)
    #       BL → smallest (x-y)
    #     """
    #     pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    #     s = pts.sum(axis=1)          # x+y
    #     xmy = pts[:, 0] - pts[:, 1]  # x-y
    #     return np.array([
    #         pts[np.argmin(s)],        # TL
    #         pts[np.argmax(xmy)],      # TR
    #         pts[np.argmax(s)],        # BR
    #         pts[np.argmin(xmy)],      # BL
    #     ], dtype=np.float32)


    # def polygon_area(pts: np.ndarray) -> float:
    #     """Shoelace formula for polygon area."""
    #     pts = np.asarray(pts, dtype=np.float32)
    #     x, y = pts[:, 0], pts[:, 1]
    #     return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


    # def angle_cos(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    #     """Cosine of interior angle at p1."""
    #     v1 = p0 - p1
    #     v2 = p2 - p1
    #     return abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))


    # def is_good_quad(
    #     quad: np.ndarray,
    #     img_shape: Tuple[int, int, int],
    #     min_area_frac: float = 0.08,
    #     max_cos: float = 0.40
    # ) -> bool:
    #     """
    #     Geometric sanity check: area, convexity, and angle constraints.
        
    #     Parameters
    #     ----------
    #     quad : (4, 2) quad corners
    #     img_shape : (H, W, C) image shape
    #     min_area_frac : quad must be ≥ this fraction of image area
    #     max_cos : max interior angle cosine (near 90° ≈ 0)
    #     """
    #     h, w = img_shape[:2]
    #     area = polygon_area(quad)
        
    #     if area < min_area_frac * h * w:
    #         return False
        
    #     cnt = quad.reshape(-1, 1, 2).astype(np.int32)
    #     if not cv2.isContourConvex(cnt):
    #         return False
        
    #     worst_cos = max(
    #         angle_cos(quad[(i - 1) % 4], quad[i], quad[(i + 1) % 4])
    #         for i in range(4)
    #     )
    #     return worst_cos <= max_cos


    # # ─────────────────────────────────────────────────────────────────────────────
    # #  EDGE DENSITY VALIDATION (NOVEL: ensures quad contains board structure)
    # # ─────────────────────────────────────────────────────────────────────────────

    # def _get_edge_density(quad: np.ndarray, edges: np.ndarray) -> float:
    #     """
    #     Compute fraction of edge pixels inside quad.
    #     High density → likely a real board. Low density → false positive.
        
    #     Returns
    #     -------
    #     density : float in [0, 1]
    #     """
    #     mask = np.zeros(edges.shape, dtype=np.uint8)
    #     pts = quad.reshape(-1, 1, 2).astype(np.int32)
    #     cv2.drawContours(mask, [pts], 0, 255, -1)
        
    #     inside = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
    #     total = cv2.countNonZero(mask)
        
    #     return inside / max(total, 1)


    # def _is_edge_dense_quad(
    #     quad: np.ndarray,
    #     edges: np.ndarray,
    #     min_density: float = 0.06
    # ) -> bool:
    #     """
    #     Check if quad interior has enough edge structure.
    #     Filters out false positives from background contours.
    #     """
    #     return _get_edge_density(quad, edges) >= min_density


    # # ─────────────────────────────────────────────────────────────────────────────
    # #  ADAPTIVE PREPROCESSING (without blocking plt.show)
    # # ─────────────────────────────────────────────────────────────────────────────

    # def _preprocess_adaptive(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Robust edge detection with adaptive CLAHE + bilateral filtering.
    #     Plots the image state at each individual step.
    #     """
    #     h, w = gray.shape[:2]
    #     base = min(h, w)
        
    #     # # 1. CLAHE
    #     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     # enhanced = clahe.apply(gray)
        
    #     # # 2. Bilateral Filter
    #     # radius = max(5, int(base * 0.01)) | 1
    #     # blurred = cv2.bilateralFilter(enhanced, radius, 75, 75)
        
    #     # # 3. Canny Edge Detection
    #     # v = np.median(blurred)
    #     # sigma = 0.33
    #     # lower = int(max(0, v * (1.0 - sigma)))
    #     # upper = int(min(255, v * (1.0 + sigma)))
    #     # edges_canny = cv2.Canny(blurred, lower, upper)
        
    #     # # 4. Morphology (Close + Dilate)
    #     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     # edges_closed = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel, iterations=1)
    #     # edges = cv2.dilate(edges_closed, kernel, iterations=1)
        
    #     # # ========== VISUALIZATION PIPELINE ==========
    #     # fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    #     # titles = [
    #     #     "1. Original Gray", 
    #     #     "2. CLAHE Enhanced", 
    #     #     "3. Bilateral Blur", 
    #     #     "4. Canny Edges", 
    #     #     "5. Morph Close", 
    #     #     "6. Final Dilated"
    #     # ]
    #     # images = [gray, enhanced, blurred, edges_canny, edges_closed, edges]
        
    #     # for ax, img, title in zip(axes, images, titles):
    #     #     ax.imshow(img, cmap='gray')
    #     #     ax.set_title(title, fontsize=10, fontweight="bold")
    #     #     ax.axis("off")
            
    #     # plt.suptitle("Adaptive Preprocessing Pipeline Steps", fontsize=14, fontweight="bold", y=0.98)
    #     # plt.tight_layout()
    #     # plt.show()
    #     # ============================================
    #     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
    #     # Canny with auto thresholds
    #     median = np.median(blurred)
    #     edges = cv2.Canny(blurred, 0.66 * median, 1.33 * median)
        
    #     # Dilate to close gaps in the board boundary
    #     kernel = np.ones((5, 5), np.uint8)
    #     dilated = cv2.dilate(edges, kernel, iterations=2)

    #     plt.imshow(dilated)
    #     plt.show()
        
    #     return edges, blurred

    # def _preprocess_contrast_aware(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Alternative preprocessing for high-contrast scenarios.
    #     Plots the image state at each individual step.
    #     """
    #     h, w = gray.shape[:2]
    #     base = min(h, w)
        
    #     # 1. Morphological Gradient
    #     kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_grad)
        
    #     # 2. OTSU Thresholding
    #     _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    #     # 3. Edge Dilation
    #     kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    #     edges = cv2.dilate(binary, kernel_dilate, iterations=2)
        
    #     # 4. Secondary Blur Path (for corner snapping alignment)
    #     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
    #     # ========== VISUALIZATION PIPELINE ==========
    #     fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    #     titles = [
    #         "1. Original Gray", 
    #         "2. Morph Gradient", 
    #         "3. OTSU Binary", 
    #         "4. Final Dilated Edges", 
    #         "5. Target Blur (Snapping)"
    #     ]
    #     images = [gray, gradient, binary, edges, blurred]
        
    #     for ax, img, title in zip(axes, images, titles):
    #         ax.imshow(img, cmap='gray')
    #         ax.set_title(title, fontsize=10, fontweight="bold")
    #         ax.axis("off")
            
    #     plt.suptitle("Contrast-Aware Preprocessing Pipeline Steps", fontsize=14, fontweight="bold", y=0.98)
    #     plt.tight_layout()
    #     plt.show()
    #     # ============================================
        
    #     return edges, blurred

    # # ─────────────────────────────────────────────────────────────────────────────
    # #  CONTOUR → QUAD EXTRACTION (consolidated, no redundancy)
    # # ─────────────────────────────────────────────────────────────────────────────

    # def _quad_from_contour(cnt: np.ndarray, method: str = "best") -> Tuple[Optional[np.ndarray], str]:
    #     """
    #     Extract quad from contour using prioritized fallback:
    #       1. Direct contour approxPolyDP
    #       2. Convex hull approxPolyDP
    #       3. Minimum area rectangle
        
    #     Returns
    #     -------
    #     quad : (4, 2) points or None
    #     source : string indicating which method succeeded
    #     """
    #     peri = cv2.arcLength(cnt, True)
        
    #     # 1) Direct contour approximation
    #     for frac in np.linspace(0.005, 0.08, 15):
    #         approx = cv2.approxPolyDP(cnt, frac * peri, True)
    #         if len(approx) == 4:
    #             return approx.reshape(4, 2).astype(np.float32), "contour"
        
    #     # 2) Convex hull approximation
    #     hull = cv2.convexHull(cnt)
    #     hull_peri = cv2.arcLength(hull, True) 
    #     for frac in np.linspace(0.005, 0.08, 15):
    #         approx = cv2.approxPolyDP(hull, frac * hull_peri, True)
    #         if len(approx) == 4:
    #             return approx.reshape(4, 2).astype(np.float32), "hull"
        
    #     # 3) Fallback: minimum area rectangle
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect).astype(np.float32)
    #     return box, "minAreaRect"


    # # ─────────────────────────────────────────────────────────────────────────────
    # #  STAGE 1: ADAPTIVE CONTOUR DETECTION
    # # ─────────────────────────────────────────────────────────────────────────────

    # def     _stage1_contour(
    #     gray: np.ndarray,
    #     img_shape: Tuple[int, int, int],
    #     min_area_frac: float = 0.08
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Multi-preprocessing strategy with edge density validation.
    #     Returns list of candidates ranked by edge density (descending).
        
    #     Returns
    #     -------
    #     candidates : list of dicts with keys: quad, score, edges, source, method
    #     """
    #     h, w = img_shape[:2]
    #     img_area = h * w
    #     candidates = []
        
    #     # Try both preprocessing strategies
    #     strategies = [
    #         ("adaptive", _preprocess_adaptive(gray)),
    #         # ("contrast", _preprocess_contrast_aware(gray)),
    #     ]
        
    #     for strategy_name, (edges, blurred) in strategies:
    #         cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            
    #         for cnt in cnts[:5]:  # Check top 5 contours
    #             area = cv2.contourArea(cnt)
    #             if area < min_area_frac * img_area:
    #                 continue
                
    #             quad, source = _quad_from_contour(cnt)
    #             quad = order_points(quad)
                
    #             # Geometric check
    #             if not is_good_quad(quad, img_shape):
    #                 continue
                
    #             # Edge density check: NEW validation ensuring quad contains board
    #             if not _is_edge_dense_quad(quad, edges, min_density=0.05):
    #                 continue
                
    #             edge_density = _get_edge_density(quad, edges)  # Use density as ranking metric
    #             # quad_area = polygon_area(quad)
    #             score = edge_density 

    #             candidates.append({
    #                 "quad": quad,
    #                 "score": score,
    #                 "edges": edges,
    #                 "source": source,
    #                 "method": strategy_name,
    #             })
        
    #     return candidates

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
        
    #     # ========== MERGED PREPROCESSING: HSV Range + Hue OTSU ==========
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
    #     # Approach 1: HSV Range (coarse)
    #     lower = np.array([0, 40, 40])
    #     upper = np.array([30, 255, 255])
    #     mask_range = cv2.inRange(hsv, lower, upper)
        
    #     # Approach 2: Hue OTSU (adaptive fine)
    #     hue = hsv[:, :, 0].astype(np.uint8)
    #     radius = max(5, int(base * 0.01)) | 1
    #     hue_filtered = cv2.bilateralFilter(hue, radius, 50, 50)
    #     _, mask_otsu = cv2.threshold(hue_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    #     # Combine both (AND = strict, OR = lenient)
    #     edges = cv2.bitwise_and(mask_range, mask_otsu)
        
    #     # Morphology cleanup
    #     mk = max(5, int(round(min(h, w) * 0.015)) | 1)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    #     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    #     edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=2)
        
    #     # ========== STAGE 1: CONTOUR DETECTION (with pre-computed edges) ==========
    #     candidates = _stage1_contour_from_edges(edges, img_shape, blurred=gray)
        
    #     if not candidates:
    #         raise RuntimeError("No board corners detected")
        
    #     # Select best candidate (largest area)
    #     best_quad = max(candidates, key=lambda c: c["score"])["quad"]
        
    #     # Refine with Harris snapping
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     snapped = snap_all_corners(best_quad, blurred)
    #     if is_good_quad(snapped, img_shape):
    #         corners = order_points(snapped)
    #     else:
    #         corners = best_quad
        
    #     if debug:
    #         _visualize_corners(img, corners)
        
    #     return corners


    # def _stage1_contour_from_edges(
    #     edges: np.ndarray,
    #     img_shape: Tuple[int, int, int],
    #     blurred: np.ndarray,
    #     min_area_frac: float = 0.08
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Stage 1 Contour Detection from pre-computed edges.
        
    #     Parameters
    #     ----------
    #     edges : pre-computed edge map (from HSV range + OTSU)
    #     img_shape : (H, W, C) image shape
    #     blurred : grayscale image for corner snapping reference
    #     min_area_frac : minimum quad area as fraction of image
        
    #     Returns
    #     -------
    #     candidates : list of dicts with quad, score, source
    #     """
    #     h, w = img_shape[:2]
    #     img_area = h * w
    #     candidates = []
        
    #     # Find contours from pre-computed edges
    #     cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
    #     for cnt in cnts[:10]:  # Check top 10 contours
    #         area = cv2.contourArea(cnt)
    #         if area < min_area_frac * img_area:
    #             continue
            
    #         # Extract quad from contour
    #         quad, source = _quad_from_contour(cnt)
    #         quad = order_points(quad)
            
    #         # Geometric validation
    #         if not is_good_quad(quad, img_shape):
    #             continue
            
    #         # Edge density validation
    #         if not _is_edge_dense_quad(quad, edges, min_density=0.05):
    #             continue
            
    #         # Score by edge density × area
    #         edge_density = _get_edge_density(quad, edges)
    #         quad_area = polygon_area(quad)
    #         score = edge_density * quad_area
            
    #         candidates.append({
    #             "quad": quad,
    #             "score": score,
    #             "source": source,
    #         })
        
    #     return candidates
    # # ─────────────────────────────────────────────────────────────────────────────
    # #  STAGE 2: HSV COLOR SEGMENTATION (improved range)
    # # ─────────────────────────────────────────────────────────────────────────────

    # def _stage2_color(
    #     img: np.ndarray,
    #     img_shape: Tuple[int, int, int]
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Segment board by warm wood tone using hybrid HSV Range + Hue OTSU.
        
    #     Combines:
    #     1. HSV range (coarse): hue 0-30, saturation 40+, value 40+
    #     2. Hue OTSU (adaptive): automatic threshold on Hue channel
    #     3. Merged: AND both masks for high confidence
        
    #     Returns
    #     -------
    #     candidates : list of dicts
    #     """
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, w = img_shape[:2]
    #     img_area = h * w
    #     base = min(h, w)
    #     candidates = []
        
    #     # ========== APPROACH 1: HSV Range (Coarse) ==========
    #     lower = np.array([0, 40, 40])
    #     upper = np.array([30, 255, 255])
    #     mask_range = cv2.inRange(hsv, lower, upper)
        
    #     # ========== APPROACH 2: Hue OTSU (Adaptive Fine) ==========
    #     hue = hsv[:, :, 0].astype(np.uint8)  # ✅ Extract Hue + explicit uint8
        
    #     # Ensure it's single-channel
    #     if len(hue.shape) == 3:
    #         hue = cv2.cvtColor(hue, cv2.COLOR_BGR2GRAY)
        
    #     # Bilateral filter on Hue
    #     radius = max(5, int(base * 0.01)) | 1
    #     hue_filtered = cv2.bilateralFilter(hue, radius, 50, 50)
        
    #     # OTSU threshold on Hue
    #     _, mask_otsu = cv2.threshold(hue_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    #     # ========== COMBINE BOTH (AND for high confidence) ==========
    #     mask = cv2.bitwise_and(mask_range, mask_otsu)
    #     # Alternative: cv2.bitwise_or(mask_range, mask_otsu) if too strict
        
    #     # ========== MORPHOLOGY: clean up combined mask ==========
    #     mk = max(5, int(round(min(h, w) * 0.015)) | 1)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
    #     # ========== CONTOUR DETECTION ==========
    #     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
    #     for cnt in cnts[:5]:
    #         if cv2.contourArea(cnt) < 0.08 * img_area:
    #             continue
            
    #         quad, source = _quad_from_contour(cnt)
    #         quad = order_points(quad)
            
    #         if is_good_quad(quad, img_shape):
    #             score = polygon_area(quad)  # Rank by size
    #             candidates.append({
    #                 "quad": quad,
    #                 "score": score,
    #                 "source": source,
    #                 "method": "color_hsv",
    #             })
        
    #     return candidates


    # # ─────────────────────────────────────────────────────────────────────────────
    # #  STAGE 3: HOUGH LINE INTERSECTION
    # # ─────────────────────────────────────────────────────────────────────────────

    # def _line_intersection(l1: Tuple[float, float], l2: Tuple[float, float]) -> Optional[np.ndarray]:
    #     """Intersect two lines in (rho, theta) form."""
    #     rho1, theta1 = l1
    #     rho2, theta2 = l2
    #     A = np.array([
    #         [np.cos(theta1), np.sin(theta1)],
    #         [np.cos(theta2), np.sin(theta2)]
    #     ])
    #     b = np.array([rho1, rho2])
    #     det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    #     if abs(det) < 1e-6:
    #         return None
    #     x = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
    #     y = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
    #     return np.array([x, y], dtype=np.float32)


    # def _stage3_hough(
    #     gray: np.ndarray,
    #     img_shape: Tuple[int, int, int]
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Detect board edges via Hough lines, intersect to find corners.
        
    #     Returns
    #     -------
    #     candidates : list of dicts
    #     """
    #     h, w = img_shape[:2]
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     edges = cv2.Canny(blurred, 50, 150)
        
    #     # Hough threshold: ~15% of longest dimension
    #     thresh = max(50, int(0.15 * max(h, w)))
    #     lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=thresh)
        
    #     if lines is None or len(lines) < 4:
    #         return []
        
    #     lines = lines[:, 0, :]  # shape (N, 2): (rho, theta)
        
    #     # Separate into horizontal and vertical families
    #     h_lines = [(r, t) for r, t in lines if 70 < np.degrees(t) < 110]
    #     v_lines = [(r, t) for r, t in lines if np.degrees(t) < 20 or np.degrees(t) > 160]
        
    #     if len(h_lines) < 2 or len(v_lines) < 2:
    #         return []
        
    #     # Keep only the 2 most extreme lines in each direction
    #     h_lines = sorted(h_lines, key=lambda l: l[0])
    #     v_lines = sorted(v_lines, key=lambda l: l[0])
    #     border_h = [h_lines[0], h_lines[-1]]
    #     border_v = [v_lines[0], v_lines[-1]]
        
    #     pts = []
    #     for hl in border_h:
    #         for vl in border_v:
    #             pt = _line_intersection(hl, vl)
    #             if pt is not None:
    #                 pts.append(pt)
        
    #     if len(pts) < 4:
    #         return []
        
    #     quad = order_points(np.array(pts[:4], dtype=np.float32))
        
    #     candidates = []
    #     if is_good_quad(quad, img_shape):
    #         candidates.append({
    #             "quad": quad,
    #             "score": polygon_area(quad),
    #             "source": "hough",
    #             "method": "hough_lines",
    #         })
        
    #     return candidates


    # # ─────────────────────────────────────────────────────────────────────────────
    # #  STAGE 4: CORNER FEATURES + CONVEX HULL
    # # ─────────────────────────────────────────────────────────────────────────────

    # def _stage4_features(
    #     gray: np.ndarray,
    #     img_shape: Tuple[int, int, int]
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Detect strong corners, compute convex hull, approximate to quad.
        
    #     Returns
    #     -------
    #     candidates : list of dicts
    #     """
    #     h, w = img_shape[:2]
    #     n_pts = max(40, min(200, (h * w) // 2000))
    #     min_d = max(10, min(h, w) // 30)
        
    #     corners = cv2.goodFeaturesToTrack(
    #         gray, maxCorners=n_pts, qualityLevel=0.01, minDistance=min_d
    #     )
        
    #     if corners is None or len(corners) < 4:
    #         return []
        
    #     pts = corners.reshape(-1, 2).astype(np.float32)
    #     hull = cv2.convexHull(pts).reshape(-1, 2)
        
    #     peri = cv2.arcLength(hull.reshape(-1, 1, 2).astype(np.float32), True)
    #     candidates = []
        
    #     for frac in np.linspace(0.01, 0.10, 20):
    #         approx = cv2.approxPolyDP(
    #             hull.reshape(-1, 1, 2).astype(np.float32), frac * peri, True
    #         )
    #         if len(approx) == 4:
    #             quad = order_points(approx.reshape(4, 2))
    #             if is_good_quad(quad, img_shape):
    #                 score = polygon_area(quad)
    #                 candidates.append({
    #                     "quad": quad,
    #                     "score": score,
    #                     "source": "convex_hull",
    #                     "method": "goodfeatures",
    #                 })
    #                 break  # Take first valid quad from features
        
    #     # Fallback: minAreaRect
    #     if not candidates:
    #         rect = cv2.minAreaRect(pts)
    #         quad = order_points(cv2.boxPoints(rect).astype(np.float32))
    #         if is_good_quad(quad, img_shape):
    #             candidates.append({
    #                 "quad": quad,
    #                 "score": polygon_area(quad),
    #                 "source": "minAreaRect",
    #                 "method": "goodfeatures",
    #             })
        
    #     return candidates


    # # ─────────────────────────────────────────────────────────────────────────────
    # #  HARRIS CORNER REFINEMENT
    # # ─────────────────────────────────────────────────────────────────────────────

    # def harris_snap(
    #     corner: np.ndarray,
    #     gray: np.ndarray,
    #     search_radius_frac: float = 0.06
    # ) -> np.ndarray:
    #     """
    #     Snap corner to nearest strong Harris corner within a local window.
    #     """
    #     h, w = gray.shape[:2]
    #     r = max(20, int(search_radius_frac * min(h, w)))
    #     cx, cy = int(corner[0]), int(corner[1])
        
    #     x0, x1 = max(0, cx - r), min(w, cx + r)
    #     y0, y1 = max(0, cy - r), min(h, cy + r)
    #     patch = gray[y0:y1, x0:x1].astype(np.float32)
        
    #     if patch.size == 0:
    #         return corner
        
    #     block = max(3, min(7, patch.shape[0] // 6) | 1)
    #     resp = cv2.cornerHarris(patch, blockSize=block, ksize=3, k=0.04)
        
    #     thresh = 0.01 * resp.max()
    #     if thresh <= 0:
    #         return corner
        
    #     ys, xs = np.where(resp > thresh)
    #     if len(xs) == 0:
    #         return corner
        
    #     strengths = resp[ys, xs]
    #     best = np.argmax(strengths)
    #     return np.array([xs[best] + x0, ys[best] + y0], dtype=np.float32)


    # def snap_all_corners(quad: np.ndarray, gray: np.ndarray) -> np.ndarray:
    #     """Refine all 4 corners via Harris snapping."""
    #     return np.array(
    #         [harris_snap(pt, gray) for pt in quad],
    #         dtype=np.float32,
    #     )