"""
Convert YOLOv8 detection results to piece_detections format.

Ultralytics Results → list of dicts with class, confidence, center
"""

import numpy as np
from typing import List, Dict, Tuple


def extract_piece_detections(yolo_results) -> List[Dict]:
    """
    Convert YOLOv8 Results object to piece_detections format.
    
    Parameters
    ----------
    yolo_results : ultralytics.engine.results.Results
        Output from model.predict() or model()
        
        Example:
            results = model(image)  # Returns list of Results
            piece_detections = extract_piece_detections(results[0])
    
    Returns
    -------
    piece_detections : list of dicts
        Each dict has:
        - 'class': int (0-11, piece class ID)
        - 'confidence': float (detection confidence 0-1)
        - 'center': tuple (x, y) pixel coordinates of bbox center
        - 'bbox': [x1, y1, x2, y2] full bounding box
        - 'name': str (piece name from model)
    
    Example
    -------
    >>> results = model(image)  # YOLOv8 model inference
    >>> piece_detections = extract_piece_detections(results[0])
    >>> piece_detections[0]
    {'class': 11, 'confidence': 0.95, 'center': (258, 148), 
     'bbox': [245, 135, 271, 161], 'name': 'white-rook'}
    """
    piece_detections = []
    
    # Get boxes from results
    boxes = yolo_results.boxes
    
    if boxes is None or len(boxes) == 0:
        print("[Warning] No detections found")
        return piece_detections
    
    # Get class name mapping
    names = yolo_results.names  # {0: 'black-bishop', 1: 'black-king', ...}
    
    # Iterate through each detection
    for i in range(len(boxes)):
        # Extract bbox in format [x1, y1, x2, y2]
        bbox = boxes.xyxy[i].cpu().numpy()  # Convert to numpy
        x1, y1, x2, y2 = bbox.astype(float)
        
        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center = (center_x, center_y)
        
        # Get class ID
        class_id = int(boxes.cls[i].item())
        
        # Get confidence
        confidence = float(boxes.conf[i].item())
        
        # Get piece name
        piece_name = names.get(class_id, 'unknown')
        
        # Create detection dict
        detection = {
            'class': class_id,
            'confidence': confidence,
            'center': center,
            'bbox': [x1, y1, x2, y2],
            'name': piece_name,
        }
        
        piece_detections.append(detection)
    
    return piece_detections


def extract_piece_detections_verbose(yolo_results, debug: bool = True) -> List[Dict]:
    """
    Extract piece detections with detailed logging.
    
    Parameters
    ----------
    yolo_results : ultralytics.engine.results.Results
    debug : bool, print detailed information
    
    Returns
    -------
    piece_detections : list of dicts
    """
    piece_detections = []
    boxes = yolo_results.boxes
    names = yolo_results.names
    
    if debug:
        print(f"\n[Extraction] Processing {len(boxes)} detections")
    
    for i in range(len(boxes)):
        bbox = boxes.xyxy[i].cpu().numpy().astype(float)
        x1, y1, x2, y2 = bbox
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        class_id = int(boxes.cls[i].item())
        confidence = float(boxes.conf[i].item())
        piece_name = names.get(class_id, 'unknown')
        
        detection = {
            'class': class_id,
            'confidence': confidence,
            'center': (center_x, center_y),
            'bbox': [x1, y1, x2, y2],
            'name': piece_name,
        }
        
        piece_detections.append(detection)
        
        if debug:
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            print(f"  [{i}] {piece_name:20} | conf: {confidence:.2f} | "
                  f"center: ({center_x:.0f}, {center_y:.0f}) | "
                  f"bbox: ({bbox_width:.0f}×{bbox_height:.0f})")
    
    if debug:
        print(f"[Extraction] Extracted {len(piece_detections)} piece detections\n")
    
    return piece_detections


def print_detections_summary(piece_detections: List[Dict]) -> None:
    """Pretty-print detection summary."""
    print(f"\n{'='*70}")
    print(f"{'PIECE DETECTION SUMMARY':^70}")
    print(f"{'='*70}")
    
    # Group by piece type
    pieces_by_type = {}
    for det in piece_detections:
        piece_name = det['name']
        if piece_name not in pieces_by_type:
            pieces_by_type[piece_name] = []
        pieces_by_type[piece_name].append(det)
    
    # Print by type
    for piece_name in sorted(pieces_by_type.keys()):
        dets = pieces_by_type[piece_name]
        print(f"\n{piece_name:20} ({len(dets)} detected)")
        
        for det in dets:
            center = det['center']
            conf = det['confidence']
            print(f"  • Center: ({center[0]:7.1f}, {center[1]:7.1f}) | "
                  f"Confidence: {conf:.2%}")
    
    print(f"\n{'='*70}")
    print(f"Total: {len(piece_detections)} pieces detected")
    print(f"{'='*70}\n")


   

# def example_batch_processing():
#     """Process multiple images."""
#     from ultralytics import YOLO
#     from pathlib import Path
    
#     model = YOLO('path/to/chess_pieces_model.pt')
    
#     image_folder = Path('path/to/images')
    
#     for image_path in image_folder.glob('*.jpg'):
#         print(f"\nProcessing: {image_path.name}")
        
#         results = model(str(image_path))
#         piece_detections = extract_piece_detections(results[0])
        
#         print(f"Found {len(piece_detections)} pieces")
        
#         # Save detections for later use
#         # np.save(f'{image_path.stem}_detections.npy', piece_detections)


