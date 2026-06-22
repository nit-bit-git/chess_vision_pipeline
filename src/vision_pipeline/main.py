import os
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
from piece_detection.chess_piece_mapping import generate_full_fen, map_pieces_to_squares, visualize_board
from piece_detection.yolo_to_detections import extract_piece_detections_verbose, print_detections_summary
import torch
import numpy as np
from vision_pipeline.inference import generate_move
from pipeline_state import PipelineState
state = PipelineState()
from .preprocessing import preprocess_image


def main():
    """Pipeline entry point / orchestrator"""
    print("Vision pipeline entry point")
    print("--- Initializing Vision Pipeline ---")
    
    # 1. Device Selection (Ensuring CUDA/GPU is utilized if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'images', 'cb3.jpg'))
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found at {image_path}")
        return
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return
    print(f"Image loaded successfully with shape: {image.shape}")

    # YOLOv8  model  
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'chess-model-yolov8m.pt'))
    if not os.path.exists(model_path):
        print(f"Error: YOLO model not found at {model_path}")
        return
    model = YOLO(model_path)
    model.to(device)

    # make prediction
    results = model(image_path) # path to test image

    pieces_found = True if len(results[0].boxes) > 0 else False

    centers, intersections, labels, sq_map, img, M = preprocess_image(image, pieces_found=pieces_found, state=state)  # Step 1: Preprocess and detect corners/gridd
   
    print(f"[Debug 2] M from warp_board: type={type(M)}, shape={M.shape if isinstance(M, np.ndarray) else 'N/A'}")
    
   # Extract with detailed logging
    piece_detections = extract_piece_detections_verbose(results[0], state=state, debug=True)
    
    # Print summary
    print_detections_summary(piece_detections)

    im_array = results[0].plot(); # plot a BGR numpy array of predictions
    # print(f"results: {results}")
    plt.imshow(im_array)
    plt.show()
    # Map pieces to squares
    piece_placement = map_pieces_to_squares(piece_detections, sq_map, debug=True)
    
    # Visualize
    print(visualize_board(piece_placement))
    
    # Generate FEN
    fen = generate_full_fen(piece_placement, debug=True)
    print(f"Generated FEN: {fen}")

    stockfish_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',"models", 'stockfish', 'stockfish.exe'))
    generate_move(fen, stockfish_path, sq_map, img, intersections)  # Assuming you want to visualize without an image)

    


if __name__ == "__main__":
    main()
