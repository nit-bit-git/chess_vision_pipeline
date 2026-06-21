import os
import cv2
import torch

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
    preprocess_image(image)

if __name__ == "__main__":
    main()
