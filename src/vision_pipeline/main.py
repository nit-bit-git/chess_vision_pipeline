import os
import cv2
import torch
from ultralytics import YOLO

from .preprocessing import testfn


def main():
    """Pipeline entry point / orchestrator"""
    print("Vision pipeline entry point")
    print("--- Initializing Vision Pipeline ---")
    
    # 1. Device Selection (Ensuring CUDA/GPU is utilized if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    testfn()

if __name__ == "__main__":
    main()
