"""
Constants and configuration values for the Chess Vision Pipeline
"""

# Preprocessing configurations for different lighting and contrast scenarios
CONFIGS = [
    {"name": "baseline",        "tone": "raw",         "blur_frac": 0.010, "morph_frac": 0.008, "canny_sigma": 0.33},
    {"name": "low_contrast",    "tone": "clahe",       "blur_frac": 0.010, "morph_frac": 0.010, "canny_sigma": 0.33},
    {"name": "shadow_tolerant", "tone": "strong_clahe","blur_frac": 0.014, "morph_frac": 0.012, "canny_sigma": 0.50},
    {"name": "edge_enhanced",   "tone": "sharp",       "blur_frac": 0.008, "morph_frac": 0.012, "canny_sigma": 0.25},
    {"name": "mixed_recovery",  "tone": "clahe_sharp", "blur_frac": 0.010, "morph_frac": 0.015, "canny_sigma": 0.40},
]


