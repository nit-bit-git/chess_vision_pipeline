import numpy as np
from vision_pipeline.utils import show_move_on_board_display
from .detector import analyze_with_python_chess


def generate_move (fen_string: str, stockfish_path: str, sq_map: dict, image: np.ndarray, intersections: list) -> str:
    """
    Generate a move for the given FEN string using the Stockfish engine.
    """
    generated_move = analyze_with_python_chess(fen_string, stockfish_path)
    print(f"Generated Move: {generated_move}")

    show_move_on_board_display(str(generated_move), sq_map, image, intersections, alpha=0.4)
    
   
    return generated_move


