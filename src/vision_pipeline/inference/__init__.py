from .detector import analyze_with_python_chess


def generate_move (fen_string: str, stockfish_path: str):
    """
    Generate a move for the given FEN string using the Stockfish engine.
    """
    return analyze_with_python_chess(fen_string, stockfish_path)