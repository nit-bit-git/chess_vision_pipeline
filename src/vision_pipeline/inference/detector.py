import chess
import chess.engine

def analyze_with_python_chess(fen_string: str, stockfish_path: str):
    # 1. Initialize the board with your FEN
    try:
        board = chess.Board(fen_string)
    except ValueError:
        print(f"Invalid FEN from vision pipeline: {fen_string}")
        return

    # 2. Start the Stockfish engine via UCI (Universal Chess Interface)
    # Use the path to the downloaded stockfish .exe
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        
        # 3. Analyze the position (e.g., search to depth 15)
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        
        # 4. Extract the data
        best_move = info["pv"][0] # Principal Variation (best move)
        score = info["score"].white() # Perspective of White
        
        print(f"Position: {fen_string}")
        print(f"Best Move: {best_move}")
        
        # Format the score nicely
        if score.is_mate():
            print(f"Evaluation: Mate in {score.mate()}")
        else:
            # Centipawns to standard pawn advantage (e.g. +1.25)
            print(f"Evaluation: {score.score() / 100.0:+.2f}")
        return best_move

# Example usage:
# analyze_with_python_chess("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "C:/path/to/stockfish.exe")