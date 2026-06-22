"""
Chess Piece Detection to FEN String Conversion

Maps detected pieces to board squares and generates FEN notation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# Class to piece mapping
CLASS_TO_PIECE = {
    0: 'b',   # black-bishop
    1: 'k',   # black-king
    2: 'n',   # black-knight
    3: 'p',   # black-pawn
    4: 'q',   # black-queen
    5: 'r',   # black-rook
    6: 'B',   # white-bishop
    7: 'K',   # white-king
    8: 'N',   # white-knight
    9: 'P',   # white-pawn
    10: 'Q',  # white-queen
    11: 'R',  # white-rook
}

PIECE_TO_CLASS = {v: k for k, v in CLASS_TO_PIECE.items()}


def map_pieces_to_squares(
    piece_detections: List[Dict],
    square_centers: Dict[str, Tuple[float, float]],
    debug: bool = False
) -> Dict[str, str]:
    """
    Map detected pieces to their closest board squares.
    
    Parameters
    ----------
    piece_detections : list of dicts with keys:
        - 'class': int (0-11, piece class index)
        - 'confidence': float (detection confidence)
        - 'bbox': [x1, y1, x2, y2] or center point
        
    square_centers : dict mapping square name (e.g., 'e4') to (x, y) pixel coords
    
    debug : print mapping information
    
    Returns
    -------
    piece_placement : dict mapping square name to piece symbol
        e.g., {'e4': 'P', 'e5': 'p', 'a1': 'R', ...}
    """
    piece_placement = {}
    
    if debug:
        print(f"\n[Mapping] {len(piece_detections)} pieces to {len(square_centers)} squares")
    
    for detection in piece_detections:
        class_id = detection['class']
        confidence = detection.get('confidence', 0.0)
        
        # Extract piece center from detection
        if 'center' in detection:
            piece_center = detection['center']
        elif 'bbox' in detection:
            bbox = detection['bbox']
            piece_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        else:
            if debug:
                print(f"  ✗ Detection missing center or bbox")
            continue
        
        # Find nearest square
        min_distance = float('inf')
        nearest_square = None
        
        for square_name, square_center in square_centers.items():
            distance = np.sqrt(
                (piece_center[0] - square_center[0])**2 + 
                (piece_center[1] - square_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_square = square_name
        
        # Convert class to piece symbol
        piece_symbol = CLASS_TO_PIECE.get(class_id, '?')
        
        # Store in placement dict
        if nearest_square in piece_placement:
            # Multiple pieces in same square - keep highest confidence
            existing_conf = detection.get('_confidence', 0)
            if confidence > existing_conf:
                piece_placement[nearest_square] = piece_symbol
                detection['_confidence'] = confidence
        else:
            piece_placement[nearest_square] = piece_symbol
            detection['_confidence'] = confidence
        
        if debug:
            piece_name = {
                'b': 'black-bishop', 'k': 'black-king', 'n': 'black-knight',
                'p': 'black-pawn', 'q': 'black-queen', 'r': 'black-rook',
                'B': 'white-bishop', 'K': 'white-king', 'N': 'white-knight',
                'P': 'white-pawn', 'Q': 'white-queen', 'R': 'white-rook'
            }.get(piece_symbol, '?')
            
            print(f"  [{piece_name}] @ pixel {piece_center} → square {nearest_square} (dist: {min_distance:.1f})")
    
    if debug:
        print(f"[Mapping] Total pieces placed: {len(piece_placement)}\n")
    
    return piece_placement


def placement_to_board_array(piece_placement: Dict[str, str]) -> np.ndarray:
    """
    Convert piece placement dict to 8×8 numpy array.
    
    Returns
    -------
    board : (8, 8) array where:
        - board[0, 0] is square 'a8' (top-left in FEN perspective)
        - board[7, 7] is square 'h1' (bottom-right)
        - Empty squares are '.'
    """
    board = np.full((8, 8), '.', dtype=object)
    
    for square_name, piece in piece_placement.items():
        # Parse square name: first char is file (a-h), second is rank (1-8)
        file_char = square_name[0]  # a-h
        rank_char = square_name[1]  # 1-8
        
        # Convert to array indices
        col = ord(file_char) - ord('a')  # 0-7
        row = 8 - int(rank_char)         # 7-0 (8→0, 1→7)
        
        board[row, col] = piece
    
    return board


def board_array_to_fen(board: np.ndarray) -> str:
    """
    Convert 8×8 board array to FEN string.
    
    Parameters
    ----------
    board : (8, 8) array of piece symbols ('.' for empty)
    
    Returns
    -------
    fen : FEN string (board position only, no move info)
    """
    fen_parts = []
    
    for row in board:
        # Process each rank (row)
        empty_count = 0
        rank_fen = ""
        
        for cell in row:
            if cell == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_fen += str(empty_count)
                    empty_count = 0
                rank_fen += cell
        
        # Add remaining empty squares
        if empty_count > 0:
            rank_fen += str(empty_count)
        
        fen_parts.append(rank_fen)
    
    # Join ranks with '/'
    fen = '/'.join(fen_parts)
    
    return fen


def generate_full_fen(
    piece_placement: Dict[str, str],
    active_color: str = 'w',
    castling: str = 'KQkq',
    en_passant: str = '-',
    halfmove: int = 0,
    fullmove: int = 1,
    debug: bool = False
) -> str:
    """
    Generate complete FEN string including move information.
    
    Parameters
    ----------
    piece_placement : dict mapping squares to pieces
    active_color : 'w' for white, 'b' for black
    castling : castling rights (e.g., 'KQkq', '-')
    en_passant : en passant square (e.g., 'e3', '-')
    halfmove : halfmove clock (moves since pawn/capture)
    fullmove : fullmove number (increments after black moves)
    debug : print FEN components
    
    Returns
    -------
    fen : complete FEN string
    """
    # Convert placement to board array
    board = placement_to_board_array(piece_placement)
    
    # Get board position FEN
    board_fen = board_array_to_fen(board)
    
    # Assemble full FEN
    fen = f"{board_fen} {active_color} {castling} {en_passant} {halfmove} {fullmove}"
    
    if debug:
        print(f"\n[FEN] Generated FEN string:")
        print(f"  Board: {board_fen}")
        print(f"  Active color: {active_color}")
        print(f"  Castling: {castling}")
        print(f"  En passant: {en_passant}")
        print(f"  Halfmove: {halfmove}")
        print(f"  Fullmove: {fullmove}")
        print(f"\n[FEN] Complete: {fen}\n")
    
    return fen


def visualize_board(piece_placement: Dict[str, str]) -> str:
    """
    ASCII visualization of the board.
    
    Returns
    -------
    board_str : pretty-printed board string
    """
    board = placement_to_board_array(piece_placement)
    
    piece_symbols = {
        'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚',
        'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
        '.': '·'
    }
    
    board_str = "  a b c d e f g h\n"
    
    for rank_idx, row in enumerate(board):
        rank_num = 8 - rank_idx
        board_str += f"{rank_num} "
        
        for cell in row:
            symbol = piece_symbols.get(cell, cell)
            board_str += f"{symbol} "
        
        board_str += f"{rank_num}\n"
    
    board_str += "  a b c d e f g h\n"
    
    return board_str

