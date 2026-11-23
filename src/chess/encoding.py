import numpy as np

from src.chess.fen import PIECES, FenError, parse_fen

FILES = "abcdefgh"
PIECE_INPUTS = len(PIECES) * 64
INPUT_SIZE = PIECE_INPUTS + len(FILES)


def encode_fens(fens):
    matrix = np.zeros((len(fens), INPUT_SIZE), dtype=np.uint8)
    for number, (row, fen) in enumerate(zip(matrix, fens), 1):
        try:
            fill(row, parse_fen(fen))
        except FenError as e:
            raise FenError(f"board {number}: {e}") from None
    return matrix


def canonical_key(fen):
    squares, en_passant = view(parse_fen(fen))
    return "".join(piece or "." for piece in squares) + (en_passant or "-")


def fill(vector, board):
    squares, en_passant = view(board)
    for square, piece in enumerate(squares):
        if piece is not None:
            vector[PIECES.index(piece) * 64 + square] = 1
    if en_passant:
        vector[PIECE_INPUTS + FILES.index(en_passant)] = 1


def view(board):
    squares = board.squares
    if board.side_to_move == "b":
        ranks = [squares[start:start + 8] for start in range(56, -1, -8)]
        squares = tuple(piece and piece.swapcase() for rank in ranks for piece in rank)
    return squares, board.en_passant and board.en_passant[0]
