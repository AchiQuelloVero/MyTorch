from dataclasses import dataclass

PIECES = "PNBRQKpnbrqk"


class FenError(ValueError):
    pass


@dataclass(frozen=True)
class Board:
    squares: tuple
    side_to_move: str
    castling: str
    en_passant: str | None
    halfmove: int
    fullmove: int


def parse_fen(fen):
    fields = fen.split()
    if len(fields) != 6:
        raise FenError(f"expected 6 fields, got {len(fields)}")
    placement, side, castling, en_passant, halfmove, fullmove = fields
    if side not in ("w", "b"):
        raise FenError(f"side to move must be 'w' or 'b', got '{side}'")
    return Board(
        squares=parse_placement(placement),
        side_to_move=side,
        castling=parse_castling(castling),
        en_passant=parse_en_passant(en_passant, side),
        halfmove=parse_counter("halfmove clock", halfmove, minimum=0),
        fullmove=parse_counter("fullmove number", fullmove, minimum=1),
    )


def parse_placement(placement):
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise FenError(f"expected 8 ranks, got {len(ranks)}")
    squares = []
    for number, rank in zip(range(8, 0, -1), ranks):
        row = []
        for char in rank:
            if char in "12345678":
                row.extend([None] * int(char))
            elif char in PIECES:
                row.append(char)
            else:
                raise FenError(f"rank {number}: invalid character '{char}'")
        if len(row) != 8:
            raise FenError(f"rank {number}: {len(row)} squares instead of 8")
        squares.extend(row)
    for king, color in (("K", "white"), ("k", "black")):
        if squares.count(king) != 1:
            raise FenError(f"expected 1 {color} king, got {squares.count(king)}")
    return tuple(squares)


def parse_castling(castling):
    if castling == "-":
        return ""
    if any(right not in "KQkq" for right in castling) or len(set(castling)) != len(castling):
        raise FenError(f"invalid castling rights '{castling}'")
    return "".join(right for right in "KQkq" if right in castling)


def parse_en_passant(square, side):
    if square == "-":
        return None
    rank = "6" if side == "w" else "3"
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] != rank:
        raise FenError(f"invalid en passant square '{square}' with '{side}' to move")
    return square


def parse_counter(name, value, minimum):
    if not (value.isascii() and value.isdigit()) or int(value) < minimum:
        raise FenError(f"{name} must be an integer >= {minimum}, got '{value}'")
    return int(value)
