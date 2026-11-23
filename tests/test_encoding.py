import unittest

import numpy as np

from src.chess.encoding import INPUT_SIZE, PIECE_INPUTS, encode_fens
from src.chess.fen import PIECES, FenError

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
AFTER_E5 = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1"
KINGS_BLACK_TO_MOVE = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"


def index(piece, square):
    return PIECES.index(piece) * 64 + square


def ones(fen):
    return set(np.flatnonzero(encode_fens([fen])[0]).tolist())


class TestEncodeFens(unittest.TestCase):
    def test_shape_and_type(self):
        matrix = encode_fens([START, AFTER_E4])
        self.assertEqual(matrix.shape, (2, INPUT_SIZE))
        self.assertEqual(matrix.dtype, np.uint8)
        self.assertEqual(INPUT_SIZE, 776)

    def test_start_position(self):
        start = ones(START)
        self.assertEqual(len(start), 32)
        self.assertIn(index("r", 0), start)
        self.assertIn(index("K", 60), start)
        self.assertIn(index("k", 4), start)
        self.assertNotIn(index("P", 36), start)

    def test_side_to_move_becomes_white(self):
        self.assertEqual(ones(KINGS_BLACK_TO_MOVE), {index("K", 60), index("k", 4)})

    def test_mirrored_positions_are_equal(self):
        self.assertEqual(ones(AFTER_E4), ones(AFTER_E5))
        self.assertEqual(ones(START), ones(START.replace(" w ", " b ")))

    def test_en_passant_file(self):
        self.assertEqual(ones(AFTER_E4) - ones(AFTER_E4.replace("e3", "-")), {PIECE_INPUTS + 4})
        self.assertFalse(encode_fens([START])[0, PIECE_INPUTS:].any())

    def test_rows_keep_the_order(self):
        matrix = encode_fens([START, KINGS_BLACK_TO_MOVE, START])
        self.assertEqual(set(np.flatnonzero(matrix[1]).tolist()), ones(KINGS_BLACK_TO_MOVE))
        self.assertTrue(np.array_equal(matrix[0], matrix[2]))

    def test_empty_list(self):
        self.assertEqual(encode_fens([]).shape, (0, INPUT_SIZE))

    def test_invalid_fen_gives_its_position(self):
        with self.assertRaisesRegex(FenError, "^board 2: "):
            encode_fens([START, "not a fen", START])


if __name__ == "__main__":
    unittest.main()
