import glob
import unittest

from src.chess.fen import FenError, parse_fen

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
DATASET = glob.glob("dataset/*/*.txt")

INVALID = {
    "5 fields": START + " w KQkq - 0",
    "7 ranks": "8/8/8/8/8/8/4K2k w - - 0 1",
    "rank too long": "rnbqkbnr/ppppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "rank too short": "rnbqkbnr/ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "invalid piece": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBXKBNR w - - 0 1",
    "digit 9": "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "no black king": "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "two white kings": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKKBNR w - - 0 1",
    "invalid side": START + " x KQkq - 0 1",
    "invalid castling": START + " w KQkx - 0 1",
    "repeated castling": START + " w KKq - 0 1",
    "en passant on arrival rank": START + " w KQkq e4 0 1",
    "en passant on wrong side": START + " w KQkq e3 0 1",
    "en passant too long": START + " b KQkq e33 0 1",
    "negative halfmove": START + " w KQkq - -1 1",
    "non-numeric halfmove": START + " w KQkq - abc 1",
    "fullmove 0": START + " w KQkq - 0 0",
    "non-ascii digit": START + " w KQkq - 0 ²",
}


class TestParseFen(unittest.TestCase):
    def test_start_position(self):
        board = parse_fen(START + " w KQkq - 0 1")
        self.assertEqual(board.squares[0], "r")
        self.assertEqual(board.squares[60], "K")
        self.assertIsNone(board.squares[36])
        self.assertEqual((board.side_to_move, board.castling, board.en_passant), ("w", "KQkq", None))
        self.assertEqual((board.halfmove, board.fullmove), (0, 1))

    def test_after_e4(self):
        board = parse_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        self.assertEqual(board.squares[36], "P")
        self.assertEqual((board.side_to_move, board.en_passant), ("b", "e3"))

    def test_castling_is_normalized(self):
        a = parse_fen("4k3/8/8/8/8/8/8/4K3 w qK - 0 1")
        b = parse_fen("4k3/8/8/8/8/8/8/4K3 w Kq - 0 1")
        self.assertEqual(a.castling, "Kq")
        self.assertEqual(a, b)

    def test_invalid_fens(self):
        for reason, fen in INVALID.items():
            with self.subTest(reason), self.assertRaises(FenError):
                parse_fen(fen)

    @unittest.skipUnless(DATASET, "dataset/ not found")
    def test_whole_dataset(self):
        for path in DATASET:
            with open(path) as f:
                for number, line in enumerate(f, 1):
                    fen = " ".join(line.split()[:6])
                    try:
                        parse_fen(fen)
                    except FenError as e:
                        self.fail(f"{path}:{number}: {e}")


if __name__ == "__main__":
    unittest.main()
