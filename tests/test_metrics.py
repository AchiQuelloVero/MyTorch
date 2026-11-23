import unittest

import numpy as np

from src.metrics import confusion_matrix, scores


class TestConfusionMatrix(unittest.TestCase):
    def test_rows_are_expected_columns_are_predicted(self):
        matrix = confusion_matrix([0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 2, 0], 3)
        self.assertEqual(matrix.tolist(), [[1, 1, 0], [0, 1, 0], [1, 0, 2]])

    def test_missing_class(self):
        self.assertEqual(confusion_matrix([0, 1], [0, 1], 3).tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 0]])


class TestScores(unittest.TestCase):
    def test_by_hand(self):
        result = scores(np.array([[1, 1, 0], [0, 1, 0], [1, 0, 2]]))
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        np.testing.assert_allclose(result["precision"], [1 / 2, 1 / 2, 1])
        np.testing.assert_allclose(result["recall"], [1 / 2, 1, 2 / 3])
        np.testing.assert_allclose(result["f1"], [1 / 2, 2 / 3, 4 / 5])
        self.assertAlmostEqual(result["macro_f1"], (1 / 2 + 2 / 3 + 4 / 5) / 3)

    def test_perfect(self):
        result = scores(np.diag([5, 3, 2]))
        self.assertEqual((result["accuracy"], result["macro_f1"]), (1.0, 1.0))

    def test_class_never_predicted_or_never_expected(self):
        result = scores(np.array([[2, 0, 1], [0, 0, 0], [0, 0, 1]]))
        np.testing.assert_allclose(result["precision"], [1, 0, 1 / 2])
        np.testing.assert_allclose(result["recall"], [2 / 3, 0, 1])
        self.assertEqual(result["f1"][1], 0)


if __name__ == "__main__":
    unittest.main()
