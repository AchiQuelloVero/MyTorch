import numpy as np


def confusion_matrix(expected, predicted, size):
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (np.asarray(expected), np.asarray(predicted)), 1)
    return matrix


def ratio(top, bottom):
    return np.divide(top, bottom, out=np.zeros(len(top)), where=bottom > 0)


def scores(matrix):
    correct = np.diag(matrix).astype(float)
    precision = ratio(correct, matrix.sum(axis=0))
    recall = ratio(correct, matrix.sum(axis=1))
    f1 = ratio(2 * precision * recall, precision + recall)
    return {"accuracy": correct.sum() / matrix.sum(), "precision": precision, "recall": recall, "f1": f1,
            "macro_f1": f1.mean()}
