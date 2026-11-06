"""Clustering metrics: Hungarian-aligned macro-F1 and macro one-vs-rest AUC, plus ACC/NMI/ARI.

For AUC the aligned hard assignments are used as one-hot class scores, so the per-class AUC equals
the mean of the positive and negative recalls.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score


def align(truth, labels):
    """Map cluster identifiers to classes by the Hungarian assignment (unmatched clusters -> -1)."""
    true_classes, true_indices = np.unique(truth, return_inverse=True)
    predicted_classes, predicted_indices = np.unique(labels, return_inverse=True)
    contingency = np.zeros((len(true_classes), len(predicted_classes)), dtype=np.int64)
    np.add.at(contingency, (true_indices, predicted_indices), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = {predicted_classes[column]: true_classes[row] for row, column in zip(rows, columns)}
    return np.asarray([mapping.get(label, -1) for label in labels]), true_classes


def evaluate(truth, labels):
    truth, labels = np.asarray(truth), np.asarray(labels)
    aligned, classes = align(truth, labels)
    aucs = []
    for cls in classes:
        positive, predicted = truth == cls, aligned == cls
        tpr = np.sum(positive & predicted) / max(int(positive.sum()), 1)
        tnr = np.sum(~positive & ~predicted) / max(int((~positive).sum()), 1)
        aucs.append(0.5 * (tpr + tnr))
    return dict(AUC=float(np.mean(aucs)),
                F1=float(f1_score(truth, aligned, labels=classes, average="macro", zero_division=0)),
                ACC=float(accuracy_score(truth, aligned)),
                NMI=float(normalized_mutual_info_score(truth, labels)),
                ARI=float(adjusted_rand_score(truth, labels)))
