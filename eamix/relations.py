"""Complementary edge relations (Section "Complementary Edge Relations").

* ``endpoint_descriptors`` -- optional helper that builds edge descriptors from
  endpoint (node) attributes with squared block weights ``share`` / ``1 - share``.
* ``shared_signal``        -- Q = RowNorm(U_r Sigma_r) of the row-normalized edge attributes.
* ``attribute_affinity``   -- symmetric approximate k_A-NN cosine affinity W_A.
* ``attribute_transition`` -- P_A = D~_A^{-1} W~_A (self-loops on isolated rows).
* ``TopologyOperator``     -- P_T = 1/2 B^T D_V^+ B applied by incidence aggregation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from sklearn.utils.extmath import randomized_svd


def row_normalize(values):
    """Rowwise Euclidean normalization; zero rows stay zero."""
    if sparse.issparse(values):
        norms = np.sqrt(np.asarray(values.multiply(values).sum(axis=1)).ravel())
        return values.multiply(1.0 / np.maximum(norms, 1e-12)[:, None])
    norms = np.linalg.norm(values, axis=1)
    return values / np.maximum(norms, 1e-12)[:, None]


def endpoint_descriptors(node_features, edges, share=0.625):
    """Concatenate endpoint attributes, weighting the blocks by sqrt(share) and sqrt(1 - share).

    The first column of ``edges`` is the endpoint that receives ``share``.
    """
    if not 0 < share < 1:
        raise ValueError("share must lie in (0, 1)")
    features = sparse.csr_matrix(node_features)
    return sparse.hstack((np.sqrt(share) * features[edges[:, 0]],
                          np.sqrt(1 - share) * features[edges[:, 1]]), format="csr")


def shared_signal(edge_attributes, rank=128, n_iter=3, random_state=0):
    """Return Q = RowNorm(U_r Sigma_r) from the truncated SVD of RowNorm(X)."""
    attributes = row_normalize(sparse.csr_matrix(edge_attributes)).tocsr()
    rank = min(rank, *attributes.shape)
    if rank == min(attributes.shape):
        left, singular, _ = np.linalg.svd(attributes.toarray(), full_matrices=False)
        coordinates = left[:, :rank] * singular[:rank]
    else:
        left, singular, _ = randomized_svd(attributes, n_components=rank, n_iter=n_iter,
                                           random_state=random_state)
        coordinates = left * singular
    return row_normalize(np.asarray(coordinates, dtype=np.float64))


def attribute_affinity(signal, n_neighbors=32, n_candidates=256, seed=0):
    """Symmetric nonnegative affinity W_A from an approximate nearest-neighbor graph in Q.

    Candidates are retrieved by random-hyperplane hashing (own bucket first, then buckets at
    Hamming distance one), capped at ``n_candidates``. Up to ``n_neighbors`` candidates with the
    largest positive cosine similarity are kept, and opposite directed weights are merged by max.
    """
    count, dimension = signal.shape
    bits = min(20, max(1, int(np.ceil(np.log2(max(count / 64.0, 1.0))))))
    random = np.random.default_rng(seed + 1729)
    planes = random.standard_normal((dimension, bits))
    codes = (signal @ planes >= 0).astype(np.uint32) @ (1 << np.arange(bits, dtype=np.uint32))
    buckets = {}
    for row, code in enumerate(codes):
        buckets.setdefault(int(code), []).append(row)
    for code, members in buckets.items():
        if len(members) > n_candidates:
            buckets[code] = random.choice(members, n_candidates, replace=False).tolist()
    rows, columns, similarities = [], [], []
    for row, code in enumerate(codes):
        candidates = list(buckets[int(code)])
        for bit in range(bits):
            if len(candidates) >= n_candidates:
                break
            candidates.extend(buckets.get(int(code) ^ 1 << bit, []))
        candidates = np.asarray(candidates, dtype=np.int64)
        candidates = candidates[candidates != row]
        if len(candidates) > n_candidates:
            candidates = random.choice(candidates, n_candidates, replace=False)
        if not len(candidates):
            continue
        scores = signal[candidates] @ signal[row]
        keep = min(n_neighbors, len(scores))
        selected = np.argpartition(scores, -keep)[-keep:]
        selected = selected[scores[selected] > 0]
        rows.extend([row] * len(selected))
        columns.extend(candidates[selected].tolist())
        similarities.extend(scores[selected].tolist())
    directed = sparse.csr_matrix((similarities, (rows, columns)), shape=(count, count))
    return directed.maximum(directed.T).tocsr()


def attribute_transition(affinity):
    """P_A = D~_A^{-1} W~_A, where W~_A adds a self-loop to every isolated row of W_A.

    Returns the transition matrix and the boolean mask of isolated rows.
    """
    affinity = affinity.tocsr().astype(np.float64)
    isolated = np.asarray(affinity.sum(axis=1)).ravel() <= 0
    completed = affinity + sparse.diags(isolated.astype(np.float64))
    degrees = np.asarray(completed.sum(axis=1)).ravel()
    return (sparse.diags(1.0 / degrees) @ completed).tocsr(), isolated


class TopologyOperator:
    """Edge topology walk P_T = 1/2 B^T D_V^+ B, applied without forming the m x m matrix.

    P_T is symmetric, so ``apply`` evaluates both P_T X and P_T^T X.
    """

    def __init__(self, edges, n_nodes):
        edges = np.asarray(edges, dtype=np.int64)
        m = len(edges)
        self.incidence = sparse.csr_matrix(
            (np.ones(2 * m, dtype=np.float64), (edges.T.reshape(-1), np.tile(np.arange(m), 2))),
            shape=(n_nodes, m))
        self.degrees = np.asarray(self.incidence.sum(axis=1)).ravel()
        self.inverse_degree = 1.0 / np.maximum(self.degrees, 1.0)

    def apply(self, values):
        return 0.5 * (self.incidence.T @ (self.inverse_degree[:, None] * (self.incidence @ values)))

    __call__ = apply
