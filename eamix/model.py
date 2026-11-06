"""EAMix: coupled dual-layer edge random walk for attributed edge clustering."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sparse

from .clustering import correct_assignments, spectral_coordinates
from .relations import (TopologyOperator, attribute_affinity, attribute_transition,
                        shared_signal)
from .walk import CoupledWalk, fuse, terminal_responses


@dataclass
class EAMixConfig:
    """Default configuration used for every graph in the paper."""

    rank: int = 128            # r, maximum number of retained attribute directions
    n_neighbors: int = 32      # k_A, attribute neighbors per edge before symmetrization
    n_candidates: int = 256    # candidate budget of the approximate neighbor search
    alpha: float = 0.85        # walk continuation probability
    kappa: float = 0.4         # coupling strength
    omega_T: float = 0.5       # topology layer weight (omega_A = 1 - omega_T)
    rho: float = 0.496         # output share of the topology response
    lam: float = 0.35          # correction strength lambda
    eta_min: float = 0.1       # fidelity floor
    n_init: int = 10           # k-Means initializations
    random_state: int = 0
    solver: str = "bicgstab"   # "bicgstab" or "fixed_point"; both solve the coupled fixed point
    correction_tol: float = 1e-6
    correction_max_iter: int = 80


class EAMix:
    """Attributed edge clustering.

    Example
    -------
    >>> labels = EAMix().fit_predict(edges, edge_attributes, n_clusters=k)
    """

    def __init__(self, config: EAMixConfig | None = None, **overrides):
        self.config = EAMixConfig(**{**asdict(config or EAMixConfig()), **overrides})

    def fit(self, edges, edge_attributes, n_clusters, n_nodes=None):
        """Cluster the ``m`` edges of an undirected graph.

        edges           : (m, 2) integer array of distinct endpoints.
        edge_attributes : (m, d) dense or sparse attribute matrix X, one row per edge.
        n_clusters      : number of clusters k, 1 <= k <= m.
        n_nodes         : number of vertices (defaults to ``edges.max() + 1``).
        """
        c = self.config
        edges = np.asarray(edges, dtype=np.int64)
        n_nodes = int(edges.max()) + 1 if n_nodes is None else int(n_nodes)
        if (edges.ndim != 2 or edges.shape[1] != 2 or not len(edges) or np.any(edges < 0)
                or np.any(edges >= n_nodes) or np.any(edges[:, 0] == edges[:, 1])):
            raise ValueError("edges must be a nonempty (m, 2) array of distinct valid endpoints")
        if edge_attributes.shape[0] != len(edges):
            raise ValueError("edge_attributes must have one row per edge")
        if not 1 <= n_clusters <= len(edges):
            raise ValueError("Require 1 <= n_clusters <= number of edges")
        if n_clusters == 1:
            self.labels_ = self.initial_labels_ = np.zeros(len(edges), dtype=np.int64)
            self.info_ = {}
            return self

        # Relations: Q, W_A, P_A and the incidence form of P_T.
        signal = shared_signal(edge_attributes, rank=c.rank, random_state=c.random_state)
        affinity = attribute_affinity(signal, c.n_neighbors, c.n_candidates, seed=c.random_state)
        transition, isolated = attribute_transition(affinity)
        topology = TopologyOperator(edges, n_nodes)

        # Coupled walk, mass normalization and fusion into H.
        walk = CoupledWalk(topology, transition, alpha=c.alpha, kappa=c.kappa, omega_T=c.omega_T)
        V_T, V_A, residual = terminal_responses(walk, signal, solver=c.solver)
        self.representation_ = fuse(V_T, V_A, rho=c.rho)

        # Joint spectral geometry, k-Means and assignment correction.
        self.coordinates_, geometry = spectral_coordinates(self.representation_, n_clusters)
        self.labels_, self.initial_labels_, correction = correct_assignments(
            self.coordinates_, n_clusters, topology, affinity, omega_T=c.omega_T, lam=c.lam,
            eta_min=c.eta_min, n_init=c.n_init, random_state=c.random_state,
            max_iter=c.correction_max_iter, tol=c.correction_tol)
        self.info_ = dict(propagation_relative_residual=residual,
                          attribute_isolated_fraction=float(isolated.mean()),
                          signal_rank=signal.shape[1], geometry=geometry, correction=correction)
        return self

    def fit_predict(self, edges, edge_attributes, n_clusters, n_nodes=None):
        return self.fit(edges, edge_attributes, n_clusters, n_nodes).labels_
