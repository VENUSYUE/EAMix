"""Joint spectral edge clustering and confidence-guided assignment correction."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .relations import row_normalize


class InsufficientRepresentation(ValueError):
    """Raised when the spectral coordinates cannot support ``k`` clusters."""


def spectral_coordinates(H, n_clusters, relative_cutoff=1e-10, absolute_cutoff=1e-12):
    """Y_sp = RowNorm(H_c V_sp) with H_c the fully centered H.

    V_sp holds the leading q = min(k, s_sp) eigenvectors of G_sp = H_c^T H_c whose eigenvalues exceed
    theta_sp = max(relative_cutoff * l_1, absolute_cutoff). No eigenvalue rescaling is applied, so the
    projected directions keep their singular-value scales before row normalization.
    """
    if H.ndim != 2 or not H.size or not np.all(np.isfinite(H)) or n_clusters < 1:
        raise ValueError("Finite nonempty matrix and positive cluster count required")
    centered = H - H.mean(axis=0, keepdims=True)
    gram = centered.T @ centered
    eigenvalues, directions = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, directions = eigenvalues[order], directions[:, order]
    cutoff = max(float(eigenvalues[0]) * relative_cutoff, absolute_cutoff)
    supported = int(np.sum(eigenvalues > cutoff))
    retained = min(int(n_clusters), supported)
    if not retained:
        raise InsufficientRepresentation("No spectral direction exceeds the support cutoff")
    coordinates = row_normalize(centered @ directions[:, :retained])
    if len(np.unique(coordinates, axis=0)) < n_clusters:
        raise InsufficientRepresentation("Fewer than k distinct spectral rows")
    return coordinates, dict(supported_rank=supported, retained_rank=retained, cutoff=cutoff,
                             retained_energy=float(eigenvalues[:retained].sum()
                                                   / max(float(np.trace(gram)), 1e-12)))


def correct_assignments(coordinates, n_clusters, topology, affinity, omega_T=0.5, lam=0.35,
                        eta_min=0.1, epsilon_D=1e-12, tau_min=1e-8, n_init=10, random_state=0,
                        max_iter=80, tol=1e-6):
    """k-Means on Y_sp followed by the confidence-guided probability correction.

    Solves (D_eta + lam L) F = D_eta C with the diagonally preconditioned fixed-point update
    F_e <- (eta_e C_e + lam (W F)_e) / (eta_e + lam d_W,e), where
    W = w_T/2 B^T diag(g) B + w_A / s_A W_A and g_v = a_v / d_v, a_v = 1 / (1 + s_v).
    Returns final labels, initial k-Means labels, and diagnostics.
    """
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    initial = model.fit_predict(coordinates)
    if n_clusters == 1:
        return initial, initial, dict(iterations=0, converged=True)
    incidence, inverse_degree = topology.incidence, topology.inverse_degree
    omega_A = 1.0 - omega_T

    # Soft anchors C and fidelity weights eta from squared center distances.
    distances = model.transform(coordinates) ** 2
    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    nearest, second = nearest_two.min(axis=1), nearest_two.max(axis=1)
    tau = max(float(np.median(nearest)), tau_min)
    anchors = np.exp(-(distances - distances.min(axis=1, keepdims=True)) / tau)
    anchors /= anchors.sum(axis=1, keepdims=True)
    fidelity = eta_min + (1 - eta_min) * np.clip((second - nearest) / (second + epsilon_D), 0, 1)

    # Endpoint agreement a_v = 1 / (1 + s_v) from first and second anchor moments.
    vertex_mean = incidence @ anchors * inverse_degree[:, None]
    dispersion = np.maximum(incidence @ np.sum(anchors * anchors, axis=1) * inverse_degree
                            - np.sum(vertex_mean * vertex_mean, axis=1), 0)
    agreement = 1.0 / (1.0 + dispersion)

    # Correction degrees d_W and the contraction factor rho_c.
    attribute_degree = np.asarray(affinity.sum(axis=1)).ravel()
    scale = max(float(attribute_degree.max()), 1.0)
    scaled_affinity = affinity / scale
    degree = omega_T * 0.5 * np.asarray(incidence.T @ agreement).ravel()
    degree += omega_A * attribute_degree / scale
    denominator = fidelity + lam * degree
    contraction = float(np.max(lam * degree / denominator))

    state, error_bound = anchors.copy(), float("inf")
    endpoint_weight = (agreement * inverse_degree)[:, None]
    for iteration in range(1, max_iter + 1):
        message = omega_T * 0.5 * (incidence.T @ (endpoint_weight * (incidence @ state)))
        message += omega_A * (scaled_affinity @ state)
        candidate = (fidelity[:, None] * anchors + lam * message) / denominator[:, None]
        change = float(np.max(np.abs(candidate - state)))
        state = candidate
        error_bound = contraction * change / max(1.0 - contraction, 1e-12)
        if error_bound < tol:
            break
    details = dict(iterations=iteration, converged=error_bound < tol, contraction=contraction,
                   error_bound=error_bound, temperature=tau)
    return state.argmax(axis=1), initial, details
