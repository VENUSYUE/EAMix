"""Coupled dual-layer edge random walk (CDLERW) and terminal-layer responses.

Both layers start from [Q | 1] and satisfy the fixed point of

    T = (1 - alpha) [Q | 1] + alpha [(1 - kappa w_A) P_T^T T + kappa w_T A]
    A = (1 - alpha) [Q | 1] + alpha [(1 - kappa w_T) P_A^T A + kappa w_A T].

The feature columns are divided by the propagated mass column (V_T, V_A) and fused into
H = RowNorm([sqrt(rho) V_T | sqrt(1 - rho) V_A]).
"""

from __future__ import annotations

import inspect

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab

from .relations import row_normalize

_TOLERANCE_KEYWORD = "rtol" if "rtol" in inspect.signature(bicgstab).parameters else "tol"


class CoupledWalk:
    """Coupled transition P_{w_T} acting on the stacked layer responses."""

    def __init__(self, topology, attribute_transition, alpha=0.85, kappa=0.4, omega_T=0.5):
        if not 0 < alpha < 1 or not 0 < kappa <= 1 or not 0 < omega_T < 1:
            raise ValueError("Require 0 < alpha < 1, 0 < kappa <= 1 and 0 < omega_T < 1")
        self.topology = topology
        self.attribute_transposed = attribute_transition.T.tocsr()
        self.alpha, self.kappa = alpha, kappa
        self.omega_T, self.omega_A = omega_T, 1.0 - omega_T

    def apply(self, topology_state, attribute_state):
        """Return the two blocks of P_{w_T}^T [T; A]."""
        kappa, omega_T, omega_A = self.kappa, self.omega_T, self.omega_A
        return ((1 - kappa * omega_A) * self.topology(topology_state) + kappa * omega_T * attribute_state,
                (1 - kappa * omega_T) * (self.attribute_transposed @ attribute_state)
                + kappa * omega_A * topology_state)

    def fixed_point(self, signal, max_iter=160, tol=1e-10):
        """Synchronous fixed-point iteration of the coupled update (Algorithm CDLERW)."""
        restart, alpha = 1.0 - self.alpha, self.alpha
        topology_state, attribute_state = signal.copy(), signal.copy()
        for _ in range(max_iter):
            applied_topology, applied_attribute = self.apply(topology_state, attribute_state)
            next_topology = restart * signal + alpha * applied_topology
            next_attribute = restart * signal + alpha * applied_attribute
            change = max(float(np.max(np.abs(next_topology - topology_state))),
                         float(np.max(np.abs(next_attribute - attribute_state))))
            topology_state, attribute_state = next_topology, next_attribute
            if change < tol:
                break
        return topology_state, attribute_state

    def _solve_block(self, signal):
        """Solve (I - alpha P^T) U = (1 - alpha) [signal; signal] by refined BiCGSTAB."""
        shape, size = signal.shape, signal.size
        alpha = self.alpha

        def operator(vector):
            topology_state = vector[:size].reshape(shape)
            attribute_state = vector[size:].reshape(shape)
            applied_topology, applied_attribute = self.apply(topology_state, attribute_state)
            return np.concatenate(((topology_state - alpha * applied_topology).ravel(),
                                   (attribute_state - alpha * applied_attribute).ravel()))

        system = LinearOperator((2 * size, 2 * size), matvec=operator, dtype=np.float64)
        initial = np.tile(signal.ravel(), 2)
        right = (1.0 - alpha) * initial
        solution, status = bicgstab(system, right, x0=initial, atol=1e-12, maxiter=240,
                                    **{_TOLERANCE_KEYWORD: 1e-14})
        for refinement in range(4):
            residual = right - operator(solution)
            relative = float(np.abs(residual).sum() / max(np.abs(solution).sum(), 1e-12))
            if status == 0 and np.max(np.abs(residual)) < 1e-10 and relative < 1e-11:
                return solution[:size].reshape(shape), solution[size:].reshape(shape)
            if refinement < 3:
                correction, status = bicgstab(system, residual, atol=1e-14, maxiter=240,
                                              **{_TOLERANCE_KEYWORD: 1e-12})
                solution += correction
        return self.fixed_point(signal)

    def solve(self, signal, solver="bicgstab", block_width=24):
        """Fixed point of the coupled update for every column of ``signal``.

        ``solver="bicgstab"`` solves the same linear system in column blocks with refined BiCGSTAB
        (falling back to fixed-point iteration); ``solver="fixed_point"`` runs the iteration
        directly. Both return the same fixed point up to the residual tolerance.
        """
        if solver == "fixed_point":
            return self.fixed_point(signal)
        if solver != "bicgstab":
            raise ValueError("solver must be 'bicgstab' or 'fixed_point'")
        topology_state, attribute_state = np.empty_like(signal), np.empty_like(signal)
        for start in range(0, signal.shape[1], block_width):
            stop = min(start + block_width, signal.shape[1])
            block = self._solve_block(np.ascontiguousarray(signal[:, start:stop]))
            topology_state[:, start:stop], attribute_state[:, start:stop] = block
        return topology_state, attribute_state

    def relative_residual(self, signal, topology_state, attribute_state):
        applied_topology, applied_attribute = self.apply(topology_state, attribute_state)
        restart = 1.0 - self.alpha
        error = (np.abs(topology_state - restart * signal - self.alpha * applied_topology).sum()
                 + np.abs(attribute_state - restart * signal - self.alpha * applied_attribute).sum())
        return float(error / max(np.abs(topology_state).sum() + np.abs(attribute_state).sum(), 1e-12))


def terminal_responses(walk, signal, solver="bicgstab", residual_tol=1e-9):
    """Propagate [Q | 1] and return the mass-normalized responses V_T, V_A."""
    augmented = np.column_stack((signal, np.ones(len(signal))))
    topology_raw, attribute_raw = walk.solve(augmented, solver=solver)
    residual = walk.relative_residual(augmented, topology_raw, attribute_raw)
    if min(topology_raw[:, -1].min(), attribute_raw[:, -1].min()) <= 0 or residual >= residual_tol:
        raise ArithmeticError(f"Coupled propagation did not converge (relative residual {residual:.3e})")
    return (topology_raw[:, :-1] / topology_raw[:, -1:],
            attribute_raw[:, :-1] / attribute_raw[:, -1:], residual)


def fuse(topology_response, attribute_response, rho=0.496):
    """H = RowNorm([sqrt(rho) V_T | sqrt(1 - rho) V_A]).

    The inner row normalization is a positive per-row rescaling, which the outer RowNorm cancels.
    """
    states = row_normalize(np.hstack((topology_response, attribute_response)))
    return row_normalize(np.hstack((np.sqrt(rho) * states[:, :topology_response.shape[1]],
                                   np.sqrt(1 - rho) * states[:, topology_response.shape[1]:])))
