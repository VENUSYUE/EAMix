"""Deterministic checks of the EAMix components. Run: python -m pytest tests  (or python tests/test_eamix.py)."""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eamix import (EAMix, CoupledWalk, InsufficientRepresentation, TopologyOperator, attribute_affinity,
                   attribute_transition, endpoint_descriptors, evaluate, row_normalize, shared_signal,
                   spectral_coordinates, terminal_responses)

FEATURES = sparse.csr_matrix([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
EDGES = np.asarray([[0, 1], [0, 2], [1, 2], [2, 3], [3, 0], [3, 1]])


def test_shared_signal_preserves_descriptor_geometry():
    descriptors = endpoint_descriptors(FEATURES, EDGES)
    signal = shared_signal(descriptors)
    direct = row_normalize(np.hstack((np.sqrt(.625) * FEATURES.toarray()[EDGES[:, 0]],
                                      np.sqrt(.375) * FEATURES.toarray()[EDGES[:, 1]])))
    np.testing.assert_allclose(signal @ signal.T, direct @ direct.T, atol=1e-12)


def test_topology_walk_is_symmetric_and_stochastic():
    matrix = TopologyOperator(EDGES, 4)(np.eye(len(EDGES)))
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


def test_attribute_transition_isolated_rows():
    transition, isolated = attribute_transition(sparse.csr_matrix((3, 3)))
    assert isolated.all()
    np.testing.assert_allclose(transition.toarray(), np.eye(3))


def test_coupled_walk_matches_dense_solve():
    signal = shared_signal(endpoint_descriptors(FEATURES, EDGES))
    topology = TopologyOperator(EDGES, 4)
    transition, _ = attribute_transition(attribute_affinity(signal))
    m, alpha, kappa, omega_T = len(EDGES), .85, .4, .7
    walk = CoupledWalk(topology, transition, alpha, kappa, omega_T)
    augmented = np.column_stack((signal, np.ones(m)))
    P_T, P_A, I = topology(np.eye(m)), transition.toarray(), np.eye(m)
    block = np.block([[(1 - kappa * (1 - omega_T)) * P_T, kappa * (1 - omega_T) * I],
                      [kappa * omega_T * I, (1 - kappa * omega_T) * P_A]])
    exact = np.linalg.solve(np.eye(2 * m) - alpha * block.T, (1 - alpha) * np.vstack((augmented, augmented)))
    for solver in ("bicgstab", "fixed_point"):
        np.testing.assert_allclose(np.vstack(walk.solve(augmented, solver=solver)), exact, atol=1e-9)
    V_T, V_A, residual = terminal_responses(walk, signal)
    assert residual < 1e-9
    np.testing.assert_allclose(V_T, exact[:m, :-1] / exact[:m, -1:], atol=1e-9)


def test_spectral_coordinates_full_center_unscaled():
    random = np.random.default_rng(1025)
    values = random.normal(size=(37, 12)) + random.normal(size=(1, 12))
    actual, info = spectral_coordinates(values, 4)
    left, singular, _ = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)
    expected = row_normalize(left[:, :4] * singular[:4])
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=1e-10)
    rotation, _ = np.linalg.qr(random.normal(size=(12, 12)))
    for transformed in (values * 7, values @ rotation, values + random.normal(size=(1, 12))):
        coordinates, _ = spectral_coordinates(transformed, 4)
        np.testing.assert_allclose(actual @ actual.T, coordinates @ coordinates.T, atol=1e-10)
    assert info["retained_rank"] == 4
    for degenerate in (np.zeros((8, 6)), np.ones((8, 6))):
        try:
            spectral_coordinates(degenerate, 2)
        except InsufficientRepresentation:
            continue
        raise AssertionError("degenerate input accepted")


def test_end_to_end():
    model = EAMix().fit(EDGES, endpoint_descriptors(FEATURES, EDGES), 2)
    assert model.labels_.shape == model.initial_labels_.shape == (len(EDGES),)
    assert model.info_["correction"]["converged"]
    assert np.all(EAMix().fit_predict(EDGES, endpoint_descriptors(FEATURES, EDGES), 1) == 0)
    scores = evaluate(model.labels_, model.labels_)
    assert all(abs(scores[name] - 1) < 1e-12 for name in ("AUC", "F1", "ACC", "NMI", "ARI"))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("passed", name)
