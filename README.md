# EAMix

Reference implementation of **EAMix: Coupled Dual-Layer Edge Random Walk for Attributed Edge Clustering**.

EAMix assigns every edge of an undirected graph to one of `k` clusters using shared endpoints and edge
attributes:

1. **Complementary edge relations**: the shared signal `Q = RowNorm(U_r Σ_r)` from a truncated SVD of the
   row-normalized edge attributes, the topology walk `P_T = ½ Bᵀ D_V⁺ B` (applied via the incidence matrix),
   and the attribute walk `P_A` from a sparse approximate `k_A`-NN cosine affinity `W_A`.
2. **Coupled dual-layer edge random walk (CDLERW)**: both layers start from `[Q | 1]` and exchange
   accumulated responses through the coupled transition `P_{ω_T}`. Mass-normalized terminal-layer
   responses `V_T`, `V_A` are fused into `H = RowNorm([√ρ V_T | √(1-ρ) V_A])`.
3. **Joint spectral geometry**: full centering `H_c = H - 1 h̄ᵀ`, projection onto at most `k` supported
   eigen-directions of `H_cᵀ H_c` (no eigenvalue rescaling), then row normalization.
4. **Assignment correction**: k-Means soft anchors and fidelity weights, endpoint-agreement weights, and the
   confidence-guided correction `(D_η + λL) F = D_η C`, solved by a certified fixed-point iteration.

## Layout

```
eamix/
  relations.py   Q, W_A, P_A and the incidence topology operator (Sec. "Complementary Edge Relations")
  walk.py        coupled walk, mass normalization and fusion into H (Sec. "Coupled Walk ...")
  clustering.py  joint spectral geometry, soft anchors, endpoint support and correction
  model.py       EAMix / EAMixConfig: the full pipeline (Algorithm "EAMix")
  metrics.py     Hungarian-aligned macro-F1 / macro one-vs-rest AUC, plus ACC, NMI, ARI
tests/
  test_eamix.py  deterministic checks of every component
```

## Installation

```bash
pip install -r requirements.txt
```

The listed versions are the ones used for the paper. The code also runs with newer NumPy/SciPy/scikit-learn.
To get bit-identical numbers, use the pinned versions and a single thread
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`).

## Usage

```python
import numpy as np
from eamix import EAMix, endpoint_descriptors, evaluate

# edges: (m, 2) int array; edge_attributes: (m, d) dense or scipy.sparse matrix; k: number of clusters
model = EAMix().fit(edges, edge_attributes, n_clusters=k)
labels = model.labels_              # corrected edge labels
initial = model.initial_labels_     # k-Means labels before correction

# If only vertex attributes are available, build edge descriptors by concatenating endpoint attributes
# (squared block weights 0.625 / 0.375; list endpoints in increasing vertex-ID order):
edge_attributes = endpoint_descriptors(node_features, edges, share=0.625)

print(evaluate(y_true, labels))     # {'AUC': ..., 'F1': ..., 'ACC': ..., 'NMI': ..., 'ARI': ...}
```

After `fit`, `model.representation_` holds `H`, `model.coordinates_` holds `Y_sp`, and `model.info_`
records the propagation residual, retained spectral rank and correction certificate.

## Configuration

The same configuration is used for every graph (`EAMixConfig` defaults):

| Parameter | Value | Meaning |
|---|---|---|
| `rank` | 128 | maximum retained attribute directions `r` |
| `n_neighbors` / `n_candidates` | 32 / 256 | attribute neighbors `k_A` per edge / candidate budget |
| `alpha` | 0.85 | walk continuation probability |
| `kappa` | 0.4 | coupling strength |
| `omega_T` | 0.5 | topology layer weight (`ω_A = 1 - ω_T`) |
| `rho` | 0.496 | output share of the topology response |
| `lam` | 0.35 | correction strength `λ` |
| `eta_min` | 0.1 | anchor-fidelity floor |
| `n_init`, `random_state` | 10, 0 | k-Means initializations and seed |
| `correction_tol`, `correction_max_iter` | 1e-6, 80 | correction stopping rule (`ε_c`, `T_c`) |

Any field can be overridden, e.g. `EAMix(kappa=0.2, lam=0.5)`.

The coupled walk has a unique fixed point. `solver="bicgstab"` (default) computes it with blocked,
residual-refined BiCGSTAB; `solver="fixed_point"` runs the synchronous iteration of Algorithm CDLERW.
Both are checked against a relative residual below `1e-9`.

## Tests

```bash
python tests/test_eamix.py      # or: python -m pytest tests
```
