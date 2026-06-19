# TT_BM

MPO-based Burer-Monteiro rank-2 SDP/QUBO solver.

Solves the Goemans-Williamson SDP relaxation of Ising/QUBO problems where the cost matrix is represented as a Matrix Product Operator (MPO), enabling sub-exponential memory scaling for structured instances.

## Installation

```bash
pip install .
```

Or in editable/development mode:

```bash
pip install -e .
```

## Quick start

```python
import torch
import tt_bm

# Suppose `mpo` is a list of MPO cores and `h` is the field vector
n_spins = 2 ** len(mpo)

alpha  = torch.randn(n_spins, device='cuda') * 0.01
alpha0 = torch.zeros(1, device='cuda')

# Optimise the rank-2 SDP relaxation
alpha_opt, obj_history = tt_bm.optimise(mpo, h, alpha, alpha0, n_iter=5000)

# GW random-hyperplane rounding
spins = tt_bm.round_gw(alpha_opt, alpha0, n_rounds=1000)

# Evaluate Ising energies of rounded solutions
energies = tt_bm.ising_energy(mpo, h, spins)
best = spins[:, energies.argmin()]
```

## API reference

| Function | Description |
|---|---|
| `mpo_vec(mpo, vec)` | Apply MPO to a vector |
| `mpo_matrix(mpo, M)` | Apply MPO to each column of a matrix |
| `sdp_r2(mpo, h, alpha, alpha0)` | Rank-2 SDP objective value |
| `grad_r2(mpo, h, alpha, alpha0)` | Gradient of the SDP objective |
| `ising_energy(mpo, h, S)` | Ising energy for spin configurations |
| `optimise(mpo, h, alpha, alpha0, n_iter, ...)` | CG optimisation loop |
| `round_gw(alpha, alpha0, n_rounds)` | GW random-hyperplane rounding |

## License

MIT
