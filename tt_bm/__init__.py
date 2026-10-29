"""
TT_BM — MPO-based Burer-Monteiro rank-2 SDP/QUBO solver.
"""

__version__ = "0.1.0"

from .core import (
    mpo_vec,
    mpo_matrix,
    sdp_r2,
    grad_r2,
    ising_energy,
    armijo,
    cg_step,
    optimise,
    round_gw,
    scale_problem
)

__all__ = [
    "mpo_vec",
    "mpo_matrix",
    "sdp_r2",
    "grad_r2",
    "ising_energy",
    "armijo",
    "cg_step",
    "optimise",
    "round_gw",
    "scale_problem"
]
