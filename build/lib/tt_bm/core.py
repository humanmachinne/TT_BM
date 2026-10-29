"""
Core routines for MPO-based Burer-Monteiro rank-2 SDP/QUBO solver.

Functions operate on MPO-structured cost matrices and rank-2
angular (alpha) parametrisations of the Burer-Monteiro relaxation.
"""

import torch
from typing import List, Optional, Tuple

Tensor = torch.Tensor
MPO = List[Tensor]  # list of 4-index cores [left, phys_out, phys_in, right]


# ---------------------------------------------------------------------------
# MPO application
# ---------------------------------------------------------------------------

def mpo_vec(mpo: MPO, vec: Tensor) -> Tensor:
    """Apply an MPO to a vector: result = MPO @ vec."""
    res = vec.clone().reshape(1, -1, 1)
    dim = len(mpo)
    phys = torch.tensor([mpo[i].shape[2] for i in range(dim)])
    left_prod = 1

    for i in range(dim):
        right_prod = phys[i+1:].prod().item()
        res = res.reshape(res.shape[0], left_prod, phys[i], right_prod, res.shape[-1])
        res = torch.einsum('inmj,lompi->lonpj', mpo[i], res)
        left_prod *= phys[i].item()

    return res.flatten()

def estimate_op_norm(mpo: MPO, n_iter: int = 20) -> float:
    """Estimate ||MPO||_op via power iteration using mpo_vec."""
    n = 1
    for core in mpo:
        n *= core.shape[2]
    dtype, device = mpo[0].dtype, mpo[0].device
    v = torch.randn(n, dtype=dtype, device=device)
    v = v / v.norm()
    for _ in range(n_iter):
        w = mpo_vec(mpo, v)
        sigma = w.norm()
        v = w / sigma
    return sigma.item()


def scale_problem(mpo: MPO, h: Tensor) -> Tuple[MPO, Tensor, float]:
    """Scale J_mpo and h by 1/||J||_op.  Returns (mpo_scaled, h_scaled, scale_factor)."""
    sigma = estimate_op_norm(mpo)
    mpo_scaled = [core.clone() for core in mpo]
    mpo_scaled[0] = mpo_scaled[0] / sigma   # absorb scalar into first core
    h_scaled = h / sigma
    return mpo_scaled, h_scaled, sigma


def mpo_matrix(mpo: MPO, M: Tensor) -> Tensor:
    """Apply an MPO to each column of a matrix: result = MPO @ M."""
    res = M.clone().reshape(1, M.shape[0], M.shape[1], 1)
    dim = len(mpo)
    phys = torch.tensor([mpo[i].shape[2] for i in range(dim)])
    left_prod = 1

    for i in range(dim):
        right_prod = phys[i+1:].prod().item()
        res = res.reshape(
            res.shape[0], left_prod, phys[i], right_prod,
            res.shape[-2], res.shape[-1],
        )
        res = torch.einsum('inmj,lompbi->lonpbj', mpo[i], res)
        left_prod *= phys[i].item()

    return res.reshape(M.shape)


# ---------------------------------------------------------------------------
# Objective & gradient (rank-2 angular parametrisation)
# ---------------------------------------------------------------------------

def sdp_r2(mpo: MPO, h: Tensor, alpha: Tensor, alpha0: Tensor) -> Tensor:
    """Rank-2 SDP objective  ⟨V, Q V⟩ + h·diag(V)  in angular form."""
    coupling = (
        torch.cos(alpha) @ mpo_vec(mpo, torch.cos(alpha))
        + torch.sin(alpha) @ mpo_vec(mpo, torch.sin(alpha))
    )
    field = h @ torch.cos(alpha - alpha0)
    return coupling + field


def grad_r2(mpo: MPO, h: Tensor, alpha: Tensor, alpha0: Tensor) -> Tensor:
    """Gradient of the rank-2 SDP objective w.r.t. alpha."""
    coupling = (
        2 * torch.cos(alpha) * mpo_vec(mpo, torch.sin(alpha))
        - 2 * torch.sin(alpha) * mpo_vec(mpo, torch.cos(alpha))
    )
    field = -h * torch.sin(alpha - alpha0)
    return coupling + field


def ising_energy(mpo: MPO, h: Tensor, S: Tensor) -> Tensor:
    """Ising energy for a set of spin configurations S (n_spins × n_samples)."""
    coupling = torch.einsum('in,in->n', S, mpo_matrix(mpo, S))
    field = torch.einsum('i,in->n', h, S)
    return coupling + field


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def armijo(
    mpo: MPO,
    h: Tensor,
    alpha: Tensor,
    alpha0: Tensor,
    p: Tensor,
    grad: Tensor,
    eta: float = 1.0,
    c: float = 0.5,
    tau: float = 0.5,
    max_iter: int = 40,
) -> float:
    """Armijo back-tracking line search along direction *p*."""
    f0 = sdp_r2(mpo, h, alpha, alpha0).item()
    slope = (grad * p).sum().item()
    for _ in range(max_iter):
        if sdp_r2(mpo, h, alpha + eta * p, alpha0).item() <= f0 + c * eta * slope:
            return eta
        eta *= tau
    return eta


def cg_step(
    mpo: MPO,
    h: Tensor,
    alpha: Tensor,
    alpha0: Tensor,
    p: Tensor,
    grad: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """One Polak-Ribière+ conjugate-gradient step."""
    eta = armijo(mpo, h, alpha, alpha0, p, grad)
    alpha = alpha + eta * p
    grad_new = grad_r2(mpo, h, alpha, alpha0)
    delta_g = grad_new - grad
    beta = (grad_new @ delta_g) / (grad @ grad)
    beta = max(beta.item(), 0.0)
    p = -grad_new + beta * p
    return alpha, p, grad_new, beta


def optimise(
    mpo: MPO,
    h: Tensor,
    alpha: Tensor,
    alpha0: Tensor,
    n_iter: int,
    tol: float = 1e-7,
    plateau_window: int = 100,
    plateau_eps: float = 1e-4,
    verbose: bool = True,
) -> Tuple[Tensor, List[float]]:
    """
    Run the CG optimisation loop.

    Parameters
    ----------
    mpo : list of Tensor
        MPO cores encoding the QUBO cost matrix.
    h : Tensor
        Linear (field) term.
    alpha, alpha0 : Tensor
        Initial angles for the rank-2 parametrisation.
    n_iter : int
        Maximum number of CG iterations.
    tol : float
        Gradient-norm convergence threshold.
    plateau_window : int
        Number of past iterations to compare for plateau detection.
    plateau_eps : float
        Plateau convergence threshold.
    verbose : bool
        Print progress every 100 iterations.

    Returns
    -------
    alpha : Tensor
        Optimised angles.
    obj : list of float
        Objective value at each iteration.
    """
    grad = grad_r2(mpo, h, alpha, alpha0)
    p = -grad.clone()
    obj: List[float] = []

    for i in range(n_iter):
        alpha, p, grad, beta = cg_step(mpo, h, alpha, alpha0, p, grad)
        grad_norm = grad.norm().item()
        obj.append(sdp_r2(mpo, h, alpha, alpha0).item())

        if verbose and i % 100 == 0:
            print(
                f"iter {i:4d} | obj {obj[-1]:10.4f} "
                f"| |g| {grad_norm:.2e} | beta {beta:.3f}"
            )

        if grad_norm < tol:
            if verbose:
                print(f"Converged at iter {i} (gradient norm)")
            break

        if i > plateau_window and abs(obj[i] - obj[i - plateau_window]) < plateau_eps:
            if verbose:
                print(f"Converged at iter {i} (plateau)")
            break

    return alpha, obj


# ---------------------------------------------------------------------------
# GW-style random-hyperplane rounding
# ---------------------------------------------------------------------------

def round_gw(
    alpha: Tensor,
    alpha0: Tensor,
    n_rounds: int,
) -> Tensor:
    """
    Goemans-Williamson random-hyperplane rounding.

    Returns an (n_spins × n_rounds) tensor of ±1 spin assignments.
    """
    alpha_ref = torch.rand(n_rounds, device=alpha.device, dtype=alpha.dtype) * 2 * torch.pi
    s_coupling = (
        torch.cos(alpha.reshape(-1, 1)) * torch.cos(alpha_ref.reshape(1, -1))
        + torch.sin(alpha.reshape(-1, 1)) * torch.sin(alpha_ref.reshape(1, -1))
    ).sign()
    s_field = (
        torch.cos(alpha0) * torch.cos(alpha_ref)
        + torch.sin(alpha0) * torch.sin(alpha_ref)
    ).sign()
    return s_coupling * s_field.reshape(1, -1)
