"""
optimizer_staging.py
===================================================================================
Reference implementation of the plateau-driven dynamic optimizer staging
algorithm described in "Adaptive Optimizer Sequencing for Kolmogorov-Arnold
Networks" (Algorithm 1). This is an independent, from-scratch implementation
written for public release: it reproduces the algorithm's logic and the
paper's seven optimizer protocols, but is not a copy of the experiments run
for the paper and does not use the paper's private datasets.

Models included are simplified reference KAN variants (Efficient-KAN-style
B-spline, ChebyKAN-style Chebyshev, RBF-KAN-style radial basis) and MLP
baselines, matching the architecture family described in the paper
(single hidden layer of width 32 for KANs; two hidden layers for MLPs).
"""

import time
import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class Config:
    hidden_dim: int = 32
    grid_size: int = 5
    spline_order: int = 3
    grid_range: Tuple[float, float] = (-1.0, 1.0)
    cheby_degree: int = 8
    rbf_centers: int = 8

    lr_adamw: float = 1e-3
    weight_decay: float = 1e-4

    lr_lbfgs: float = 1.0
    lbfgs_max_iter: int = 10
    lbfgs_history_size: int = 15

    sam_rho: float = 0.01

    plateau_patience: int = 10
    early_stopping_patience: int = 25
    time_budget_seconds: float = 60.0

    test_size: float = 0.2
    random_state: int = 42


# =====================================================================
# Basis function layers (simplified reference implementations)
# =====================================================================

class BSplineBasis(nn.Module):
    """B-spline basis, as used by the original KAN / Efficient-KAN."""

    def __init__(self, grid_size: int, spline_order: int, grid_range: Tuple[float, float]):
        super().__init__()
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        grid = self.grid.to(x.device)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            ld = grid[k:-1] - grid[:-k - 1]
            ld[ld == 0] = 1.0
            rd = grid[k + 1:] - grid[1:-k]
            rd[rd == 0] = 1.0
            bases = ((x - grid[:-k - 1]) / ld) * bases[..., :-1] + \
                    ((grid[k + 1:] - x) / rd) * bases[..., 1:]
        return bases


class EfficientKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, cfg: Config):
        super().__init__()
        self.basis = BSplineBasis(cfg.grid_size, cfg.spline_order, cfg.grid_range)
        num_coef = cfg.grid_size + cfg.spline_order
        self.coefficients = nn.Parameter(torch.randn(in_features, out_features, num_coef) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bic,ioc->bo", self.basis(x), self.coefficients)


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, cfg: Config):
        super().__init__()
        self.degree = cfg.cheby_degree
        self.weights = nn.Parameter(torch.randn(in_features, out_features, self.degree) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + 1e-4, 1.0 - 1e-4)
        cheby = [torch.ones_like(x)]
        if self.degree > 1:
            cheby.append(x)
        for i in range(2, self.degree):
            cheby.append(2 * x * cheby[i - 1] - cheby[i - 2])
        cheby = torch.stack(cheby, dim=-1)
        return torch.einsum("bid,iod->bo", cheby, self.weights)


class RBFKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, cfg: Config):
        super().__init__()
        self.num_centers = cfg.rbf_centers
        centers = torch.linspace(cfg.grid_range[0], cfg.grid_range[1], self.num_centers)
        self.register_buffer("centers", centers)
        width = (cfg.grid_range[1] - cfg.grid_range[0]) / max(self.num_centers - 1, 1)
        self.sigma = nn.Parameter(torch.full((in_features, self.num_centers), width * 1.5))
        self.weights = nn.Parameter(torch.randn(in_features, out_features, self.num_centers) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = (x.unsqueeze(-1) - self.centers) ** 2
        basis = torch.exp(-dist / (self.sigma.unsqueeze(0) ** 2 + 1e-8))
        return torch.einsum("bic,ioc->bo", basis, self.weights)


# =====================================================================
# Model wrappers (KAN: single hidden layer of width 32; MLP: two hidden layers)
# =====================================================================

class BaseModel(nn.Module):
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EfficientKANModel(BaseModel):
    def __init__(self, in_dim: int, cfg: Config):
        super().__init__()
        self.l1 = EfficientKANLayer(in_dim, cfg.hidden_dim, cfg)
        self.l2 = EfficientKANLayer(cfg.hidden_dim, 1, cfg)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


class ChebyKANModel(BaseModel):
    def __init__(self, in_dim: int, cfg: Config):
        super().__init__()
        self.l1 = ChebyKANLayer(in_dim, cfg.hidden_dim, cfg)
        self.l2 = ChebyKANLayer(cfg.hidden_dim, 1, cfg)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


class RBFKANModel(BaseModel):
    def __init__(self, in_dim: int, cfg: Config):
        super().__init__()
        self.l1 = RBFKANLayer(in_dim, cfg.hidden_dim, cfg)
        self.l2 = RBFKANLayer(cfg.hidden_dim, 1, cfg)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


class MLPModel(BaseModel):
    def __init__(self, in_dim: int, cfg: Config, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x)


ARCHITECTURES = {
    "Efficient-KAN": EfficientKANModel,
    "ChebyKAN": ChebyKANModel,
    "RBF-KAN": RBFKANModel,
    "MLP": MLPModel,
}


# =====================================================================
# SAM + AdamW
# =====================================================================

class SAMAdamW:
    """Sharpness-Aware Minimization wrapping AdamW (Foret et al., 2021)."""

    def __init__(self, model: nn.Module, cfg: Config, rho: Optional[float] = None):
        self.model = model
        self.rho = rho if rho is not None else cfg.sam_rho
        self.base_opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_adamw, weight_decay=cfg.weight_decay)
        self._e_ws: List[Optional[torch.Tensor]] = []

    @torch.no_grad()
    def first_step(self):
        grad_norm = torch.norm(
            torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None])
        ).clamp(min=1e-12)
        self._e_ws = []
        for p in self.model.parameters():
            if p.grad is None:
                self._e_ws.append(None)
                continue
            e_w = p.grad * (self.rho / grad_norm)
            p.add_(e_w)
            self._e_ws.append(e_w)
        # Clear gradients from the clean-point pass so the base optimizer
        # step only uses the gradient at the perturbed point.
        self.base_opt.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for p, e_w in zip(self.model.parameters(), self._e_ws):
            if e_w is not None:
                p.sub_(e_w)
        self.base_opt.step()
        self.base_opt.zero_grad()

    def zero_grad(self):
        self.base_opt.zero_grad()


# =====================================================================
# Protocol definitions (Section 3.3 of the paper)
# =====================================================================

@dataclass
class Protocol:
    name: str
    phases: List[str]  # each phase is "adamw", "lbfgs", or "sam_adamw"


PROTOCOLS: List[Protocol] = [
    Protocol("AdamW Only", ["adamw"]),
    Protocol("L-BFGS Only", ["lbfgs"]),
    Protocol("AdamW -> L-BFGS", ["adamw", "lbfgs"]),
    Protocol("L-BFGS -> AdamW", ["lbfgs", "adamw"]),
    Protocol("SAM + AdamW", ["sam_adamw"]),
    Protocol("SAM+AdamW -> L-BFGS", ["sam_adamw", "lbfgs"]),
    Protocol("L-BFGS -> SAM+AdamW", ["lbfgs", "sam_adamw"]),
]

# L-BFGS protocols are excluded from MLP evaluation (Section 3.3, point 8).
MLP_ALLOWED_PROTOCOLS = {"AdamW Only", "SAM + AdamW"}


def make_optimizer(kind: str, model: nn.Module, cfg: Config, sam_rho: Optional[float] = None):
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr_adamw, weight_decay=cfg.weight_decay)
    if kind == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(), lr=cfg.lr_lbfgs, max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history_size, line_search_fn="strong_wolfe",
        )
    if kind == "sam_adamw":
        return SAMAdamW(model, cfg, rho=sam_rho)
    raise ValueError(f"Unknown optimizer kind: {kind}")


def optimizer_step(kind: str, opt, model, x, y, loss_fn):
    if kind == "adamw":
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    elif kind == "lbfgs":
        def closure():
            opt.zero_grad()
            l = loss_fn(model(x), y)
            l.backward()
            return l
        opt.step(closure)
    elif kind == "sam_adamw":
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.first_step()
        loss_fn(model(x), y).backward()
        opt.second_step()
    else:
        raise ValueError(f"Unknown optimizer kind: {kind}")


# =====================================================================
# Algorithm 1: Dynamic Plateau-Driven Optimizer Staging
# =====================================================================

def train_staged(model: nn.Module, x_train, y_train, x_val, y_val,
                  protocol: Protocol, cfg: Config, sam_rho: Optional[float] = None) -> Dict:
    """
    Implements Algorithm 1 from the paper: switches optimizers on
    plateau_patience epochs of stagnation, halts on early_stopping_patience
    or the wall-clock budget, and always returns the best checkpoint seen
    across all phases (not the final iteration).
    """
    def mse(pred, target):
        return torch.mean((pred - target) ** 2)

    phase_idx = 0
    opt = make_optimizer(protocol.phases[phase_idx], model, cfg, sam_rho=sam_rho)

    best_weights = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    phase_best = float("inf")
    no_improve = 0
    epoch = 0
    switch_epochs = []
    t0 = time.time()
    stop_reason = "exhausted_phases"

    while True:
        if (time.time() - t0) > cfg.time_budget_seconds:
            stop_reason = "time_budget"
            break

        optimizer_step(protocol.phases[phase_idx], opt, model, x_train, y_train, mse)

        with torch.no_grad():
            val_loss = mse(model(x_val), y_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        if val_loss < phase_best:
            phase_best = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stopping_patience:
                stop_reason = "early_stopping"
                break
            elif no_improve >= cfg.plateau_patience and phase_idx < len(protocol.phases) - 1:
                phase_idx += 1
                switch_epochs.append(epoch)
                opt = make_optimizer(protocol.phases[phase_idx], model, cfg, sam_rho=sam_rho)
                phase_best = float("inf")
                no_improve = 0

        epoch += 1

    model.load_state_dict(best_weights)
    wall_time = time.time() - t0

    return {
        "epochs": epoch,
        "wall_time": wall_time,
        "stop_reason": stop_reason,
        "switch_epochs": switch_epochs,
        "n_phases_reached": phase_idx + 1,
        "n_phases_total": len(protocol.phases),
        "best_val_loss": best_loss,
    }
