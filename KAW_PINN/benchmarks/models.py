import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class InputScaler(nn.Module):
    def __init__(self, in_dim: int, ranges: Optional[Sequence[Tuple[float, float]]] = None):
        super().__init__()
        if ranges is None:
            lows = torch.zeros(in_dim)
            highs = torch.ones(in_dim)
        else:
            lows = torch.tensor([item[0] for item in ranges], dtype=torch.float32)
            highs = torch.tensor([item[1] for item in ranges], dtype=torch.float32)
        self.register_buffer("x_low", lows.view(1, in_dim))
        self.register_buffer("x_scale", (highs - lows).clamp_min(1e-12).view(1, in_dim))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.x_low) / self.x_scale - 1.0


class OutputRescale(nn.Module):
    """Map a model's dimensionless output back to physical field units."""

    def __init__(self, model: nn.Module, mean: Sequence[float], scale: Sequence[float]):
        super().__init__()
        self.model = model
        self.register_buffer("output_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("output_scale", torch.tensor(scale, dtype=torch.float32).clamp_min(1.0e-12).view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_mean + self.output_scale * self.model(x)


class MLP(InputScaler):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 4,
        ranges: Optional[Sequence[Tuple[float, float]]] = None,
        normalize: bool = True,
    ):
        super().__init__(in_dim, ranges)
        self.normalize = normalize
        layers: List[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = self.normalize_input(x)
        return self.net(x)


class FourierFeaturePINN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 3,
        features: int = 4,
        sigma: float = 1.0,
        ranges: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        super().__init__()
        if ranges is None:
            lows = torch.zeros(in_dim)
            highs = torch.ones(in_dim)
        else:
            lows = torch.tensor([item[0] for item in ranges], dtype=torch.float32)
            highs = torch.tensor([item[1] for item in ranges], dtype=torch.float32)
        self.register_buffer("x_low", lows.view(1, in_dim))
        self.register_buffer("x_scale", (highs - lows).clamp_min(1e-12).view(1, in_dim))
        freqs = sigma * (2.0 ** torch.arange(features, dtype=torch.float32))
        self.register_buffer("freqs", freqs.view(1, 1, features))
        self.mlp = MLP(in_dim * (1 + 2 * features), out_dim, width=width, depth=depth, ranges=None, normalize=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = 2.0 * (x - self.x_low) / self.x_scale - 1.0
        z = math.pi * x_norm.unsqueeze(-1) * self.freqs
        encoded = torch.cat([x_norm, torch.sin(z).flatten(1), torch.cos(z).flatten(1)], dim=-1)
        return self.mlp(encoded)


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, grid_size))
        self.coeff = nn.Parameter(torch.randn(in_dim, out_dim, grid_size) * 0.03)
        self.base = nn.Linear(in_dim, out_dim)
        self.log_scale = nn.Parameter(torch.zeros(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(torch.tanh(x))
        z = x.unsqueeze(2).unsqueeze(3) - self.grid.view(1, 1, 1, -1)
        scale = torch.nn.functional.softplus(self.log_scale).view(1, self.in_dim, self.out_dim, 1) + 1e-4
        basis = torch.exp(-(scale * z) ** 2)
        spline = (basis * self.coeff.unsqueeze(0)).sum(dim=(1, 3))
        return base + spline


class KANPINN(InputScaler):
    def __init__(self, in_dim: int, out_dim: int, width: int = 32, depth: int = 3, grid_size: int = 16, ranges: Optional[Sequence[Tuple[float, float]]] = None):
        super().__init__(in_dim, ranges)
        dims: Iterable[int] = [in_dim] + [width] * depth + [out_dim]
        layers = []
        dims = list(dims)
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i + 1], grid_size=grid_size))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize_input(x))


class KAWPINNAD(InputScaler):
    """KAW-style Fourier/KAN architecture evaluated with automatic derivatives."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 8,
        modes: int = 24,
        phi_modes: int = 16,
        ranges: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        super().__init__(in_dim, ranges)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.modes = modes
        self.phi_modes = phi_modes
        self.a0 = nn.Parameter(torch.zeros(in_dim, rank))
        self.am = nn.Parameter(torch.randn(in_dim, rank, modes) * 1e-2)
        self.bm = nn.Parameter(torch.randn(in_dim, rank, modes) * 1e-2)
        self.mix = nn.Parameter(torch.randn(in_dim, rank) * 0.1)
        self.p0 = nn.Parameter(torch.zeros(out_dim, rank))
        self.pm = nn.Parameter(torch.randn(out_dim, rank, phi_modes) * 1e-2)
        self.qm = nn.Parameter(torch.randn(out_dim, rank, phi_modes) * 1e-2)
        self.register_buffer("freq", torch.arange(1, modes + 1).view(1, 1, 1, -1).float())
        self.register_buffer("phi_freq", torch.arange(1, phi_modes + 1).view(1, 1, 1, -1).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize_input(x)
        z = x.unsqueeze(2).unsqueeze(3) * self.freq
        psi = self.a0.unsqueeze(0) + (self.am.unsqueeze(0) * torch.sin(z)).sum(-1)
        psi = psi + (self.bm.unsqueeze(0) * torch.cos(z)).sum(-1)
        s = (psi * self.mix.unsqueeze(0)).sum(1) / math.sqrt(max(1, self.in_dim * self.modes))
        phi_z = s.unsqueeze(1).unsqueeze(-1) * self.phi_freq
        out = self.p0.unsqueeze(0) + (self.pm.unsqueeze(0) * torch.sin(phi_z)).sum(-1)
        out = out + (self.qm.unsqueeze(0) * torch.cos(phi_z)).sum(-1)
        return out.sum(-1) / math.sqrt(max(1, self.rank * self.phi_modes))


def _fit_model_to_budget(candidates: List[nn.Module], param_budget: int) -> nn.Module:
    if not candidates:
        raise ValueError("No model candidates were provided.")
    return min(candidates, key=lambda model: abs(count_parameters(model) - param_budget))


def make_model(
    name: str,
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    ranges: Optional[Sequence[Tuple[float, float]]] = None,
    fourier_sigma: float = 1.0,
    param_budget: Optional[int] = None,
) -> nn.Module:
    key = name.lower().replace("_", "-")
    if key == "kaw-pinn":
        return KAWPINNAD(in_dim, out_dim, rank=10, modes=32, phi_modes=24, ranges=ranges)
    if key == "mlp-pinn":
        if param_budget is not None:
            candidates = [MLP(in_dim, out_dim, width=w, depth=depth, ranges=ranges) for w in range(4, max(5, width + 1), 2)]
            return _fit_model_to_budget(candidates, param_budget)
        return MLP(in_dim, out_dim, width=width, depth=depth, ranges=ranges)
    if key == "fourier-feature-pinn":
        if param_budget is not None:
            candidates = []
            for features in range(1, 9):
                for w in range(4, max(5, width + 1), 2):
                    candidates.append(
                        FourierFeaturePINN(
                            in_dim,
                            out_dim,
                            width=w,
                            depth=max(2, depth - 1),
                            features=features,
                            sigma=fourier_sigma,
                            ranges=ranges,
                        )
                    )
            return _fit_model_to_budget(candidates, param_budget)
        return FourierFeaturePINN(
            in_dim,
            out_dim,
            width=width,
            depth=max(2, depth - 1),
            sigma=fourier_sigma,
            ranges=ranges,
        )
    if key == "kan-pinn":
        if param_budget is not None:
            candidates = []
            for grid_size in (4, 6, 8, 10, 12, 16):
                for w in range(4, max(5, width + 1), 2):
                    candidates.append(
                        KANPINN(in_dim, out_dim, width=w, depth=max(1, depth - 2), grid_size=grid_size, ranges=ranges)
                    )
            return _fit_model_to_budget(candidates, param_budget)
        return KANPINN(in_dim, out_dim, width=max(16, width // 2), depth=max(2, depth - 1), ranges=ranges)
    if key in {"kaw-pinn-ad", "kaw-pinn+ad"}:
        if param_budget is not None:
            candidates = []
            for rank in range(4, 34, 2):
                for modes in (24, 32, 48, 64, 80, 96, 120, 160, 200):
                    for phi_modes in (16, 24, 32, 48, 64, 72, 80, 96, 120, 160, 200):
                        candidates.append(
                            KAWPINNAD(
                                in_dim,
                                out_dim,
                                rank=rank,
                                modes=modes,
                                phi_modes=phi_modes,
                                ranges=ranges,
                            )
                        )
            return _fit_model_to_budget(candidates, param_budget)
        return KAWPINNAD(in_dim, out_dim, rank=10, modes=32, phi_modes=24, ranges=ranges)
    raise ValueError(f"Unknown baseline model: {name}")
