import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except ImportError:  # The benchmark can still write metrics in minimal Python environments.
    plt = None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KAW_PINN.benchmarks.models import make_model  # noqa: E402


BASELINE_MODELS = ["KAW-PINN", "MLP-PINN", "Fourier-feature-PINN", "KAN-PINN", "KAW-PINN-AD"]


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    """Manufactured high-frequency solution on [0, 1]."""
    return (
        torch.sin(math.pi * x)
        + 0.25 * torch.sin(8.0 * math.pi * x)
        + 0.10 * torch.sin(20.0 * math.pi * x)
    )


def forcing(x: torch.Tensor) -> torch.Tensor:
    """Right-hand side for -u_xx = f."""
    return (
        (math.pi**2) * torch.sin(math.pi * x)
        + 0.25 * (8.0 * math.pi) ** 2 * torch.sin(8.0 * math.pi * x)
        + 0.10 * (20.0 * math.pi) ** 2 * torch.sin(20.0 * math.pi * x)
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class KAWPINN1D(nn.Module):
    """
    One-dimensional KAW-PINN specialization with explicit trigonometric channels.

    The sine envelope enforces homogeneous Dirichlet boundaries, and the second
    derivative is evaluated by a closed-form formula instead of autograd.
    """

    def __init__(self, modes: int = 24):
        super().__init__()
        self.modes = modes
        self.coeff = nn.Parameter(torch.zeros(modes, 1))
        self.register_buffer("freq", torch.arange(1, modes + 1, dtype=torch.float32).view(1, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.sin(math.pi * x * self.freq)
        return basis @ self.coeff

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        basis_xx = -((math.pi * self.freq) ** 2) * torch.sin(math.pi * x * self.freq)
        return basis_xx @ self.coeff

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        return -self.second_derivative(x) - forcing(x)


class MLPPINN(nn.Module):
    """Small MLP baseline that uses automatic differentiation for u_xx."""

    def __init__(self, width: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - x) * self.net(x)

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone().requires_grad_(True)
        u = self.forward(x)
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
        return -uxx - forcing(x)


def pde_residual(model: nn.Module, x: torch.Tensor, analytic_if_available: bool) -> torch.Tensor:
    if analytic_if_available and hasattr(model, "compute_pde_residual"):
        return model.compute_pde_residual(x)
    x = x.detach().clone().requires_grad_(True)
    u = model(x)
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return -uxx - forcing(x)


def relative_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - truth) / torch.linalg.norm(truth).clamp_min(1e-12)).item()


def train(
    model: nn.Module,
    n_domain: int,
    adam_epochs: int,
    lbfgs_steps: int,
    lr: float,
    analytic_residual: bool,
) -> tuple[dict, float]:
    x = torch.linspace(0.0, 1.0, n_domain, device=device).view(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"total": [], "pde": [], "bc": []}
    start = time.perf_counter()

    for _ in range(adam_epochs):
        optimizer.zero_grad(set_to_none=True)
        residual = pde_residual(model, x, analytic_residual)
        pde_loss = torch.mean(residual.pow(2))
        bc = model(torch.tensor([[0.0], [1.0]], device=device))
        bc_loss = torch.mean(bc.pow(2))
        loss = pde_loss + 100.0 * bc_loss
        loss.backward()
        optimizer.step()
        history["total"].append(float(loss.detach().cpu()))
        history["pde"].append(float(pde_loss.detach().cpu()))
        history["bc"].append(float(bc_loss.detach().cpu()))

    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            residual = pde_residual(model, x, analytic_residual)
            pde_loss = torch.mean(residual.pow(2))
            bc = model(torch.tensor([[0.0], [1.0]], device=device))
            bc_loss = torch.mean(bc.pow(2))
            loss = pde_loss + 100.0 * bc_loss
            loss.backward()
            history["total"].append(float(loss.detach().cpu()))
            history["pde"].append(float(pde_loss.detach().cpu()))
            history["bc"].append(float(bc_loss.detach().cpu()))
            return loss

        lbfgs.step(closure)

    return history, time.perf_counter() - start


def evaluate(model: nn.Module, analytic_residual: bool) -> dict:
    x = torch.linspace(0.0, 1.0, 1000, device=device).view(-1, 1)
    with torch.no_grad():
        pred = model(x)
        truth = exact_solution(x)
    residual = pde_residual(model, x, analytic_residual)
    return {
        "relative_l2_error": relative_l2(pred, truth),
        "max_error": torch.max(torch.abs(pred - truth)).item(),
        "pde_residual_mse": torch.mean(residual.detach().pow(2)).item(),
        "parameters": count_parameters(model),
    }


def plot_result(models: dict[str, nn.Module], histories: dict[str, dict], out_dir: Path) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(0.0, 1.0, 1000, device=device).view(-1, 1)
    truth = exact_solution(x).detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.8), sharex=False)
    axes[0].plot(x_np, truth, "k-", linewidth=2.2, label="Exact")
    for name, model in models.items():
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy()
        axes[0].plot(x_np, pred, "--", linewidth=1.6, label=name)
        axes[1].semilogy(x_np, np.abs(pred - truth) + 1e-12, linewidth=1.4, label=name)
    axes[0].set_title("Solution reconstruction")
    axes[0].set_ylabel("$u(x)$")
    axes[1].set_title("Pointwise absolute error")
    axes[1].set_ylabel("Absolute error")
    axes[2].set_title("Training loss")
    for name, hist in histories.items():
        axes[2].semilogy(hist["total"], linewidth=1.2, label=name)
    axes[2].set_xlabel("Epoch / L-BFGS step")
    axes[2].set_ylabel("Loss")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "high_frequency_poisson_1d_results.png", dpi=300)
    fig.savefig(out_dir / "high_frequency_poisson_1d_results.pdf")
    plt.close(fig)


def make_benchmark_model(name: str, param_budget: int) -> tuple[nn.Module, bool]:
    if name == "KAW-PINN":
        return KAWPINN1D(modes=24).to(device), True
    model = make_model(
        name,
        in_dim=1,
        out_dim=1,
        width=64,
        depth=4,
        ranges=[(0.0, 1.0)],
        fourier_sigma=1.0,
        param_budget=param_budget,
    ).to(device)
    return model, False


def main() -> None:
    parser = argparse.ArgumentParser(description="1D high-frequency Poisson benchmark for KAW-PINN.")
    parser.add_argument("--out-dir", default="one_dimensional/results_highfreq_poisson")
    parser.add_argument("--adam-epochs", type=int, default=1200)
    parser.add_argument("--lbfgs-steps", type=int, default=80)
    parser.add_argument("--n-domain", type=int, default=512)
    parser.add_argument(
        "--models",
        nargs="+",
        default=BASELINE_MODELS,
        choices=BASELINE_MODELS,
        help="Models to train and plot in one combined figure.",
    )
    parser.add_argument("--run-mlp", action="store_true", help="Backward-compatible shortcut: run only KAW-PINN and MLP-PINN.")
    args = parser.parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)
    out_dir = Path(args.out_dir)
    selected_models = ["KAW-PINN", "MLP-PINN"] if args.run_mlp else args.models
    param_budget = count_parameters(KAWPINN1D(modes=24))
    models: dict[str, nn.Module] = {}
    residual_modes: dict[str, bool] = {}
    for name in selected_models:
        model, analytic_residual = make_benchmark_model(name, param_budget)
        models[name] = model
        residual_modes[name] = analytic_residual
    if args.run_mlp:
        models["MLP-PINN"] = MLPPINN(width=64, depth=3).to(device)
        residual_modes["MLP-PINN"] = False

    metrics = []
    histories = {}
    for name, model in models.items():
        lr = 1e-2 if name == "KAW-PINN" else 1e-3
        history, elapsed = train(model, args.n_domain, args.adam_epochs, args.lbfgs_steps, lr=lr, analytic_residual=residual_modes[name])
        item = {"model": name, "AD": not residual_modes[name], "total_time_sec": elapsed, **evaluate(model, residual_modes[name])}
        metrics.append(item)
        histories[name] = history
        print(json.dumps(item, indent=2, ensure_ascii=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    plot_result(models, histories, out_dir)


model = KAWPINN1D(modes=24).to(device)


if __name__ == "__main__":
    main()
