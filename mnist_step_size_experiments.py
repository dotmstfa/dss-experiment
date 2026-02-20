import argparse
import copy
import importlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def binarize_tensor(t: torch.Tensor) -> torch.Tensor:
    return (t > 0.5).float()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def choose_scaled_hidden_dims(
    base_hidden_dims: List[int],
    target_ratio: float,
    input_dim: int,
    output_dim: int,
) -> List[int]:
    base_model = BaseMLP(input_dim, base_hidden_dims, output_dim)
    base_count = count_params(base_model)
    target = int(base_count * target_ratio)

    best_dims = base_hidden_dims
    best_diff = abs(base_count - target)

    for scale in torch.linspace(0.2, 1.0, 161):
        dims = [max(8, int(round(h * float(scale)))) for h in base_hidden_dims]
        candidate = BaseMLP(input_dim, dims, output_dim)
        c = count_params(candidate)
        diff = abs(c - target)
        if diff < best_diff:
            best_diff = diff
            best_dims = dims
    return best_dims


@dataclass
class TrainResult:
    name: str
    losses: List[float]
    epoch_times: List[float]
    cumulative_times: List[float]
    convergence_time: float
    convergence_epoch: int
    converged: bool
    final_loss: float
    nn_params: int
    total_params: int


def convergence_from_curve(
    losses: List[float], cumulative_times: List[float], target_loss: float
) -> Tuple[float, int, bool]:
    for i, v in enumerate(losses):
        if v <= target_loss:
            return cumulative_times[i], i + 1, True
    return cumulative_times[-1], len(losses), False


def init_step_tensors(model: nn.Module, init_value: float, device: torch.device) -> List[torch.Tensor]:
    step_tensors: List[torch.Tensor] = []
    raw = math.log(math.exp(init_value) - 1.0)
    for p in model.parameters():
        t = torch.full_like(p, fill_value=raw, device=device)
        step_tensors.append(t)
    return step_tensors


def init_decay_tensors(model: nn.Module, init_value: float, device: torch.device) -> List[torch.Tensor]:
    decay_tensors: List[torch.Tensor] = []
    raw = math.log(init_value / (1.0 - init_value))
    for p in model.parameters():
        t = torch.full_like(p, fill_value=raw, device=device)
        decay_tensors.append(t)
    return decay_tensors

def train_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    mode: str,
    base_lr: float,
    step_lr: float,
    decay_lr: float,
    step_init: float,
    decay_init: float,
    min_step: float = 1e-6,
) -> Tuple[List[float], List[float], int]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())

    step_tensors: List[torch.Tensor] = []
    decay_tensors: List[torch.Tensor] = []
    optimizer: torch.optim.Optimizer | None = None

    if mode == "scalar_lr":
        optimizer = torch.optim.Adam(params, lr=base_lr)

    if mode in {"hessian_step", "grad_step", "grad_step_hessian_decay"}:
        step_tensors = init_step_tensors(model, step_init, device)
    if mode == "grad_step_hessian_decay":
        decay_tensors = init_decay_tensors(model, decay_init, device)

    losses: List[float] = []
    epoch_times: List[float] = []

    for _ in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mode in {"scalar_lr"}:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                batch_size = x.shape[0]
                epoch_loss += float(loss.detach().item()) * batch_size
                seen += batch_size
                continue

            logits = model(x)
            loss = criterion(logits, y)

            need_hessian = mode in {"hessian_step", "grad_step_hessian_decay"}
            grads = torch.autograd.grad(loss, params, create_graph=need_hessian)

            hess_proxy = None
            if need_hessian:
                grad_sum = None
                for g in grads:
                    s = g.sum()
                    grad_sum = s if grad_sum is None else grad_sum + s
                hess_proxy = torch.autograd.grad(grad_sum, params)

            with torch.no_grad():
                if mode == "hessian_step":
                    assert hess_proxy is not None
                    for p, g, s, h in zip(params, grads, step_tensors, hess_proxy):
                        alpha = F.softplus(s) + min_step
                        p -= alpha * g
                        s -= step_lr * torch.tanh(h)
                        s.clamp_(-10.0, 6.0)

                elif mode == "grad_step":
                    for p, g, s in zip(params, grads, step_tensors):
                        alpha = F.softplus(s) + min_step
                        p -= alpha * g
                        s -= step_lr * torch.tanh(g)
                        s.clamp_(-10.0, 6.0)

                elif mode == "grad_step_hessian_decay":
                    assert hess_proxy is not None
                    for p, g, s, d, h in zip(params, grads, step_tensors, decay_tensors, hess_proxy):
                        alpha = F.softplus(s) + min_step
                        decay = torch.sigmoid(d)
                        p -= alpha * g
                        s.mul_(1.0 - decay)
                        s.add_(-step_lr * torch.tanh(g))
                        s.clamp_(-10.0, 6.0)
                        d -= decay_lr * torch.tanh(h)
                        d.clamp_(-8.0, 8.0)

                else:
                    raise ValueError(f"Unknown mode: {mode}")

            batch_size = x.shape[0]
            epoch_loss += float(loss.detach().item()) * batch_size
            seen += batch_size

        losses.append(epoch_loss / max(1, seen))
        epoch_times.append(time.perf_counter() - epoch_start)

    total_params = count_params(model)
    if mode in {"hessian_step", "grad_step", "grad_step_hessian_decay"}:
        total_params += sum(t.numel() for t in step_tensors)
    if mode == "grad_step_hessian_decay":
        total_params += sum(t.numel() for t in decay_tensors)

    return losses, epoch_times, total_params


def make_train_loader(data_dir: Path, batch_size: int, subset_size: int, seed: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(binarize_tensor),
        ]
    )
    train_dataset = datasets.MNIST(root=str(data_dir), train=True, transform=transform, download=True)

    g = torch.Generator().manual_seed(seed)
    subset_size = min(subset_size, len(train_dataset))
    indices = torch.randperm(len(train_dataset), generator=g)[:subset_size].tolist()
    subset = Subset(train_dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def plot_single_loss(
    output_dir: Path,
    exp_name: str,
    losses: List[float],
    convergence_epoch: int,
    converged: bool,
) -> None:
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses, marker="o", linewidth=2)
    if converged:
        plt.axvline(convergence_epoch, linestyle="--", linewidth=1, label=f"Converged at epoch {convergence_epoch}")
    else:
        plt.axvline(convergence_epoch, linestyle="--", linewidth=1, label="Did not converge")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(f"{exp_name} Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{exp_name.lower().replace(' ', '_')}_loss.png", dpi=200)
    plt.close()


def plot_combined_losses(output_dir: Path, results: List[TrainResult]) -> None:
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.plot(r.cumulative_times, r.losses, marker="o", linewidth=1.8, label=r.name)
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Training loss")
    plt.title("All Experiments: Loss vs Time")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "all_experiments_loss_over_time.png", dpi=220)
    plt.close()


def plot_convergence_times(output_dir: Path, results: List[TrainResult]) -> None:
    plt.figure(figsize=(10, 5))
    names = [r.name for r in results]
    times = [r.convergence_time for r in results]
    plt.bar(names, times)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Time to convergence (s)")
    plt.title("Training Time to Convergence")
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_times.png", dpi=220)
    plt.close()


def plot_lr_sweep(output_dir: Path, sweep_losses: Dict[float, List[float]]) -> None:
    plt.figure(figsize=(8, 5))
    for lr, losses in sweep_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.8, label=f"lr={lr:g}")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Experiment 1 LR Sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "exp1_lr_sweep.png", dpi=220)
    plt.close()


def cumulative(values: List[float]) -> List[float]:
    out: List[float] = []
    s = 0.0
    for v in values:
        s += v
        out.append(s)
    return out


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Re-run on a CUDA machine or pass --allow-cpu.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_train_loader(Path(args.data_dir), args.batch_size, args.subset_size, args.seed)

    input_dim, output_dim = 28 * 28, 10
    base_hidden = [192, 128]
    half_hidden = choose_scaled_hidden_dims(base_hidden, 0.5, input_dim, output_dim)
    third_hidden = choose_scaled_hidden_dims(base_hidden, 1.0 / 3.0, input_dim, output_dim)

    set_seed(args.seed)
    base_template = BaseMLP(input_dim, base_hidden, output_dim)
    base_state = copy.deepcopy(base_template.state_dict())

    set_seed(args.seed)
    half_template = BaseMLP(input_dim, half_hidden, output_dim)
    half_state = copy.deepcopy(half_template.state_dict())

    set_seed(args.seed)
    third_template = BaseMLP(input_dim, third_hidden, output_dim)
    third_state = copy.deepcopy(third_template.state_dict())

    print(f"Base hidden dims: {base_hidden}, NN params={count_params(base_template)}")
    print(f"Half-budget hidden dims: {half_hidden}, NN params={count_params(half_template)}")
    print(f"Third-budget hidden dims: {third_hidden}, NN params={count_params(third_template)}")

    # Experiment 1: regular network with Adam LR sweep
    sweep_lrs = [float(x) for x in args.lr_sweep.split(",")]
    sweep_losses: Dict[float, List[float]] = {}
    sweep_epoch_times: Dict[float, List[float]] = {}

    for lr in sweep_lrs:
        model = BaseMLP(input_dim, base_hidden, output_dim)
        model.load_state_dict(base_state)
        losses, epoch_times, _ = train_experiment(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            mode="scalar_lr",
            base_lr=lr,
            step_lr=args.step_lr,
            decay_lr=args.decay_lr,
            step_init=args.step_init,
            decay_init=args.decay_init,
        )
        sweep_losses[lr] = losses
        sweep_epoch_times[lr] = epoch_times

    plot_lr_sweep(output_dir, sweep_losses)

    best_lr = min(sweep_lrs, key=lambda lr: sweep_losses[lr][-1])
    e1_losses = sweep_losses[best_lr]
    e1_epoch_times = sweep_epoch_times[best_lr]

    results: List[TrainResult] = []

    e1_cum = cumulative(e1_epoch_times)
    e1_conv_time, e1_conv_epoch, e1_converged = convergence_from_curve(e1_losses, e1_cum, args.convergence_loss)
    results.append(
        TrainResult(
            name=f"Exp1 Adam LR sweep best (lr={best_lr:g})",
            losses=e1_losses,
            epoch_times=e1_epoch_times,
            cumulative_times=e1_cum,
            convergence_time=e1_conv_time,
            convergence_epoch=e1_conv_epoch,
            converged=e1_converged,
            final_loss=e1_losses[-1],
            nn_params=count_params(base_template),
            total_params=count_params(base_template),
        )
    )

    exp_configs = [
        ("Exp2 Gradient step-size", "grad_step", base_hidden, base_state),
        ("Exp3 Grad step-size + Hessian decay", "grad_step_hessian_decay", base_hidden, base_state),
        ("Exp4 Hessian step-size (x budget)", "hessian_step", half_hidden, half_state),
        ("Exp5 Gradient step-size (x budget)", "grad_step", half_hidden, half_state),
        ("Exp6 Grad step-size + Hessian decay (x budget)", "grad_step_hessian_decay", third_hidden, third_state),
    ]

    for name, mode, hidden_dims, init_state in exp_configs:
        model = BaseMLP(input_dim, hidden_dims, output_dim)
        model.load_state_dict(init_state)
        try:
            losses, epoch_times, total_params = train_experiment(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=args.epochs,
                mode=mode,
                base_lr=args.base_lr,
                step_lr=args.step_lr,
                decay_lr=args.decay_lr,
                step_init=args.step_init,
                decay_init=args.decay_init,
            )
        except RuntimeError as exc:
            print(f"Skipping {name}: {exc}")
            continue

        cum = cumulative(epoch_times)
        conv_time, conv_epoch, converged = convergence_from_curve(losses, cum, args.convergence_loss)
        results.append(
            TrainResult(
                name=name,
                losses=losses,
                epoch_times=epoch_times,
                cumulative_times=cum,
                convergence_time=conv_time,
                convergence_epoch=conv_epoch,
                converged=converged,
                final_loss=losses[-1],
                nn_params=count_params(model),
                total_params=total_params,
            )
        )

    for r in results:
        plot_single_loss(output_dir, r.name, r.losses, r.convergence_epoch, r.converged)

    plot_combined_losses(output_dir, results)
    plot_convergence_times(output_dir, results)

    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "subset_size": args.subset_size,
        "base_hidden": base_hidden,
        "half_hidden": half_hidden,
        "third_hidden": third_hidden,
        "results": [
            {
                "name": r.name,
                "final_loss": r.final_loss,
                "convergence_time_sec": r.convergence_time,
                "convergence_epoch": r.convergence_epoch,
                "converged": r.converged,
                "nn_params": r.nn_params,
                "total_params": r.total_params,
            }
            for r in results
        ],
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Experiment summary ===")
    for r in results:
        print(
            f"{r.name}: final_loss={r.final_loss:.4f}, conv_time={r.convergence_time:.2f}s, "
            f"conv_epoch={r.convergence_epoch}, converged={r.converged}, "
            f"nn_params={r.nn_params}, total_params={r.total_params}"
        )
    print(f"\nSaved plots and summary to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="MNIST dedicated step-size experiments")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subset-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-lr", type=float, default=0.01)
    parser.add_argument("--lr-sweep", type=str, default="0.001,0.003,0.01,0.03")
    parser.add_argument("--step-lr", type=float, default=0.01)
    parser.add_argument("--decay-lr", type=float, default=0.005)
    parser.add_argument("--step-init", type=float, default=0.01)
    parser.add_argument("--decay-init", type=float, default=0.05)
    parser.add_argument("--convergence-loss", type=float, default=0.6)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
