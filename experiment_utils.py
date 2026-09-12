import json
import os
import random

import numpy as np
import torch
import firedrake as fd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_exp_dir(base_dir: str, pde: str, params: dict) -> str:
    """Create EXPS/<pde>_<k=v>_..._<seed>/ and dump config.json inside."""
    name = "_".join(
        [pde] + [f"{k}={v}" for k, v in params.items()]
    )
    exp_dir = os.path.join(base_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=2, default=str)
    return exp_dir


def save_checkpoint(model, optimizer, epoch, losses, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "losses": losses,
        },
        path,
    )


def rollout_ground_truth_on_grid(stepper, u0, n_steps, point_grid):
    """Pure-solver reference evaluated on *point_grid* -> list of [H, W] arrays.

    Uses the stepper's FE mesh for the rollout and evaluates each
    time-step on a VertexOnlyMesh built from *point_grid*.
    """
    spatial_shape = point_grid.shape[:2]  # (H, W)
    ndim = point_grid.shape[-1]

    vom = fd.VertexOnlyMesh(
        stepper.V.mesh(),
        point_grid.reshape(-1, ndim),
        reorder=False,
    )
    P0DG = fd.FunctionSpace(vom, "DG", 0)

    u = fd.Function(stepper.V, name="gt_state").assign(u0)
    gt = []
    for _ in range(n_steps):
        u = stepper.step(u)
        vals = fd.assemble(fd.interpolate(u, P0DG)).dat.data_ro
        gt.append(vals.reshape(spatial_shape))
    return gt  # list of ndarray [H, W]


def gt_error_metrics(pred_states, gt_grids):
    """Per-step error metrics against ground truth.

    Parameters
    ----------
    pred_states : np.ndarray or torch.Tensor  [T, P, V]
        Prediction tensor where the last channel (``V-1``) is the scalar u.
    gt_grids : list[np.ndarray]  (length T)
        Reference solution on the same spatial grid, each of shape [H, W].

    Returns
    -------
    dict with per-step and summary scalars.
    """
    if hasattr(pred_states, "detach"):
        pred_states = pred_states.detach().cpu().numpy()
    # u-channel is last column
    u_pred = pred_states[..., -1]  # [T, P]
    gt_flat = np.stack([g.reshape(-1) for g in gt_grids])  # [T, P]

    err = u_pred - gt_flat
    rmse = np.sqrt((err ** 2).mean(axis=-1))
    denom = np.sqrt((gt_flat ** 2).mean(axis=-1))
    denom = np.where(denom == 0.0, 1.0, denom)
    rel_rmse = rmse / denom
    linf = np.abs(err).max(axis=-1)

    return {
        "rmse_per_step": rmse.tolist(),
        "rel_rmse_per_step": rel_rmse.tolist(),
        "linf_per_step": linf.tolist(),
        "rel_rmse_mean": float(rel_rmse.mean()),
        "rel_rmse_last": float(rel_rmse[-1]),
        "linf_max": float(linf.max()),
    }


def grids_from_prediction_list(pred_states, grid_shape):
    """Convert rollout prediction tensors to a list of [H, W] grids."""
    if hasattr(pred_states, "detach"):
        pred_states = pred_states.detach().cpu().numpy()
    H, W = grid_shape
    u_ch = pred_states[..., -1]
    return [np.asarray(u_ch[i]).reshape(H, W) for i in range(len(u_ch))]


def plot_learning_process(losses, train_errors, output_path):
    """Two-panel learning-process figure: loss/epoch + rollout error vs GT."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    if losses:
        ax1.plot(np.arange(1, len(losses) + 1), losses, "-", color="#1f77b4", lw=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training loss")
        ax1.set_title("Learning process: training loss")
    ax1.grid(True, alpha=0.3)

    if train_errors:
        eps = [e["epoch"] for e in train_errors]
        n_steps = int(train_errors[0].get("n_steps", 1))
        for k in range(n_steps):
            vals = [e["rel_rmse_per_step"][k] for e in train_errors]
            ax2.plot(eps, vals, "o-", lw=1.2, ms=4, alpha=0.75, label=f"rollout step {k + 1}")
        ax2.plot(eps, [e["rel_rmse_mean"] for e in train_errors],
                 "s--", color="k", ms=5, label="mean rel. RMSE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("rel. RMSE vs GT")
        ax2.set_title("Learning process: corrected-model rollout error")
        ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_solution_snapshots(pred_grids, gt_grids, times, output_path, n_snap=2):
    """Prediction vs ground-truth field snapshots with |error| maps."""
    T = len(pred_grids)
    if T == 0:
        return
    idx = sorted({0, T - 1})
    for i in range(1, n_snap - 1):
        idx.add(int(round((T - 1) * i / max(1, n_snap - 1))))
    idx = sorted(idx)[:n_snap]
    n = len(idx)

    gt_vals = np.concatenate([np.asarray(gt_grids[i]).ravel() for i in idx])
    pr_vals = np.concatenate([np.asarray(pred_grids[i]).ravel() for i in idx])
    vmin, vmax = float(gt_vals.min()), float(gt_vals.max())

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    for r, i in enumerate(idx):
        gt = np.asarray(gt_grids[i])
        pr = np.asarray(pred_grids[i])
        er = np.abs(pr - gt)
        t = times[i] if times is not None and i < len(times) else i

        im0 = axes[r, 0].imshow(gt, origin="lower", extent=(0, 1, 0, 1),
                                vmin=vmin, vmax=vmax, cmap="viridis")
        axes[r, 0].set_title(f"Ground truth  t={float(t):.4f}")
        plt.colorbar(im0, ax=axes[r, 0], fraction=0.046)

        axes[r, 1].imshow(pr, origin="lower", extent=(0, 1, 0, 1),
                          vmin=vmin, vmax=vmax, cmap="viridis")
        axes[r, 1].set_title("Predicted (corrected)")

        im2 = axes[r, 2].imshow(er, origin="lower", extent=(0, 1, 0, 1), cmap="Reds")
        axes[r, 2].set_title(f"|err|  max={er.max():.3e}")
        plt.colorbar(im2, ax=axes[r, 2], fraction=0.046)

        for ax in axes[r]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

    fig.suptitle("Solution quality: predicted vs ground truth", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _training_error_with_grids(trainer, u0, n_steps, point_grid):
    """Corrected-model rollout error + extracted grids/times on *point_grid*."""
    pred, _, _, times, _ = trainer.predict_rollout(u0, 0.0, n_steps,
                                                   spatial_sample=point_grid)
    gt = rollout_ground_truth_on_grid(trainer.physical_model, u0, len(pred),
                                      point_grid)
    metrics = gt_error_metrics(pred, gt)
    metrics["n_steps"] = int(len(pred))
    pred_grids = grids_from_prediction_list(pred, point_grid.shape[:2])
    return metrics, pred_grids, gt, times


def compute_training_error(trainer, u0, n_steps, point_grid):
    """Corrected-model rollout error vs ground truth evaluated on *point_grid*.

    Rolls out the trained (corrected) model for ``n_steps`` and compares the
    predicted ``u`` field against the pure-solver ground truth sampled on the
    same grid. Returns the ``gt_error_metrics`` dict plus the number of steps.
    """
    metrics, _, _, _ = _training_error_with_grids(trainer, u0, n_steps, point_grid)
    return metrics


def train_with_error_report(trainer, u0, n_steps, point_grid,
                            n_epochs, batch_size, save_every, exp_dir,
                            checkpoint_callback=None):
    """Train with a ground-truth training-error report at each checkpoint.

    Runs the same checkpoint loop as the individual train scripts but, in
    addition to the residual ``loss``, computes the corrected-model rollout
    error against ground truth after each checkpoint, records it, and emits
    per-checkpoint images for the learning process and solution quality.

    ``checkpoint_callback(trainer, epoch, losses, train_errors)`` is invoked
    after each checkpoint's images so callers can regenerate posterior plots
    with the just-trained model state.

    Returns
    -------
    (losses, train_errors) where losses is the per-epoch residual loss and
    train_errors is a list (one entry per checkpoint) of metric dicts from
    :func:`compute_training_error`.
    """
    losses = []
    train_errors = []
    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    def _emit_images(epoch):
        """Regenerate images for the latest checkpoint state.

        The learning-process image is replaced on every checkpoint, while the
        solution-quality snapshot is written to an epoch-specific file that is
        never overwritten.
        """
        lpath = os.path.join(plot_dir, "learning.png")
        spath = os.path.join(plot_dir, f"solution_epoch_{epoch:03d}.png")
        plot_learning_process(losses, train_errors, lpath)
        plot_solution_snapshots(_cur_pred, _cur_gt, _cur_times, spath)
        print(f"  [images] {lpath} | {spath}")

    _cur_pred, _cur_gt, _cur_times = [], [], []
    metrics, _cur_pred, _cur_gt, _cur_times = _training_error_with_grids(
        trainer, u0, n_steps, point_grid)
    train_errors.append({"epoch": 0, **metrics})
    print(f"  [train-error] epoch 0/{n_epochs}  "
          f"rel_rmse={metrics['rel_rmse_mean']:.4f}  "
          f"linf={metrics['linf_max']:.4f}")
    _emit_images(0)

    for start in range(0, n_epochs, save_every):
        n = min(save_every, n_epochs - start)
        chunk_losses = trainer.train(epochs=n, batch_size=batch_size)
        losses.extend(chunk_losses)
        save_checkpoint(trainer.st_model, trainer.optimizer, start + n, losses,
                        os.path.join(exp_dir, "checkpoint.pt"))

        metrics, _cur_pred, _cur_gt, _cur_times = _training_error_with_grids(
            trainer, u0, n_steps, point_grid)
        train_errors.append({"epoch": start + n, **metrics})
        print(f"  [checkpoint] epoch {start+n}/{n_epochs}  "
              f"loss={chunk_losses[-1]:.6f}  "
              f"rel_rmse={metrics['rel_rmse_mean']:.4f}  "
              f"linf={metrics['linf_max']:.4f}")
        _emit_images(start + n)
        if checkpoint_callback is not None:
            checkpoint_callback(trainer, start + n, losses, train_errors)

    np.save(os.path.join(exp_dir, "train_errors.npy"), train_errors)
    with open(os.path.join(exp_dir, "train_errors.json"), "w") as f:
        json.dump(train_errors, f, indent=2)
    plot_learning_process(losses, train_errors,
                          os.path.join(plot_dir, "learning.png"))
    return losses, train_errors
