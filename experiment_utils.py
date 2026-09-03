import json
import os
import random

import numpy as np
import torch
import firedrake as fd


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


def compute_training_error(trainer, u0, n_steps, point_grid):
    """Corrected-model rollout error vs ground truth evaluated on *point_grid*.

    Rolls out the trained (corrected) model for ``n_steps`` and compares the
    predicted ``u`` field against the pure-solver ground truth sampled on the
    same grid. Returns the ``gt_error_metrics`` dict plus the number of steps.
    """
    pred, _, _, _, _ = trainer.predict_rollout(u0, 0.0, n_steps,
                                               spatial_sample=point_grid)
    gt = rollout_ground_truth_on_grid(trainer.physical_model, u0, len(pred),
                                      point_grid)
    metrics = gt_error_metrics(pred, gt)
    metrics["n_steps"] = int(len(pred))
    return metrics


def train_with_error_report(trainer, u0, n_steps, point_grid,
                            n_epochs, batch_size, save_every, exp_dir):
    """Train with a ground-truth training-error report at each checkpoint.

    Runs the same checkpoint loop as the individual train scripts but, in
    addition to the residual ``loss``, computes the corrected-model rollout
    error against ground truth after each checkpoint and records it.

    Returns
    -------
    (losses, train_errors) where losses is the per-epoch residual loss and
    train_errors is a list (one entry per checkpoint) of metric dicts from
    :func:`compute_training_error`.
    """
    losses = []
    train_errors = []
    pre_metrics = compute_training_error(trainer, u0, n_steps, point_grid)
    train_errors.append({"epoch": 0, **pre_metrics})
    print(f"  [train-error] epoch 0/{n_epochs}  "
          f"rel_rmse={pre_metrics['rel_rmse_mean']:.4f}  "
          f"linf={pre_metrics['linf_max']:.4f}")

    for start in range(0, n_epochs, save_every):
        n = min(save_every, n_epochs - start)
        chunk_losses = trainer.train(epochs=n, batch_size=batch_size)
        losses.extend(chunk_losses)
        save_checkpoint(trainer.st_model, trainer.optimizer, start + n, losses,
                        os.path.join(exp_dir, "checkpoint.pt"))

        metrics = compute_training_error(trainer, u0, n_steps, point_grid)
        train_errors.append({"epoch": start + n, **metrics})
        print(f"  [checkpoint] epoch {start+n}/{n_epochs}  "
              f"loss={chunk_losses[-1]:.6f}  "
              f"rel_rmse={metrics['rel_rmse_mean']:.4f}  "
              f"linf={metrics['linf_max']:.4f}")

    np.save(os.path.join(exp_dir, "train_errors.npy"), train_errors)
    with open(os.path.join(exp_dir, "train_errors.json"), "w") as f:
        json.dump(train_errors, f, indent=2)
    return losses, train_errors
