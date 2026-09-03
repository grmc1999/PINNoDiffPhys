import argparse
import json
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import firedrake as fd

from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
from trainer.Trainer import IterativePoissonSolverStepper, FiredrakePINNSBasedSOLTrainerCNN
from DL_models.PINNS.Residual_losses import poisson_residual_loss
from experiment_utils import (set_seed, make_exp_dir, save_checkpoint,
                               rollout_ground_truth_on_grid, gt_error_metrics,
                               train_with_error_report)


def make_point_grid(n: int, P_min: List[float] = [0.0, 0.0],
                    P_max: List[float] = [1.0, 1.0]):
    grid = np.stack(
        np.meshgrid(
            *tuple(np.linspace(p_min, p_max, n)
                   for p_min, p_max in zip(P_min, P_max)),
            indexing="xy",
        ),
        axis=-1,
    )
    return grid


def make_ic(V):
    X = fd.SpatialCoordinate(V.mesh())
    u0 = fd.Function(V).interpolate(
        0.5 * fd.exp(0.5 * ((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 - 0.1) ** 2 - 1.0)
    )
    return u0


def tensor_state_to_grid(state_tensor, grid_shape):
    return state_tensor.reshape(grid_shape)


def compute_residual_curve(trainer, pred_states, input_states):
    val = trainer.loss(pred_states, input_states)

    if torch.is_tensor(val):
        val_h = torch.mean(val, axis=-1).detach().cpu().numpy()
    else:
        val_h = torch.mean(val, axis=-1).numpy()

    return {
        "residual": val.detach().cpu().numpy(),
        "residual_decay": val_h,
        "residual_mean": float(np.mean(val_h)),
        "residual_last": float(val_h[-1]),
        "residual_max": float(np.max(val_h)),
    }


def build_trainer(mesh, point_grid, simulation_steps, st_model,
                  m_iters=5, relaxation=1.0, forcing=0.0, lr=1e-4):
    ph_model = IterativePoissonSolverStepper(
        mesh=mesh,
        m_iters=m_iters,
        relaxation=relaxation,
        diffusivity=1.0,
        forcing=forcing,
        bc_value=0.0,
        degree=1,
        point_evaluator=point_grid,
    )

    trainer = FiredrakePINNSBasedSOLTrainerCNN(
        physical_model=ph_model,
        statistical_model=st_model,
        optimizer=torch.optim.Adam(st_model.parameters(), lr=lr),
        simulation_steps=simulation_steps,
        dt=1.0,
        loss=lambda u, x: (poisson_residual_loss(u, x, K=1.0)) ** 2,
    )
    return trainer


def grids_from_prediction_list(pred_states, point_grid):
    H, W = point_grid
    return [tensor_state_to_grid(s, (H, W)) for s in pred_states]


# ============================================================
# Plotting
# ============================================================

def plot_residual_curves(time_dict, output_path):
    plt.figure(figsize=(8, 5))
    for name, report in time_dict.items():
        plt.plot(report["iterations"], report["residual"], linewidth=2, label=name)
    plt.xlabel("Iteration step")
    plt.ylabel("PDE residual")
    plt.title("Posterior testing: Poisson residual vs iteration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_curve(losses, output_path):
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_error_curves(time_dict, output_path):
    plt.figure(figsize=(8, 5))
    for name, report in time_dict.items():
        plt.plot(report["iterations"], report["rel_rmse"], linewidth=2, label=name)
    plt.xlabel("Iteration step")
    plt.ylabel("Relative RMSE")
    plt.title("Posterior testing: error vs iteration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_residual(report, output_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(report["iterations"], report["residual_decay"])
    axes[0].set_xlabel("Iteration step")
    axes[0].set_ylabel("Residual loss")
    axes[0].set_title("Poisson residual\n")
    axes[0].grid(True, alpha=0.3)

    res_arr = np.asarray(report["residual"])
    grid_shape = report["grid_shape"]

    im1 = axes[1].imshow(res_arr[0].reshape(grid_shape[:2]),
                          origin="lower", extent=(0, 1, 0, 1))
    axes[1].set_title("Residual spatial map, iter 0")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(res_arr[-1].reshape(grid_shape[:2]),
                          origin="lower", extent=(0, 1, 0, 1))
    axes[2].set_title("Residual spatial map, last iter")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_report_json(report, output_path):
    serializable = {}
    for k, v in report.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


# ============================================================
# Experiments
# ============================================================

def run_spatial_interpolation_experiment(mesh, trained_model, u0, args):
    """Same solver config, but denser spatial point sampling."""
    fine_grid = make_point_grid(args.spatial_test_n)
    grid = make_point_grid(args.train_grid_n)
    n_steps = args.num_rollout

    test_trainer = build_trainer(
        mesh=mesh, point_grid=grid,
        simulation_steps=n_steps, st_model=trained_model, lr=0.0,
        m_iters=args.m_iters,
    )

    pred_states, input_states, corr_states, pred_times, uncorrected_sol = \
        test_trainer.predict_rollout(u0, t0=0.0, n_steps=n_steps,
                                     spatial_sample=fine_grid)

    pred_grids = grids_from_prediction_list(pred_states[:, :, [-1]],
                                            fine_grid.shape[:2])
    report = compute_residual_curve(test_trainer, pred_states[:, :, -1:],
                                    uncorrected_sol)
    report["iterations"] = np.arange(1, len(pred_states) + 1)
    report["pred_grids"] = pred_grids
    report["grid_shape"] = fine_grid.shape

    gt_grids = rollout_ground_truth_on_grid(test_trainer.physical_model,
                                             u0, n_steps, fine_grid)
    report["gt_error"] = gt_error_metrics(pred_states, gt_grids)
    return report


def run_budget_shift_experiment(mesh, trained_model, u0, args):
    """Fewer Richardson iterations (coarser solver) at test time."""
    grid = make_point_grid(args.train_grid_n)
    n_steps = args.num_rollout
    m_test = max(1, args.m_iters // args.budget_refinement)

    test_trainer = build_trainer(
        mesh=mesh, point_grid=grid,
        simulation_steps=n_steps, st_model=trained_model, lr=0.0,
        m_iters=m_test,
    )

    pred_states, input_states, corr_states, pred_times, uncorrected_sol = \
        test_trainer.predict_rollout(u0, t0=0.0, n_steps=n_steps,
                                     spatial_sample=grid)

    pred_grids = grids_from_prediction_list(pred_states[:, :, [-1]],
                                            grid.shape[:2])
    report = compute_residual_curve(test_trainer, pred_states[:, :, -1:],
                                    uncorrected_sol)
    report["iterations"] = np.arange(1, len(pred_states) + 1)
    report["m_train"] = args.m_iters
    report["m_test"] = m_test
    report["pred_grids"] = pred_grids
    report["grid_shape"] = grid.shape

    gt_grids = rollout_ground_truth_on_grid(test_trainer.physical_model,
                                             u0, n_steps, grid)
    report["gt_error"] = gt_error_metrics(pred_states, gt_grids)
    return report


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_definition", type=str,
                        default="fd.UnitSquareMesh(10,10)")
    parser.add_argument("--num_rollout", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--m_iters", type=int, default=5,
                        help="Richardson iterations per step (training)")
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--forcing", type=float, default=1.0,
                        help="Source term f in -Lap(u) = f")
    parser.add_argument("--bc_value", type=float, default=0.0)

    # spatial setup
    parser.add_argument("--train_grid_n", type=int, default=11)
    parser.add_argument("--spatial_test_n", type=int, default=41)

    # budget shift test
    parser.add_argument("--budget_refinement", type=int, default=2,
                        help="m_test = m_iters // budget_refinement")

    # experiment infra
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--base_output_dir", type=str, default="EXPS")
    parser.add_argument("--save_every", type=int, default=10)

    args = parser.parse_args()
    set_seed(args.seed)

    params = {
        "m_iters": args.m_iters,
        "forcing": args.forcing,
        "num_rollout": args.num_rollout,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "train_grid_n": args.train_grid_n,
        "seed": args.seed,
    }

    if args.exp_name is not None:
        exp_dir = os.path.join(args.base_output_dir, args.exp_name)
    else:
        exp_dir = make_exp_dir(args.base_output_dir, "poisson", params)
    os.makedirs(exp_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    st_model = simple_dual_space_with_time_derivative_cnn_model()
    mesh = eval(args.mesh_definition)
    train_grid = make_point_grid(args.train_grid_n)

    train_trainer = build_trainer(
        mesh=mesh, point_grid=train_grid,
        simulation_steps=5, st_model=st_model, lr=1e-4,
        m_iters=args.m_iters, relaxation=args.relaxation,
        forcing=args.forcing,
    )

    u0 = make_ic(train_trainer.physical_model.V)
    train_trainer.generate_ground_truth(u0, args.num_rollout)

    losses = []
    train_errors = []
    train_error_steps = 3
    losses, train_errors = train_with_error_report(
        trainer=train_trainer,
        u0=u0,
        n_steps=train_error_steps,
        point_grid=train_grid,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
        exp_dir=exp_dir,
    )

    np.save(os.path.join(exp_dir, "train_losses.npy"), np.asarray(losses))
    torch.save(st_model.state_dict(), os.path.join(exp_dir, "trained_model.pt"))

    if len(losses) > 0:
        plot_training_curve(losses, os.path.join(plot_dir, "training_curve.png"))

    # --------------------------------------------------------
    # 1. Spatial interpolation
    # --------------------------------------------------------
    spatial_report = run_spatial_interpolation_experiment(
        mesh=mesh, trained_model=st_model, u0=u0, args=args,
    )
    plot_residual(spatial_report,
                  os.path.join(plot_dir, "spatial_interpolation.png"),
                  title="Spatial interpolation")

    # --------------------------------------------------------
    # 2. Iteration budget shift
    # --------------------------------------------------------
    budget_report = run_budget_shift_experiment(
        mesh=mesh, trained_model=st_model, u0=u0, args=args,
    )
    plot_residual(budget_report,
                  os.path.join(plot_dir, "budget_shift.png"),
                  title="Iteration budget shift")

    # --------------------------------------------------------
    # Combined residual + error curves
    # --------------------------------------------------------
    posterior_residual_curves = {
        "spatial interpolation": {
            "iterations": spatial_report["iterations"],
            "residual": spatial_report["residual"],
        },
        "budget shift": {
            "iterations": budget_report["iterations"],
            "residual": budget_report["residual"],
        },
    }
    plot_residual_curves(posterior_residual_curves,
                         os.path.join(plot_dir, "posterior_test_residual_curves.png"))

    posterior_error_curves = {
        "spatial interpolation": {
            "iterations": spatial_report["iterations"],
            "rel_rmse": spatial_report["gt_error"]["rel_rmse_per_step"],
        },
        "budget shift": {
            "iterations": budget_report["iterations"],
            "rel_rmse": budget_report["gt_error"]["rel_rmse_per_step"],
        },
    }
    plot_error_curves(posterior_error_curves,
                      os.path.join(plot_dir, "posterior_test_error_curves.png"))

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary = {
        "training": {
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "m_iters_train": args.m_iters,
            "num_rollout_train": args.num_rollout,
            "train_grid_n": args.train_grid_n,
            "final_loss": float(losses[-1]) if len(losses) > 0 else None,
            "train_errors": train_errors,
        },
        "spatial_interpolation": {
            "grid_test_n": args.spatial_test_n,
            "residual_mean": spatial_report["residual_mean"],
            "residual_last": spatial_report["residual_last"],
            "residual_max": spatial_report["residual_max"],
            "gt_rel_rmse_mean": spatial_report["gt_error"]["rel_rmse_mean"],
            "gt_rel_rmse_last": spatial_report["gt_error"]["rel_rmse_last"],
            "gt_linf_max": spatial_report["gt_error"]["linf_max"],
        },
        "budget_shift": {
            "m_train": budget_report["m_train"],
            "m_test": budget_report["m_test"],
            "residual_mean": budget_report["residual_mean"],
            "residual_last": budget_report["residual_last"],
            "residual_max": budget_report["residual_max"],
            "gt_rel_rmse_mean": budget_report["gt_error"]["rel_rmse_mean"],
            "gt_rel_rmse_last": budget_report["gt_error"]["rel_rmse_last"],
            "gt_linf_max": budget_report["gt_error"]["linf_max"],
        },
    }

    save_report_json(summary, os.path.join(exp_dir, "summary.json"))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
