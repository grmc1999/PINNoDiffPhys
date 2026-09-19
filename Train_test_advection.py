import argparse
import json
import os
from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import firedrake as fd

from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
from trainer.Trainer import ImplicitLinearAdvectionStepper, FiredrakePINNSBasedSOLTrainerCNN
from trainer.PurePINNTrainer import CoordinateMLP, PurePINNTrainer
from DL_models.PINNS.Residual_losses import advection_loss
from experiment_utils import (set_seed, make_exp_dir, save_checkpoint,
                              rollout_ground_truth_on_grid, fine_reference_on_grid,
                              gt_error_metrics, train_with_error_report)

def make_point_grid(n: int, P_min: List[float] = [0.0,0.0], P_max: List[float] = [1.0,1.0]):
    """
    Returns a regular grid of points in [0,1]^2 with shape [n, n, 2].
    """
    grid = np.stack(
        np.meshgrid(
            *tuple(np.linspace(p_min,p_max,n) for p_min,p_max in zip(P_min,P_max)),
            #np.linspace(0.0, 1.0, n),
            #np.linspace(0.0, 1.0, n),
            indexing="xy",
        ),
        axis=-1,
    )
    return grid


def make_ic(V):
    """
    Same IC you used in the example.
    """
    X = fd.SpatialCoordinate(V.mesh())
    u0 = fd.Function(V).interpolate(
        0.5 * fd.exp(0.5 * ((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 - 0.1) ** 2 - 1.0)
    )
    return u0


def tensor_state_to_grid(state_tensor, grid_shape):
    """
    Convert trainer output tensor to [H, W] grid.
    Assumes scalar field output.

    Supports common formats:
      [1, 1, H, W]
      [1, H, W]
      [H, W]
      [1, H*W, 1]
      [H*W, 1]
    """
    return state_tensor.reshape(grid_shape)

def fem_residual_curve(pred_grids, dt, velocity=(1.0, 0.0)):
    """Numeric PDE residual of the pure-FEM field on the point grid.

    Same report-dict contract as compute_residual_curve; used for fem mode
    where the interpolated coarse field has no autograd linkage.
    """
    u = np.stack([g.detach().cpu().numpy() if torch.is_tensor(g)
                  else np.asarray(g) for g in pred_grids])   # [T, H, W]
    T, H, W = u.shape
    du_dx = np.gradient(u, axis=2) / (1.0 / (W - 1))
    du_dy = np.gradient(u, axis=1) / (1.0 / (H - 1))
    u_t = np.gradient(u, axis=0) / dt if T > 1 else np.zeros_like(u)
    res = u_t + velocity[0] * du_dx + velocity[1] * du_dy
    res2 = res ** 2
    val_h = np.mean(res2, axis=(1, 2))
    return {
        "residual": res2,
        "residual_decay": val_h,
        "residual_mean": float(np.mean(val_h)),
        "residual_last": float(val_h[-1]),
        "residual_max": float(np.max(val_h)),
    }


def report_from_predictions(trainer, args, pred_states, uncorrected_sol,
                            pred_grids, dt):
    if args.mode == "fem":
        return fem_residual_curve(pred_grids, dt, velocity=(1.0, 0.0))
    return compute_residual_curve(trainer, pred_states, uncorrected_sol)


def compute_residual_curve(trainer, pred_states, input_states):
    """
    Evaluate the same residual-based loss used during training over a rollout.

    Returns:
        dict with per-step residual values and summary statistics
    """
    val = trainer.loss(pred_states, input_states)

    if torch.is_tensor(val):
        val_h = torch.mean(val,axis = -1 ).detach().cpu().numpy()
    else:
        val_h = torch.mean(val,axis = -1 ).numpy()

    return {
        "residual": val.detach().cpu().numpy(),
        "residual_decay": val_h,
        "residual_mean": float(np.mean(val_h)),
        "residual_last": float(val_h[-1]),
        "residual_max": float(np.max(val_h)),
    }


def rollout_ground_truth(stepper, u0: fd.Function, n_steps: int):
    """
    Rollout PDE ground truth directly with the physical model.
    Returns states for times t = dt, 2dt, ..., n_steps*dt.
    """
    states = []
    u = fd.Function(stepper.V).assign(u0)

    for _ in range(n_steps):
        u = stepper.step(u)
        u_next = fd.Function(stepper.V).assign(u)
        states.append(u_next)

    return states


def build_trainer(mesh, point_grid, dt, simulation_steps, st_model, lr=1e-4,
                  mode="hybrid"):

    ph_model = ImplicitLinearAdvectionStepper(
    mesh=mesh,
    dt=dt,
    velocity=(1.0, 0.0),
    inflow_value=0.0,
    degree=1,
    point_evaluator=point_grid,
    )

    if mode == "pinn":
        trainer = PurePINNTrainer(
            physical_model=ph_model,
            statistical_model=st_model,
            optimizer=torch.optim.Adam(st_model.parameters(), lr=lr),
            simulation_steps=simulation_steps,
            dt=dt,
            loss=lambda u, x: (advection_loss(u, x, velocity=(1.0, 0.0))) ** 2,
            eval_grid=point_grid,
        )
    else:
        trainer = FiredrakePINNSBasedSOLTrainerCNN(
            physical_model=ph_model,
            statistical_model=st_model,
            optimizer=torch.optim.Adam(st_model.parameters(), lr=lr),
            simulation_steps=simulation_steps,
            dt=dt,
            loss=lambda u, x: (advection_loss(u, x, velocity=(1.0, 0.0))) ** 2,
            correction_enabled=(mode == "hybrid"),
        )
    return trainer


def build_ref_stepper(mesh_def, dt, point_grid):
    """Finer-mesh stepper of the same family, used as the refined truth."""
    mesh = eval(mesh_def)
    return ImplicitLinearAdvectionStepper(
        mesh=mesh,
        dt=dt,
        velocity=(1.0, 0.0),
        inflow_value=0.0,
        degree=1,
        point_evaluator=point_grid,
    )


def grids_from_prediction_list(pred_states, point_grid):
    H, W = point_grid
    return [tensor_state_to_grid(s, (H, W)) for s in pred_states]


#def grids_from_gt_fields(gt_fields, point_grid):
#    return [evaluate_field_on_grid(f, point_grid) for f in gt_fields]


# ============================================================
# Plotting
# ============================================================

def plot_residual_curves(time_dict, output_path, train_horizon=None):
    plt.figure(figsize=(8, 5))

    for name, report in time_dict.items():
        plt.plot(report["times"], report["residual"], linewidth=2, label=name)

    if train_horizon is not None:
        plt.axvline(train_horizon, linestyle="--", linewidth=1.5, label="train horizon")

    plt.xlabel("Time")
    plt.ylabel("PDE residual")
    plt.title("Posterior testing: PDE residual vs time")
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


def plot_error_curves(time_dict, output_path, train_horizon=None):
    plt.figure(figsize=(8, 5))

    for name, report in time_dict.items():
        plt.plot(report["times"], report["rel_rmse"], linewidth=2, label=name)

    if train_horizon is not None:
        plt.axvline(train_horizon, linestyle="--", linewidth=1.5, label="train horizon")

    plt.xlabel("Time")
    plt.ylabel("Relative RMSE")
    plt.title("Posterior testing: error vs time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_residual(report, output_path, title, test_limit = None):

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im0 = axes[0].plot(report["times"], report["residual_decay"])
    if isinstance(test_limit,float):
        axes[0].axvline(x=test_limit, color='r', linestyle='--', label='extrapolation horizon')

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residual loss")
    axes[0].set_title(f"time MSE residual \n")
    axes[0].grid(True, alpha=0.3)

    im1 = axes[1].imshow(report["residual"][0].reshape(report["grid_shape"][:2]), origin="lower", extent=(0, 1, 0, 1))
    axes[1].set_title("Residual spatial mal at t = 0")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(report["residual"][-1].reshape(report["grid_shape"][:2]), origin="lower", extent=(0, 1, 0, 1))
    axes[2].set_title("Residual spatial mal at t = T")
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

def run_spatial_interpolation_experiment(test_trainer, u0, args, ref_stepper=None):
    """
    Same dt and same time horizon, but denser spatial point sampling.

    *test_trainer* is prebuilt once outside the training loop and reused at
    every posterior refresh so pyadjoint blocks are recorded only once.
    """
    fine_grid = make_point_grid(args.spatial_test_n)
    n_steps = args.num_rollout

    pred_states, input_states, corr_states, pred_times, uncorrected_sol = test_trainer.predict_rollout( # Output should be in original resolution
        u0, t0=0.0, n_steps=n_steps, spatial_sample=fine_grid
    )
    pred_grids = grids_from_prediction_list(pred_states[:,:,[-1]], fine_grid.shape[:2])

    report = report_from_predictions(test_trainer, args, pred_states[:,:,-1:],
                                     uncorrected_sol, pred_grids, args.dt)

    report["times"] = np.asarray(pred_times)
    report["pred_grids"] = pred_grids
    report["grid_shape"] = fine_grid.shape

    gt_grids = rollout_ground_truth_on_grid(test_trainer.physical_model, u0, n_steps, fine_grid)
    report["gt_error"] = gt_error_metrics(pred_states[:,:,-1:], gt_grids)

    if ref_stepper is not None:
        gt_fine_grids = fine_reference_on_grid(ref_stepper, u0, n_steps, fine_grid)
        report["gt_error_fine"] = gt_error_metrics(pred_states[:,:,-1:], gt_fine_grids)

    return report


def run_temporal_interpolation_experiment(test_trainer, u0, args, ref_stepper=None):
    """
    Smaller dt within the same training horizon.
    """
    grid = make_point_grid(args.train_grid_n)
    dt_test = args.dt / args.temporal_refinement
    train_horizon = args.num_rollout * args.dt
    n_steps = int(round(train_horizon / dt_test))

    pred_states, input_states, corr_states, pred_times, uncorrected_sol = test_trainer.predict_rollout( # Output should be in original resolution
        u0, t0=0.0, n_steps=n_steps, spatial_sample=grid
    )
    pred_grids = grids_from_prediction_list(pred_states[:,:,[-1]], grid.shape[:2])

    report = report_from_predictions(test_trainer, args, pred_states[:,:,-1:],
                                     uncorrected_sol, pred_grids, dt_test)

    report["times"] = np.asarray(pred_times)
    report["dt_test"] = dt_test
    report["pred_grids"] = pred_grids
    report["grid_shape"] = grid.shape

    gt_grids = rollout_ground_truth_on_grid(test_trainer.physical_model, u0, n_steps, grid)
    report["gt_error"] = gt_error_metrics(pred_states[:,:,-1:], gt_grids)

    if ref_stepper is not None:
        gt_fine_grids = fine_reference_on_grid(ref_stepper, u0, n_steps, grid)
        report["gt_error_fine"] = gt_error_metrics(pred_states[:,:,-1:], gt_fine_grids)

    return report


def run_temporal_extrapolation_experiment(test_trainer, u0, args, ref_stepper=None):
    """
    Same dt as training, but rollout beyond the training horizon.
    """
    grid = make_point_grid(args.train_grid_n)
    train_horizon = args.num_rollout * args.dt
    test_horizon = args.extrapolation_factor * train_horizon
    n_steps = int(round(test_horizon / args.dt))

    pred_states, input_states, corr_states, pred_times, uncorrected_sol = test_trainer.predict_rollout( # Output should be in original resolution
        u0, t0=0.0, n_steps=n_steps, spatial_sample=grid
    )
    pred_grids = grids_from_prediction_list(pred_states[:,:,[-1]], grid.shape[:2])

    report = report_from_predictions(test_trainer, args, pred_states[:,:,[-1]],
                                     uncorrected_sol, pred_grids, args.dt)

    report["times"] = np.asarray(pred_times)
    report["pred_grids"] = pred_grids
    report["grid_shape"] = grid.shape
    report["train_horizon"] = train_horizon
    report["test_horizon"] = test_horizon

    gt_grids = rollout_ground_truth_on_grid(test_trainer.physical_model, u0, n_steps, grid)
    report["gt_error"] = gt_error_metrics(pred_states[:,:,-1:], gt_grids)

    if ref_stepper is not None:
        gt_fine_grids = fine_reference_on_grid(ref_stepper, u0, n_steps, grid)
        report["gt_error_fine"] = gt_error_metrics(pred_states[:,:,-1:], gt_fine_grids)

    return report


# ============================================================
# Posterior refresh: regenerate all posterior plots + summary.json
# with the current model state (called after every checkpoint).
# ============================================================

def refresh_posterior(post_spatial, post_temporal_interp, post_temporal_extra,
                      u0, args, exp_dir, plot_dir, losses, train_errors):
    point_grid = make_point_grid(args.train_grid_n)
    dt_test = args.dt / args.temporal_refinement
    ref_spatial = build_ref_stepper(
        mesh_def=f"fd.UnitSquareMesh({args.refine_mesh_n},{args.refine_mesh_n})",
        dt=args.dt, point_grid=point_grid,
    )
    ref_temp_interp = build_ref_stepper(
        mesh_def=f"fd.UnitSquareMesh({args.refine_mesh_n},{args.refine_mesh_n})",
        dt=dt_test, point_grid=point_grid,
    )
    ref_temp_extra = build_ref_stepper(
        mesh_def=f"fd.UnitSquareMesh({args.refine_mesh_n},{args.refine_mesh_n})",
        dt=args.dt, point_grid=point_grid,
    )
    spatial_report = run_spatial_interpolation_experiment(
        post_spatial, u0=u0, args=args, ref_stepper=ref_spatial)
    temporal_interp_report = run_temporal_interpolation_experiment(
        post_temporal_interp, u0=u0, args=args, ref_stepper=ref_temp_interp)
    temporal_extra_report = run_temporal_extrapolation_experiment(
        post_temporal_extra, u0=u0, args=args, ref_stepper=ref_temp_extra)

    plot_residual(spatial_report,
                  os.path.join(plot_dir, "spatial_interpolation.png"),
                  title="spatial interpolation")
    plot_residual(temporal_interp_report,
                  os.path.join(plot_dir, "temporal_interpolation.png"),
                  title="temporal interpolation")
    plot_residual(temporal_extra_report,
                  os.path.join(plot_dir, "temporal_extrapolation.png"),
                  title="temporal extrapolation",
                  test_limit=float(temporal_extra_report["train_horizon"]))

    posterior_residual_curves = {
        "spatial interpolation": {
            "times": spatial_report["times"],
            "residual": spatial_report["residual_decay"],
        },
        "temporal interpolation": {
            "times": temporal_interp_report["times"],
            "residual": temporal_interp_report["residual_decay"],
        },
        "temporal extrapolation": {
            "times": temporal_extra_report["times"],
            "residual": temporal_extra_report["residual_decay"],
        },
    }
    plot_residual_curves(
        posterior_residual_curves,
        os.path.join(plot_dir, "posterior_test_residual_curves.png"),
        train_horizon=args.num_rollout * args.dt,
    )

    posterior_error_curves = {
        "spatial interpolation": {
            "times": spatial_report["times"],
            "rel_rmse": spatial_report["gt_error"]["rel_rmse_per_step"],
        },
        "temporal interpolation": {
            "times": temporal_interp_report["times"],
            "rel_rmse": temporal_interp_report["gt_error"]["rel_rmse_per_step"],
        },
        "temporal extrapolation": {
            "times": temporal_extra_report["times"],
            "rel_rmse": temporal_extra_report["gt_error"]["rel_rmse_per_step"],
        },
    }
    plot_error_curves(
        posterior_error_curves,
        os.path.join(plot_dir, "posterior_test_error_curves.png"),
        train_horizon=args.num_rollout * args.dt,
    )

    if len(losses) > 0:
        plot_training_curve(losses, os.path.join(plot_dir, "training_curve.png"))

    summary = {
        "training": {
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "dt_train": args.dt,
            "num_rollout_train": args.num_rollout,
            "train_grid_n": args.train_grid_n,
            "mode": args.mode,
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
        "temporal_interpolation": {
            "dt_test": temporal_interp_report["dt_test"],
            "residual_mean": temporal_interp_report["residual_mean"],
            "residual_last": temporal_interp_report["residual_last"],
            "residual_max": temporal_interp_report["residual_max"],
            "gt_rel_rmse_mean": temporal_interp_report["gt_error"]["rel_rmse_mean"],
            "gt_rel_rmse_last": temporal_interp_report["gt_error"]["rel_rmse_last"],
            "gt_linf_max": temporal_interp_report["gt_error"]["linf_max"],
        },
        "temporal_extrapolation": {
            "train_horizon": temporal_extra_report["train_horizon"],
            "test_horizon": temporal_extra_report["test_horizon"],
            "residual_mean": temporal_extra_report["residual_mean"],
            "residual_last": temporal_extra_report["residual_last"],
            "residual_max": temporal_extra_report["residual_max"],
            "gt_rel_rmse_mean": temporal_extra_report["gt_error"]["rel_rmse_mean"],
            "gt_rel_rmse_last": temporal_extra_report["gt_error"]["rel_rmse_last"],
            "gt_linf_max": temporal_extra_report["gt_error"]["linf_max"],
        },
    }
    if "gt_error_fine" in spatial_report:
        summary["spatial_interpolation"].update({
            "gt_rel_rmse_mean_fine": spatial_report["gt_error_fine"]["rel_rmse_mean"],
            "gt_rel_rmse_last_fine": spatial_report["gt_error_fine"]["rel_rmse_last"],
            "gt_linf_max_fine": spatial_report["gt_error_fine"]["linf_max"],
        })
    if "gt_error_fine" in temporal_interp_report:
        summary["temporal_interpolation"].update({
            "gt_rel_rmse_mean_fine": temporal_interp_report["gt_error_fine"]["rel_rmse_mean"],
            "gt_rel_rmse_last_fine": temporal_interp_report["gt_error_fine"]["rel_rmse_last"],
            "gt_linf_max_fine": temporal_interp_report["gt_error_fine"]["linf_max"],
        })
    if "gt_error_fine" in temporal_extra_report:
        summary["temporal_extrapolation"].update({
            "gt_rel_rmse_mean_fine": temporal_extra_report["gt_error_fine"]["rel_rmse_mean"],
            "gt_rel_rmse_last_fine": temporal_extra_report["gt_error_fine"]["rel_rmse_last"],
            "gt_linf_max_fine": temporal_extra_report["gt_error_fine"]["linf_max"],
        })
    save_report_json(summary, os.path.join(exp_dir, "summary.json"))
    epoch = train_errors[-1]["epoch"] if train_errors else 0
    print(f"  [posterior] refreshed at epoch {epoch}"
          f"  rel_rmse_mean={spatial_report['gt_error']['rel_rmse_mean']:.4f}"
          f"  resid_mean={spatial_report['residual_mean']:.3f}")
    return summary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh_definition", type=str, default="fd.UnitSquareMesh(10,10)")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num_rollout", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)

    # spatial setup
    parser.add_argument("--train_grid_n", type=int, default=11)
    parser.add_argument("--spatial_test_n", type=int, default=41)

    # ablation mode: hybrid (CNN corrector, default), fem (corrected coarse
    # solver upstream baseline, no training), pinn (pure PINN MLP)
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["hybrid", "fem", "pinn"])
    parser.add_argument("--refine_mesh_n", type=int, default=40)

    # temporal tests
    parser.add_argument("--temporal_refinement", type=int, default=4)
    parser.add_argument("--extrapolation_factor", type=float, default=2.0)

    parser.add_argument("--output_dir", type=str, default="results_diffusion_experiments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--base_output_dir", type=str, default="EXPS")
    parser.add_argument("--save_every", type=int, default=10)

    args = parser.parse_args()
    set_seed(args.seed)

    params = {
        "dt": args.dt,
        "num_rollout": args.num_rollout,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "train_grid_n": args.train_grid_n,
        "seed": args.seed,
    }

    if args.exp_name is not None:
        exp_dir = os.path.join(args.base_output_dir, args.exp_name)
    else:
        exp_dir = make_exp_dir(args.base_output_dir, "advection", params)

    os.makedirs(exp_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    if args.mode == "pinn":
        st_model = CoordinateMLP()
    else:
        st_model = simple_dual_space_with_time_derivative_cnn_model()
    mesh = eval(args.mesh_definition)

    train_grid = make_point_grid(args.train_grid_n)

    train_trainer = build_trainer(
        mesh=mesh,
        point_grid=train_grid,
        dt=args.dt,
        simulation_steps=5,
        st_model=st_model,
        lr=1e-4,
        mode=args.mode,
    )

    # Posterior test trainers: built ONCE (annotation ON, blocks recorded a
    # single time) and reused at every checkpoint so the pyadjoint tape stays
    # constant instead of growing per checkpoint.
    post_spatial = build_trainer(
        mesh=mesh,
        point_grid=train_grid,
        dt=args.dt,
        simulation_steps=args.num_rollout,
        st_model=st_model,
        lr=0.0,
        mode=args.mode,
    )
    dt_test = args.dt / args.temporal_refinement
    post_temporal_interp = build_trainer(
        mesh=mesh,
        point_grid=train_grid,
        dt=dt_test,
        simulation_steps=int(round(args.num_rollout * args.dt / dt_test)),
        st_model=st_model,
        lr=0.0,
        mode=args.mode,
    )
    post_temporal_extra = build_trainer(
        mesh=mesh,
        point_grid=train_grid,
        dt=args.dt,
        simulation_steps=int(round(args.extrapolation_factor * args.num_rollout)),
        st_model=st_model,
        lr=0.0,
        mode=args.mode,
    )

    if args.mode == "fem":
        args.n_epochs = 0

    # Freeze the pyadjoint tape for the remainder of the run.
    fd.adjoint.stop_annotating()

    u0 = make_ic(train_trainer.physical_model.V)

    train_trainer.generate_ground_truth(u0, args.num_rollout)

    losses = []
    train_errors = []
    train_error_steps = 3
    def _refresh_cb(trainer, epoch, losses, train_errors):
        if os.environ.get("PINNO_SKIP_POSTERIOR") == "1":
            return
        if epoch != args.n_epochs:
            return
        refresh_posterior(post_spatial, post_temporal_interp, post_temporal_extra,
                          u0, args, exp_dir, plot_dir, losses, train_errors)

    losses, train_errors = train_with_error_report(
        trainer=train_trainer,
        u0=u0,
        n_steps=train_error_steps,
        point_grid=train_grid,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
        exp_dir=exp_dir,
        checkpoint_callback=_refresh_cb,
    )

    if losses is None:
        losses = []

    np.save(os.path.join(exp_dir, "train_losses.npy"), np.asarray(losses))
    torch.save(st_model.state_dict(), os.path.join(exp_dir, "trained_model.pt"))

    if len(losses) > 0:
        plot_training_curve(
            losses,
            os.path.join(plot_dir, "training_curve.png"),
        )

    if (args.n_epochs % args.save_every != 0 or args.mode == "fem") \
            and os.environ.get("PINNO_SKIP_POSTERIOR") != "1":
        refresh_posterior(post_spatial, post_temporal_interp, post_temporal_extra,
                          u0, args, exp_dir, plot_dir, losses, train_errors)

    with open(os.path.join(exp_dir, "summary.json")) as f:
        summary = json.load(f)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
