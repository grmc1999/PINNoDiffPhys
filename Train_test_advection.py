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
from DL_models.PINNS.Residual_losses import diffusion_loss



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
    state_tensor.reshape(grid_shape)

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


def build_trainer(mesh, point_grid, dt, simulation_steps, st_model, lr=1e-4):

    print(point_grid)
    ph_model = ImplicitLinearAdvectionStepper(
    mesh=mesh,
    dt=0.01,
    velocity=(1.0, 0.0),
    inflow_value=0.0,
    degree=1,
    point_evaluator=point_grid,
    )
    print(ph_model.point_evaluator)

    trainer = FiredrakePINNSBasedSOLTrainerCNN(
        physical_model=ph_model,
        statistical_model=st_model,
        optimizer=torch.optim.Adam(st_model.parameters(), lr=lr),
        simulation_steps=simulation_steps,
        dt=dt,
        loss=lambda u, x: (diffusion_loss(u, x, K=1.0))**2,
        feature_builder=None,
    )
    return trainer


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

def run_spatial_interpolation_experiment(mesh, trained_model, u0, args):
    """
    Same dt and same time horizon, but denser spatial point sampling.
    """
    fine_grid = make_point_grid(args.spatial_test_n)
    grid = make_point_grid(args.train_grid_n)
    n_steps = args.num_rollout

    test_trainer = build_trainer(
        mesh=mesh,
        point_grid=grid, # Use same grid
        dt=args.dt,
        simulation_steps=n_steps,
        st_model=trained_model,
        lr=0.0,
    )

    pred_states, input_states, corr_states, pred_times, uncorrected_sol = test_trainer.predict_rollout( # Output should be in original resolution
        u0, t0=0.0, n_steps=n_steps, spatial_sample=fine_grid
    )
    pred_grids = grids_from_prediction_list(pred_states[:,:,[-1]], fine_grid.shape[:2])

    report = compute_residual_curve(test_trainer, pred_states, uncorrected_sol)

    report["times"] = np.asarray(pred_times)
    report["pred_grids"] = pred_grids
    report["grid_shape"] = fine_grid.shape
    return report


def run_temporal_interpolation_experiment(mesh, trained_model, u0, args):
    """
    Smaller dt within the same training horizon.
    """
    grid = make_point_grid(args.train_grid_n)
    dt_test = args.dt / args.temporal_refinement
    train_horizon = args.num_rollout * args.dt
    n_steps = int(round(train_horizon / dt_test))

    test_trainer = build_trainer(
        mesh=mesh,
        point_grid=grid,
        dt=dt_test,
        simulation_steps=n_steps,
        st_model=trained_model,
        lr=0.0,
    )

    pred_states, input_states, corr_states, pred_times, uncorrected_sol = test_trainer.predict_rollout( # Output should be in original resolution
        u0, t0=0.0, n_steps=n_steps, spatial_sample=grid
    )
    pred_grids = grids_from_prediction_list(pred_states[:,:,[-1]], grid.shape[:2])

    report = compute_residual_curve(test_trainer, pred_states, uncorrected_sol)

    report["times"] = np.asarray(pred_times)
    report["dt_test"] = dt_test
    report["pred_grids"] = pred_grids
    report["grid_shape"] = grid.shape
    return report


def run_temporal_extrapolation_experiment(mesh, trained_model, u0, args):
    """
    Same dt as training, but rollout beyond the training horizon.
    """
    grid = make_point_grid(args.train_grid_n)
    train_horizon = args.num_rollout * args.dt
    test_horizon = args.extrapolation_factor * train_horizon
    n_steps = int(round(test_horizon / args.dt))

    test_trainer = build_trainer(
        mesh=mesh,
        point_grid=grid,
        dt=args.dt,
        simulation_steps=n_steps,
        st_model=trained_model,
        lr=0.0,
    )

    pred_states, input_states, corr_states, pred_times, uncorrected_sol = test_trainer.predict_rollout( # Output should be in original resolution
        u0, t0=0.0, n_steps=n_steps, spatial_sample=grid
    )
    pred_grids = grids_from_prediction_list(pred_states[:,:,[-1]], grid.shape[:2])

    report = compute_residual_curve(test_trainer, pred_states[:,:,[-1]], uncorrected_sol)

    report["times"] = np.asarray(pred_times)
    report["pred_grids"] = pred_grids
    report["grid_shape"] = grid.shape
    report["train_horizon"] = train_horizon
    report["test_horizon"] = test_horizon

    return report


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

    # temporal tests
    parser.add_argument("--temporal_refinement", type=int, default=4)
    parser.add_argument("--extrapolation_factor", type=float, default=2.0)

    parser.add_argument("--output_dir", type=str, default="results_diffusion_experiments")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    st_model = simple_dual_space_with_time_derivative_cnn_model()
    mesh = eval(args.mesh_definition)

    train_grid = make_point_grid(args.train_grid_n)
    print(train_grid)

    train_trainer = build_trainer(
        mesh=mesh,
        point_grid=train_grid,
        dt=args.dt,
        simulation_steps=5,
        st_model=st_model,
        lr=1e-4,
    )

    u0 = make_ic(train_trainer.physical_model.V)

    train_trainer.generate_ground_truth(u0, args.num_rollout)
    losses = train_trainer.train(epochs=args.n_epochs, batch_size=args.batch_size)

    if losses is None:
        losses = []

    if len(losses) > 0:
        plot_training_curve(
            losses,
            os.path.join(args.output_dir, "training_curve.png"),
        )

    # --------------------------------------------------------
    # 1. Spatial interpolation
    # --------------------------------------------------------
    spatial_report = run_spatial_interpolation_experiment(
        mesh=mesh,
        trained_model=st_model,
        u0=u0,
        args=args,
    )

    plot_residual(spatial_report,
                os.path.join(args.output_dir, "spatial_interpolation.png"),
                title = "spatial interpolation"
                )

    # --------------------------------------------------------
    # 2. Temporal interpolation
    # --------------------------------------------------------
    temporal_interp_report = run_temporal_interpolation_experiment(
        mesh=mesh,
        trained_model=st_model,
        u0=u0,
        args=args,
    )

    plot_residual(temporal_interp_report,
                os.path.join(args.output_dir, "temporal_interpolation.png"),
                title = "temporal interpolation"
                )

    # --------------------------------------------------------
    # 3. Temporal extrapolation
    # --------------------------------------------------------
    temporal_extra_report = run_temporal_extrapolation_experiment(
        mesh=mesh,
        trained_model=st_model,
        u0=u0,
        args=args,
    )

    plot_residual(temporal_extra_report,
                os.path.join(args.output_dir, "temporal_extrapolation.png"),
                title = "temporal extarpolation",
                test_limit=float(temporal_extra_report["train_horizon"])
                )


    posterior_residual_curves = {
    "spatial interpolation": {
        "times": spatial_report["times"],
        "residual": spatial_report["residual"],
    },
    "temporal interpolation": {
        "times": temporal_interp_report["times"],
        "residual": temporal_interp_report["residual"],
    },
    "temporal extrapolation": {
        "times": temporal_extra_report["times"],
        "residual": temporal_extra_report["residual"],
    },
}

    plot_residual_curves(
        posterior_residual_curves,
        os.path.join(args.output_dir, "posterior_test_residual_curves.png"),
        train_horizon=args.num_rollout * args.dt,
    )


    # --------------------------------------------------------
    # Save quantitative summaries
    # --------------------------------------------------------
    summary = {
    "training": {
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "dt_train": args.dt,
        "num_rollout_train": args.num_rollout,
        "train_grid_n": args.train_grid_n,
        "final_loss": float(losses[-1]) if len(losses) > 0 else None,
    },
    "spatial_interpolation": {
        "grid_test_n": args.spatial_test_n,
        #"rmse_mean": spatial_report["rmse_mean"],
        #"rmse_last": spatial_report["rmse_last"],
        #"rel_rmse_mean": spatial_report["rel_rmse_mean"],
        #"rel_rmse_last": spatial_report["rel_rmse_last"],
        #"linf_max": spatial_report["linf_max"],
        "residual_mean": spatial_report["residual_mean"],
        "residual_last": spatial_report["residual_last"],
        "residual_max": spatial_report["residual_max"],
    },
    "temporal_interpolation": {
        "dt_test": temporal_interp_report["dt_test"],
        #"rmse_mean": temporal_interp_report["rmse_mean"],
        #"rmse_last": temporal_interp_report["rmse_last"],
        #"rel_rmse_mean": temporal_interp_report["rel_rmse_mean"],
        #"rel_rmse_last": temporal_interp_report["rel_rmse_last"],
        #"linf_max": temporal_interp_report["linf_max"],
        "residual_mean": temporal_interp_report["residual_mean"],
        "residual_last": temporal_interp_report["residual_last"],
        "residual_max": temporal_interp_report["residual_max"],
    },
    "temporal_extrapolation": {
        "train_horizon": temporal_extra_report["train_horizon"],
        "test_horizon": temporal_extra_report["test_horizon"],
        #"rmse_mean": temporal_extra_report["rmse_mean"],
        #"rmse_last": temporal_extra_report["rmse_last"],
        #"rel_rmse_mean": temporal_extra_report["rel_rmse_mean"],
        #"rel_rmse_last": temporal_extra_report["rel_rmse_last"],
        #"linf_max": temporal_extra_report["linf_max"],
        "residual_mean": temporal_extra_report["residual_mean"],
        "residual_last": temporal_extra_report["residual_last"],
        "residual_max": temporal_extra_report["residual_max"],
    },
}

    save_report_json(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))