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
from trainer.Trainer import ImplicitDiffusionStepper, FiredrakePINNSBasedSOLTrainerCNN
from DL_models.PINNS.Residual_losses import diffusion_loss


# ============================================================
# Utilities
# ============================================================
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


def evaluate_field_on_grid(field: fd.Function, point_grid: np.ndarray) -> np.ndarray:
    """
    Evaluate a Firedrake field on a regular point grid using PointEvaluator.

    Returns:
        values with shape [H, W] for scalar fields.
    """
    points = point_grid.reshape(-1, point_grid.shape[-1])
    pe = fd.PointEvaluator(field.function_space().mesh(), points)
    values = pe.evaluate(field)

    vom = fd.VertexOnlyMesh(
        field.function_space().mesh(),
        points.reshape(-1,field.function_space().mesh().geometric_dimension()),
        reorder = False
    )
    
    field = fd.assemble(fd.interpolate(field, fd.FunctionSpace(vom, "DG", 0)))

    values = field.dat.data
    if values.ndim == 1:
        return values.reshape(point_grid.shape[:2])
    elif values.ndim == 2 and values.shape[-1] == 1:
        return values[:, 0].reshape(point_grid.shape[:2])
    else:
        # vector/tensor fallback
        return values.reshape(point_grid.shape[:2] + values.shape[1:])


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
    residual_values = []

    with torch.no_grad():
        for u_pred, x_in in zip(pred_states, input_states):
            val = trainer.loss(u_pred, x_in)

            if torch.is_tensor(val):
                val = float(torch.mean(val).detach().cpu().item())
            else:
                val = float(torch.mean(val))

            residual_values.append(val)

    residual_values = np.asarray(residual_values, dtype=float)

    return {
        "residual": residual_values,
        "residual_mean": float(np.mean(residual_values)),
        "residual_last": float(residual_values[-1]),
        "residual_max": float(np.max(residual_values)),
    }

def compute_error_curve(pred_grids, gt_grids, eps=1e-12):
    """
    pred_grids, gt_grids: list of [H, W] arrays
    """
    rmse = []
    rel_rmse = []
    linf = []

    for pred, gt in zip(pred_grids, gt_grids):
        err = pred - gt
        rmse_i = np.sqrt(np.mean(err ** 2))
        gt_norm = np.sqrt(np.mean(gt ** 2)) + eps
        rmse.append(rmse_i)
        rel_rmse.append(rmse_i / gt_norm)
        linf.append(np.max(np.abs(err)))

    return {
        "rmse": np.asarray(rmse),
        "rel_rmse": np.asarray(rel_rmse),
        "linf": np.asarray(linf),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_last": float(rmse[-1]),
        "rel_rmse_mean": float(np.mean(rel_rmse)),
        "rel_rmse_last": float(rel_rmse[-1]),
        "linf_max": float(np.max(linf)),
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
    ph_model = ImplicitDiffusionStepper(
        mesh=mesh,
        dt=dt,
        point_evaluator=point_grid,
    )

    trainer = FiredrakePINNSBasedSOLTrainerCNN(
        physical_model=ph_model,
        statistical_model=st_model,
        optimizer=torch.optim.Adam(st_model.parameters(), lr=lr),
        simulation_steps=simulation_steps,
        dt=dt,
        loss=lambda u, x: diffusion_loss(u, x, K=1.0),
        feature_builder=None,
    )
    return trainer


def grids_from_prediction_list(pred_states, point_grid):
    H, W = point_grid
    return [tensor_state_to_grid(s, (H, W)) for s in pred_states]


def grids_from_gt_fields(gt_fields, point_grid):
    return [evaluate_field_on_grid(f, point_grid) for f in gt_fields]


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


def plot_snapshot(gt, pred, time_value, title, output_path):
    err = np.abs(pred - gt)
    vmax = max(float(np.max(gt)), float(np.max(pred)))
    emin = float(np.min([gt.min(), pred.min()]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(gt, origin="lower", extent=(0, 1, 0, 1), vmin=emin, vmax=vmax)
    axes[0].set_title(f"Ground truth\n t={time_value:.4f}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred, origin="lower", extent=(0, 1, 0, 1), vmin=emin, vmax=vmax)
    axes[1].set_title("Prediction")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(err, origin="lower", extent=(0, 1, 0, 1))
    axes[2].set_title("|Error|")
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
    pred_grids = grids_from_prediction_list(pred_states[-1], fine_grid.shape[:2])

    #gt_fields = rollout_ground_truth(test_trainer.physical_model, u0, n_steps=n_steps)
    #gt_grids = grids_from_gt_fields(gt_fields, fine_grid)

    #report = compute_error_curve(pred_grids, gt_grids)
    report = compute_residual_curve(test_trainer, pred_states, uncorrected_sol)

    #report.update(residual_report)
    report["times"] = np.asarray(pred_times)
    report["pred_grids"] = pred_grids
    #report["gt_grids"] = gt_grids
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
    pred_grids = grids_from_prediction_list(pred_states[-1], grid.shape[:2])

    #gt_fields = rollout_ground_truth(test_trainer.physical_model, u0, n_steps=n_steps)
    #gt_grids = grids_from_gt_fields(gt_fields, grid)

    #report = compute_error_curve(pred_grids, gt_grids)
    report = compute_residual_curve(test_trainer, pred_states, uncorrected_sol)

    #report.update(residual_report)
    report["times"] = np.asarray(pred_times)
    report["dt_test"] = dt_test
    report["pred_grids"] = pred_grids
    #report["gt_grids"] = gt_grids
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
    #pred_grids = grids_from_prediction_list(pred_states[-1], grid)
    pred_grids = grids_from_prediction_list(pred_states[-1], grid.shape[:2])

    #gt_fields = rollout_ground_truth(test_trainer.physical_model, u0, n_steps=n_steps)
    #gt_grids = grids_from_gt_fields(gt_fields, grid)

    #full_report = compute_error_curve(pred_grids, gt_grids)
    report = compute_residual_curve(test_trainer, pred_states, uncorrected_sol)

    report["times"] = np.asarray(pred_times)
    report["pred_grids"] = pred_grids
    #full_report["gt_grids"] = gt_grids

    mask = np.asarray(pred_times) > train_horizon

    extra_report = {
        "times": np.asarray(pred_times)[mask],
        #"rmse": report["rmse"][mask],
        #"rel_rmse": report["rel_rmse"][mask],
        "linf": report["linf"][mask],
        "residual": report["residual"][mask],
        "residual_mean": float(np.mean(report["residual"][mask])),
        "residual_last": float(report["residual"][mask][-1]),
        "residual_max": float(np.max(report["residual"][mask])),
        "train_horizon": train_horizon,
        "test_horizon": test_horizon,
    }
    return extra_report, report


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

    plot_snapshot(
        spatial_report["gt_grids"][-1],
        spatial_report["pred_grids"][-1],
        spatial_report["times"][-1],
        title="Spatial interpolation test",
        output_path=os.path.join(args.output_dir, "snapshot_spatial_interpolation.png"),
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

    plot_snapshot(
        temporal_interp_report["gt_grids"][-1],
        temporal_interp_report["pred_grids"][-1],
        temporal_interp_report["times"][-1],
        title="Temporal interpolation test",
        output_path=os.path.join(args.output_dir, "snapshot_temporal_interpolation.png"),
    )

    # --------------------------------------------------------
    # 3. Temporal extrapolation
    # --------------------------------------------------------
    temporal_extra_report, temporal_extra_full_report = run_temporal_extrapolation_experiment(
        mesh=mesh,
        trained_model=st_model,
        u0=u0,
        args=args,
    )

    plot_snapshot(
        temporal_extra_report["gt_grids"][-1],
        temporal_extra_report["pred_grids"][-1],
        temporal_extra_report["times"][-1],
        title="Temporal extrapolation test",
        output_path=os.path.join(args.output_dir, "snapshot_temporal_extrapolation.png"),
    )

    # --------------------------------------------------------
    # Posterior testing plot
    # --------------------------------------------------------
    #posterior_curves = {
    #    "spatial interpolation": {
    #        "times": spatial_report["times"],
    #        "rel_rmse": spatial_report["rel_rmse"],
    #    },
    #    "temporal interpolation": {
    #        "times": temporal_interp_report["times"],
    #        "rel_rmse": temporal_interp_report["rel_rmse"],
    #    },
    #    "temporal extrapolation": {
    #        "times": temporal_extra_full_report["times"],
    #        "rel_rmse": temporal_extra_full_report["rel_rmse"],
    #    },
    #}

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
        "times": temporal_extra_full_report["times"],
        "residual": temporal_extra_full_report["residual"],
    },
}

    plot_residual_curves(
        posterior_residual_curves,
        os.path.join(args.output_dir, "posterior_test_residual_curves.png"),
        train_horizon=args.num_rollout * args.dt,
    )

#    plot_error_curves(
#        posterior_curves,
#        os.path.join(args.output_dir, "posterior_test_error_curves.png"),
#        train_horizon=args.num_rollout * args.dt,
#    )

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
        "linf_max": spatial_report["linf_max"],
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
        #"test_horizon": temporal_extra_report["test_horizon"],
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