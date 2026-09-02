#!/usr/bin/env python3
"""Smoke tests for PINNoDiffPhys M1 modifications.

Builds every stepper on a tiny 8x8 mesh, runs 2 training epochs,
one rollout per regime, and asserts finite losses.
Exit 1 on any failure; exit 0 on success.
"""
import sys
import traceback
import numpy as np
import torch
import firedrake as fd

from experiment_utils import set_seed, rollout_ground_truth_on_grid, gt_error_metrics


def _ok(msg):
    print(f"  OK: {msg}")


def _fail(msg, exc):
    print(f"  FAIL: {msg} -- {exc}")
    traceback.print_exc()
    return False


def test_diffusion():
    from trainer.Trainer import ImplicitDiffusionStepper, FiredrakePINNSBasedSOLTrainerCNN
    from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
    from DL_models.PINNS.Residual_losses import diffusion_loss

    mesh = fd.UnitSquareMesh(8, 8)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9)),
                    axis=-1)
    model = simple_dual_space_with_time_derivative_cnn_model()
    stepper = ImplicitDiffusionStepper(mesh=mesh, dt=0.1, point_evaluator=grid)
    trainer = FiredrakePINNSBasedSOLTrainerCNN(
        physical_model=stepper, statistical_model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        simulation_steps=2, dt=0.1,
        loss=lambda u, x: (diffusion_loss(u, x, K=1.0)) ** 2,
    )
    X = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(stepper.V).interpolate(
        0.5 * fd.exp(0.5 * ((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 - 0.1) ** 2 - 1.0)
    )
    trainer.generate_ground_truth(u0, 4)
    losses = trainer.train(epochs=2, batch_size=2)
    assert all(np.isfinite(l) for l in losses), "non-finite loss"
    # predict_rollout
    pred, inp, corr, times, unc = trainer.predict_rollout(u0, 0.0, 3,
                                                          spatial_sample=grid)
    assert pred.shape[0] == 3, f"expected 3 steps, got {pred.shape[0]}"
    _ok("diffusion")


def test_advection():
    from trainer.Trainer import ImplicitLinearAdvectionStepper, FiredrakePINNSBasedSOLTrainerCNN
    from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
    from DL_models.PINNS.Residual_losses import diffusion_loss

    mesh = fd.UnitSquareMesh(8, 8)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9)),
                    axis=-1)
    model = simple_dual_space_with_time_derivative_cnn_model()
    stepper = ImplicitLinearAdvectionStepper(
        mesh=mesh, dt=0.01, velocity=(1.0, 0.0),
        inflow_value=0.0, degree=1, point_evaluator=grid,
    )
    trainer = FiredrakePINNSBasedSOLTrainerCNN(
        physical_model=stepper, statistical_model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        simulation_steps=2, dt=0.01,
        loss=lambda u, x: (diffusion_loss(u, x, K=1.0)) ** 2,
    )
    X = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(stepper.V).interpolate(
        0.5 * fd.exp(0.5 * ((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 - 0.1) ** 2 - 1.0)
    )
    trainer.generate_ground_truth(u0, 4)
    losses = trainer.train(epochs=2, batch_size=2)
    assert all(np.isfinite(l) for l in losses), "non-finite loss"
    _ok("advection")


def test_poisson():
    from trainer.Trainer import IterativePoissonSolverStepper, FiredrakePINNSBasedSOLTrainerCNN
    from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
    from DL_models.PINNS.Residual_losses import poisson_residual_loss

    mesh = fd.UnitSquareMesh(8, 8)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9)),
                    axis=-1)
    model = simple_dual_space_with_time_derivative_cnn_model()
    stepper = IterativePoissonSolverStepper(
        mesh=mesh, m_iters=5, relaxation=1.0,
        forcing=1.0, bc_value=0.0,
        degree=1, point_evaluator=grid,
    )
    trainer = FiredrakePINNSBasedSOLTrainerCNN(
        physical_model=stepper, statistical_model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        simulation_steps=2, dt=1.0,
        loss=lambda u, x: (poisson_residual_loss(u, x, K=1.0)) ** 2,
    )
    X = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(stepper.V).interpolate(
        0.5 * fd.exp(0.5 * ((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 - 0.1) ** 2 - 1.0)
    )
    trainer.generate_ground_truth(u0, 4)
    losses = trainer.train(epochs=2, batch_size=2)
    assert all(np.isfinite(l) for l in losses), "non-finite loss"
    _ok("poisson")


def test_gt_metrics():
    mesh = fd.UnitSquareMesh(8, 8)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9)),
                    axis=-1)
    from trainer.Trainer import ImplicitDiffusionStepper
    stepper = ImplicitDiffusionStepper(mesh=mesh, dt=0.1, point_evaluator=grid)
    X = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(stepper.V).interpolate(
        0.5 * fd.exp(0.5 * ((X[0] - 0.5) ** 2 + (X[1] - 0.5) ** 2 - 0.1) ** 2 - 1.0)
    )
    gt_grids = rollout_ground_truth_on_grid(stepper, u0, 3, grid)
    assert len(gt_grids) == 3
    assert gt_grids[0].shape == (9, 9)
    # fake pred
    pred = np.random.randn(3, 81, 4).astype(np.float32)
    m = gt_error_metrics(pred, gt_grids)
    assert "rel_rmse_mean" in m
    assert np.isfinite(m["linf_max"])
    _ok("gt_metrics")


def test_experiment_utils():
    from experiment_utils import set_seed, make_exp_dir, save_checkpoint
    import tempfile, os
    set_seed(42)
    with tempfile.TemporaryDirectory() as d:
        ed = make_exp_dir(d, "test_pde", {"dt": 0.1, "seed": 42})
        assert os.path.isdir(ed)
        assert os.path.isfile(os.path.join(ed, "config.json"))
        save_checkpoint(torch.nn.Linear(2, 2), None, 1, [1.0, 0.5],
                        os.path.join(ed, "ckpt.pt"))
        assert os.path.isfile(os.path.join(ed, "ckpt.pt"))
    _ok("experiment_utils")


if __name__ == "__main__":
    set_seed(0)
    tests = [
        ("experiment_utils", test_experiment_utils),
        ("gt_metrics", test_gt_metrics),
        ("diffusion", test_diffusion),
        ("advection", test_advection),
        ("poisson", test_poisson),
    ]
    results = {}
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            results[name] = True
        except Exception as e:
            _fail(name, e)
            results[name] = False

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if not all(results.values()):
        print("\nSome tests FAILED.")
        sys.exit(1)
    print("\nAll tests PASSED.")
    sys.exit(0)
