#!/usr/bin/env python3
"""Probe 1c: bisect the flip — plumbing vs learned weights.

Runs the SAME fine-branch predict_rollout path with the CNN weights ZEROED.
If  zero-pred == gt      -> the plumbing (from_torch/interpolate/einops/observe)
                             preserves orientation; the transpose lives in the
                             LEARNED weights (model learned to output gt.T).
If  zero-pred == gt.T    -> a primitive in the default pipeline flips the axes
                             (NOT the CNN); the learned model is consistent with
                             the plumbing and simply learns what it sees.
"""
import json
import os

import numpy as np
import torch
import firedrake as fd

from Train_test_advection import make_point_grid, make_ic, build_trainer
from experiment_utils import (
    rollout_ground_truth_on_grid,
    grids_from_prediction_list,
    gt_error_metrics,
    set_seed,
)
from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model


EXP_DIR = os.environ.get("EXP_DIR", "/code/EXPS/advection_grid16_seed0")
N_STEPS = int(os.environ.get("N_STEPS", "10"))


def zero_model(model):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()


def run_bisect(trainer, u0, n_steps, grid, label):
    pred, _, _, _, _ = trainer.predict_rollout(
        u0, t0=0.0, n_steps=n_steps, spatial_sample=grid)
    pred_grids = grids_from_prediction_list(pred[:, :, [-1]], grid.shape[:2])
    gt_grids = rollout_ground_truth_on_grid(
        trainer.physical_model, u0, n_steps, grid)

    print("\n" + "="*68)
    print(f" {label}")
    print("="*68)
    print(f"{'step':>4} | {'rel_rmse(zero,gt)':>17} | {'rel_rmse(zero,gt.T)':>19} | {'winner':>12}")
    print("-"*68)
    winners = []
    for i in range(n_steps):
        mg   = gt_error_metrics(pred[i:i+1, :, -1:], [gt_grids[i]])
        mg_t = gt_error_metrics(pred[i:i+1, :, -1:], [gt_grids[i].T])
        r1, r2 = mg["rel_rmse_mean"], mg_t["rel_rmse_mean"]
        w = "zero==gt" if r1 <= r2 else "zero==gt.T"
        winners.append(w)
        print(f"{i+1:>4} | {r1:>17.6f} | {r2:>19.6f} | {w:>12}")
    overall = ("zero==gt" if winners.count("zero==gt") >= n_steps//2
               else "zero==gt.T")
    print(f"\n  >>> {label} VERDICT: {overall}")
    return overall


def main():
    with open(os.path.join(EXP_DIR, "config.json")) as f:
        cfg = json.load(f)
    set_seed(cfg.get("seed", 0))

    print(f"EXP_DIR = {EXP_DIR}  config dt={cfg['dt']} train_grid_n={cfg['train_grid_n']}"
          f" spatial_test_n={cfg.get('spatial_test_n',41)} seed={cfg.get('seed',0)}")
    print(f"N_STEPS = {N_STEPS}\n")

    mesh = fd.UnitSquareMesh(10, 10)
    train_grid = make_point_grid(cfg["train_grid_n"])
    fine_grid = make_point_grid(cfg.get("spatial_test_n", 41))

    st_model = simple_dual_space_with_time_derivative_cnn_model()
    zero_model(st_model)

    trainer = build_trainer(
        mesh=mesh, point_grid=train_grid, dt=cfg["dt"],
        simulation_steps=N_STEPS, st_model=st_model, lr=0.0,
    )
    u0 = make_ic(trainer.physical_model.V)

    # Sanity: zero-CNN, single-step on train grid; also confirm model actually zero.
    tot = sum(p.abs().sum().item() for p in st_model.parameters())
    print(f"zeroed model total |w| = {tot:.6f} (expect 0)")

    v_fine = run_bisect(trainer, u0, N_STEPS, fine_grid,
                        "FINE GRID  (zeroed CNN)")
    v_train = run_bisect(trainer, u0, N_STEPS, train_grid,
                         "TRAIN GRID (zeroed CNN)")

    print("\n" + "="*68)
    if v_fine == "zero==gt" and v_train == "zero==gt":
        print(" PLUMBING IS ORIENTATION-CORRECT."
              "\n => The transpose is embedded in the LEARNED weights:"
              "\n    during training the features must have been laid out"
              "\n    transposed relative to this evaluation path (or the"
              "\n    observation/lift operators deliver transposed gradients),"
              "\n    so the CNN learned the transposed association that"
              "\n    reproduces it at eval.")
    else:
        print(" PLUMBING FLIPS THE AXES (zero-CNN output already == gt.T)."
              "\n => A primitive in the default pipeline (from_torch /")
        print("    interpolate / observe / einops) transposes the field;"
              "\n    the trained CNN is merely consistent with the plumbing.")


if __name__ == "__main__":
    main()