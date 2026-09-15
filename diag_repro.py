#!/usr/bin/env python3
"""Probe 1b: reproduce predict_rollout from a saved checkpoint and compare
pred vs gt vs gt.T to determine the flip direction.

Uses the same code path that produces solution_epoch_*.png and posterior figures.
"""
import json
import os
import sys

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


def main():
    with open(os.path.join(EXP_DIR, "config.json")) as f:
        cfg = json.load(f)

    set_seed(cfg.get("seed", 0))

    print(f"EXP_DIR   = {EXP_DIR}")
    print(f"config    = dt={cfg['dt']}, train_grid_n={cfg['train_grid_n']}, "
          f"spatial_test_n={cfg.get('spatial_test_n',41)}, seed={cfg.get('seed',0)}")
    print(f"N_STEPS   = {N_STEPS}")

    mesh = fd.UnitSquareMesh(10, 10)
    train_grid = make_point_grid(cfg["train_grid_n"])
    fine_grid = make_point_grid(cfg.get("spatial_test_n", 41))

    st_model = simple_dual_space_with_time_derivative_cnn_model()
    ckpt_path = os.path.join(EXP_DIR, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st_model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded checkpoint epoch={ckpt.get('epoch')} from {ckpt_path}")

    trainer = build_trainer(
        mesh=mesh,
        point_grid=train_grid,
        dt=cfg["dt"],
        simulation_steps=N_STEPS,
        st_model=st_model,
        lr=0.0,
    )
    u0 = make_ic(trainer.physical_model.V)

    # ------------------------------------------------------------------
    # A) Fine grid (spatial interpolation — the summary posterior path)
    # ------------------------------------------------------------------
    pred_f, _, _, times_f, _ = trainer.predict_rollout(
        u0, t0=0.0, n_steps=N_STEPS, spatial_sample=fine_grid)
    pred_grids_f = grids_from_prediction_list(pred_f[:, :, [-1]], fine_grid.shape[:2])
    gt_grids_f = rollout_ground_truth_on_grid(
        trainer.physical_model, u0, N_STEPS, fine_grid)

    print("\n" + "="*68)
    print(" FINE GRID  (spatial interpolation — what the user sees)")
    print("="*68)
    print(f"{'step':>4} | {'rel_rmse(pred,gt)':>17} | {'rel_rmse(pred,gt.T)':>19} | {'winner':>12}")
    print("-"*68)
    winner_f = []
    for i in range(len(pred_grids_f)):
        mg   = gt_error_metrics(pred_f[i:i+1, :, -1:], [gt_grids_f[i]])
        mg_t = gt_error_metrics(pred_f[i:i+1, :, -1:], [gt_grids_f[i].T])
        r1 = mg["rel_rmse_mean"]
        r2 = mg_t["rel_rmse_mean"]
        w = "pred==gt" if r1 <= r2 else "pred==gt.T"
        winner_f.append(w)
        print(f"{i+1:>4} | {r1:>17.6f} | {r2:>19.6f} | {w:>12}")

    overall = "pred==gt" if sum(v == "pred==gt" for v in winner_f) >= len(winner_f)//2 else "pred==gt.T"
    print(f"\n  >>> FINE GRID VERDICT:  {overall}")

    # ------------------------------------------------------------------
    # B) Train grid (solution_epoch_*.png path)
    # ------------------------------------------------------------------
    pred_t, _, _, times_t, _ = trainer.predict_rollout(
        u0, t0=0.0, n_steps=N_STEPS, spatial_sample=train_grid)
    pred_grids_t = grids_from_prediction_list(pred_t[:, :, [-1]], train_grid.shape[:2])
    gt_grids_t = rollout_ground_truth_on_grid(
        trainer.physical_model, u0, N_STEPS, train_grid)

    print("\n" + "="*68)
    print(" TRAIN GRID  (solution_epoch_*.png path)")
    print("="*68)
    print(f"{'step':>4} | {'rel_rmse(pred,gt)':>17} | {'rel_rmse(pred,gt.T)':>19} | {'winner':>12}")
    print("-"*68)
    winner_t = []
    for i in range(len(pred_grids_t)):
        mg   = gt_error_metrics(pred_t[i:i+1, :, -1:], [gt_grids_t[i]])
        mg_t = gt_error_metrics(pred_t[i:i+1, :, -1:], [gt_grids_t[i].T])
        r1 = mg["rel_rmse_mean"]
        r2 = mg_t["rel_rmse_mean"]
        w = "pred==gt" if r1 <= r2 else "pred==gt.T"
        winner_t.append(w)
        print(f"{i+1:>4} | {r1:>17.6f} | {r2:>19.6f} | {w:>12}")

    overall_t = "pred==gt" if sum(v == "pred==gt" for v in winner_t) >= len(winner_t)//2 else "pred==gt.T"
    print(f"\n  >>> TRAIN GRID VERDICT:  {overall_t}")

    # ------------------------------------------------------------------
    # C) Corner peek for manual inspection
    # ------------------------------------------------------------------
    print("\n" + "="*68)
    print(" CORNER PEEK  (step 0)")
    print("="*68)
    print(f"pred[0, :3, :3] = {repr(pred_grids_f[0][:3, :3])}")
    print(f"gt  [0, :3, :3] = {repr(gt_grids_f[0][:3, :3])}")
    print(f"gt.T[:3, :3]    = {repr(gt_grids_f[0].T[:3, :3])}")
    print(f"\nabs-mean(pred-gt)    = {np.abs(pred_grids_f[0] - gt_grids_f[0]).mean():.6f}")
    print(f"abs-mean(pred-gt.T)  = {np.abs(pred_grids_f[0] - gt_grids_f[0].T).mean():.6f}")

    print("\n" + "="*68)
    print(" OVERALL VERDICT")
    print("="*68)
    print(f"  Fine grid  : {overall}")
    print(f"  Train grid : {overall_t}")


if __name__ == "__main__":
    main()
