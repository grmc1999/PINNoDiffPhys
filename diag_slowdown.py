import argparse
import time

import numpy as np
import torch
import firedrake as fd
from pyadjoint import get_working_tape

from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
from trainer.Trainer import ImplicitDiffusionStepper, FiredrakePINNSBasedSOLTrainerCNN
from DL_models.PINNS.Residual_losses import diffusion_loss
from experiment_utils import set_seed
from Train_test_diffusion import make_point_grid, make_ic


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return -1


def build_trainer(grid_n, seed):
    set_seed(seed)
    mesh = fd.UnitSquareMesh(10, 10)
    train_grid = make_point_grid(grid_n)
    ph = ImplicitDiffusionStepper(mesh=mesh, dt=0.1, point_evaluator=train_grid)
    st = simple_dual_space_with_time_derivative_cnn_model()
    tr = FiredrakePINNSBasedSOLTrainerCNN(
        physical_model=ph, statistical_model=st,
        optimizer=torch.optim.Adam(st.parameters(), lr=1e-4),
        simulation_steps=5, dt=0.1,
        loss=lambda u, x: (diffusion_loss(u, x, K=1.0)) ** 2)
    u0 = make_ic(ph.V)
    tr.generate_ground_truth(u0, 10)
    return tr


def run_phase(name, trainer, epochs):
    print(f"--- {name} ---")
    for e in range(epochs):
        t = time.perf_counter()
        trainer.train(epochs=1, batch_size=8)
        dt = time.perf_counter() - t
        tape = len(get_working_tape().get_blocks())
        print(f"  epoch {e+1}: {dt:6.2f}s  tape_blocks={tape:6d}  rss={rss_mb()}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_n", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    trainer = build_trainer(args.grid_n, seed=0)
    run_phase("CURRENT (as shipped)", trainer, args.epochs)

    fresh = build_trainer(args.grid_n, seed=1)
    fd.adjoint.stop_annotating()
    run_phase("GUARDED (stop_annotating)", fresh, args.epochs)
    fd.adjoint.continue_annotation()


if __name__ == "__main__":
    main()