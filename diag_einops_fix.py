#!/usr/bin/env python3
"""Probe 1f: which einops pattern restores VOM point order?

Confirmed: `rearrange(feats_t, "V x y t -> t (x y) V")` produces a
TRANSPOSED flat order vs point grid (== gt.T).
Test alternative patterns to find the one producing == gt on both grids:
   P1: "V x y t -> t (x y) V"    (current, expects transposed)
   P2: "V x y t -> t (y x) V"    (swap paren order)
   P3: "V y x t -> t (x y) V"    (swap input spatial axes)
"""
import numpy as np
import torch
import firedrake as fd
from einops import rearrange
from firedrake.ml.pytorch.fem_operator import to_torch


def make_point_grid(n):
    return np.stack(
        np.meshgrid(*tuple(np.linspace(0.0, 1.0, n) for _ in range(2)),
                     indexing="xy"),
        axis=-1,
    )


def feature_builder_finer(u, t, eval_points, fs):
    u = u.reshape(eval_points.shape[:-1] + (1,))
    V = fd.VectorFunctionSpace(fs.mesh(), "DG", 0)
    X = to_torch(fd.Function(V).interpolate(fd.SpatialCoordinate(fs.mesh())))
    X = X.reshape(eval_points.shape)
    t = torch.tile(torch.tensor(t), (eval_points.shape[:2]) + (1,))
    return torch.concat((X, t, u), axis=-1).transpose(0, -1).float()


PATTERNS = [
    ("P1 current", "V x y t -> t (x y) V"),
    ("P2 paren-swap", "V x y t -> t (y x) V"),
    ("P3 axis-swap", "V y x t -> t (x y) V"),
]


def check(n, label, mesh, u_fd):
    grid = make_point_grid(n)
    vom = fd.VertexOnlyMesh(mesh, grid.reshape(-1, 2), reorder=False)
    P0DG = fd.FunctionSpace(vom, "DG", 0)

    up = fd.assemble(fd.interpolate(u_fd, P0DG)).dat.data_ro
    gt = up.reshape(grid.shape[:2])

    print(f"\n{'='*60}\n {label} ({n}x{n})\n{'='*60}")
    u_tensor = to_torch(fd.assemble(fd.interpolate(u_fd, P0DG))).requires_grad_(True)
    feats = feature_builder_finer(u_tensor, 0.01, grid, P0DG)  # (4,H,W)
    feats_t = torch.stack([feats], axis=-1)                    # (4,H,W,T)

    for name, pattern in PATTERNS:
        try:
            u_sol = rearrange(feats_t, pattern)                # (T,P,V)
            u_ein = u_sol[0, :, -1].detach().cpu().numpy().reshape(grid.shape[:2])
        except Exception as e:
            print(f"  {name:14s} ERROR: {e}")
            continue
        m = np.allclose(u_ein, gt)
        t = np.allclose(u_ein, gt.T)
        print(f"  {name:14s} == gt?: {m}   == gt.T?: {t}   -> {'OK' if m else ('TRANSPOSED' if t else 'OTHER')}")


def main():
    mesh = fd.UnitSquareMesh(10, 10)
    X, Y = fd.SpatialCoordinate(mesh)
    u_fd = fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(0.3 * X + 0.7 * Y)
    print("field = 0.3*x + 0.7*y  (asymmetric)\n")
    check(11, "COARSE", mesh, u_fd)
    check(41, "FINE", mesh, u_fd)


if __name__ == "__main__":
    main()