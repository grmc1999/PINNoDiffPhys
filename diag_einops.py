#!/usr/bin/env python3
"""Probe 1e: isolate the flatten stage — einops rearrange vs data_ro reshape.

Builds an asymmetric FE field, evaluates it on the (H,W) point grid, and
pushes it through the THREE competing pipelines to see which one disagrees:
  A) data_ro + reshape(H,W)                 (GT path)
  B) to_torch(interp) + reshape(H,W)        (probe 1d path — shown OK on coarse)
  C) feature_builder_finer + torch.stack + einops "V x y t -> t (x y) V"
     + [..., -1] + reshape(H,W)             (actual predict_rollout fine branch)
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


def check(n, label, mesh, u_fd):
    grid = make_point_grid(n)
    vom = fd.VertexOnlyMesh(mesh, grid.reshape(-1, 2), reorder=False)
    P0DG = fd.FunctionSpace(vom, "DG", 0)

    up = fd.assemble(fd.interpolate(u_fd, P0DG)).dat.data_ro  # point order

    print(f"\n{'='*60}\n {label}  ({n}x{n})\n{'='*60}")

    # A) GT path
    gt = up.reshape(grid.shape[:2])

    # B) to_torch path
    u_tt = to_torch(fd.assemble(fd.interpolate(u_fd, P0DG))).reshape(grid.shape[:2])

    # C) actual fine branch
    u_tensor = to_torch(fd.assemble(fd.interpolate(u_fd, P0DG))).requires_grad_(True)
    feats = feature_builder_finer(u_tensor, 0.01, grid, P0DG)   # (4,H,W)
    feats_t = torch.stack([feats], axis=-1)                     # (4,H,W,T=1)
    u_sol = rearrange(feats_t, "V x y t -> t (x y) V")          # (1, P, 4)
    u_ein = u_sol[0, :, -1].detach().cpu().numpy().reshape(grid.shape[:2])

    print(f"  A) gt          = up.reshape(H,W)            (reference)")
    print(f"  B) u_tt        = to_torch(...).reshape(H,W)")
    print(f"  C) u_ein       = feature_builder_finer + einops + reshape")

    b_m = np.allclose(u_tt, gt)
    b_t = np.allclose(u_tt, gt.T)
    c_m = np.allclose(u_ein, gt)
    c_t = np.allclose(u_ein, gt.T)
    print(f"\n  B == gt?: {b_m}    B == gt.T?: {b_t}")
    print(f"  C == gt?: {c_m}    C == gt.T?: {c_t}")

    print(f"\n  corner gt    [0:3,0:3] = (data_ro path):")
    print(gt[:3, :3])
    print(f"  corner u_ein [0:3,0:3] = (einops path):")
    print(u_ein[:3, :3])
    print(f"  corner u_tt  [0:3,0:3] = (to_torch path):")
    print(u_tt[:3, :3])

    return c_m, c_t


def main():
    mesh = fd.UnitSquareMesh(10, 10)
    X, Y = fd.SpatialCoordinate(mesh)
    Vmesh = fd.FunctionSpace(mesh, "DG", 0)
    u_fd = fd.Function(Vmesh).interpolate(0.3 * X + 0.7 * Y)

    print("field = 0.3*x + 0.7*y  (asymmetric)")
    c11_m, c11_t = check(11, "COARSE (11x11)", mesh, u_fd)
    c41_m, c41_t = check(41, "FINE   (41x41)", mesh, u_fd)

    print("\n" + "="*60)
    print(" SUMMARY")
    print(f"{'='*60}")
    print(f"  COARSE: einops==gt={c11_m}  einops==gt.T={c11_t}")
    print(f"  FINE  : einops==gt={c41_m}  einops==gt.T={c41_t}")


if __name__ == "__main__":
    main()