#!/usr/bin/env python3
"""Probe 1d: is fd.ml.pytorch.to_torch ordering the culprit?

feature_builder / feature_builder_finer build BOTH the X channel and the u
channel via `fd.ml.pytorch.to_torch` and then `.reshape(grid_shape)`.
Probe 1a verified that the *interpolate*.dat.data_ro ordering is row-major;
this probe checks whether the ACTUAL `to_torch(...)` tensors used in the real
pipeline have the same ordering or are TRANSPOSED relative to the point grid.
"""
import numpy as np
import firedrake as fd
from firedrake.ml.pytorch.fem_operator import fem_operator, to_torch


def make_point_grid(n):
    return np.stack(
        np.meshgrid(*tuple(np.linspace(0.0, 1.0, n) for _ in range(2)),
                     indexing="xy"),
        axis=-1,
    )


def check(n, label, mesh, field_expr):
    grid = make_point_grid(n)
    vom = fd.VertexOnlyMesh(mesh, grid.reshape(-1, 2), reorder=False)
    Vx = fd.VectorFunctionSpace(vom, "DG", 0)
    P0DG = fd.FunctionSpace(vom, "DG", 0)

    print(f"\n{'='*60}")
    print(f" {label}  ({n}x{n})")
    print(f"{'='*60}")

    # --- X channel: EXACT feature_builder construction (to_torch) ---
    X_func = fd.Function(Vx).interpolate(fd.SpatialCoordinate(vom))
    X_tt = to_torch(X_func).reshape(grid.shape)
    row_match = np.allclose(X_tt[:, :, 0], grid[:, :, 0]) and \
                np.allclose(X_tt[:, :, 1], grid[:, :, 1])
    trans_match = np.allclose(X_tt[:, :, 0], grid[:, :, 1]) and \
                  np.allclose(X_tt[:, :, 1], grid[:, :, 0])
    print(f"X<to_torch> shape      : {tuple(X_tt.shape)}")
    print(f"X<to_torch>[0,0]       : ({X_tt[0,0,0]:.6f}, {X_tt[0,0,1]:.6f})")
    print(f"X<to_torch>[0,1]       : ({X_tt[0,1,0]:.6f}, {X_tt[0,1,1]:.6f})")
    print(f"X<to_torch>[1,0]       : ({X_tt[1,0,0]:.6f}, {X_tt[1,0,1]:.6f})")
    print(f"X<to_torch> row-major match = {row_match} | transposed match = {trans_match}")

    # --- u channel: EXACT feature_builder construction (to_torch(assemble(interp))) ---
    u_fd = fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(field_expr)
    u_points = fd.assemble(fd.interpolate(u_fd, P0DG))
    u_tt = to_torch(u_points).reshape(grid.shape[:2])
    u_dat = u_points.dat.data_ro.reshape(grid.shape[:2])

    gt = fd.assemble(fd.interpolate(
        fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(field_expr),
        fd.FunctionSpace(vom, "DG", 0))).dat.data_ro.reshape(grid.shape[:2])

    u_tt_match = np.allclose(u_tt, gt)
    u_tt_t_match = np.allclose(u_tt, gt.T)
    u_dat_match = np.allclose(u_dat, gt)
    print(f"u<to_torch> == gt        : {u_tt_match}")
    print(f"u<to_torch> == gt.T      : {u_tt_t_match}")
    print(f"u<dat.data_ro> == gt     : {u_dat_match}   (reference path, probe 1a style)")

    # corner printout
    print(f"u<to_torch> corner [2:5,2:5]:")
    print(np.array2string(np.round(u_tt[2:5, 2:5], 6)))
    print(f"gt corner [2:5,2:5] (data_ro):")
    print(np.array2string(np.round(gt[2:5, 2:5], 6)))


def main():
    mesh = fd.UnitSquareMesh(10, 10)
    X, Y = fd.SpatialCoordinate(mesh)
    field_expr = 0.3 * X + 0.7 * Y   # asymmetric: u(x,y)=0.3x+0.7y

    print("field_expr = 0.3*x + 0.7*y   (asymmetric, must NOT look same under transpose)")
    check(11, "COARSE (train grid)", mesh, field_expr)
    check(41, "FINE (posterior grid)", mesh, field_expr)


if __name__ == "__main__":
    main()