#!/usr/bin/env python3
"""Probe 1a: verify X-coordinate layout from DG0-on-VOM interpolation.

The hypothesis: `to_torch(Function(Vx).interpolate(SpatialCoordinate(vom)))
    .reshape(eval_shape)` produces X[i,j] = (x_j, y_i) — matching row-major
point order where row = y-index, col = x-index.

We test this against two alternatives:
  A) X[i,j] == (x_j, y_i)  — row-major, consistent with grid + u channel
  B) X[i,j] == (x_i, y_j)  — transposed (flipped across main diagonal)
"""
import numpy as np
import firedrake as fd


def make_point_grid(n):
    """Exact copy of Train_test_advection.make_point_grid."""
    return np.stack(
        np.meshgrid(*tuple(np.linspace(0.0, 1.0, n) for _ in range(2)),
                     indexing="xy"),
        axis=-1,
    )


def check_layout(n, label, mesh):
    grid = make_point_grid(n)
    vom = fd.VertexOnlyMesh(mesh, grid.reshape(-1, 2), reorder=False)
    Vx = fd.VectorFunctionSpace(vom, "DG", 0)
    X = fd.assemble(fd.interpolate(fd.SpatialCoordinate(vom), Vx)).dat.data_ro

    print(f"\n{'='*50}")
    print(f" {label}  ({n}x{n},  {X.shape[0]} VOM points)")
    print(f"{'='*50}")
    print(f"grid shape       : {grid.shape}")
    print(f"VOM data shape   : {X.shape}")
    X2 = X.reshape(grid.shape)
    print(f"reshaped X shape : {X2.shape}")

    row_match = np.allclose(X2[:, :, 0], grid[:, :, 0]) and \
                np.allclose(X2[:, :, 1], grid[:, :, 1])
    transp_match = np.allclose(X2[:, :, 0], grid[:, :, 1]) and \
                   np.allclose(X2[:, :, 1], grid[:, :, 0])

    print(f"\nX[0,0]     = ({X2[0,0,0]:.6f}, {X2[0,0,1]:.6f})"
          f"   grid[0,0]     = ({grid[0,0,0]:.6f}, {grid[0,0,1]:.6f})")
    print(f"X[0,1]     = ({X2[0,1,0]:.6f}, {X2[0,1,1]:.6f})"
          f"   grid[0,1]     = ({grid[0,1,0]:.6f}, {grid[0,1,1]:.6f})")
    print(f"X[1,0]     = ({X2[1,0,0]:.6f}, {X2[1,0,1]:.6f})"
          f"   grid[1,0]     = ({grid[1,0,0]:.6f}, {grid[1,0,1]:.6f})")
    print(f"X[0,-1]    = ({X2[0,-1,0]:.6f}, {X2[0,-1,1]:.6f})"
          f"   grid[0,-1]    = ({grid[0,-1,0]:.6f}, {grid[0,-1,1]:.6f})")
    print(f"X[-1,0]    = ({X2[-1,0,0]:.6f}, {X2[-1,0,1]:.6f})"
          f"   grid[-1,0]    = ({grid[-1,0,0]:.6f}, {grid[-1,0,1]:.6f})")
    print(f"X[-1,-1]   = ({X2[-1,-1,0]:.6f}, {X2[-1,-1,1]:.6f})"
          f"   grid[-1,-1]   = ({grid[-1,-1,0]:.6f}, {grid[-1,-1,1]:.6f})")

    print(f"\nX-x-coord row 0 : {np.round(X2[0,:,0], 4).tolist()}")
    print(f"grid x row 0    : {np.round(grid[0,:,0], 4).tolist()}")
    print(f"X-y-coord col 0 : {np.round(X2[:,0,1], 4).tolist()}")
    print(f"grid y col 0    : {np.round(grid[:,0,1], 4).tolist()}")

    print(f"\nVERDICT:  row-major match = {row_match}  |  transposed match = {transp_match}")
    if row_match:
        print("  => X[i,j] = (x_j, y_i)  — CONSISTENT with row-major point order")
    elif transp_match:
        print("  => X[i,j] = (x_i, y_j)  — TRANSPOSED relative to point order")
    else:
        print("  => NEITHER simple hypothesis holds — inspect output above")

    return row_match, transp_match


if __name__ == "__main__":
    mesh = fd.UnitSquareMesh(10, 10)
    print(f"Advection mesh: UnitSquareMesh(10,10), {mesh.num_vertices()} vertices, "
          f"{mesh.num_cells()} cells")

    ok11, tr11 = check_layout(11, "COARSE (advection train grid)", mesh)
    ok41, tr41 = check_layout(41, "FINE (advection posterior grid)", mesh)

    print("\n" + "="*50)
    print(" SUMMARY")
    print(f"{'='*50}")
    print(f"  Coarse (11x11) :  row-major={ok11}, transposed={tr11}")
    print(f"  Fine   (41x41) :  row-major={ok41}, transposed={tr41}")
    if ok11 and ok41:
        print("\n  All checks passed — X layout is row-major consistent on both grids.")
    elif tr11 or tr41:
        print("\n  One or both grids have TRANSPOSED X layout — likely root cause of flipped images.")
    else:
        print("\n  Mixed or unexpected results — investigate further.")
