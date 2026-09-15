#!/usr/bin/env python3
"""Probe 1g (v3): pick the einops strings so that
  (i)   flattening a TRUE-physical (c,H,W) grid yields VOM point-order flat,
  (ii)  expanding a point-order flat yields a TRUE-physical (c,H,W) image,
  (iii) flatten(C) and expand(C) are exact inverses for candidate C.

Anchor (1f): on "V x y t" (true phys, axis1=row[y], axis2=col[x]),
  "(x y)" -> gt.T ;  "(y x)" -> gt. This script maps that onto the exact
  "(c,h,w)" / "(c,w,h)" strings used at Trainer.py lines 700/701/727 and the
  fine-branch "(V,x,y,t)" strings at 815/818/822.
"""
import numpy as np
import torch
import firedrake as fd
from einops import rearrange


def make_point_grid(n):
    return np.stack(
        np.meshgrid(*tuple(np.linspace(0.0, 1.0, n) for _ in range(2)),
                     indexing="xy"),
        axis=-1,
    )


def main():
    mesh = fd.UnitSquareMesh(10, 10)
    X, Y = fd.SpatialCoordinate(mesh)
    u_fd = fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(0.3 * X + 0.7 * Y)

    for n in (11, 41):
        grid = make_point_grid(n)
        vom = fd.VertexOnlyMesh(mesh, grid.reshape(-1, 2), reorder=False)
        P0DG = fd.FunctionSpace(vom, "DG", 0)
        Vc = fd.VectorFunctionSpace(vom, "DG", 0)
        up = torch.tensor(fd.assemble(fd.interpolate(u_fd, P0DG)).dat.data_ro).float()
        gt = up.numpy().reshape((n, n))  # VOM point order, row-major
        Xp = fd.Function(Vc).interpolate(fd.SpatialCoordinate(vom))
        Xt = torch.stack([torch.tensor(Xp.dat.data_ro[:, i]) for i in range(2)], -1).float()
        tch = torch.zeros_like(up)[:, None] + 0.01
        ue = up[:, None]

        # TRUE-physical (c,H,W) block: pixel (r,c) holds (x_c, y_r, t, u(x_c,y_r))
        feats_tp = torch.concat((Xt, tch, ue), -1).reshape((n, n, 4)).transpose(0, -1).float().contiguous()
        # POINT-ORDER flat: k = r*n + c
        flat_po = torch.concat((Xt, tch, ue), -1).float().contiguous()  # (n*n,4)

        print(f"\n=== {n}x{n} ===")
        ok_c = []
        for inp in ('c h w', 'c w h'):
            grid_pat = f"1 (h w) c -> {inp}" if '%' not in inp else None
            # flatten candidates enumerated on the (true-physical) grid
        cands = {
            'c h w -> 1 (h w) c': lambda f: rearrange(f, "c h w -> 1 (h w) c"),
            'c h w -> 1 (w h) c': lambda f: rearrange(f, "c h w -> 1 (w h) c"),
            'c w h -> 1 (h w) c': lambda f: rearrange(f, "c w h -> 1 (h w) c"),
            'c w h -> 1 (w h) c': lambda f: rearrange(f, "c w h -> 1 (w h) c"),
        }
        print("  [A] flatten TRUE-physical (c,H,W) -> flat; is flat == point-order?")
        for pat, fn in cands.items():
            fl = fn(feats_tp)
            u_flat = fl[0, :, -1].numpy()
            r = u_flat.reshape((n, n))
            m = np.allclose(r, gt)
            t = np.allclose(r, gt.T)
            note = 'OK' if m else ('TRANSPOSED' if t else 'OTHER')
            print(f"    {pat:34s} reshape==gt:{m}  ==gt.T:{t}  [{note}]")
            if m:
                ok_c.append(pat)

        print("  [B] expand point-order flat -> img; is img TRUE-physical?")
        # try every expand string; check X corners of the rebuilt image
        for pat in ('1 (h w) c -> c h w', '1 (h w) c -> c w h',
                    '1 (w h) c -> c h w', '1 (w h) c -> c w h'):
            try:
                img = rearrange(flat_po[None], pat, h=n, w=n)
                # corner pixels of channel 0 (X-coord): (0,0)=x0, (0,W-1)=x_last
                x00 = img[0, 0, 0].item()
                x0W = img[0, 0, -1].item()
                xH0 = img[0, -1, 0].item()
                true_phys = (abs(x00 - 0.0) < 1e-6 and abs(x0W - 1.0) < 1e-6
                             and abs(xH0 - 0.0) < 1e-6)
                print(f"    {pat:28s} X@(0,0)={x00:.4f} X@(0,W-1)={x0W:.4f} "
                      f"X@(H-1,0)={xH0:.4f}  true-phys:{true_phys}")
            except Exception as e:
                print(f"    {pat:28s} FAILED {e}")

        print("  [C] roundtrip with the OK flatten: expand(inverse) then flatten == point order")
        if ok_c:
            pat = ok_c[0]
            print(f"    winning flatten: {pat}")
            fn = cands[pat]
            inv = {
                'c h w -> 1 (h w) c': '1 (h w) c -> c h w',
                'c h w -> 1 (w h) c': '1 (w h) c -> c h w',
                'c w h -> 1 (h w) c': '1 (h w) c -> c w h',
                'c w h -> 1 (w h) c': '1 (w h) c -> c w h',
            }[pat]
            img = rearrange(flat_po[None], inv, h=n, w=n)
            back = rearrange(img, pat)
            u_flat = back[0, :, -1].numpy().reshape((n, n))
            print(f"    inverse {inv:28s} roundtrip reshape==gt: "
                  f"{np.allclose(u_flat, gt)}")


if __name__ == "__main__":
    main()