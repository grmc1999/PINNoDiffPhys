"""Tabulate hybrid vs fem vs pinn ablation results from summary.json files.

Usage:
    python3 srm_routines/compare_modes.py [EXPS_DIR] [--pdes advection diffusion poisson]
                                          [--grids 11 16 21] [--seeds 0 1 2]

Scans *EXPS_DIR*/{pde}_grid{g}_seed{s}/summary.json for each pde/grid/seed and
prints mode-vs-mode comparisons for each pairwise regime (rel_rmse_last, plus
the refined-reference variant gt_rel_rmse_last_fine and final residual).
"""
import argparse
import json
import os


REGIMES = {
    "diffusion": ["spatial_interpolation", "temporal_interpolation", "temporal_extrapolation"],
    "advection": ["spatial_interpolation", "temporal_interpolation", "temporal_extrapolation"],
    "poisson": ["spatial_interpolation", "budget_shift"],
}
LABELS = {
    "spatial_interpolation": "spatial interp",
    "temporal_interpolation": "temporal interp",
    "temporal_extrapolation": "temporal extrap",
    "budget_shift": "budget shift",
}


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def load_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps_dir", nargs="?", default="EXPS")
    ap.add_argument("--pdes", nargs="+", default=["advection", "diffusion", "poisson"])
    ap.add_argument("--grids", nargs="+", type=int, default=[11, 16, 21])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    modes = ("hybrid", "fem", "pinn")
    for pde in args.pdes:
        regimes = REGIMES.get(pde, [])
        print(f"\n## {pde}")
        for g in args.grids:
            for s in args.seeds:
                summaries = {}
                for mode in modes:
                    d = os.path.join(args.exps_dir, f"{pde}_grid{g}_seed{s}",
                                     f"mode_{mode}", "summary.json")
                    if not os.path.exists(d):
                        d = os.path.join(args.exps_dir, f"{pde}_grid{g}_seed{s}",
                                         "summary.json")
                    summaries[mode] = load_summary(d)

                print(f"\n### {pde} grid={g} seed={s}")
                for reg in regimes:
                    cols = []
                    for mode in modes:
                        s_ = summaries[mode]
                        if s_ is None or reg not in s_:
                            cols.append("mode: -")
                            continue
                        r = s_[reg]
                        coarse = fmt(r.get("gt_rel_rmse_last"))
                        fine = fmt(r.get("gt_rel_rmse_last_fine"))
                        resid = fmt(r.get("residual_last"))
                        cols.append(f"{mode}: rmse={coarse} (fine={fine}, resid={resid})")
                    print(f"  {LABELS[reg]:>16}: " + " | ".join(cols))


if __name__ == "__main__":
    main()