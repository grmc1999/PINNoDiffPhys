#!/usr/bin/env python3
"""Generate experiment configs and submit PINNoDiffPhys jobs to clusters.

Usage:
    python submit_wave1.py --dry-run        # generate configs only
    python submit_wave1.py                  # generate + submit
    python submit_wave1.py --cluster ICA    # submit to ICA only
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

# ── experiment grid ──────────────────────────────────────────────
EXPERIMENTS = {
    "diffusion": {
        "script": "Train_test_diffusion.py",
        "pde": "diffusion",
        "dt": 0.1,
        "seeds": [0, 1, 2],
        "train_grid_ns": [11, 16, 21],
        "extra_args": {},
        "cluster": "SD2_h100",
    },
    "advection": {
        "script": "Train_test_advection.py",
        "pde": "advection",
        "dt": 0.01,
        "seeds": [0, 1, 2],
        "train_grid_ns": [11, 16, 21],
        "extra_args": {},
        "cluster": "SD2_h100",
    },
    "poisson": {
        "script": "Train_test_poisson.py",
        "pde": "poisson",
        "m_iters": 5,
        "seeds": [0, 1, 2],
        "train_grid_ns": [11, 16, 21],
        "extra_args": {"forcing": 1.0},
        "cluster": "ICA",
    },
}

# ── cluster connection info ──────────────────────────────────────
CLUSTERS = {
    "SD2_h100": {
        "host": "146.134.176.5",
        "user": "guillermo.carrillo",
        "srm": "srm_routines/PINNoDiffPhys_train_SD2_h100.srm",
    },
    "SD2_gh200": {
        "host": "146.134.176.5",
        "user": "guillermo.carrillo",
        "srm": "srm_routines/PINNoDiffPhys_infer_SD2_gh200.srm",
    },
    "ICA": {
        "host": "139.82.152.10",
        "user": "gmorenoc",
        "srm": "srm_routines/PINNoDiffPhys_ICA_cpu.srm",
    },
}

DEFAULTS = {
    "n_epochs": 100,
    "batch_size": 16,
    "num_rollout": 10,
    "base_output_dir": "EXPS",
    "spatial_test_n": 41,
    "temporal_refinement": 4,
    "extrapolation_factor": 2.0,
}


def make_config(pde_name, seed, grid_n, overrides=None):
    cfg = {
        "pde": pde_name,
        "seed": seed,
        "train_grid_n": grid_n,
        **DEFAULTS,
    }
    pde_cfg = EXPERIMENTS[pde_name]
    if "dt" in pde_cfg:
        cfg["dt"] = pde_cfg["dt"]
    if "m_iters" in pde_cfg:
        cfg["m_iters"] = pde_cfg["m_iters"]
    if "extra_args" in pde_cfg:
        cfg.update(pde_cfg["extra_args"])
    if overrides:
        cfg.update(overrides)
    return cfg


def exp_name(pde_name, seed, grid_n):
    return f"{pde_name}_grid{grid_n}_seed{seed}"


def create_experiment(base_dir, pde_name, seed, grid_n, overrides=None):
    name = exp_name(pde_name, seed, grid_n)
    exp_dir = os.path.join(base_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    cfg = make_config(pde_name, seed, grid_n, overrides)
    cfg_path = os.path.join(exp_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return exp_dir, cfg


def submit_job(cluster_name, exp_dir):
    cl = CLUSTERS[cluster_name]
    ssh_target = f"{cl['user']}@{cl['host']}"
    srm = cl["srm"]

    cmd = (
        f'cd {os.path.basename(os.getcwd())} && '
        f'sbatch --output={exp_dir}/log.out {srm} {exp_dir} {cluster_name}'
    )
    # Use ssh to submit
    full_cmd = ["ssh", "-o", "ConnectTimeout=10", ssh_target, cmd]
    print(f"  Submitting: ssh {ssh_target} '{cmd}'")
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"    -> job {job_id}")
            return job_id
        else:
            print(f"    -> FAILED: {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        print("    -> TIMEOUT")
        return None


def main():
    parser = argparse.ArgumentParser(description="Submit PINNoDiffPhys wave 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs only, don't submit")
    parser.add_argument("--cluster", type=str, default=None,
                        help="Submit to specific cluster only")
    parser.add_argument("--base_output_dir", type=str, default="EXPS")
    parser.add_argument("--pde", type=str, default=None,
                        help="Submit specific PDE only (diffusion/advection/poisson)")
    args = parser.parse_args()

    os.makedirs(args.base_output_dir, exist_ok=True)
    tracking = {}

    for pde_name, pde_cfg in EXPERIMENTS.items():
        if args.pde and pde_name != args.pde:
            continue
        cluster = pde_cfg["cluster"]
        if args.cluster and cluster != args.cluster:
            continue

        print(f"\n=== {pde_name} (cluster: {cluster}) ===")
        for seed in pde_cfg["seeds"]:
            for grid_n in pde_cfg["train_grid_ns"]:
                exp_dir, cfg = create_experiment(
                    args.base_output_dir, pde_name, seed, grid_n
                )
                print(f"  Config: {exp_dir}/config.json")

                if not args.dry_run:
                    job_id = submit_job(cluster, exp_dir)
                    tracking[exp_name(pde_name, seed, grid_n)] = {
                        "cluster": cluster,
                        "job_id": job_id,
                        "config": cfg,
                    }

    # Save tracking
    tracking_path = os.path.join(args.base_output_dir, "tracking.json")
    with open(tracking_path, "w") as f:
        json.dump(tracking, f, indent=2)
    print(f"\nTracking saved to {tracking_path}")

    if args.dry_run:
        print("\n[DRY RUN] Configs generated. Remove --dry-run to submit.")
    else:
        n_jobs = sum(1 for v in tracking.values() if v["job_id"])
        n_fail = sum(1 for v in tracking.values() if not v["job_id"])
        print(f"\nSubmitted {n_jobs} jobs ({n_fail} failed)")


if __name__ == "__main__":
    main()
