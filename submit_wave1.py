#!/usr/bin/env python3
"""Generate experiment configs and submit PINNoDiffPhys jobs to clusters.

Usage:
    python submit_wave1.py --dry-run        # generate configs only
    python submit_wave1.py                  # generate + submit (paramiko)
    python submit_wave1.py --cluster ICA    # submit to ICA only
    python submit_wave1.py --pde diffusion  # submit specific PDE only
"""
import argparse
import json
import os
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
# path_code = canonical deploy dir on the remote (srm_routines/set_env.sh)
CLUSTERS = {
    "SD2_h100": {
        "host": "146.134.176.5",
        "user": "guillermo.carrillo",
        "srm": "srm_routines/PINNoDiffPhys_train_SD2_h100.srm",
        "path_code": "/petrobr/parceirosbr/proxy-sim/users/guillermo.carrillo/PINNoDiffPhys",
    },
    "SD2_gh200": {
        "host": "146.134.176.5",
        "user": "guillermo.carrillo",
        "srm": "srm_routines/PINNoDiffPhys_infer_SD2_gh200.srm",
        "path_code": "/petrobr/parceirosbr/proxy-sim/users/guillermo.carrillo/PINNoDiffPhys",
    },
    "ICA": {
        "host": "139.82.152.10",
        "user": "gmorenoc",
        "srm": "srm_routines/PINNoDiffPhys_ICA_cpu.srm",
        "path_code": "/share_zeta/Proxy-Sim/guillermo.carrillo/PINNoDiffPhys",
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


def _ica_client():
    from srm_routines.ica_ssh import ICA

    return ICA(host=CLUSTERS["ICA"]["host"], user=CLUSTERS["ICA"]["user"])


def submit_ica(exp_dir, cluster_name):
    """Upload config to the ICA deploy dir and sbatch the SRM via paramiko."""
    from srm_routines.ica_ssh import ICA

    cl = CLUSTERS[cluster_name]
    path_code = cl["path_code"]

    # relative exp dir (e.g. EXPS/diffusion_grid11_seed0) on the remote
    rel_exp = exp_dir.replace("\\", "/")

    with ICA(host=cl["host"], user=cl["user"]) as ica:
        # ensure remote EXPS/<exp> and upload config.json
        ica.run(f"mkdir -p {path_code}/{rel_exp}")
        remote_cfg = f"{path_code}/{rel_exp}/config.json"
        local_cfg = os.path.join(exp_dir, "config.json")
        ica.sftp_put(local_cfg, remote_cfg)
        print(f"    -> uploaded {local_cfg} -> {remote_cfg}")

        # submit from PATH_CODE
        st, out, err = ica.run(
            f"cd {path_code} && sbatch {cl['srm']} {rel_exp} {cluster_name}",
            timeout=60,
        )
        # stdout has "Submitted batch job <id>"
        job_id = None
        for token in out.split():
            if token.isdigit():
                job_id = token
                break
        print(f"    -> job {job_id}" if job_id else f"    -> ??? {out} {err}")
        return job_id


def submit_job(cluster_name, exp_dir):
    cl = CLUSTERS[cluster_name]
    if cluster_name == "ICA":
        return submit_ica(exp_dir, cluster_name)
    # Non-ICA clusters not wired to paramiko yet.
    print(f"    -> SKIP: cluster '{cluster_name}' not yet wired to paramiko")
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
    parser.add_argument("--force-cluster", type=str, default=None,
                        help="Override the target cluster for every selected experiment")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override n_epochs in every generated config")
    args = parser.parse_args()

    os.makedirs(args.base_output_dir, exist_ok=True)
    tracking = {}

    for pde_name, pde_cfg in EXPERIMENTS.items():
        if args.pde and pde_name != args.pde:
            continue
        cluster = args.force_cluster or pde_cfg["cluster"]
        if args.cluster and cluster != args.cluster:
            continue

        print(f"\n=== {pde_name} (cluster: {cluster}) ===")
        for seed in pde_cfg["seeds"]:
            for grid_n in pde_cfg["train_grid_ns"]:
                overrides = {"n_epochs": args.epochs} if args.epochs else None
                exp_dir, cfg = create_experiment(
                    args.base_output_dir, pde_name, seed, grid_n, overrides
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
