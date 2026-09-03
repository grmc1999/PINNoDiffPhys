#!/usr/bin/env python3
"""Submit a small, explicit list of wave-1 configs to ICA (capped concurrency).

Usage:
    python srm_routines/submit_n.py  # submits the EXPS python in EXP_LIST
"""
import json
import os

from srm_routines.ica_ssh import ICA

PATH_CODE = "/share_zeta/Proxy-Sim/guillermo.carrillo/PINNoDiffPhys"
CLUSTER = "ICA"

# Representative scout: diffusion (2 grids) + advection (1 grid), seed 0.
EXP_LIST = [
    "EXPS/diffusion_grid11_seed0",
    "EXPS/diffusion_grid21_seed0",
    "EXPS/advection_grid11_seed0",
]


def ensure_config(rel_exp, epochs=20):
    """Regenerate the config json locally so it always carries the right epochs."""
    parts = rel_exp.replace("\\", "/").split("/")
    # rel_exp like EXPS/<pde>_grid<N>_seed<S>
    import re
    m = re.match(r"EXPS/(\w+)_grid(\d+)_seed(\d+)", rel_exp)
    if not m:
        raise ValueError(f"cannot parse exp dir: {rel_exp}")
    pde, grid, seed = m.group(1), int(m.group(2)), int(m.group(3))
    # reuse the config generator from submit_wave1
    import submit_wave1 as sw
    exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    os.chdir(exp_dir)
    _, cfg = sw.create_experiment(
        "EXPS", pde, seed, grid, overrides={"n_epochs": epochs}
    )
    return rel_exp


def main():
    submitted = []
    with ICA() as ica:
        for rel_exp in EXP_LIST:
            ensure_config(rel_exp, epochs=20)
            local_cfg = os.path.join(
                "EXPS", rel_exp.replace("EXPS/", ""), "config.json"
            )
            ica.run(f"mkdir -p {PATH_CODE}/{rel_exp}")
            ica.sftp_put(local_cfg, f"{PATH_CODE}/{rel_exp}/config.json")
            st, out, err = ica.run(
                f"cd {PATH_CODE} && sbatch srm_routines/PINNoDiffPhys_ICA_cpu.srm {rel_exp} {CLUSTER}",
                timeout=60,
            )
            job_id = None
            for tok in out.split():
                if tok.isdigit():
                    job_id = tok
                    break
            submitted.append((rel_exp, job_id))
            print(f"{rel_exp} -> job {job_id}")
    print("\nSUMMARY:")
    for e, j in submitted:
        print(f"  {e}: {j}")


if __name__ == "__main__":
    main()
