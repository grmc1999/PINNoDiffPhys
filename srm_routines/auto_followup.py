#!/usr/bin/env python3
"""Wait for the current 50-epoch wave to finish, then submit the fix runs
(advection + poisson with the corrected losses/adjoint wiring) into fresh dirs.

Idempotent: records submitted job ids in a local state file, so re-running
after a tool timeout only continues what is missing.

Usage:
    python -m srm_routines.auto_followup
"""
import json
import os
import sys
import time
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from srm_routines.ica_ssh import ICA

PATH_CODE = "/share_zeta/Proxy-Sim/guillermo.carrillo/PINNoDiffPhys"
WAVE = ["601130", "601131", "601132"]
FOLLOWUPS = [
    ("EXPS/advection_grid11_seed0_fixed", 50),
    ("EXPS/poisson_grid11_seed0_fixed", 50),
]
STATE = os.path.join(tempfile.gettempdir(), "followup_state.json")
CAPTURE = os.path.join(tempfile.gettempdir(), "followup_capture.txt")
LOG = "/share_zeta/Proxy-Sim/guillermo.carrillo/PINNo_ICA_%s.log"


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE, "w") as f:
        json.dump(state, f, indent=2)


def wave_done(ica):
    st, out, err = ica.run(
        "squeue -u gmorenoc -o '%i %j %t' 2>&1", timeout=30
    )
    running = set(out.split())
    return not any(j in running for j in (WAVE + ["60113"]))


def capture_tails(ica):
    with open(CAPTURE, "a", encoding="utf-8") as f:
        f.write(f"\n===== capture {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        for j in WAVE:
            st, out, err = ica.run(
                "sacct -j %s --format=JobID,State,Elapsed,ExitCode 2>&1" % j,
                timeout=30,
            )
            f.write(f"\n--- {j} sacct ---\n{out.strip()}\n")
            st, out, err = ica.run("tail -n 80 %s 2>&1" % (LOG % j), timeout=30)
            f.write(f"--- {j} log tail ---\n{out.rstrip()}\n")
    print("captured log tails ->", CAPTURE)


def submit_one(ica, rel_exp, epochs):
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "srm_routines.submit_n",
         "--only", rel_exp, "--epochs", str(epochs)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=ROOT,
    )
    return proc.stdout, proc.stderr


def main():
    state = load_state()
    with ICA() as ica:
        print("polling for wave completion:", WAVE)
        while not wave_done(ica):
            print(time.strftime("%H:%M:%S"), "still running ...")
            time.sleep(60)
        print("wave done, capturing tails")
        capture_tails(ica)
        state["captured"] = True
        save_state(state)
        for rel_exp, epochs in FOLLOWUPS:
            if state.get(rel_exp):
                print("already submitted:", rel_exp, "->", state[rel_exp])
                continue
            out, err = submit_one(ica, rel_exp, epochs)
            print(f"--- {rel_exp} submit ---")
            print(out)
            if err.strip():
                print("ERR:", err[-2000:])
            job_id = None
            for tok in out.split():
                if tok.replace(".", "").isdigit():
                    job_id = tok
                    break
            state[rel_exp] = job_id
            save_state(state)
        print("done. state:", state)


if __name__ == "__main__":
    main()