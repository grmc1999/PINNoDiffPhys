#!/usr/bin/env python3
"""Lightweight ICA job monitor for PINNoDiffPhys wave-1.

Usage:
    python srm_routines/monitor_wave1.py                # full queue + status vs EXPS/tracking.json
    python srm_routines/monitor_wave1.py 593143          # specific job id
"""
import json
import os
import sys

from srm_routines.ica_ssh import ICA


def main():
    tracker_path = os.path.join("EXPS", "tracking.json")
    jobids = set()
    if os.path.exists(tracker_path):
        with open(tracker_path) as f:
            tracking = json.load(f)
        jobids = {v["job_id"] for v in tracking.values() if v.get("job_id")}

    with ICA() as ica:
        if len(sys.argv) > 1:
            j = sys.argv[1]
            st, out, err = ica.run(f"squeue -j {j} 2>&1")
            st2, out2, err2 = ica.run(f"sacct -j {j} --format=JobID,State,ExitCode,Elapsed 2>&1")
            print(out)
            print("--- sacct ---")
            print(out2)
            return

        st, out, err = ica.run("squeue -u gmorenoc 2>&1")
        lines = out.splitlines()
        print("=== queue (all user jobs) ===")
        print(out)
        mine = [ln for ln in lines[1:] if ln.split() and ln.split()[0] in jobids] if lines else []
        print(f"\n=== wave-1 jobs: running/pending in tracking ({len(mine)}/{len(jobids)}) ===")
        for ln in mine:
            print(ln)
        done_count = len(jobids) - len(mine)
        print(f"tracked wave-1 total: {len(jobids)}; still active: {len(mine)}; finished/absent: {done_count}")


if __name__ == "__main__":
    main()
