#!/usr/bin/env python3
"""Dump sacct state + tail remote logs for the wave-1 jobs.

Usage:
    python job_status.py [jobid ...]
"""
import os
import sys
import tempfile

from srm_routines.ica_ssh import ICA

LOG = "/share_zeta/Proxy-Sim/guillermo.carrillo/PINNo_ICA_%s.log"


def main():
    jobids = sys.argv[1:]
    to_file = "-file" in jobids
    jobids = [j for j in jobids if j and j != "-file"] or ["593171", "593172", "593173"]
    buf = []
    with ICA() as ica:
        st, out, err = ica.run(
            "squeue -u gmorenoc -o '%i %j %t %N' 2>&1", timeout=30
        )
        buf.append("=== queue ===")
        buf.append(out)
        buf.append("=== per-job sacct + log tail ===")
        for j in jobids:
            st, out, err = ica.run(
                "sacct -j %s --format=JobID,State,Elapsed,ExitCode 2>&1" % j,
                timeout=30,
            )
            buf.append(f"\n--- {j} sacct ---")
            buf.append(out.strip())
            st, out, err = ica.run(
                "tail -n 40 %s 2>&1" % (LOG % j), timeout=30
            )
            buf.append(f"--- {j} log tail ---")
            buf.append(out.rstrip())
    result = "\n".join(buf)
    if to_file:
        outp = os.path.join(tempfile.gettempdir(), "job_status.txt")
        with open(outp, "w", encoding="utf-8") as f:
            f.write(result)
        print("WROTE", outp)
    else:
        print(result.encode("utf-8", "replace").decode("utf-8", "replace"))


if __name__ == "__main__":
    main()
