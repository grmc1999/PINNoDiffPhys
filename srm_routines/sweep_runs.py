#!/usr/bin/env python3
"""On-demand grouped submitter for the full 27-run sweep (fixed code, ICA cpu).

NOT a daemon: nothing runs in the background. You must invoke it explicitly:
    python -m srm_routines.sweep_runs status          # show current group + job states
    python -m srm_routines.sweep_runs next            # submit next group (only if current is terminal)
    python -m srm_routines.sweep_runs next <idx>      # force-submit group <idx> (0-8)

State is kept in %TEMP%\\opencode\\sweep27.json so progress survives between
sessions. Groups are 3 experiments each, submitted 3-at-a-time.
"""
import json
import os
import sys

from srm_routines.ica_ssh import ICA

PATH_CODE = "/share_zeta/Proxy-Sim/guillermo.carrillo/PINNoDiffPhys"
SRM = "srm_routines/PINNoDiffPhys_ICA_cpu.srm"

GROUPS = []
for pde in ["advection", "diffusion", "poisson"]:
    for g in [11, 16, 21]:
        GROUPS.append([f"EXPS/{pde}_grid{g}_seed{s}" for s in [0, 1, 2]])

STATE = os.path.join(
    os.environ.get("TEMP", "."), "opencode", "sweep27.json"
)
TERMINAL = {"COMPLETED", "CANCELLED", "TIMEOUT", "FAILED", "OUT_OF_MEMORY"}


def load():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"group": 0, "jobids": [], "phase": "submit"}


def save(st):
    with open(STATE, "w") as f:
        json.dump(st, f)


def batch_states(ica, jids):
    state = {}
    for i in range(0, len(jids), 200):
        ids = ",".join(jids[i:i + 200])
        st, out, _ = ica.run(
            f"sacct -j {ids} --format=JobID,State --noheader 2>&1 | grep '.batch '",
            timeout=60,
        )
        for line in (out or "").splitlines():
            p = line.split()
            if len(p) >= 2:
                state.setdefault(p[0].split(".")[0], p[1])
    return state


def submit_group(ica, group_idx, names):
    jids = []
    for rel in names:
        st, out, err = ica.run(
            f"cd {PATH_CODE} && sbatch {SRM} {rel} ICA", timeout=90
        )
        jid = next((t for t in (out or "").split() if t.isdigit()), None)
        jids.append(jid)
        print(f"    -> submitted {rel:30s} job {jid}" + (f"  ERR:{err[:120]}" if err and jid is None else ""))
    return jids


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    st = load()
    g = st.get("group", 0)

    with ICA() as ica:
        if cmd == "status" or cmd == "next":
            names = GROUPS[g] if g < len(GROUPS) else None
            if names is None:
                print("All 9 groups already processed.")
                return
            print(f"=== group {g + 1}/{len(GROUPS)}: {names} ===")

            if st.get("phase") == "wait" and st.get("jobids"):
                states = batch_states(ica, st["jobids"])
                cur = [states.get(j, "?") for j in st["jobids"]]
                print(f"  jobs {st['jobids']} -> {cur}")
                if g >= len(GROUPS) - 1 and all(s in TERMINAL for s in cur):
                    print("  (last group ended — sweep complete)")
                if cmd == "status":
                    return
                # next: only proceed when terminal
                if not all(s in TERMINAL for s in cur):
                    print("  >> still running; NOT submitting next group")
                    return
                # report artifacts of finished group
                for rel in names:
                    ck = ica.run(f"test -s {PATH_CODE}/{rel}/summary.json && echo OK || echo MISSING")[1].strip()
                    print(f"   summary.json {rel}: {ck}")
                g += 1
                if g >= len(GROUPS):
                    print("ALL 9 GROUPS DONE")
                    return
                names = GROUPS[g]

            # force index override
            if len(sys.argv) > 2 and sys.argv[1] == "next" and sys.argv[2].isdigit():
                g = int(sys.argv[2])
                names = GROUPS[g]
                print(f"  (force) switching to group {g + 1}: {names}")

            print(f"  submitting group {g + 1}/{len(GROUPS)}:")
            jids = submit_group(ica, g, names)
            save({"group": g, "jobids": jids, "phase": "wait"})
            print(f"  state saved -> {STATE}")
        else:
            print("usage: python -m srm_routines.sweep_runs status|next [group_idx]")


if __name__ == "__main__":
    main()