# Research Plan: PINNoDiffPhys → ICLR 2027

**Target:** ICLR 2027 main track
**Deadlines:** Abstract **Sep 18, 2026** · Full paper **Sep 25, 2026** (AoE)
**Baseline publication:** *Bridging Continuous and Discrete Physics: A Hybrid PINN
Framework with Differentiable Solvers* — DiffSys Workshop @ EurIPS 2025
([OpenReview](https://openreview.net/forum?id=uIFgXZEq2I))

---

## 1. Goal

Extend the workshop version into a main-track submission:

- Migrate from PhiFlow to a **Firedrake/FEM** differentiable backend (already in progress).
- Evaluate on the three canonical linear PDE families: **parabolic, hyperbolic, elliptic**.
- Demonstrate generalization across **spatial resolution**, **temporal resolution**, and
  **temporal extrapolation** (correcting future steps beyond the training horizon).
- All experiments run directly on clusters (SDumont h100/gh200, ICA).

## 2. Scope

| Family | PDE | Stepper | Status |
|---|---|---|---|
| Parabolic | Diffusion | `ImplicitDiffusionStepper` (CG, backward Euler) | done |
| Hyperbolic | Advection | `ImplicitLinearAdvectionStepper` (DG upwind BE) | done |
| Hyperbolic | Wave | implicit first-order system (u, uₜ) | new (M1.5) |
| Elliptic | Poisson | `ImplicitPoissonSolverStepper`: step = m CG/Jacobi iterations from previous iterate | new (M1) |

Generalization protocol per family:
1. **Spatial interpolation** — train grid n∈{11}, test on finer grids (41+).
2. **Temporal interpolation** — smaller dt within the same training horizon.
3. **Temporal extrapolation** — rollout beyond training horizon ("future-step correction").
4. **Elliptic axis** — solver tolerance / iteration-budget shift (trained high-tolerance,
   tested coarse), mirroring the workshop's Poisson experiment.

**Out of scope (journal extension later):** NS, Darcy with heterogeneous K(x),
inverse/control problems, full theory section.

## 3. Milestones

### M1 — Code hardening + elliptic stepper (Aug 24–28) ✅ DONE
- Bug fixes:
  - hardcoded `dt=0.01` in `Train_test_advection.py:102` (must use `args.dt`) ✅
  - missing `return` in `tensor_state_to_grid` (both test scripts) ✅
  - restore GT-error metrics (rel-RMSE / L∞ vs `rollout_ground_truth`) in summaries ✅
- Seeding (torch/numpy/firedrake), per-run checkpointing, config-driven `EXPS/<exp>` layout. ✅
- Implement `ImplicitPoissonSolverStepper` (iterative-solve-as-step design). ✅

### M1.5 — Wave stepper (Aug 29 – Sep 1)
- Reformulate mass-matrix prototype (`hyperbolic/wave.py`) as an implicit first-order
  system fitting the single-control `FiredrakeTimeStepper` API.
- Smoke-test on ICA cpu queue.
- **Timebox:** if not stable by Sep 3, ship advection-only hyperbolic.

### M2 — Cluster enablement (Aug 31 – Sep 5)
- Build Apptainer `.sif`: Firedrake base image + PyTorch/einops/tqdm/matplotlib/pandas;
  deploy to `$PATH_ENV` on SDumont + ICA.
- New `srm_routines/PINNoDiffPhys_train_SD2_h100.srm`,
  `PINNoDiffPhys_infer_SD2_gh200.srm`, `PINNoDiffPhys_ICA_cpu.srm`
  (+ smoke script for fast correctness checks).
- Submission driver + tracking table (`agent/complete_table.md` style).
- Launch wave 1: diffusion + advection × seeds{3} × train grids{11,16,21}.

### M3 — Benchmark + baselines (Sep 7–12)
- Poisson + wave experiment runs.
- Baselines:
  1. uncorrected coarse solver (anchor)
  2. pure PINN (MLP, residual-only)
  3. supervised CNN corrector (same net, GT loss)
  4. *(stretch)* supervised FNO

### M4 — Ablations + analysis (Sep 14–18)
- Correction placement C∘P vs P∘C; lift regularization ε; rollout length N;
  train-grid sweep aggregation.
- Figures/tables auto-generated from `summary.json` artifacts.
- **Submit abstract by Sep 18.**

### M5 — Writing + buffer (Sep 21–25)
- Full paper (9 pp + refs), internal review, **submit by Sep 25**.

## 4. Compute & dev loop

- **Clusters only**: every code change validated by a fast ICA smoke job
  (tiny mesh, ~1 epoch, rollout + report) before any h100 allocation.
- Production runs: SDumont h100 (train), gh200 (inference/tests); ICA cpu as fallback.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Container build friction on clusters | Start `.sif` build early (overlaps M1.5); ICA-only fallback |
| Adjoint memory growth in long rollouts | Modest meshes first (train ≤ ~32², test ≤ ~65²); scale only if time allows |
| Wave stepper rework slips | Timeboxed to Sep 3; advection covers the hyperbolic family otherwise |
| Single-author bandwidth | Strict scope table above; deferred items go to journal extension |

## 6. Repository map (current state)

- `trainer/Trainer.py` — Firedrake↔PyTorch bridge: differentiable steppers,
  point-observation operator (VertexOnlyMesh), `TorchPointCloudLift`, hybrid trainers.
- `Train_test_diffusion.py`, `Train_test_advection.py` — full train + 3-regime test pipelines.
- `elliptic/`, `parabolic/`, `hyperbolic/` — PDE prototypes (control problems, wave, nonlinear Poisson).
- `DL_models/` — CNN/MLP/PINN models + residual-loss library (incl. NS, Darcy).
- `Diff_phys/*.ipynb` — legacy PhiFlow experiments from the workshop paper.
