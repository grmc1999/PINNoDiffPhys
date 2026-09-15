# Diagnostic Plan: Transposed Reconstruction in Advection `solution_epoch_*.png`

**Status:** Root cause CONFIRMED + fix applied (commit `xxx`) · **Owner:** Guillermo · **Tracker:** `PLAN.md`

## 0. Verdict

**Root cause:** the einops collapses in `trainer/Trainer.py` flatten the `(h,w)` spatial
grid the wrong way: `"c h w -> 1 (h w) c"` (and the fine-branch `"V x y t -> t (x y) V"`)
emit the point vector in the *transposed* (column-major-equivalent) order relative to the
VOM/data `data_ro` row-major point order. Every downstream consumer (`grids_from_prediction_list`
`reshape(H,W)`, the FE lift, and the GT comparison) assumes row-major, so `pred` appears as
`gt.T`. All PDEs carry the flip; diffusion/poisson just hide it visually.

**Empirical proof (in-container probes, commit-independent):**
- Zero-CNN probe: `pred ≡ gt.T` exactly (`rel_rmse ≡ 0.0`) → flip is in the plumbing.
- einops isolation probe: for true-physical blocks, flatten `(h w)`/`(x y)` → transposed,
  `(w h)`/`(y x)` → correct point order (both 11×11 and 41×41).
- CNN roundtrip: expand `"1 (h w) c -> c h w"` from point-order flat yields a TRUE-physical
  image (corner check), so this expand is left unchanged and pairs with the new flattens.

**Fix (single commit, retraining required):** swap the parenthesized axis order in all 4
active einops flattens (`Trainer.py:701`, `:729`, `:815`, `:822`) and the fine-branch expand
(`:818`) so every flat is VOM point-order; the coarse expand `:700` is already correct.

| Location | Pattern before | Pattern after |
|---|---|---|
| `Trainer.py:701` corr out flatten | `"c h w -> 1 (h w) c"` | `"c h w -> 1 (w h) c"` |
| `Trainer.py:729` feat flatten | `"c h w -> 1 (h w) c"` | `"c h w -> 1 (w h) c"` |
| `Trainer.py:815` fine feat flatten | `"V x y t -> t (x y) V"` | `"V x y t -> t (y x) V"` |
| `Trainer.py:818` fine CNN expand | `"(x y) V -> V x y"` | `"(y x) V -> V x y"` |
| `Trainer.py:822` fine CNN flatten | `"V x y -> (x y) V"` | `"V x y -> (y x) V"` |
| `Trainer.py:700` coarse CNN expand | `"1 (h w) c -> c h w"` | *(unchanged, verified)* |

> Because the correction CNN was trained in the transposed frame, checkpoints from
> pre-fix code must be **retrained** (in-flight reruns 601926-928 use the old code and will
> be re-oriented offline or regenerated for the paper).

## 1. Problem statement

Advection runs (`solution_epoch_*.png`, produced per checkpoint) show the predicted field
appearing as the **transpose (flip across the main diagonal)** of the ground truth.
Diffusion and Poisson runs do not visibly present the artifact — likely because their
solutions are smooth / near-circularly-symmetric, so a spatial flip is less obvious,
while the advection profile is sharp and anisotropic, making any flip unmistakable.

The artifact is a *relative* flip: `pred ≈ gt.T` in the emitted figures. It does not
affect scalar error metrics (which are index-wise), but it would invalidate any
figure intended for the ICLR paper.

## 2. What has been ruled out (static analysis)

The pure-Python plumbing was traced end-to-end and is internally row-major consistent:

| Stage | Path | Consistency |
|---|---|---|
| Train points | `make_point_grid(...)` (`indexing="xy"`) -> `ravel()` | x varies fastest (point order) |
| Train VOM | `VertexOnlyMesh(mesh, grid.reshape(-1,ndim), reorder=False)` (`Trainer.py:108-112`) | point order preserved in `data_ro` |
| Fine VOM | `VertexOnlyMesh(V.mesh(), spatial_sample.reshape(-1,ndim), reorder=False)` (`Trainer.py:802-806`) | point order preserved |
| Coarse features | `feature_builder` (`trainer/Trainer.py:847-854`) | X = DG0-on-VOM coords, `reshape(eval_shape)` row-major |
| Fine features | `feature_builder_finer` (`trainer/Trainer.py:856-863`) | X = DG0-on-VOM coords, `reshape(eval_points.shape)` row-major |
| Rollout flatten | `rearrange(..., "V x y t -> t (x y) V")` (`Trainer.py:815`) | `(x y)` = row-major index (`row*W + col`) |
| CNN in/out | `"(x y) V -> V x y"` / `" V x y -> (x y) V"` (`Trainer.py:817-822`) | bijective, index-consistent |
| Pred grids | `grids_from_prediction_list` (`experiment_utils.py:108-114`) | `reshape(H, W)` row-major |
| GT grids | `rollout_ground_truth_on_grid` (`experiment_utils.py:46-68`) | VOM `data_ro` point order -> `reshape(H,W)` row-major |

A pure-numpy/einops-free emulation of the fine-branch flatten order round-trips
`pred == gt` for row-major **and** for X-channel-transposed inputs — i.e. the
Python-level indexing cannot introduce a flip by itself.

**Conclusion:** the flip was confirmed empirically to be an **einops axis-order issue**
(not a Firedrake primitive). VOM, `data_ro`, and `to_torch`/`from_torch` are exonerated
(probe 1a/1d). The offending primitives are the einops collapses listed in §0.

## 3. Step 1 — Locate the flip with an in-container probe (sbatch job)

### 3.0 Status: COMPLETE — verdict in §0. Probe jobs and results:

| Probe | Job | Result |
|---|---|---|
| 1a layout (X VOM row/col) | 602147 | VOM `data_ro` points + X channel row-major-correct on coarse & fine |
| 1b repro from checkpoint | 602151 | `pred ≈ gt.T` at every step (train + fine), `rel_rmse(pred,gt.T)` ≈ half |
| 1c zero-CNN | 602155 | `pred ≡ gt.T` exactly (`rel_rmse = 0.000000`) → plumbing, not learned |
| 1d to_torch | 602156/157 | `to_torch` exonerated; coarse X & u match GT |
| 1e einops isolation | 602161 | einops flatten `(x y)` = `gt.T`; `to_torch`/reshape paths = `gt` |
| 1f candidate patterns | 602162 | `(y x)` and input-axis-swap both restore `== gt` (fine branch) |
| 1g all call sites | 602168 | flatten `(h w)`=T, `(w h)`=OK; expand `1 (h w) c -> c h w`=true-physical (11 & 41) |

### 3.1 Environment constraints
- `singularity` exists only on **compute nodes** (inside sbatch jobs), not on the login node.
- Container: `$PATH_ENV/fem_pytorch/envs/firedrake.sif` (exists, ~2.6 GB; Firedrake +
  torch 2.4.1+cu121 + numpy + matplotlib + einops + scipy; entrypoint `python3`).
- `bash_routines/set_env.sh ICA`: `PATH_ENV=/share_zeta/Proxy-Sim/guillermo.carrillo`,
  `CONTAINER_PATH=$PATH_ENV/envs/ICA_v4.sif`.
- Probe writes **nothing to the repo**; script + artifacts live in temp dirs under
  `%TEMP%\opencode\` and `$PATH_ENV/diag/` (wiped after).

### 3.2 Probe 1a — primitive layout checks (~1 min cpu job, no weights needed)
Script `diag_layout.py`, run as `singularity exec $CONTAINER python3 diag_layout.py`:

1. Build the advection train point grid `make_point_grid(11)` (P_min/P_max as in
   `Train_test_advection.py`).
2. Build `VertexOnlyMesh(UnitSquareMesh(10,10), grid.reshape(-1,2), reorder=False)`.
3. `Vx = VectorFunctionSpace(vom, "DG", 0)`;
   `X = Function(Vx).interpolate(SpatialCoordinate(vom))`.
4. Inspect `X.dat.data_ro.reshape(11,11,2)` corners: `[0,0]`, `[0,1]`, `[1,0]`, `[0,-1]`.
5. Compare against the two hypotheses:
   - **H-rowmajor:** `X[i,j] == (x_j, y_i)` (grid point order)
   - **H-transposed:** `X[i,j] == (x_i, y_j)` (flipped)
6. Repeat 2–5 for the **fine** grid `make_point_grid(41)`.
7. Print a verdict table: coarse row/col, fine row/col, and their agreement.

### 3.3 Probe 1b — decisive reproduction with a real checkpoint
Use an existing advection checkpoint (e.g. from canonical runs 601677–601685, saved
30–40/50 epochs; artifacts under `EXPS/advection_*`):

1. Load the trainer/model exactly as `Train_test_advection.py` does.
2. `pred = predict_rollout(u0, 0.0, n_steps, spatial_sample=fine_grid(41))`.
3. `gt = rollout_ground_truth_on_grid(stepper, u0, n_steps, fine_grid(41))`.
4. Compare `pred_grids[i]` against `gt[i]` **and** `gt[i].T`, reporting per-step
   (`rel_rmse(pred, gt)`, `rel_rmse(pred, gt.T)`) for every saved checkpoint.
5. Also dump one 3×3 corner of an asymmetric probe field (`u0 = exp(x) + 2 exp(y)`)
   round-tripped through both paths, printed lossless (repr).
6. **Success criteria:** either `pred≈gt` (artifact is in old checkpoints only / already
   gone) or `pred≈gt.T` with the same flip across all checkpoints — confirming the bug
   and giving the exact orientation sign for the fix.

### 3.4 Deliverables
- `diag_summary.txt` (captured job stdout) placed in the run's temp dir.
- If 1b reproduces: a one-line verdict `FLIP_DIRECTION = transpose(pred)` vs
  `transpose(gt)` and the exact offending primitive name (flagged by WHICH check in 1a
  disagreed).

## 4. Step 2 — Fix (code change) — APPLIED

### 4.1 Fix (see §0 table)
The einops collapses at `Trainer.py:701/729/815/822` and expand at `:818` were switched to
the probe-verified point-order forms. Coarse expand `:700` verified unchanged.

### 4.2 Regression validation
- [ ] Rerun **one** advection seed (grid11 seed0) briefly with the fixed code; confirm
  `pred == gt` (not `gt.T`) via `diag_repro.py`-style comparison and image orientation.
- [ ] Confirm diffusion/poisson images/metrics still fine (guard passes).
- [ ] Existing checkpoints are stale (CNN learned in transposed frame) → retrain required
  for paper-quality results.

## 5. Step 3 — Keep advection reruns monitored (independent)

- Grouped driver `group_adv.py` (state `%TEMP%\opencode\grp_adv.json`) submits 3-by-3
  (grid11 → grid16 → grid21), polls `sacct`; the 9 reruns are at 48 h wall
  (`#SBATCH --time=48:00:00`, commit `0e0c088`).
- Group 1 = 601926/601927/601928 currently in flight.
- Interaction with the bug: reruns execute the current code, so their images carry the
  same artifact; if Step 1 confirms a pure layout flip, we re-orient those figures
  offline (no recompute). If a fix is merged before later rerun groups finish, note the
  code version per group in the tracking table.
- Paper figures: decide after Step 1 whether to regenerate one seed post-fix.

## 6. Open decisions (confirmed with user)
- [x] Submit probe 1a/1b job now → done, full probe suite 1a-1g (jobs 602147→602168).
- [x] Guard disposition → n/a; root cause was einops flatten order, fixed directly.
- [x] Keep reruns untouched while investigating → yes; 601926-928 run old code, re-orient/regenerate offline.
- [ ] Validation run after fix (single seed, short).

## 7. Reference map
| Item | Location |
|---|---|
| Coarse features | `trainer/Trainer.py:847-854` |
| Fine features | `trainer/Trainer.py:856-863` |
| Fine rollout branch | `trainer/Trainer.py:795-824` |
| GT on grid | `experiment_utils.py:46-68` |
| Pred grids | `experiment_utils.py:108-114` |
| Image emission | `experiment_utils.py:147+` (`plot_solution_snapshots`) |
| Train/fine grids | `Train_test_advection.py:235-236` |
| SRM (48 h) | `srm_routines/PINNoDiffPhys_ICA_cpu.srm` |
| Container env | `bash_routines/set_env.sh ICA` |
| Canonical runs | `sacct` ids 601668–601694 (advection 601677–601685) |
| Rerun group 1 | `sacct` ids 601926/601927/601928 |