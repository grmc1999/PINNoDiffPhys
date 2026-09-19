import numpy as np
import torch

from experiment_utils import set_seed


class CoordinateMLP(torch.nn.Module):
    """Simple coordinate MLP: (x, y, t) -> scalar u.

    Sized to stay near the CNN corrector's parameter budget (~180k).
    """

    def __init__(self, dim_in: int = 3, hidden: tuple = (256, 256, 256), dim_out: int = 1):
        super().__init__()
        layers = []
        prev = dim_in
        for h in hidden:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.Tanh())
            prev = h
        layers.append(torch.nn.Linear(prev, dim_out))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xt: torch.Tensor):
        return self.net(xt)


class PurePINNTrainer:
    """Residual-only physics-informed neural network (no FEM stepping).

    The statistical model is a continuous map u(x, y, t); training minimizes
    the squared PDE residual on interior collocation points penalized by the
    initial and (zero-Dirichlet) boundary conditions.  ``physical_model`` is
    retained only as a reference-solver handle so the posterior codepaths
    (``grid_input/rollout_ground_truth_on_grid``) work unchanged.

    API contract mirrors ``FiredrakePINNSBasedSOLTrainer`` where the shared
    harness uses it: ``.st_model``, ``.physical_model``, ``.loss``,
    ``.optimizer``, ``.n_steps``, ``.dt``, ``.train(epochs, batch_size)`` and
    ``.predict_rollout(u0, t0, n_steps, spatial_sample)``.
    """

    def __init__(
        self,
        physical_model,
        statistical_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        simulation_steps: int,
        dt: float,
        loss: callable,
        u0_fn=None,
        eval_grid: np.ndarray = None,
        w_ic: float = 1.0,
        w_bc: float = 1.0,
        t_max: float = None,
    ):
        self.physical_model = physical_model
        self.st_model = statistical_model
        self.optimizer = optimizer
        self.n_steps = simulation_steps
        self.dt = dt
        self.loss = loss
        self.correction_enabled = True

        self.w_ic = w_ic
        self.w_bc = w_bc
        self.t_max = t_max if t_max is not None else float(simulation_steps * dt)

        self.eval_grid = eval_grid  # [H, W, 2] point grid; None disables IC/BC splits
        self.u0_grid = None
        if eval_grid is not None:
            self.u0_grid = self._sample_u0(eval_grid)
        self.interior_mask = None
        self.boundary_mask = None
        self._radius = 1.0

    def _sample_u0(self, grid: np.ndarray):
        """u0 evaluated on *grid* -> torch tensor [P] (exact analytic, no VOM)."""
        X = grid[..., 0]
        Y = grid[..., 1]
        v = 0.5 * np.exp(0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2 - 0.1) ** 2 - 1.0)
        return torch.as_tensor(v.reshape(-1), dtype=torch.float32)

    def _split_masks(self, n: int):
        """Interior (PDE residual) and boundary (Dirichlet-BC) point masks."""
        pts = self.eval_grid.reshape(-1, 2)
        tol = 1e-9
        on_bdry = (
            (np.abs(pts[:, 0]) < tol) | (np.abs(pts[:, 0] - 1.0) < tol)
            | (np.abs(pts[:, 1]) < tol) | (np.abs(pts[:, 1] - 1.0) < tol)
        )
        n_int = int(np.count_nonzero(~on_bdry))
        self.interior_mask = torch.as_tensor(~np.array(on_bdry, dtype=bool))
        self.boundary_mask = torch.as_tensor(np.array(on_bdry, dtype=bool))

    def _coords(self, t: float):
        """[P, 3] leaf coordinates (x, y, t) requiring grad + [P, 1] features."""
        pts = torch.as_tensor(self.eval_grid.reshape(-1, 2), dtype=torch.float32)
        c = torch.cat([pts, torch.full((pts.shape[0], 1), float(t))], dim=-1)
        c = c.requires_grad_(True)
        return c

    def _residual_grid(self, u: torch.Tensor, xt: torch.Tensor):
        """Squared PDE residual per point via the script's loss callable."""
        # loss(u, xt) contract: u [B, P, 1], xt [B, P, C] with t at channel `dim`.
        r = self.loss(u, xt)  # [B, P]
        return r

    def generate_ground_truth(self, u0: np.ndarray, n_rollout: int):
        """No-op: the PINN evaluates the PDE residual directly, no coarse
        rollout is needed as reference."""
        return

    def train(self, epochs: int, batch_size: int = 8):
        losses = []
        P = self.eval_grid.reshape(-1, 2).shape[0]
        self._split_masks(P)
        pts = torch.as_tensor(self.eval_grid.reshape(-1, 2), dtype=torch.float32)
        idx_int = torch.nonzero(self.interior_mask).squeeze(-1)
        idx_bc = torch.nonzero(self.boundary_mask).squeeze(-1)
        pts_int = pts.index_select(0, idx_int).clone().requires_grad_(True)
        pts_bc = pts.index_select(0, idx_bc).clone().requires_grad_(True)

        for _ in range(epochs):
            total = torch.zeros(1, dtype=torch.float32)
            ts = np.random.uniform(0.0, self.t_max, size=batch_size)
            for ti in ts:
                t = float(ti)
                # interior residual at collocation time t (fresh leaf coords)
                c_int = torch.cat(
                    [pts_int, torch.full((pts_int.shape[0], 1), t, dtype=torch.float32)],
                    dim=-1,
                )
                u_int = self.st_model(c_int)                 # [P_i, 1]
                rloss = torch.mean(self._residual_grid(
                    u_int.unsqueeze(0), c_int.unsqueeze(0)))
                total = total + rloss

                # initial condition (t = 0) on the full grid
                c_ic = torch.cat(
                    [pts, torch.zeros((pts.shape[0], 1), dtype=torch.float32)],
                    dim=-1,
                ).requires_grad_(True)
                u_ic = self.st_model(c_ic).squeeze(-1)       # [P]
                ic_err = torch.mean((u_ic - self.u0_grid) ** 2)
                total = total + self.w_ic * ic_err

                # boundary condition u|_dOmega = 0
                c_bc = torch.cat(
                    [pts_bc, torch.full((pts_bc.shape[0], 1), t, dtype=torch.float32)],
                    dim=-1,
                )
                u_bc = self.st_model(c_bc).squeeze(-1)
                bc_err = torch.mean(u_bc ** 2)
                total = total + self.w_bc * bc_err

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            losses.append(float(total.detach()))

        return losses

    def predict_rollout(self, u0, t0: float, n_steps: int, spatial_sample=None):
        """Evaluate the continuous PINN field at times t0+dt, ..., t0+n*dt.

        Returns (states_pred [T, P, V], states_in, states_corr, times,
        uncorrected_sol) with V = 4 channels [x, y, t, u] so the shared
        ``gt_error_metrics``/``compute_residual_curve`` consumers work unchanged
        (last channel is the predicted scalar field).
        """
        grid = spatial_sample if spatial_sample is not None else self.eval_grid
        H, W = grid.shape[:2]
        pts = grid.reshape(-1, 2)
        times = [t0 + (k + 1) * self.dt for k in range(n_steps)]

        states_pred = []
        states_feat = []
        for t in times:
            c = torch.as_tensor(np.concatenate(
                [pts, np.full((pts.shape[0], 1), float(t))], axis=-1),
                dtype=torch.float32).requires_grad_(True)
            u = self.st_model(c)                    # [P, 1]
            # keep c (not detached) so residual losses that differentiate the
            # predicted field w.r.t. the feature channels still have a graph
            feat = torch.cat([c, u], dim=-1)        # [P, 4]
            states_pred.append(feat)
            states_feat.append(feat)

        states_pred = torch.stack(states_pred, axis=0)   # [T, P, 4]
        states_in = torch.stack(states_feat, axis=0)     # [T, P, 4]
        states_corr = torch.zeros_like(states_pred[..., -1:])  # [T, P, 1]
        uncorrected_sol = states_in                      # residual input features

        return states_pred, states_in, states_corr, times, uncorrected_sol


class FiredrakePINNSBasedSOLTrainerPurePINN(PurePINNTrainer):
    """Alias kept for drop-in ``build_trainer(...)`` parity naming."""
    pass