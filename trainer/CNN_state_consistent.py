from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import firedrake as fd
from einops import rearrange
from firedrake.adjoint import Control, ReducedFunctional
from firedrake.ml.pytorch import fem_operator, to_torch
from tqdm import tqdm

from trainer.Trainer import FiredrakeTimeStepper


class TorchPointCloudLift(torch.nn.Module):
    """
    Differentiable torch lift from point-grid values to Firedrake state DOFs.

    Given a fixed Firedrake point-evaluation matrix

        E : V_dofs -> point_values,

    this module applies the Tikhonov-regularized pseudo-inverse

        L = (E^T E + reg I)^(-1) E^T,

    so that a CNN correction living on the VertexOnlyMesh / CNN grid can be
    lifted back to the original finite-element space V before the next PDE step.

    The matrix is fixed, but the operation q -> L q is differentiable with
    respect to q, so gradients from the recurrent PDE rollout still reach the
    CNN parameters.
    """

    def __init__(self, point_eval_matrix: torch.Tensor, reg: float = 1.0e-6):
        super().__init__()
        if point_eval_matrix.ndim != 2:
            raise ValueError("point_eval_matrix must have shape [n_points, n_dofs].")

        E = point_eval_matrix.detach()
        n_points, n_dofs = E.shape
        eye = torch.eye(n_dofs, dtype=E.dtype, device=E.device)
        lift = torch.linalg.solve(E.T @ E + float(reg) * eye, E.T)  # [n_dofs, n_points]

        self.n_points = n_points
        self.n_dofs = n_dofs
        self.register_buffer("lift", lift)

    def forward(self, q_points: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        q_points:
            Shape [P], [B, P], or [B, P, 1].

        Returns
        -------
        q_dofs:
            Shape [B, n_dofs].
        """
        if q_points.ndim == 1:
            q_points = q_points.unsqueeze(0)
        if q_points.ndim == 3:
            if q_points.shape[-1] != 1:
                raise ValueError("Only scalar point fields with shape [B, P, 1] are supported.")
            q_points = q_points[..., 0]
        if q_points.ndim != 2:
            raise ValueError("q_points must have shape [P], [B, P], or [B, P, 1].")
        if q_points.shape[-1] != self.n_points:
            raise ValueError(
                f"Expected {self.n_points} point values, got {q_points.shape[-1]}."
            )
        return q_points @ self.lift.T


def build_torch_state_step_operator(model: FiredrakeTimeStepper) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a PyTorch operator for one Firedrake PDE step, but keep the output in
    the original Firedrake function space model.V.

    This is the critical difference from the original implementation, which
    returned point-grid values when point_evaluator was present.
    """
    fd.adjoint.continue_annotation()

    u_n = fd.Function(model.V, name="u_n_control_state")
    u_np1 = fd.Function(model.V, name="u_np1_state")

    F = model.residual(u_np1, u_n)
    fd.solve(
        F == 0,
        u_np1,
        bcs=model.bcs,
        solver_parameters=model.solver_parameters,
    )

    red = ReducedFunctional(u_np1, Control(u_n))
    fd.adjoint.stop_annotating()
    return fem_operator(red)


def build_torch_point_observation_operator(model: FiredrakeTimeStepper) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a differentiable observation operator

        V dofs -> values on model.P0DG, i.e. the CNN grid.

    Firedrake implements this as interpolation onto a VertexOnlyMesh. This is a
    Firedrake/pyadjoint operation, not a NumPy detour.
    """
    if not hasattr(model, "P0DG"):
        raise ValueError("model must be constructed with point_evaluator to build point observations.")

    fd.adjoint.continue_annotation()

    u = fd.Function(model.V, name="u_state_for_point_observation")
    u_points = fd.assemble(fd.interpolate(u, model.P0DG))

    red = ReducedFunctional(u_points, Control(u))
    fd.adjoint.stop_annotating()
    return fem_operator(red)


def build_dense_point_eval_matrix(
    observation_op: Callable[[torch.Tensor], torch.Tensor],
    n_dofs: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Construct the dense matrix E corresponding to the differentiable observation
    operator V -> point grid, without using NumPy conversion in the training loop.

    E has shape [n_points, n_dofs]. It is obtained by applying the observation
    operator to canonical basis vectors of V.
    """
    rows_by_basis_batch = []
    device = torch.device(device)

    with torch.no_grad():
        for start in range(0, n_dofs, chunk_size):
            end = min(start + chunk_size, n_dofs)
            x = torch.zeros((end - start, n_dofs), dtype=dtype, device=device)
            x[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = 1.0

            y = observation_op(x)
            if y.ndim == 1:
                y = y.unsqueeze(0)
            y = y.reshape(y.shape[0], -1)
            rows_by_basis_batch.append(y.detach().cpu())

    # Currently: [n_dofs, n_points]. We need [n_points, n_dofs].
    return torch.cat(rows_by_basis_batch, dim=0).T.contiguous()


class FiredrakePINNSBasedSOLTrainerCNNStateConsistent:
    """
    CNN trainer that keeps the recurrent physical state in the Firedrake space V.

    Data flow per time step:

        current_v       : tensor in Firedrake V DOF layout
        phys_next_v     = PDE_step(current_v)                 in V
        phys_next_grid  = observe(phys_next_v)                on VertexOnlyMesh/CNN grid
        corr_grid       = CNN(x, y, t, phys_next_grid)        on CNN grid
        corr_v          = lift(corr_grid)                     in V
        current_v       = phys_next_v + corr_v                in V

    Therefore the next PDE step always receives a tensor compatible with model.V.
    """

    def __init__(
        self,
        physical_model: FiredrakeTimeStepper,
        statistical_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        simulation_steps: int,
        dt: float,
        loss: Callable,
        lift_regularization: float = 1.0e-6,
        lift_chunk_size: int = 64,
    ):
        if not hasattr(physical_model, "P0DG"):
            raise ValueError("physical_model must be constructed with point_evaluator for CNN training.")

        self.physical_model = physical_model
        self.st_model = statistical_model
        self.optimizer = optimizer
        self.n_steps = simulation_steps
        self.dt = dt
        self.loss = loss

        # 1) PDE recurrence: V -> V.
        self.step_op = build_torch_state_step_operator(physical_model)

        # 2) Differentiable observation: V -> point grid.
        self.observe_op = build_torch_point_observation_operator(physical_model)

        # 3) Differentiable lift: point-grid correction -> V correction.
        n_dofs = physical_model.V.dim()
        E = build_dense_point_eval_matrix(
            self.observe_op,
            n_dofs,
            dtype=torch.float32,
            device="cpu",
            chunk_size=lift_chunk_size,
        )
        self.grid_to_state_lift = TorchPointCloudLift(E, reg=lift_regularization)

        self.init_states_gt: List[fd.Function] = []
        self.T: List[float] = [0.0]

        # Filled after forward_prediction_correction_from_state; useful for posterior evaluation.
        self._last_phys_v: List[torch.Tensor] = []
        self._last_corrected_v: List[torch.Tensor] = []

    def generate_ground_truth(self, u0: fd.Function, n_rollout: int):
        self.init_states_gt = [fd.Function(u0, name="gt_0")]
        self.T = [0.0]

        u = fd.Function(u0, name="gt_state")
        for _ in range(n_rollout):
            u = self.physical_model.step(u)
            self.init_states_gt.append(fd.Function(u))
            self.T.append(self.T[-1] + self.dt)

    def feature_builder(self, u_points: torch.Tensor, t: float) -> torch.Tensor:
        """
        Build CNN features [x, y, t, u] with shape [C, H, W].

        u_points must be the scalar field evaluated on physical_model.P0DG, not
        the original Firedrake DOF tensor.
        """
        eval_shape = self.physical_model.evaluation_shape
        spatial_shape = eval_shape[:-1]

        if u_points.ndim == 2:
            # [B, P]; current implementation uses B=1 during rollout.
            u_points = u_points[0]
        u = u_points.reshape(spatial_shape + (1,))

        Vx = fd.VectorFunctionSpace(self.physical_model.P0DG.mesh(), "DG", 0)
        X = to_torch(fd.Function(Vx).interpolate(fd.SpatialCoordinate(self.physical_model.mesh)))
        X = X.reshape(eval_shape)

        t_channel = torch.full(
            spatial_shape + (1,),
            fill_value=float(t),
            dtype=u.dtype,
            device=u.device,
        )
        X = X.to(dtype=u.dtype, device=u.device)
        return torch.concat((X, t_channel, u), axis=-1).transpose(0, -1).float()

    def feature_builder_finer(
        self,
        u_points: torch.Tensor,
        t: float,
        eval_points: np.ndarray,
        fs: fd.FunctionSpace,
    ) -> torch.Tensor:
        if u_points.ndim == 2:
            u_points = u_points[0]
        u = u_points.reshape(eval_points.shape[:-1] + (1,))

        Vx = fd.VectorFunctionSpace(fs.mesh(), "DG", 0)
        X = to_torch(fd.Function(Vx).interpolate(fd.SpatialCoordinate(fs.mesh())))
        X = X.reshape(eval_points.shape)

        t_channel = torch.full(
            eval_points.shape[:-1] + (1,),
            fill_value=float(t),
            dtype=u.dtype,
            device=u.device,
        )
        X = X.to(dtype=u.dtype, device=u.device)
        return torch.concat((X, t_channel, u), axis=-1).transpose(0, -1).float()

    def _cnn_correction_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [1, P, C]
        returns scalar correction on grid: [1, P, 1]
        """
        H, W = self.physical_model.evaluation_shape[:2]
        cnn_in = rearrange(features, "1 (h w) c -> c h w", h=H, w=W)
        corr = self.st_model(cnn_in)
        corr = rearrange(corr, "c h w -> 1 (h w) c")

        if corr.shape[-1] != 1:
            # The physical state is scalar. Keep only the scalar-state correction.
            corr = corr[..., -1:]
        return corr

    def correct(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        corr = self._cnn_correction_from_features(features)
        corrected = features[:, :, -1:] + corr
        return corrected, corr, features

    def forward_prediction_correction_from_state(self, state0_tensor: torch.Tensor, t0: float):
        states_pred = []
        states_corr = []
        states_in = []
        states_phys_v = []
        states_corrected_v = []

        current_t = t0
        current_v = state0_tensor
        if current_v.ndim == 1:
            current_v = current_v.unsqueeze(0)
        current_v = current_v.float()

        for _ in range(self.n_steps):
            # V -> V. This is the physical recurrence.
            phys_next_v = self.step_op(current_v)

            # V -> point grid. This is only an observation for the CNN/loss.
            current_t = current_t + self.dt
            phys_next_grid = self.observe_op(phys_next_v)

            features = rearrange(
                self.feature_builder(phys_next_grid, current_t),
                "c h w -> 1 (h w) c",
            ).requires_grad_(True)

            corrected_grid, corr_grid, features = self.correct(features)

            # point-grid correction -> V correction.
            corr_v = self.grid_to_state_lift(corr_grid).to(
                dtype=phys_next_v.dtype,
                device=phys_next_v.device,
            )
            corrected_v = phys_next_v + corr_v

            states_in.append(features)
            states_corr.append(corr_grid)
            states_pred.append(corrected_grid)
            states_phys_v.append(phys_next_v)
            states_corrected_v.append(corrected_v)

            # The next PDE step receives a valid V-layout tensor.
            current_v = corrected_v

        self._last_phys_v = states_phys_v
        self._last_corrected_v = states_corrected_v
        return states_pred, states_corr, states_in

    def train(self, epochs: int, batch_size: int = 8):
        losses = []
        for _ in tqdm(range(epochs)):
            batch_pred = []
            batch_in = []

            for _b in range(batch_size):
                idx = torch.randint(low=0, high=len(self.init_states_gt), size=(1,)).item()
                u0_fd = self.init_states_gt[idx]
                t0 = self.T[idx]

                u0_torch = to_torch(u0_fd, batched=True).float()
                states_pred, _, states_in = self.forward_prediction_correction_from_state(u0_torch, t0)

                batch_pred.extend(states_pred)
                batch_in.extend(states_in)

            total_loss = 0.0
            for u_pred, u_in in zip(batch_pred, batch_in):
                total_loss = total_loss + torch.mean(self.loss(u_pred, u_in))

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            losses.append(float(total_loss.detach().cpu()))

        return losses

    def predict_rollout(
        self,
        u0: fd.Function,
        t0: float,
        n_steps: int,
        spatial_sample: Optional[np.ndarray] = None,
    ):
        old_n_steps = self.n_steps
        self.n_steps = n_steps

        states_pred, states_corr, states_in = self.forward_prediction_correction_from_state(
            to_torch(u0, batched=True).float(),
            t0,
        )

        # These are true Firedrake-space states, so from_torch(..., V) is valid.
        uncorrected_fd = [
            fd.ml.pytorch.from_torch(state_v, self.physical_model.V)
            for state_v in self._last_phys_v
        ]

        uncorrected_sol = uncorrected_fd

        if isinstance(spatial_sample, np.ndarray):
            vom = fd.VertexOnlyMesh(
                self.physical_model.V.mesh(),
                spatial_sample.reshape(-1, self.physical_model.V.mesh().geometric_dimension()),
                reorder=False,
            )
            P0DG_fine = fd.FunctionSpace(vom, "DG", 0)

            fine_pred = []
            fine_input = []
            for i, u_sol in enumerate(uncorrected_fd):
                u_fine = to_torch(fd.assemble(fd.interpolate(u_sol, P0DG_fine))).requires_grad_(True)
                feats = rearrange(
                    self.feature_builder_finer(
                        u_fine,
                        t0 + self.dt * (i + 1),
                        spatial_sample,
                        P0DG_fine,
                    ),
                    "c h w -> 1 (h w) c",
                )

                H, W = spatial_sample.shape[:2]
                cnn_in = rearrange(feats, "1 (h w) c -> c h w", h=H, w=W)
                corr = rearrange(self.st_model(cnn_in), "c h w -> 1 (h w) c")
                if corr.shape[-1] != 1:
                    corr = corr[..., -1:]
                pred = feats[:, :, -1:] + corr
                fine_input.append(feats)
                fine_pred.append(pred)

            states_in = fine_input
            states_pred = fine_pred
            uncorrected_sol = uncorrected_fd

        self.n_steps = old_n_steps
        times = [t0 + (k + 1) * self.dt for k in range(len(states_pred))]
        return states_pred, states_in, states_corr, times, uncorrected_sol
