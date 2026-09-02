from __future__ import annotations

from typing import Optional, Union, Callable
import firedrake as fd

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from einops import rearrange
import torch
import firedrake as fd

from firedrake.adjoint import Control, ReducedFunctional
from firedrake.ml.pytorch.fem_operator import fem_operator, to_torch
from tqdm import tqdm

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

        E = point_eval_matrix.detach().to(torch.float32)
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

class FiredrakeTimeStepper_(ABC):
    """
    Abstract differentiable Firedrake time-stepper.

    A subclass must define:
      - function space
      - boundary conditions
      - variational residual for one time step
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        dt: float,
        solver_parameters: Optional[dict] = None,
        point_evaluator: np.ndarray = None,
    ):
        self.mesh = mesh
        self.dt = fd.Constant(dt)
        self.solver_parameters = solver_parameters or {}

        self.V = self.build_function_space(mesh)
        self.bcs = self.build_bcs()
        self.point_evaluator = point_evaluator

        self.ori_points = fd.Function(fd.VectorFunctionSpace(self.V.mesh(), "DG", 0)).interpolate(fd.SpatialCoordinate(self.mesh))

        if isinstance(self.point_evaluator, np.ndarray):
            self.evaluation_shape = self.point_evaluator.shape
            vom = fd.VertexOnlyMesh(
                                    mesh,
                                    self.point_evaluator.reshape(-1,mesh.geometric_dimension()),
                                    reorder = False
                                    )
            self.P0DG = fd.FunctionSpace(vom, "DG", 0)

            vom = fd.VertexOnlyMesh(
                                    mesh,
                                    self.ori_points.dat.data,
                                    )
            self.P0DG_ori = fd.FunctionSpace(vom, "DG", 0)

    @abstractmethod
    def build_function_space(self, mesh: fd.MeshGeometry):
        ...

    @abstractmethod
    def build_bcs(self):
        ...

    @abstractmethod
    def residual(self, u_np1: fd.Function, u_n: fd.Function):
        """
        Return the weak residual F(u_{n+1}; v, u_n) = 0 for one implicit step.
        """
        ...

    def step(self, u_n: fd.Function) -> fd.Function:
        """
        Pure Firedrake step: u_n -> u_{n+1}
        """
        u_np1 = fd.Function(self.V, name="u_np1")
        F = self.residual(u_np1, u_n)
        fd.solve(
            F == 0,
            u_np1,
            bcs=self.bcs,
            solver_parameters=self.solver_parameters,
        )
        return u_np1

    def build_torch_step_operator(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Build a PyTorch-callable operator corresponding to one Firedrake step.

        The control is u_n, and the output is u_{n+1}.
        """
        fd.adjoint.continue_annotation()
        u_n = fd.Function(self.V, name="u_n_control")
        u_np1 = fd.Function(self.V, name="u_np1_state")

        #if isinstance(self.point_evaluator, np.ndarray):
        #    #u_n = fd.assemble(fd.project(u_n, self.V))
        #    #u_n = fd.assemble(fd.interpolate(u_n, self.P0DG_ori))
        #    u_n = fd.assemble(fd.project(u_n,self.V))
        F = self.residual(u_np1, u_n)
        fd.solve(
            F == 0,
            u_np1,
            bcs=self.bcs,
            solver_parameters=self.solver_parameters,
        )
        if isinstance(self.point_evaluator, np.ndarray):
            u_np1 = fd.assemble(fd.interpolate(u_np1, self.P0DG))

        # Reduced functional whose "output" is the field u_{n+1}
        red = ReducedFunctional(u_np1, Control(u_n))
        fd.adjoint.stop_annotating()
        return fem_operator(red)
    
class FiredrakeTimeStepper(ABC):
    """
    Abstract differentiable Firedrake time-stepper.

    A subclass must define:
      - function space
      - boundary conditions
      - variational residual for one time step
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        dt: float,
        solver_parameters: Optional[dict] = None,
        point_evaluator: np.ndarray = None,
    ):
        self.mesh = mesh
        self.dt = fd.Constant(dt)
        self.solver_parameters = solver_parameters or {}

        self.V = self.build_function_space(mesh)
        self.bcs = self.build_bcs()
        self.point_evaluator = point_evaluator

        self.ori_points = fd.Function(fd.VectorFunctionSpace(self.V.mesh(), "DG", 0)).interpolate(fd.SpatialCoordinate(self.mesh))

        if isinstance(self.point_evaluator, np.ndarray):
            self.evaluation_shape = self.point_evaluator.shape
            vom = fd.VertexOnlyMesh(
                                    mesh,
                                    self.point_evaluator.reshape(-1,mesh.geometric_dimension()),
                                    reorder = False
                                    )
            self.P0DG = fd.FunctionSpace(vom, "DG", 0)

            vom = fd.VertexOnlyMesh(
                                    mesh,
                                    self.ori_points.dat.data,
                                    )
            self.P0DG_ori = fd.FunctionSpace(vom, "DG", 0)

    @abstractmethod
    def build_function_space(self, mesh: fd.MeshGeometry):
        ...

    @abstractmethod
    def build_bcs(self):
        ...

    @abstractmethod
    def residual(self, u_np1: fd.Function, u_n: fd.Function):
        """
        Return the weak residual F(u_{n+1}; v, u_n) = 0 for one implicit step.
        """
        ...

    def step(self, u_n: fd.Function) -> fd.Function:
        """
        Pure Firedrake step: u_n -> u_{n+1}
        """
        u_np1 = fd.Function(self.V, name="u_np1")
        F = self.residual(u_np1, u_n)
        fd.solve(
            F == 0,
            u_np1,
            bcs=self.bcs,
            solver_parameters=self.solver_parameters,
        )
        return u_np1
    
    def build_torch_state_step_operator(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        """
        fd.adjoint.continue_annotation()

        u_n = fd.Function(self.V, name="u_n_control_state")
        u_np1 = fd.Function(self.V, name="u_np1_state")

        F = self.residual(u_np1, u_n)
        fd.solve(
            F == 0,
            u_np1,
            bcs=self.bcs,
            solver_parameters=self.solver_parameters,
        )

        red = ReducedFunctional(u_np1, Control(u_n))
        fd.adjoint.stop_annotating()
        return fem_operator(red)

    def build_torch_point_observation_operator(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        """
        if not hasattr(self, "P0DG"):
            raise ValueError("model must be constructed with point_evaluator to build point observations.")

        fd.adjoint.continue_annotation()

        u = fd.Function(self.V, name="u_state_for_point_observation")
        u_points = fd.assemble(fd.interpolate(u, self.P0DG))

        red = ReducedFunctional(u_points, Control(u))
        fd.adjoint.stop_annotating()
        return fem_operator(red)
    
    def build_dense_point_eval_matrix(self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """
        Construct the dense matrix E corresponding to the differentiable observation
        operator V -> point grid, without using NumPy conversion in the training loop.

        Rows are assembled column-wise; each column is the point grid
        produced by observing a single canonical basis vector of V.
        """
        device = torch.device(device)

        n_dofs = self.V.dim()

        observation_op = self.build_torch_point_observation_operator()

        cols = []
        with torch.no_grad():
            for i in range(n_dofs):
                x = torch.zeros(n_dofs, dtype=dtype, device=device)
                x[i] = 1.0
                y = observation_op(x)  # [n_points] (single observation)
                cols.append(y.detach().cpu().reshape(-1))

        return torch.stack(cols, dim=1).contiguous()


class ImplicitLinearAdvectionStepper(FiredrakeTimeStepper):
    """
    Backward-Euler DG upwind stepper for the scalar hyperbolic PDE

        u_t + div(beta * u) = 0

    Weak form:
        ((u^{n+1} - u^n)/dt, v)
        - (u^{n+1}, div(v beta))
        + exterior upwind fluxes
        + interior upwind fluxes
        = 0

    Notes
    -----
    - For hyperbolic transport, boundary conditions are imposed weakly
      through inflow/outflow flux terms, not with DirichletBC.
    - DG is the natural discretization here.
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        dt: float,
        velocity: Optional[Union[fd.Function, tuple, list, Callable]] = None,
        inflow_value: Union[float, fd.Constant, fd.Function] = 0.0,
        degree: int = 1,
        solver_parameters: Optional[dict] = None,
        point_evaluator: np.ndarray = None,
    ):
        self.degree = degree
        
        self.inflow_value = inflow_value

        super().__init__(
            mesh=mesh,
            dt=dt,
            point_evaluator = point_evaluator,
            solver_parameters=solver_parameters
            )

        # Build/store velocity field before calling parent constructor
        self._velocity_input = velocity

        default_solver_parameters = {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }
        if solver_parameters is not None:
            default_solver_parameters.update(solver_parameters)

        # Build velocity field once, after V exists
        self.W = fd.VectorFunctionSpace(mesh, "CG", max(1, degree))
        self.beta = self.build_velocity_field()

    def build_function_space(self, mesh):
        cell_name = mesh.ufl_cell().cellname()
        family = "DQ" if cell_name in ("quadrilateral", "hexahedron") else "DG"
        return fd.FunctionSpace(mesh, family, self.degree)

    def build_bcs(self):
        # Hyperbolic inflow/outflow is handled weakly in the residual.
        return []

    def build_velocity_field(self):
        """
        Build beta as a Firedrake Function in a vector CG space.
        """
        beta = fd.Function(self.W, name="velocity")

        if self._velocity_input is None:
            x = fd.SpatialCoordinate(self.mesh)
            if self.mesh.geometric_dimension() == 1:
                expr = fd.as_vector((fd.Constant(1.0),))
            elif self.mesh.geometric_dimension() == 2:
                expr = fd.as_vector((fd.Constant(1.0), fd.Constant(0.0))) ######################### DEFINITION
            else:
                expr = fd.as_vector(
                    tuple(fd.Constant(1.0 if i == 0 else 0.0)
                          for i in range(self.mesh.geometric_dimension()))
                )
            beta.interpolate(expr)
            return beta

        if isinstance(self._velocity_input, fd.Function):
            beta.assign(self._velocity_input)
            return beta

        if callable(self._velocity_input):
            x = fd.SpatialCoordinate(self.mesh)
            expr = self._velocity_input(x)
            beta.interpolate(expr)
            return beta

        if isinstance(self._velocity_input, (tuple, list)):
            expr = fd.as_vector(tuple(self._velocity_input))
            beta.interpolate(expr)
            return beta

        # Assume UFL-compatible expression
        beta.interpolate(self._velocity_input)
        return beta

    def residual(self, u_np1: fd.Function, u_n: fd.Function):
        v = fd.TestFunction(self.V)
        n = fd.FacetNormal(self.mesh)
        beta = self.beta

        # Positive outgoing normal flux, as in Firedrake's DG advection demo
        un = 0.5 * (fd.dot(beta, n) + abs(fd.dot(beta, n)))

        # Inflow boundary value
        if isinstance(self.inflow_value, (int, float)):
            u_in = fd.Constant(float(self.inflow_value))
        else:
            u_in = self.inflow_value

        return (
            ((u_np1 - u_n) / self.dt) * v * fd.dx
            - u_np1 * fd.div(v * beta) * fd.dx
            + fd.conditional(fd.dot(beta, n) < 0, v * fd.dot(beta, n) * u_in, 0.0) * fd.ds
            + fd.conditional(fd.dot(beta, n) > 0, v * fd.dot(beta, n) * u_np1, 0.0) * fd.ds
            + (v("+") - v("-"))
              * (un("+") * u_np1("+") - un("-") * u_np1("-")) * fd.dS
        )


class ImplicitDiffusionStepper(FiredrakeTimeStepper):
    """
    u_t - div(k grad u) = f
    Backward Euler:
        (u^{n+1} - u^n)/dt - div(k grad u^{n+1}) = f
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        dt: float,
        point_evaluator: np.ndarray = None,
        diffusivity: float = 1.0,
        forcing: float = 0.0,
        degree: int = 1,
        solver_parameters: Optional[dict] = None,
    ):
        self.degree = degree
        self.k = fd.Constant(diffusivity)
        self.f = fd.Constant(forcing)
        super().__init__(
            mesh=mesh,
            dt=dt,
            point_evaluator = point_evaluator,
            solver_parameters=solver_parameters
            )

    def build_function_space(self, mesh):
        return fd.FunctionSpace(mesh, "CG", self.degree)

    def build_bcs(self):
        # Replace as needed
        return [fd.DirichletBC(self.V, fd.Constant(0.0), "on_boundary")]

    def residual(self, u_np1: fd.Function, u_n: fd.Function):
        v = fd.TestFunction(self.V)

        return (
            ((u_np1 - u_n) / self.dt) * v * fd.dx
            + self.k * fd.dot(fd.grad(u_np1), fd.grad(v)) * fd.dx
            - self.f * v * fd.dx
        )


class IterativePoissonSolverStepper(FiredrakeTimeStepper):
    """
    Elliptic Poisson problem treated as a sequence of iterative-solver steps:

        -div(k grad u) = f  on unit square, u|_bdy = g

    One differentiable "step" applies *m* mass-matrix-preconditioned Richardson
    iterations (Jacobi-style warm start) starting from the previous iterate.

    Trained against high-tolerance references; generalized across iteration
    budgets (smaller *m* at test time = "coarser" solver = imperfect representation).
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        m_iters: int = 5,
        relaxation: float = 1.0,
        point_evaluator: np.ndarray = None,
        diffusivity: float = 1.0,
        forcing: float = 0.0,
        bc_value: float = 0.0,
        degree: int = 1,
        solver_parameters: Optional[dict] = None,
    ):
        self.degree = degree
        self.m_iters = m_iters
        self.relaxation = relaxation
        self.k = fd.Constant(diffusivity)
        self.f = fd.Constant(forcing)
        self.bc_value = bc_value
        super().__init__(
            mesh=mesh,
            dt=1.0,
            point_evaluator=point_evaluator,
            solver_parameters=solver_parameters,
        )

    def build_function_space(self, mesh):
        return fd.FunctionSpace(mesh, "CG", self.degree)

    def build_bcs(self):
        return [fd.DirichletBC(self.V, fd.Constant(self.bc_value), "on_boundary")]

    def residual(self, u_np1: fd.Function, u_n: fd.Function):
        """Exact variational residual (= solved by inherited .step())."""
        v = fd.TestFunction(self.V)
        return (
            fd.inner(self.k * fd.grad(u_np1), fd.grad(v)) * fd.dx
            - self.f * v * fd.dx
        )

    def _a_bilinear(self, u, v):
        return fd.inner(self.k * fd.grad(u), fd.grad(v)) * fd.dx

    def _L_linear(self, v):
        return self.f * v * fd.dx

    def iterative_step(self, u_n: fd.Function) -> fd.Function:
        """Apply *m_iters* mass-preconditioned Richardson correction iterations.

        Each iteration: r = b - A u;  M du = r;  u <- u + omega du.
        This is a differentiable operation through firedrake adjoint (each
        linear solve is taped as a KSP solve node in the annotation tape).
        """
        v = fd.TestFunction(self.V)
        u = fd.Function(self.V, name="poisson_iterate")
        u.assign(u_n)

        M_mat = fd.assemble(fd.inner(fd.TrialFunction(self.V), v) * fd.dx)
        du = fd.Function(self.V)

        inner_sp = {
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }

        for _ in range(self.m_iters):
            r = fd.assemble(
                self._L_linear(v) - fd.action(self._a_bilinear(u, v), v)
            )
            fd.solve(M_mat, du, r, solver_parameters=inner_sp)
            u.dat.data[:] += self.relaxation * du.dat.data[:]
            for bc in self.bcs:
                bc.apply(u)

        return u

    def build_torch_step_operator(self):
        fd.adjoint.continue_annotation()
        u_n = fd.Function(self.V, name="u_n_control_poisson")
        u_out = self.iterative_step(u_n)
        if isinstance(self.point_evaluator, np.ndarray):
            u_out = fd.assemble(fd.interpolate(u_out, self.P0DG))
        red = ReducedFunctional(u_out, Control(u_n))
        fd.adjoint.stop_annotating()
        return fem_operator(red)


def firedrake_field_to_torch(u: fd.Function, batched: bool = True) -> torch.Tensor:
    """
    Convert Firedrake Function to torch tensor using Firedrake's helper.
    Shape depends on the backend helper and space; for scalar CG it is typically flat.
    """
    return to_torch(u, batched=batched)


def append_time_channel(x_tensor: torch.Tensor, t: float) -> torch.Tensor:
    """
    Minimal helper if your NN expects [state, time].
    Extend later with coordinates if you want x,y,t,u explicitly.
    """
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(0)
    t_channel = torch.full(
        (x_tensor.shape[0], 1),
        fill_value=float(t),
        dtype=x_tensor.dtype,
        device=x_tensor.device,
    )
    return torch.cat([x_tensor, t_channel], dim=-1)

class FiredrakePINNSBasedSOLTrainer:
    """
    Hybrid trainer:
        u_{n+1}^{phys} = PDE(u_n)
        u_{n+1}^{corr} = u_{n+1}^{phys} + NN(features(u_{n+1}^{phys}, t_{n+1}))
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
        lift_chunk_size: int = 64
    ):
        self.physical_model = physical_model
        self.st_model = statistical_model
        self.optimizer = optimizer
        self.n_steps = simulation_steps
        self.dt = dt
        self.loss = loss
        #self.feature_builder = feature_builder or append_time_channel
        self.step_op = physical_model.build_torch_state_step_operator()
        self.observe_op = physical_model.build_torch_point_observation_operator()
        n_dofs = physical_model.V.dim()
        E = physical_model.build_dense_point_eval_matrix(
            dtype=torch.float32,
            device="cpu",
            chunk_size=lift_chunk_size,
        )
        self.grid_to_state_lift = TorchPointCloudLift(E, reg=lift_regularization)

        self.init_states_gt: List[fd.Function] = []
        self.T: List[float] = [0.0]

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

    def correct(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Model-corrector should implement the coordinate message passing
        #features = rearrange(self.feature_builder(state_tensor, t),"c h w-> 1 (h w) c").requires_grad_(True)
        correction = rearrange(self.st_model(
            rearrange(features,"1 (h w) c -> c h w", h = self.physical_model.evaluation_shape[0] ,w = self.physical_model.evaluation_shape[1])
            ), " c h w -> 1 (h w) c") # [u x t] 
        corrected = features[:,:,-1:] + correction # [c h w]
        return corrected, correction, features

    def forward_prediction_correction_from_state(
        self,
        state0_tensor: torch.Tensor,
        t0: float,
    ):
        states_pred = []
        states_corr = []
        states_in = []
        states_phys_v = []
        states_corrected_v = []

        current_t = t0
        current_v = state0_tensor
        current_v = current_v.float()
        

        for _ in range(self.n_steps):
            # Firedrake differentiable step
            phys_next_v = self.step_op(current_v) # [B p]

            current_t = current_t + self.dt
            phys_next_grid = self.observe_op(phys_next_v)
            features = rearrange(
                self.feature_builder(phys_next_grid, current_t),
                "c h w-> 1 (h w) c"
                ).requires_grad_(True)
            corrected_grid, corr_grid, features = self.correct(features) # [b p v], ?, [b p v]
            # corrected to embeded feature

            corr_v = self.grid_to_state_lift(corr_grid).to(
                dtype=phys_next_v.dtype,
                device=phys_next_v.device,
            )
            corrected_v = phys_next_v + corr_v

            states_in.append(features) # states_in.append(XTUp_1)
            states_corr.append(corr_grid)
            states_pred.append(corrected_grid)
            states_phys_v.append(phys_next_v)
            states_corrected_v.append(corrected_v)

            #current = corrected[:,:,0] # [B p v] # "squeeze"
            current_v = corrected_v

        return states_pred, states_corr, states_in

    def train(self, epochs: int, batch_size: int = 8):
        losses = []

        for _ in tqdm(range(epochs)):
            batch_pred = []
            batch_in = []

            for b in range(batch_size):
                idx = torch.randint(low=0, high=len(self.init_states_gt), size=(1,)).item()
                u0_fd = self.init_states_gt[idx]
                t0 = self.T[idx]

                u0_torch = firedrake_field_to_torch(u0_fd, batched=True).float()
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
    

    def predict_rollout(self, u0: fd.Function, t0: float, n_steps: int, spatial_sample: Optional[np.ndarray] = None):
        """
        Uses the trainer's internal forward_prediction_correction().

        Returns:
            pred_states : corrected predicted states
            input_states: corresponding trainer inputs used in loss(u, x)
            corr_states : corrections
            times       : times of predicted states
        """
        old_n_steps = self.n_steps
        self.n_steps = n_steps

        #with torch.no_grad():
        states_pred, states_corr, states_in = self.forward_prediction_correction_from_state(
            fd.ml.pytorch.to_torch(u0),t0)
        #uncorrected_sol = list(fd.ml.pytorch.from_torch(pred - corr, self.physical_model.V) for pred, corr in zip(states_pred, states_corr))
        uncorrected_sol = list(fd.ml.pytorch.from_torch(state_in[:,:,-1], self.physical_model.V) for state_in in states_in)
        # over sample
        if isinstance(spatial_sample,np.ndarray):
            vom = fd.VertexOnlyMesh(
                                self.physical_model.V.mesh(),
                                spatial_sample.reshape(-1,self.physical_model.V.mesh().geometric_dimension()),
                                reorder = False
                                )
            P0DG_ = fd.FunctionSpace(vom, "DG", 0)

            # List[ [b x y (xytv)] ] - [b x y (xytv) t]
            uncorrected_sol_h = torch.stack(list(
                self.feature_builder_finer(
                        fd.ml.pytorch.to_torch(fd.assemble(fd.interpolate(u_sol, P0DG_))).requires_grad_(True), (t0 + self.physical_model.dt.values()*(i+1)), spatial_sample, P0DG_
                            ) for i,u_sol in enumerate(uncorrected_sol)), axis = -1 )
            
            uncorrected_sol = rearrange(uncorrected_sol_h, "V x y t -> t (x y) V")

            states_pred = list(u_sol + \
                               rearrange(self.st_model(rearrange(u_sol,"(x y) V -> V x y",
                                                               x = spatial_sample.shape[0],
                                                               y = spatial_sample.shape[1],
                                                               V = (self.physical_model.V.mesh().geometric_dimension() + 1 + 1) # TODO: extend to multiple output space
                                                               ))," V x y -> (x y) V") for u_sol in uncorrected_sol)
            
            states_pred = torch.stack(states_pred,axis = 0) # [t p V]
                
            

        self.n_steps = old_n_steps

        # If implementation includes corrected initial state, align lengths
        if len(states_pred) == n_steps + 1:
            states_pred = states_pred[1:]

        if len(states_in) > len(states_pred):
            states_in = states_in[-len(states_pred):]

        if len(states_corr) > len(states_pred):
            states_corr = states_corr[-len(states_pred):]

        times = [t0 + (k + 1) * self.dt for k in range(len(states_pred))]
        return states_pred, states_in, states_corr, times, uncorrected_sol
    
class FiredrakePINNSBasedSOLTrainerCNN(FiredrakePINNSBasedSOLTrainer):
  def __init__(self,**args):
    super().__init__(**args)

  def feature_builder(self,u: torch.Tensor,t: float):
    u = u.reshape(self.physical_model.evaluation_shape[:-1]+(1,))

    V = fd.VectorFunctionSpace(self.physical_model.P0DG.mesh(), "DG", 0)
    X = fd.ml.pytorch.to_torch(fd.Function(V).interpolate(fd.SpatialCoordinate(self.physical_model.mesh))) # [eval_points dim]
    X = X.reshape(self.physical_model.evaluation_shape) # [p_dims x y]
    t = torch.tile(torch.tensor(t),(self.physical_model.evaluation_shape[:2])+(1,))
    return torch.concat((X,t,u),axis=-1).transpose(0,-1).float()
  
  def feature_builder_finer(self,u: torch.Tensor, t: float, eval_points: np.ndarray, fs: fd.FunctionSpace):
    u = u.reshape(eval_points.shape[:-1]+(1,))

    V = fd.VectorFunctionSpace(fs.mesh(), "DG", 0)
    X = fd.ml.pytorch.to_torch(fd.Function(V).interpolate(fd.SpatialCoordinate(fs.mesh()))) # [eval_points dim]
    X = X.reshape(eval_points.shape) # [p_dims x y]
    t = torch.tile(torch.tensor(t),(eval_points.shape[:2])+(1,))
    return torch.concat((X,t,u),axis=-1).transpose(0,-1).float()
  

class FiredrakePINNSBasedSOLTrainerConsistentCNN(FiredrakePINNSBasedSOLTrainer):
    def __init__(self,**args):
        super().__init__(**args)

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
        #features = self.feature_builder_finer(u_points, t, )
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
