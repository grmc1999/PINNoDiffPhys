from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from einops import rearrange
import torch
import firedrake as fd

from firedrake.adjoint import Control, ReducedFunctional
from firedrake.ml.pytorch.fem_operator import fem_operator, to_torch

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

        if isinstance(self.point_evaluator, np.ndarray):
            self.evaluation_shape = self.point_evaluator.shape
            vom = fd.VertexOnlyMesh(
                                    mesh,
                                    self.point_evaluator.reshape(-1,mesh.geometric_dimension()),
                                    reorder = False
                                    )
            self.P0DG = fd.FunctionSpace(vom, "DG", 0)

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
        super().__init__(mesh=mesh, dt=dt,point_evaluator = point_evaluator, solver_parameters=solver_parameters)

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
        feature_builder: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    ):
        self.physical_model = physical_model
        self.step_op = physical_model.build_torch_step_operator()

        self.st_model = statistical_model
        self.optimizer = optimizer
        self.n_steps = simulation_steps
        self.dt = dt
        self.loss = loss
        self.feature_builder = feature_builder or append_time_channel

        self.init_states_gt: List[fd.Function] = []
        self.T: List[float] = [0.0]

    def generate_ground_truth(self, u0: fd.Function, n_rollout: int):
        self.init_states_gt = [fd.Function(u0, name="gt_0")]
        self.T = [0.0]

        u = fd.Function(u0, name="gt_state")
        for _ in range(n_rollout):
            u = self.physical_model.step(u)
            self.init_states_gt.append(fd.Function(u))
            self.T.append(self.T[-1] + self.dt)

    def correct(self, state_tensor: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Model-corrector should implement the coordinate message passing
        #features = self.feature_builder(state_tensor, t).requires_grad_(True) # [c h w]
        features = rearrange(self.feature_builder(state_tensor, t),"c h w-> 1 (h w) c").requires_grad_(True)
        # Reshape
        #state_tensor = rearrange(state_tensor,"B p->")
        correction = self.st_model(
            rearrange(features)
            ) # [u x t] 
        corrected = state_tensor + correction # [c h w]
        corrected = rearrange(corrected,"c h w -> 1 (h w) c") # [b p v]
        return corrected, correction,rearrange(features,"c h w -> 1 (h w) c")

    def forward_prediction_correction_from_state(
        self,
        state0_tensor: torch.Tensor,
        t0: float,
    ):
        states_pred = []
        states_corr = []
        states_in = []

        current = rearrange(self.feature_builder(state0_tensor,current_t),"c h w -> 1 (h w) c").requires_grad_(True) # [B p] -> [B p v]
        current_t = t0

        for _ in range(self.n_steps):
            # Firedrake differentiable step
            phys_next = self.step_op(current[:,:,-1]) # [B p]

            current_t = current_t + self.dt
            # TODO: Coordinate encoder should be outside of the corrector (model)
            corrected, corr, features = self.correct(phys_next, current_t) # [b p v], ?, [b p v]
            # corrected to embeded feature

            states_in.append(features) # states_in.append(XTUp_1)
            states_corr.append(corr)
            states_pred.append(corrected)

            current = rearrange(self.feature_builder(corrected,current_t),"c h w -> 1 (h w) c").requires_grad_(True)
            #current = corrected # [B p v]

        return states_pred, states_corr, states_in

    def train(self, epochs: int, batch_size: int = 8):
        losses = []

        for _ in range(epochs):
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
                total_loss = total_loss + self.loss(u_pred.unsqueeze(-1), u_in)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            losses.append(float(total_loss.detach().cpu()))

        return losses
    
class FiredrakePINNSBasedSOLTrainerCNN(FiredrakePINNSBasedSOLTrainer):
  def __init__(self,**args):
    super().__init__(**args)
    del self.feature_builder

  def feature_builder(self,u,t):
    u = u.reshape(self.physical_model.evaluation_shape[:-1]+(1,))

    V = fd.VectorFunctionSpace(self.physical_model.P0DG.mesh(), "DG", 0)
    X = fd.ml.pytorch.to_torch(fd.Function(V).interpolate(fd.SpatialCoordinate(self.physical_model.mesh))) # [eval_points dim]
    X = X.reshape(self.physical_model.evaluation_shape) # [p_dims x y]
    t = torch.tile(torch.tensor(t),(self.physical_model.evaluation_shape[:2])+(1,))
    return torch.concat((X,t,u),axis=-1).transpose(0,-1).float()