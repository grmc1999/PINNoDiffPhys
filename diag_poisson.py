import sys
import numpy as np
import torch
import firedrake as fd
from firedrake.adjoint import Control, ReducedFunctional
from firedrake.ml.pytorch.fem_operator import fem_operator
from pyadjoint import get_working_tape

import trainer.Trainer as T

fd.adjoint.pause_annotation()


def main():
    mesh = fd.UnitSquareMesh(4, 4)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3)), axis=-1)
    stepper = T.IterativePoissonSolverStepper(
        mesh=mesh, m_iters=3, relaxation=1.0, diffusivity=1.0,
        forcing=1.0, bc_value=0.0, degree=1, point_evaluator=grid,
    )
    n_dofs = stepper.V.dim()
    print("n_dofs", n_dofs)

    tape = get_working_tape()
    base_blocks = len(tape.get_blocks())

    op = stepper.build_torch_step_operator()
    print("blocks right after build:", [type(b).__name__ for b in tape.get_blocks()[base_blocks:]])

    x = torch.rand(n_dofs, dtype=torch.float32)
    print("op(x):", op(x))

    print("=== grad test: full step operator (with interpolate to P0DG grid) ===")
    base_blocks = len(tape.get_blocks())
    x = torch.rand(n_dofs, dtype=torch.float32)
    y = op(x) if hasattr(op, "forward") else op.__call__(x)
    xr = x.detach().clone().requires_grad_(True)
    y2 = op(xr)
    print("y.requires_grad", y2.requires_grad, "grad_fn", type(y2.grad_fn).__name__)
    if y2.requires_grad:
        g = torch.autograd.grad(y2.sum(), xr)[0]
        print("grad norm", float(g.norm()), "nz", int((g != 0).sum()))
    print("blocks for forward:", [(type(b).__name__, getattr(b, "dependencies", ["?"])) for b in tape.get_blocks()[base_blocks:]])

    print("=== grad test B: red on raw Function u_out (NO interpolate) ===")
    fd.adjoint.continue_annotation()
    base_blocks = len(tape.get_blocks())
    u_n = fd.Function(stepper.V, name="u_n_control_poisson")
    u_out = stepper.iterative_step(u_n)
    red = ReducedFunctional(u_out, Control(u_n))
    fd.adjoint.stop_annotating()
    opB = fem_operator(red)
    print("blocks for iterative_step chain:", [type(b).__name__ for b in tape.get_blocks()[base_blocks:]])
    xr = torch.rand(n_dofs, dtype=torch.float32).requires_grad_(True)
    yB = opB(xr)
    print("yB.requires_grad", yB.requires_grad, [type(b).__name__ for b in tape.get_blocks()[base_blocks:]])
    if yB.requires_grad:
        gB = torch.autograd.grad(yB.sum(), xr)[0]
        print("gradB norm", float(gB.norm()), "nz", int((gB != 0).sum()))

    print("=== grad test C: diffusion-style direct variational solve ===")
    class DirectPoisson(T.FiredrakeTimeStepper):
        def build_function_space(self, mesh):
            return fd.FunctionSpace(mesh, "CG", 1)
        def build_bcs(self):
            return [fd.DirichletBC(self.V, fd.Constant(0.0), "on_boundary")]
        def residual(self, u_np1, u_n):
            v = fd.TestFunction(self.V)
            return (fd.inner(fd.grad(u_np1), fd.grad(v)) * fd.dx
                    - fd.Constant(1.0) * v * fd.dx)
    dp = DirectPoisson(mesh=mesh, dt=1.0, point_evaluator=grid)
    opC = dp.build_torch_state_step_operator()
    xr = torch.rand(n_dofs, dtype=torch.float32).requires_grad_(True)
    yC = opC(xr)
    print("yC.requires_grad", yC.requires_grad)
    if yC.requires_grad:
        gC = torch.autograd.grad(yC.sum(), xr)[0]
        print("gradC norm", float(gC.norm()), "nz", int((gC != 0).sum()))
        print("gradC head", gC[:6].tolist())

    print("DONE")


if __name__ == "__main__":
    main()