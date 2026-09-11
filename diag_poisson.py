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

    print("=== grad test C: residual WITHOUT u_n dependence (root-cause demo) ===")
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
    fd.adjoint.continue_annotation()
    opC = dp.build_torch_state_step_operator()
    xr = torch.rand(n_dofs, dtype=torch.float32).requires_grad_(True)
    yC = opC(xr)
    print("yC.requires_grad", yC.requires_grad)
    if yC.requires_grad:
        gC = torch.autograd.grad(yC.sum(), xr)[0]
        print("gradC norm", float(gC.norm()), "nz", int((gC != 0).sum()))
        print("gradC head", gC[:6].tolist())

    print("=== grad test D: real trainer forward+backward ===")
    import logging
    warns = []
    class _H(logging.Handler):
        def emit(self, record):
            warns.append(record.getMessage())
    hlog = _H()
    _root = logging.getLogger()
    _root.setLevel(logging.WARNING)
    _root.addHandler(hlog)

    from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model as cnn_model
    from DL_models.PINNS.Residual_losses import poisson_residual_loss
    from trainer.Trainer import FiredrakePINNSBasedSOLTrainerCNN

    cnn = cnn_model()
    for p in cnn.parameters():
        if p.dim() >= 2:
            torch.nn.init.kaiming_normal_(p)
        else:
            torch.nn.init.normal_(p)
    tr = FiredrakePINNSBasedSOLTrainerCNN(
        physical_model=stepper,
        statistical_model=cnn,
        optimizer=torch.optim.Adam(cnn.parameters(), lr=1e-4),
        simulation_steps=2,
        dt=1.0,
        loss=lambda u, x: (poisson_residual_loss(u, x, K=1.0, f=1.0)) ** 2,
    )
    u0_f = fd.Function(stepper.V).interpolate(fd.Constant(0.0))
    u0t = fd.ml.pytorch.to_torch(u0_f).float()
    pred, _, sin_, _ = tr.forward_prediction_correction_from_state(u0t, 0.0)
    tot = sum(torch.mean(tr.loss(p, i)) for p, i in zip(pred, sin_))
    tot.backward()
    nz = [int((p.grad is not None and p.grad.sum() != 0)) for p in cnn.parameters() if p.requires_grad]
    print("test D total_loss", float(tot.detach().cpu()))
    print("test D cnn grad nz per param:", nz, "n_trainable", sum(1 for p in cnn.parameters() if p.requires_grad))
    print("test D 'Adjoint value is None' count:", sum(1 for w in warns if "Adjoint value is None" in w))
    with torch.no_grad():
        pv = pred[0]
        print("test D pred0 has_nan", bool(pv.isnan().any()), "max", float(pv.abs().max()), "mean", float(pv.mean()))
    print("test D pred0 grad_fn:", type(pred[0].grad_fn).__name__ if pred[0].grad_fn else None)
    _root.removeHandler(hlog)

    print("=== grad test E: stability scan (m=5, 10x10 CG1) ===")
    def iter_stats(pc, rel):
        mesh10 = fd.UnitSquareMesh(10, 10)
        g11 = np.stack(np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11)), axis=-1)
        ste = T.IterativePoissonSolverStepper(
            mesh=mesh10, m_iters=5, relaxation=rel, diffusivity=1.0,
            forcing=1.0, bc_value=0.0, degree=1, point_evaluator=g11,
            solver_parameters={"snes_type": "ksponly", "ksp_type": "preonly", "pc_type": pc},
        )
        out = ste.iterative_step(fd.Function(ste.V))
        d = np.asarray(out.dat.data)
        return float(np.max(d)), float(np.min(d)), bool(np.isnan(d).any())
    fd.adjoint.pause_annotation()
    try:
        for pc in ["lu", "jacobi"]:
            for rel in [1.0, 0.5, 0.1]:
                try:
                    mx, mn, nan = iter_stats(pc, rel)
                    print("pc", pc, "rel", rel, "max", mx, "min", mn, "nan", nan)
                except Exception as ex:
                    print("pc", pc, "rel", rel, "ERR", type(ex).__name__, str(ex)[:100])
    finally:
        fd.adjoint.continue_annotation()

    print("DONE")


if __name__ == "__main__":
    main()