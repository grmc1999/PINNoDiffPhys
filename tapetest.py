import firedrake as fd
import torch

from pyadjoint import get_working_tape, annotate_tape
from firedrake.adjoint import ReducedFunctional, Control
from firedrake.ml.pytorch import fem_operator

import sys

_log = open("/code/tapetest_out.txt", "w", buffering=1)

def log(*a):
    print(*a)
    print(*a, file=_log)

sys.stdout = _log

mesh = fd.UnitSquareMesh(4, 4)
P1 = fd.FunctionSpace(mesh, "CG", 1)
vom = fd.VertexOnlyMesh(mesh, [[0.1, 0.1], [0.3, 0.4], [0.9, 0.9]], reorder=False)
DG0 = fd.FunctionSpace(vom, "DG", 0)

def nb():
    t = get_working_tape()
    return len(t.get_blocks()) if t is not None else -1

log("PYADJOINT version:", __import__("pyadjoint").__version__)

try:
    src = fd.adjoint.stop_annotating
    log("stop_annotating type:", type(src).__name__)
except Exception as e:
    log("no fd.adjoint.stop_annotating:", e)

fd.adjoint.continue_annotation()
log("annotate after continue_annotation():", annotate_tape())

u = fd.Function(P1, name="u_test")
u.interpolate(fd.SpatialCoordinate(mesh)[0] + 2 * fd.SpatialCoordinate(mesh)[1])
interp = fd.assemble(fd.interpolate(u, DG0))
rf = ReducedFunctional(interp, Control(u))
op = fem_operator(rf)
log("blocks after build:", nb())

x = torch.randn(P1.dim(), dtype=torch.float32)
y1 = op(x)
log("blocks after replay1:", nb(), "y1sum=", float(y1.sum()))
y2 = op(x)
log("blocks after replay2:", nb(), "y2sum=", float(y2.sum()))

fd.adjoint.stop_annotating()
log("annotate after stop_annotating():", annotate_tape())

v2 = fd.Function(P1, name="v2_test")
i2 = fd.assemble(fd.interpolate(v2, DG0))
log("blocks after guarded interpolate:", nb())
op2 = fem_operator(ReducedFunctional(i2, Control(v2)))
y3 = op2(x)
log("blocks after guarded replay:", nb(), "y3sum=", float(y3.sum()))

fd.adjoint.continue_annotation()
log("annotate after continue_annotation() again:", annotate_tape())
log("TAPETEST_DONE")
_log.close()