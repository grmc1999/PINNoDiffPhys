from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type

from phi.torch.flow import Field, Solve, advect, diffuse, fluid, math


class BasePDEModel:
    def __init__(self, domain: Field, dt: float) -> None:
        self.domain = domain
        self.dt = dt

    def step(self, field: Field) -> Field:
        raise NotImplementedError


class NavierStokesModel(BasePDEModel):
    def __init__(self, domain: Field, dt: float, diffusivity: float = 0.01) -> None:
        super().__init__(domain, dt)
        self.diffusivity = diffusivity

    def momentum_eq(self, u: Field, u_prev: Field, dt: float) -> Field:
        diffusion_term = dt * diffuse.implicit(u, self.diffusivity, dt=dt, correct_skew=False)
        advection_term = dt * advect.semi_lagrangian(u, u_prev, dt)
        return u + advection_term + diffusion_term

    def implicit_time_step(self, v: Field, dt: float) -> Field:
        v = math.solve_linear(
            self.momentum_eq,
            v,
            Solve("CG-adaptive", 1e-2, 1e-2, x0=v),
            u_prev=v,
            dt=-dt,
        )
        v, _p = fluid.make_incompressible(
            v,
            solve=Solve("CG-adaptive", 1e-2, 1e-2),
        )
        return v

    def step(self, v: Field) -> Field:
        return self.implicit_time_step(v, self.dt)


class DiffusionModel(BasePDEModel):
    def __init__(
        self,
        domain: Field,
        dt: float,
        diffusivity: float = 0.01,
        use_implicit: bool = True,
    ) -> None:
        super().__init__(domain, dt)
        self.diffusivity = diffusivity
        self.use_implicit = use_implicit

    def step(self, field: Field) -> Field:
        if self.use_implicit:
            return diffuse.implicit(field, self.diffusivity, dt=self.dt, correct_skew=False)
        return diffuse.explicit(field, self.diffusivity, dt=self.dt, correct_skew=False)


PDE_MODEL_REGISTRY: Dict[str, Type[BasePDEModel]] = {
    "navier_stokes": NavierStokesModel,
    "diffusion": DiffusionModel,
}


def create_pde_model(name: str, domain: Field, dt: float, **kwargs: Any) -> BasePDEModel:
    try:
        model_class = PDE_MODEL_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PDE_MODEL_REGISTRY))
        raise ValueError(f"Unknown PDE model '{name}'. Available: {valid}.") from exc
    return model_class(domain, dt=dt, **kwargs)


@dataclass(frozen=True)
class PDEModelConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self, domain: Field, dt: float) -> BasePDEModel:
        return create_pde_model(self.name, domain, dt=dt, **self.kwargs)
