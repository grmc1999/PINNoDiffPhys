from phi.torch.flow import Box, Field, UniformGrid, tensor, vec
import torch

from DL_models.Models.CNN_models import simple_cnn_model
from Physical_models.pde_models import PDEModelConfig
from Train_differentiable_physics_with_PINNs import PINNS_based_SOL_trainer, PINNSTrainerDependencies


def advection_loss(u, _x):
    return torch.mean(u**2)


def build_trainer(grid_size=64, time_step=0.05):
    geometry = UniformGrid(x=grid_size, y=grid_size, bounds=Box(x=1.0, y=1.0))
    field = Field(geometry, values=tensor(0.0))
    model = simple_cnn_model().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    physical_model = PDEModelConfig(name="advection", kwargs={"velocity": vec(x=1.0, y=0.0)})
    dependencies = PINNSTrainerDependencies(
        field=field,
        physical_model=physical_model,
        statistical_model=model,
        optimizer=optimizer,
        simulation_steps=5,
        time_step=time_step,
        loss=advection_loss,
    )
    return PINNS_based_SOL_trainer.from_dependencies(dependencies)


if __name__ == "__main__":
    trainer = build_trainer()
    print("Initialized advection (hyperbolic) trainer:", trainer)
