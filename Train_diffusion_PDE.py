from PINNoDiffPhys.DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
import numpy as np
import torch
import firedrake as fd
from PINNoDiffPhys.trainer.Trainer import ImplicitDiffusionStepper,FiredrakePINNSBasedSOLTrainerCNN
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_definition", type=str, default="fd.UnitSquareMesh(10,10)")
    parser.add_argument("--dt", type=float, default=0.1)

    args = parser.parse_args()
    st_model = simple_dual_space_with_time_derivative_cnn_model()

    mesh = eval(args.mesh_definition)

    X = np.stack(np.meshgrid(
        np.linspace(0,1,11),
        np.linspace(0,1,11)
        ),axis = -1)
    
    
    ph_model = ImplicitDiffusionStepper(
        mesh = mesh,
        dt = args.dt,
        point_evaluator = X,
    )

    # Creating IC
    X = fd.SpatialCoordinate(mesh)
    un = fd.Function(ph_model.V).interpolate(0.5*fd.exp(.5*((X[0]-0.5)**2 + (X[1]-0.5)**2 - 0.1)**2 - 1))

    T = FiredrakePINNSBasedSOLTrainerCNN(
            physical_model = ph_model,
            statistical_model = st_model,
            optimizer = torch.optim.Adam(st_model.parameters(),lr=1e-4),
            simulation_steps = 5,
            dt = 0.1,
            loss = lambda u,x: diffusion_loss(u,x,K = 1.0),
            feature_builder = None,
    )
