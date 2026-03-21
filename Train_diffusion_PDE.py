from PINNoDiffPhys.DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
import numpy as np
import torch
import firedrake as fd
from PINNoDiffPhys.trainer.Trainer import ImplicitDiffusionStepper,FiredrakePINNSBasedSOLTrainer
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
