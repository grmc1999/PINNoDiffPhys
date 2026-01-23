import sys
import os
import torch
from DL_models.PINNS.losses import PDE_res,Discriminator_loss,Generator_loss,PDE_GAN_loss
from .PointNet import *
from .MLP import *
sys.path.append(os.path.join("..","..","DL_models"))
from DL_models.PINNS.losses import PINN_loss

class PINN_base(torch.nn.Module):
    def __init__(self):
        """
        Provides basic functionalities for PDE behavior
        """
        super(PINN_base,self).__init__()
        print("PINNS_init")
    def derivatives(self,u,x,n):
        print(n)
        if n==0:
            return u
        else:
            du=torch.autograd.grad(u,x,
            grad_outputs=torch.ones_like(x).to(u.device),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
            )[0]
        return self.derivatives(du,x,n-1)


class PINN_vanilla(PINN_base):
    def __init__(self,G_expr,u_dim,args_PDE_res,args_PDE_sup,weights={"Residual_loss":1.,"Supervised_loss":1.}):
        super(PINN_vanilla,self).__init__()
        self.G_model=eval(G_expr)
        self.loss=PINN_loss(args_PDE_res,args_PDE_sup,weights)
        self.u_dims=u_dim

    def forward(self,X):
        u=self.G_model(X)
        return u

    def compute_loss(self,X,U):
        U_=self.forward(X)
        return self.loss(X,U,U_)