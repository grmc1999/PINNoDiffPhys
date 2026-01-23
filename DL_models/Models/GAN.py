import sys
import os
import torch
from .MLP import MLP
sys.path.append(os.path.join("..","..","DL_models"))
from DL_models.PINNS.losses import PDE_res,Discriminator_loss,Generator_loss,PDE_GAN_loss
from .PINN import PINN_base
from .PointNet import *



class GAN_PI_base(PINN_base):
    def __init__(self,u_dim,args_Gen,args_PDE_res,args_PDE_sup,distribution_args, weights={"generative_posterior_loss":1.,
                                                                "generative_entropy_loss":1.,
                                                                "PDE_residual_loss":1.,
                                                                "PDE_supervised_loss":1.}):
        super(GAN_PI_base,self).__init__()
        self.loss=PDE_GAN_loss(args_Gen,args_PDE_res,args_PDE_sup,weights=weights)
        self.distribution_args=distribution_args
        self.u_dims=u_dim

    
    def Generate(self,z,X): # X [batch, n_points, coords]
        u=self.G_model(torch.concatenate([z,X],axis=-1))

    def Generate_forward(self,X):
        #z=torch.normal(**self.distribution_args)
        z=torch.normal(
            #list(torch.ones_like(torch.zeros((3,4,6))).shape[:-1])+[4]
            mean=torch.zeros(list(X.shape[:-1])+[self.u_dims]),
            std=torch.ones(list(X.shape[:-1])+[self.u_dims])
        ).to(X.device)
        u=self.G_model(torch.concatenate([z,X],axis=-1))
        return u
    
    def Posterior_forward(self,X,u_):
        z=self.P_model(torch.concatenate([u_,X],axis=-1))
        return z

    def Discriminate(self,u):
        y=self.D_model(u)
        return y

    def compute_loss(self,X,U):
        logits_G=self.Generate_forward(X)
        logits_P=self.Posterior_forward(X,logits_G)
        logits_F=self.Discriminate(logits_G)
        logits_R=self.Discriminate(U)
        return self.loss(logits_G,logits_P,logits_F,logits_R,X,U)


class GAN_PI(GAN_PI_base):
    def __init__(self,G_params,P_params,D_params,args_Gen,args_PDE_res,args_PDE_sup,distribution_args,weights={"generative_posterior_loss":1.,
                                                                "generative_entropy_loss":1.,
                                                                "PDE_residual_loss":1.,
                                                                "PDE_supervised_loss":1.}):
        super(GAN_PI,self).__init__(0,args_Gen,args_PDE_res,args_PDE_sup,distribution_args,weights)
        self.G_model=MLP(**G_params)
        self.P_model=MLP(**P_params)
        self.D_model=MLP(**D_params)
        #self.loss=PDE_GAN_loss(args_Gen,args_PDE_res,args_PDE_sup,weights=weights)
        #self.distribution_args=distribution_args
        self.u_dims=self.G_model.layer_sizes[-1]

    
class General_architecture_GAN_PI(GAN_PI_base):
    def __init__(self,G_expr,P_expr,D_expr,**args):
        super(General_architecture_GAN_PI,self).__init__(**args)
        self.G_model=eval(G_expr)
        self.P_model=eval(P_expr)
        self.D_model=eval(D_expr)
        




