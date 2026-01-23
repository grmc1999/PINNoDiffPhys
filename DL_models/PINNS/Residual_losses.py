import torch
from .utils import x_grad,vector_grad


def incompresibble_fluid_loss(up,xt,mu=1,rho=1):
    l=0
    # x-velocity components
    l+=x_grad(up,xt,0,1)[...,2] # dudt
    l+=torch.sum(up[...,:1]*x_grad(up,xt,0,1)[...,:2],axis=-1) # u * grad u
    l+=(mu/rho)*(x_grad(up,xt,2,1)[...,0]) #  dpdx
    l-=(mu/rho)*torch.sum(x_grad(up,xt,0,2)[...,:2],axis=-1) # grad**2 u
    # y-velocity components
    l+=x_grad(up,xt,1,1)[...,2] # dvdt
    l+=torch.sum(up[...,1:2]*x_grad(up,xt,0,1)[...,:2],axis=-1) # v * grad v
    l+=(mu/rho)*(x_grad(up,xt,2,1)[...,1]) #  dpdy
    l-=(mu/rho)*torch.sum(x_grad(up,xt,1,2)[...,:2],axis=-1) # grad**2 v
    return l

def incompresibble_fluid_3D_loss(up,xt,mu=1,rho=1):
    """
    assumes: up [u v w p] , xt [x y z t]"""
    l=0
    # x-velocity components
    l+=x_grad(up,xt,0,1)[...,3] # dudt
    l+=torch.sum(up[...,:1]*x_grad(up,xt,0,1)[...,:3],axis=-1) # u * grad u
    l+=(mu/rho)*(x_grad(up,xt,3,1)[...,0]) #  dpdx
    l-=(mu/rho)*torch.sum(x_grad(up,xt,0,2)[...,:3],axis=-1) # grad**2 u
    # y-velocity components
    l+=x_grad(up,xt,1,1)[...,3] # dvdt
    l+=torch.sum(up[...,1:2]*x_grad(up,xt,0,1)[...,:3],axis=-1) # v * grad v
    l+=(mu/rho)*(x_grad(up,xt,3,1)[...,1]) #  dpdy
    l-=(mu/rho)*torch.sum(x_grad(up,xt,1,2)[...,:3],axis=-1) # grad**2 v
    # z-velocity components
    l+=x_grad(up,xt,2,1)[...,3] # dvdt
    l+=torch.sum(up[...,2:3]*x_grad(up,xt,0,1)[...,:3],axis=-1) # w * grad w
    l+=(mu/rho)*(x_grad(up,xt,3,1)[...,2]) #  dpdz
    l-=(mu/rho)*torch.sum(x_grad(up,xt,1,2)[...,:3],axis=-1) # grad**2 w
    return l



#OsWsPoPwBo
def one_phase_darcy_flow_loss_deterministic_K(Uv,xtk,mu=0.3,porosity=0.15):
    """
    assumes: up [p] , xt [x y t]
    """
    l=0
    # grad n of U-ith comp wrt to x, indexing to choose x-ith derivative
    #K=0.5*torch.exp(-1.*((xtk[...,0:1]-0.5)**2 + (xtk[...,1:2]-0.5)**2)/0.1)
    K=0.5*torch.exp(-1.*torch.sum((xtk-0.5)**2)/0.1)
    #K= 1.0/(torch.exp(-1*(((xtk[...,1:2]-0.5 -0.1*torch.sin(10*xtk[...,0:1]))/0.1)*((xtk[...,1:2]-0.5 -0.1*torch.sin(10*xtk[...,0:1]))/0.1))))
    
    l+=vector_grad( # oil pressure gradient
        K*x_grad(Uv,xtk,0,1)[...,:2]
            ,xtk).squeeze(-1).sum(-1)/mu
    
    l+=porosity * x_grad(Uv,xtk,0,1)[...,2] # Oil saturatin change
    
    return l

def one_phase_darcy_flow_loss(Uv,xtk,mu=0.3,porosity=0.15):
    """
    assumes: up [p] , xt [x y ki kj t]
    """
    l=0
    # grad n of U-ith comp wrt to x, indexing to choose x-ith derivative
    #K=torch.stack([xtk[...,2],xtk[...,3]],axis=2)
    K=1.0 # Constant
    
    l+=vector_grad( # oil pressure gradient
        K*x_grad(Uv,xtk,0,1)[...,:2]
            ,xtk).squeeze(-1).sum(-1)/mu
    
    l+=porosity * x_grad(Uv,xtk,0,1)[...,2] # Oil saturatin change
    
    return l

#OsWsPoPwBo
def two_phase_darcy_flow_loss(Uv,xtk,muw=0.32,muo=1.295,porosity=0.2):
    l=0
    # grad n of U-ith comp wrt to x, indexing to choose x-ith derivative
    
    #Ko=torch.stack(
    #[torch.stack([xtk[...,3],torch.zeros_like(xtk[...,3])],axis=2),
    # torch.stack([torch.zeros_like(xtk[...,3]),xtk[...,4]],axis=2)],axis=3)
    Ko=torch.stack([xtk[...,3],xtk[...,4]],axis=2)
    
    #Kw=torch.stack(
    #[torch.stack([torch.ones_like(xtk[...,3]),torch.zeros_like(xtk[...,3])],axis=2),
    # torch.stack([torch.zeros_like(xtk[...,3]),torch.ones_like(xtk[...,4])],axis=2)],axis=3)
    Kw=torch.stack([xtk[...,3],xtk[...,4]],axis=2)
    #l+=vector_grad( # oil pressure gradient
    #        torch.tensordot(
    #        x_grad(Uv,xtk,2,1)[...,:2],
    #        Ko,dims=([-1],[1])),xtk).sum(-1)/muo
    
    l+=vector_grad( # oil pressure gradient
        Ko*x_grad(Uv,xtk,2,1)[...,:2]
            ,xtk).squeeze(-1).sum(-1)/muo
    
    l+=porosity * x_grad(Uv,xtk,0,1)[...,2] # Oil saturatin change
#    l+=vector_grad( # oil pressure gradient
#            torch.tensordot(
#            x_grad(Uv,xtk,3,1)[...,:2],
#            Ko,dims=([-1],[1])),xtk).sum(-1)/muw
    
    l+=vector_grad( # water pressure gradient
        Kw*x_grad(Uv,xtk,3,1)[...,:2]
            ,xtk).squeeze(-1).sum(-1)/muo
    l+=porosity * x_grad(Uv,xtk,1,1)[...,2] # water saturatin change
    
    return l