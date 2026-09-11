import torch
from .utils import x_grad,vector_grad


def diffusion_loss(u, xt, K, dim=2):
    """
    r = du/dt - K * Laplacian(u) = 0

    Feature/coordinate tensor `xt` has channels [x, y, t, u] for a 2D PDE,
    so the temporal coordinate sits at index `dim` and spatial coords at 0..dim-1.
    """
    du = x_grad(u, xt, 0, 1)                  # first partials wrt [x, y, t, u]
    u_t = du[..., dim]                        # du/dt (t channel is at index dim)
    d2u = x_grad(u, xt, 0, 2)                 # second partials wrt [x, y, t, u]
    lap = torch.sum(d2u[..., :dim], axis=-1)  # d2u/dx2 + d2u/dy2
    return u_t - K * lap

def poisson_residual_loss(u, xt, K=1.0, f=1.0, dim=2):
    """
    r = -K * Laplacian(u) - f   (interior grid points)

    Evaluated as a *finite-difference* Laplacian of the grid values u (not via
    autograd wrt coordinates): u is [.., P, 1] field values on a uniform 2D
    grid, xt carries [.., P, C] features whose first dim coordinates are x,y
    (row-major from np.meshgrid(..., indexing='xy')). This keeps the residual
    fully torch-differentiable into the CNN correction regardless of how u was
    obtained.
    """
    if u.ndim != 3:
        u = u.reshape(u.shape[0], -1, 1)
    B, P, _ = u.shape
    xc = xt[..., 0]
    yc = xt[..., 1]
    xs = torch.sort(xc[0, :].unique())[0]
    ys = torch.sort(yc[0, :].unique())[0]
    H, W = int(len(ys)), int(len(xs))
    if H * W != P:
        H = int(round(P ** 0.5))
        W = int(round(P / H))
    if H < 3 or W < 3:
        dxx = torch.zeros_like(u[..., 0])
        return -K * dxx - f
    hx = float((xs[1] - xs[0]))
    hy = float((ys[1] - ys[0]))
    u2 = u.reshape(B, H, W)
    lap = (
        (u2[..., 2:, 1:-1] - 2 * u2[..., 1:-1, 1:-1] + u2[..., :-2, 1:-1]) / (hx * hx)
        + (u2[..., 1:-1, 2:] - 2 * u2[..., 1:-1, 1:-1] + u2[..., 1:-1, :-2]) / (hy * hy)
    )
    return -K * lap - f

def advection_loss(u, xt, velocity=(1.0, 0.0), dim=2):
    """
    r = du/dt + v . grad(u) = 0  (divergence-free velocity assumed)

    spatial channel indices are 0..dim-1, t sits at `dim`.
    """
    du = x_grad(u, xt, 0, 1)
    u_t = du[..., dim]
    vel = torch.as_tensor(list(velocity), dtype=u.dtype, device=u.device)
    conv = torch.sum(vel[None, None, :] * du[..., :dim], axis=-1)
    return u_t + conv

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