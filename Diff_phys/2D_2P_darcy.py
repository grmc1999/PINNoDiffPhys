# Auto-generated from 2D_2P_darcy.ipynb

# %%
# ! pwd
# ! pip uninstall sympytorch -y

# %%
import os
import sys


ROOT="/home/guillermo.carrilho/PhysicsSimulationDeepLearning"
#ROOT="/share_zeta/Proxy-Sim/PhysicsSimulationDeepLearning"

sys.path.append(os.path.join(ROOT,"Physical_models"))

# %%
from Differentiable_simulation import dK_w,K_w,K_o,grad_phi_dK
from phi.torch.flow import *


geo=UniformGrid(x=500, y=500,bounds=Box(x=5e3, y=5e3))
phi_w=Field( geo,values=tensor(0.0),
      boundary= {
          'x-':4e3,
          'x+': 4e3,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': ZERO_GRADIENT,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

phi_o=Field( geo,values=tensor(1142.0),
      boundary= {
          'x-': 2e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
         'y+': ZERO_GRADIENT
 })#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': ZERO_GRADIENT,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

phy=two_phase_flow(
   phi_w,
    phi_o,
    dtphi_w_1,
    dtphi_o_1,
    dt=0.01,
    #w_advection_solver=lambda v: Solve('CG',1e-4,1e-4,x0=v),
    w_advection_solver=lambda v: Solve('CG-adaptive',1e-4,1e-4,x0=v),
    #o_advection_solver=lambda v: Solve('CG',1e-4,1e-4,x0=v)
    o_advection_solver=lambda v: Solve('CG-adaptive',1e-4,1e-4,x0=v)
)

from Differentiable_simulation import K_o_f_t,K_w_f_t,S_w,dK_o,dK_w,dK_o,dsdpc
#from Differentiable_simulation import *
#dsdpc=(lambda p_c:math.clip((-1*LAMBDA)*((S_w(p_c)-SWR)/PD),lower_limit=1e-6))
import anisotropic_diffusion
print(phy.compute_p_c(phi_w,phi_o))
print(S_w(phy.compute_p_c(phi_w,phi_o)))
print(K_w_f_t(S_w(phy.compute_p_c(phi_w,phi_o))))
print(dsdpc(phy.compute_p_c(phi_w,phi_o)))

print(dK_w(phy.compute_p_c(phi_w,phi_o)))
print(K_w(phy.compute_p_c(phi_w,phi_o)))
print(phy.compute_convective_velocity(phi_o,phi_w,dK_w,dK_o).sample(phi_w.geometry))
print(K_o(phy.compute_p_c(phi_w,phi_o)))
print(dK_o(phy.compute_p_c(phi_w,phi_o)))
#print(phy.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o).sample(phi_o.geometry))
print(phy.phi_w_momentum_eq(phi_w,phi_o, 1e-4).sample(phi_o.geometry))

PD=2e3
SWR=0.3
SOR=0.1
print((lambda s_w:PD*((s_w-SWR)/(1-SWR)))(0.3))
print((lambda s_w:PD*((s_w-SWR)/(1-SWR)))(0.7))

# %%
import matplotlib.pyplot as plt

class two_phase_flow(object):
  def __init__(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,w_advection_solver,o_advection_solver):
    #self.v0=v0
    self.phi_w=phi_w
    self.phi_o=phi_o
    self.dtphi_o_1=dtphi_o_1
    self.dtphi_w_1=dtphi_w_1
    self.dt=dt
    self.p=None
    self.w_advection_solver=w_advection_solver
    self.o_advection_solver=o_advection_solver


  def compute_p_c(self,phi_w,phi_o):
    p_c=phi_o.sample(phi_o.geometry) -\
    phi_w.sample(phi_w.geometry)
    return p_c

  def compute_convective_velocity(self,phi_a,phi_b,dK_a,dK_b):
    p_c=self.compute_p_c(self.phi_w,self.phi_o)
    convective_velocity = grad_phi_dK(phi_a,dK_a(p_c))\
                         - grad_phi_dK(phi_b,dK_b(p_c))

    V=unstack(convective_velocity,"dk")
    convective_velocity=Field(self.phi_o.geometry,values=vec(x=V[0],y=V[1]))
    return convective_velocity

  #def compute_anisotropic_viscosity_effect(self):
    # reformulate differential solver
    
  def phi_w_momentum_eq(self,phi_w,phi_o, dt):
    #grad_phi_w=field.spatial_gradient(self.phi_w,self.phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)
    w_advection_term = dt * advect.semi_lagrangian((phi_w),
                                                    self.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o),
                                                    dt).sample(phi_w.geometry)

    w_diffusion_term = dt * anisotropic_diffusion.implicit(phi_w,-1*K_w(p_c), dt=dt,correct_skew=False).sample(phi_w.geometry)
    #o_diffusion_term = dt * anisotropic_diffusion.implicit(phi_o,K_o(p_c), dt=dt,correct_skew=False).sample(phi_w.geometry)

    pressure_chage_term = dt * (self.dtphi_o_1)

    return phi_w + phi_w.with_values(pressure_chage_term) + phi_w.with_values(w_advection_term) + phi_w.with_values(w_diffusion_term)
    #return phi_w + phi_w.with_values(pressure_chage_term) + phi_w.with_values(w_diffusion_term)
    #return phi_w + phi_w.with_values(w_diffusion_term)
  
  def phi_o_momentum_eq(self,phi_o,phi_w, dt):
    #grad_phi_w=field.spatial_gradient(phi_w,phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)

    o_advection_term = dt * advect.semi_lagrangian((phi_o),
                                                    self.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o),
                                                    dt).sample(phi_o.geometry)

    o_diffusion_term = dt * anisotropic_diffusion.implicit(phi_o,-1*K_o(p_c), dt=dt,correct_skew=False).sample(phi_o.geometry)

    pressure_chage_term = dt * (self.dtphi_w_1)

    return phi_o + phi_o.with_values(pressure_chage_term) + phi_o.with_values(o_advection_term) + phi_o.with_values(o_diffusion_term)
    #return phi_o + phi_o.with_values(pressure_chage_term) + phi_o.with_values(o_diffusion_term)
    #return phi_o + phi_o.with_values(o_diffusion_term)
  
  def compute_phi_k(self,phi_w,phi_o,phi_w_1,phi_o_1,dt):
    print(((phi_o-phi_o_1)/dt).sample(geo))
    return (phi_w-phi_w_1)/dt,(phi_o-phi_o_1)/dt


  def implicit_time_step(self, phi_w,phi_o, dt):
    new_phi_w = math.solve_linear(self.phi_w_momentum_eq, phi_w, self.w_advection_solver(phi_w),phi_o, dt=-dt)
    print("w phse solved")
    new_phi_o = math.solve_linear(self.phi_o_momentum_eq, phi_o, self.o_advection_solver(phi_o),phi_w, dt=-dt)
    print("o phse solved")
    print(phi_o.sample(geo))
    print(new_phi_o.sample(geo))
    self.dtphi_w_1,self.dtphi_o_1=self.compute_phi_k(new_phi_w,new_phi_o,phi_w,phi_o, dt)
    return new_phi_w,new_phi_o

# %%
phi=(phi_w,phi_o)

# %%
print(phy.dtphi_o_1.sample(geo))
print(phy.dtphi_w_1.sample(geo))
print(dsdpc(phy.compute_p_c(*phi)))
print(phy.dtphi_o_1.sample(geo)/(dsdpc(phy.compute_p_c(*phi))))
print(phy.dtphi_w_1.sample(geo)/(dsdpc(phy.compute_p_c(*phi))))

# %%
phi=(phi_w,phi_o)
for i in range(1):
    phi=phy.implicit_time_step(*phi,1e-4)

# %%
#plt.imshow(phy.dtphi_o_1.sample(geo))
plt.imshow(phy.dtphi_w_1.sample(geo))
plt.colorbar()
print(phy.dtphi_w_1.sample(geo))
print(phy.dtphi_o_1.sample(geo))
print(phy.phi_o.sample(geo))

# %%
print(phi[0].sample(geo))
plt.imshow(phi[0].sample(geo).native("x,y"))
plt.colorbar()
plt.show()
print(phi[1].sample(geo))
plt.imshow(phi[1].sample(geo).native("x,y"))
plt.colorbar()
plt.show()
plt.imshow(S_w(phy.compute_p_c(*phi)))
plt.colorbar()
print(S_w(phy.compute_p_c(*phi)))
plt.show()
print(phy.compute_p_c(*phi))
plt.imshow(dsdpc(phy.compute_p_c(*phi)))
plt.colorbar()
plt.show()
print(dsdpc(phy.compute_p_c(*phi)))

plt.imshow(phy.compute_convective_velocity(*phi,dK_w,dK_o).sample(phi_w.geometry).native("x,y")[:,:,0])
plt.colorbar()
plt.show()
print(phy.compute_convective_velocity(*phi,dK_w,dK_o).sample(phi_w.geometry))
plt.imshow(phy.compute_convective_velocity(*phi,dK_w,dK_o).sample(phi_w.geometry).native("x,y")[:,:,1])
plt.colorbar()
plt.show()
print(phy.compute_convective_velocity(*phi,dK_w,dK_o).sample(phi_w.geometry))

print(S_w(phy.compute_p_c(*phi)))
print(K_w_f_t(S_w(phy.compute_p_c(*phi))))
print(dsdpc(phy.compute_p_c(*phi)))
print(phy.compute_p_c(*phi))
print(dK_w(phy.compute_p_c(*phi)))
print(K_w(phy.compute_p_c(*phi)))
print(phy.compute_convective_velocity(*phi,dK_w,dK_o).sample(phi_w.geometry))
print(K_o(phy.compute_p_c(*phi)))
print(dK_o(phy.compute_p_c(*phi)))
#print(phy.compute_convective_velocity(*phi,dK_w,dK_o).sample(phi_o.geometry))
print(phy.phi_w_momentum_eq(*phi, 1e-5).sample(phi_o.geometry))

# %%
print(phy.dtphi_w_1.sample(geo))

# %%
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

# %%
x1 = torch.randn((3,1,3)).requires_grad_(True)  #
print(x1)
print(x1.shape)

u1 = torch.stack([
    x1[:,:,0]**0.5,
    torch.sin(x1[:,:,1])+x1[:,:,0]**0.5,
    torch.cos(x1[:,:,0]),
    ],axis=2)

print(u1.shape)

# %%
incompresibble_fluid_loss(u1,x1,1,1)-0
#x_grad(u1,x1,0,1)[...,0]
#print(x1[:,:,:1])
#print(x_grad(u1,x1,0,1)[...,:2])
#x_grad(u1,x1,0,1)[...,2]
#x_grad(u1,x1,0,2)[...,:2]

# %%
from Differentiable_simulation import two_phase_flow_ReactionDiffusion
from phi.torch.flow import *
import matplotlib.pyplot as plt

# %%

@jit_compile
def reaction_diffusion(u, v, du, dv, f, k, dt):
    uvv = u * v**2
    su = du * field.laplace(u) - uvv + f * (1 - u)
    sv = dv * field.laplace(v) + uvv - (f + k) * v
    return u + dt * su, v + dt * sv


def reaction_diffusion(u, v, du, dv, f, k, dt):
    uvv = u * v**2
    su = du * field.laplace(u) - uvv + f * (1 - u)
    sv = dv * field.laplace(v) + uvv - (f + k) * v
    return u + dt * su, v + dt * sv

# %%
u0 = [
    CenteredGrid(Noise(scale=20, smoothness=1.3), x=100, y=100) * .2 + .1,
    CenteredGrid(lambda x: math.exp(-0.5 * math.sum((x - 50)**2) / 3**2), x=100, y=100),
    CenteredGrid(lambda x: math.cos(math.vec_length(x-50)/3), x=100, y=100) * .5,
]
u0 = stack(u0, batch('initialization'))
plt.imshow(u0)

# %%
maze = {'du': 0.19, 'dv': 0.05, 'f': 0.06, 'k': 0.062}
u_trj, v_trj = iterate(reaction_diffusion, batch(time=100), u0, u0, dt=.5, f_kwargs=maze, substeps=20)

# %%

time=50
plt.imshow(u_trj.initialization[0].time[time].values)
plt.show()
plt.imshow(u_trj.initialization[1].time[time].values)
plt.show()
plt.imshow(u_trj.initialization[2].time[time].values)
plt.show()

# %%
# !ls Data/
import pandas as pd
data=pd.read_csv("Data/60x60_T240_2D_2P_Darcy_flow.csv")

# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy
from copy import deepcopy
from scipy import ndimage

data["Ki"]

K_s=np.ones((60,60))
for x,y,k in zip(data.x.values[:],data.y.values[:],data["Ki"].values[:]):
    #K_s[int(x/50)-1,int(y/50)-1]
    K_s[int((np.unique(x/50)-1)/2),int((np.unique(y/50)-1)/2)]=k
#K_s=ndimage.zoom(K_s,(0.6,0.6)) * 9.869233e-13
plt.imshow(K_s)
K_s

# %%
from Differentiable_simulation import two_phase_flow_ReactionDiffusion
from phi.torch.flow import *

Sw_args={
    "SWR":0.01,
    "SOR":0.20
}
Pc_args=Sw_args.copy()
Pc_args.update({
    "lam":0.5,
    "Pi":1e3
    })

p_o=2000.0 # TEMP

geo=UniformGrid(x=60, y=60,bounds=Box(x=6e3, y=6e3))
phi_w = Field( geo,values=tensor(0.0),
      boundary= {
          'x-':5e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })

phi_o=Field( geo,values=tensor(p_o),
      boundary= {
          'x-': p_o + 5e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
         'y+': ZERO_GRADIENT
 })#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })




phy=two_phase_flow_ReactionDiffusion(
   phi_w,
    phi_o,
    dtphi_w_1,
    dtphi_o_1,
    dt=0.01,
    por=0.1,
    mu_w=0.32,
    mu_o=1.295,
    K_s=K_s * 9.869233e-13,
    kr_w=0.4,
    kr_o=0.3,
    Pc_args=Pc_args
)


# %%
phi_t=[]
phi=(phi_w,phi_o)
phi_t.append(phi)
for i in range(240):
    print(i)
    
    phi=phy.RK4(*phi,1e3)
    #phi=phy.implicit_time_step(*phi,1e5)
    phi_t.append(phi)
    
    if np.isnan(np.sum(phy.compute_p_c(*phi).numpy("x,y"))):
        print("nan")
        print(i)
        break

# %%
#for f in np.array(phi_t):
#for t,f in enumerate(phi_t[::10]):
for t,f in enumerate(phi_t[::-10]):
    fig,axs=plt.subplots(1,5,figsize=(15,5))
    #axs[0].imshow(Sw_Pc_f(phy.compute_p_c(*f)))
    pcm = axs[0].pcolormesh(phy.Sw_Pc_f(phy.compute_p_c(*f)).numpy("y,x"))
    axs[0].title.set_text("Sw")
    fig.colorbar(pcm,ax=axs[0])
    pcm = axs[1].pcolormesh(phy.compute_p_c(*f).numpy("y,x"))
    axs[1].title.set_text("P_c")
    fig.colorbar(pcm,ax=axs[1])

    #pcm = axs[2].pcolormesh(1/dScdPc_f(phy.compute_p_c(*f)).numpy("y,x"))
    #axs[2].title.set_text("1/dS")
    #fig.colorbar(pcm,ax=axs[2])

    pcm = axs[2].pcolormesh(phi_sum(
      phy.compute_convective_velocity(*f,phy.dK_w,phy.dK_o)*f[1].gradient(),
      "vector").numpy("y,x"))
    axs[2].title.set_text("1/dS")
    fig.colorbar(pcm,ax=axs[2])

    pcm = axs[3].pcolormesh(f[0].numpy("y,x"))
    axs[3].title.set_text("phi_w")
    fig.colorbar(pcm,ax=axs[3])

    pcm = axs[4].pcolormesh(f[1].numpy("y,x"))
    axs[4].title.set_text("phi_o")
    fig.colorbar(pcm,ax=axs[4])


    print(t)
    plt.show()

# %%
#**$      p  kpa      Rs        Bo        Bg      viso      visg
#        1.000         1         1         1     1.295       0.1
#      173.352        10         1       0.1     1.294       0.1
#      400.000        20         1      0.01     1.293       0.1

#Sw   krw      krow
SKr=np.array([[0.300, 0., 1.000000],
[0.310, 0.000346, 0.934804],
[0.320, 0.001383, 0.872505],
[0.330, 0.003111, 0.813037],
[0.340, 0.005531, 0.756335],
[0.350, 0.008642, 0.702332],
[0.360, 0.012444, 0.650963],
[0.370, 0.016938, 0.602162],
[0.380, 0.022123, 0.555863],
[0.390, 0.028000, 0.512000],
[0.400, 0.034568, 0.470508],
[0.410, 0.041827, 0.431320],
[0.420, 0.049778, 0.394370],
[0.430, 0.058420, 0.359594],
[0.440, 0.067753, 0.326925],
[0.450, 0.077778, 0.296296],
[0.460, 0.088494, 0.267643],
[0.470, 0.099901, 0.240900],
[0.480, 0.112000, 0.216000],
[0.490, 0.124790, 0.192878],
[0.500, 0.138272, 0.171468],
[0.510, 0.152444, 0.151704],
[0.520, 0.167309, 0.133520],
[0.530, 0.182864, 0.116850],
[0.540, 0.199111, 0.101630],
[0.550, 0.216049, 0.087791],
[0.560, 0.233679, 0.075270],
[0.570, 0.252000, 0.064000],
[0.580, 0.271012, 0.053915],
[0.590, 0.290716, 0.044949],
[0.600, 0.311111, 0.037037],
[0.610, 0.332198, 0.030112],
[0.620, 0.353975, 0.024110],
[0.630, 0.376444, 0.018963],
[0.640, 0.399605, 0.014606],
[0.650, 0.423457, 0.010974],
[0.660, 0.448000, 0.008000],
[0.670, 0.473235, 0.005619],
[0.680, 0.499160, 0.003764],
[0.690, 0.525778, 0.002370],
[0.700, 0.553086, 0.001372],
[0.710, 0.581086, 0.000702],
[0.720, 0.609778, 0.000296],
[0.730, 0.639160, 0.000088],
[0.740, 0.669235, 0.000011],
[0.750, 0.700000, 0.000000]])

plt.plot(SKr[:,0],SKr[:,1])
plt.plot(SKr[:,0],SKr[:,2])


# %%
#from Differentiable_simulation import dK_w,K_w,K_o,grad_phi_dK
from Differentiable_simulation import two_phase_flow_RD_TBK
from phi.torch.flow import *

Sw_args={
    "SWR":0.01,
    "SOR":0.20
}
Pc_args=Sw_args.copy()
Pc_args.update({
    "lam":0.5,
    "Pi":1e3
    })
SWC=sympy.symbols("S_{wc}")
SOR=sympy.symbols("S_{or}")
Sw=sympy.symbols("S_w")
lam=sympy.symbols("\lambda")
Pi=sympy.symbols("P_i")
Sc=(Sw-SWC)/(1-SWC-SOR)
Pc=Pi*Sc**(-1/lam)
Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,*tuple(Pc_args.values()))
p_o=Pc_f(0.05)

geo=UniformGrid(x=240, y=240,bounds=Box(x=6e3, y=6e3))
phi_w=Field( geo,values=tensor(0.0),
      boundary= {
          'x-':2e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })

phi_o=Field( geo,values=tensor(p_o),
      boundary= {
          'x-': p_o + 2e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
         'y+': ZERO_GRADIENT
 })#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })




phy=two_phase_flow_RD_TBK(
#phy=two_phase_flow_RD_decoupled_DT(
   phi_w,
    phi_o,
    dtphi_w_1,
    dtphi_o_1,
    dt=1e3,
    por=0.1,
    mu_w=0.32,
    mu_o=1.295,
    Pc_args=Pc_args,
    K_s=ndimage.zoom(K_s,(4.,4.)) * 9.869233e-13,
    krwo=SKr
)

# %%
import torch
from einops import rearrange
import matplotlib.pyplot as plt

class simple_cnn_model(torch.nn.Module):
  def __init__(self):
    super(simple_cnn_model,self).__init__()

    self.conv1=torch.nn.Conv2d(2, 32, (9,9), padding=4)
    self.conv2=torch.nn.Conv2d(32, 64, (9,9), padding=4)
    self.conv3=torch.nn.Conv2d(64, 2, (9,9), padding=4)
    self.act1=torch.nn.ReLU()
    self.act2=torch.nn.ReLU()
    self.act3=torch.nn.Tanh()
  def forward(self,x):
    x=self.act1(self.conv1(x))
    x=self.act2(self.conv2(x))
    x=self.act3(self.conv3(x))*0.01
    #x=torch.clamp(x, min=-0.5, max=0.5)
    return x
  
class simple_dual_space_cnn_model(torch.nn.Module):
  def __init__(self):
    super(simple_dual_space_cnn_model,self).__init__()

    self.conv1=torch.nn.Conv2d(2, 32, (9,9), padding=4)
    self.conv2=torch.nn.Conv2d(32, 64, (9,9), padding=4)
    self.conv3=torch.nn.Conv2d(64, 2, (9,9), padding=4)
    self.act1=torch.nn.ReLU()
    self.act2=torch.nn.ReLU()
    self.act3=torch.nn.Tanh()
  def forward(self,x1,x2):
    x=torch.concatenate((x1,x2),axis=1)
    x=self.act1(self.conv1(x))
    x=self.act2(self.conv2(x))
    x=self.act3(self.conv3(x))*0.01

    #x=torch.clamp(x, min=-0.5, max=0.5)
    return x[:,0],x[:,1]

# %%
from random import choice,sample
from phi.torch.flow import *

def Space2Tensor(Space,geometry,space_signature='x,y,vector',tensor_signature="b x y c->b c x y"):
  return rearrange(Space.sample(geometry).native(space_signature).unsqueeze(0),tensor_signature)


def Tensor2Space(Tensor,geometry,tensor_signature='c x y->x y c',space_signature="x:s,y:s,vector:c"):
  #return math.wrap(rearrange(Tensor[0],'c x y->x y c'),"x:s,y:s,vector:c")
  return Field(geometry=geometry,values=math.wrap(rearrange(Tensor,tensor_signature),space_signature))

class SOL_trainer(object):
    def __init__(self,ph_model,coarse_model,model,optimizer,simulation_steps,time_step,gt_factor,batch_size,train_horizon):

      self.Space2Tensor= lambda Space,geometry:Space2Tensor(Space,geometry,space_signature='x,y,vector',tensor_signature="b x y c ->b c x y")
      self.Tensor2Space= lambda Tensor,geometry:Tensor2Space(Tensor,geometry,space_signature="x:s,y:s",tensor_signature="x y -> x y")
      
      self.co_dt=time_step
      self.gt_dt=time_step*gt_factor
      self.gt_factor=gt_factor

      self.v_co=coarse_model  # (Field (x y) Field (x y))

      self.v_gt=copy(coarse_model)

      self.init_states_gt=[self.v_gt]
      self.ph_model=ph_model

      for i in range(train_horizon):
        self.init_states_gt.append(self.ph_model.RK4(*self.init_states_gt[-1],1e5)) # list [ horizon (Field[x y] Field[x y]) ]

      self.n_steps=simulation_steps
      self.st_model=model
      self.batch_size=batch_size

      self.loss=(lambda y_,y: torch.sum((y-y_)**2)/self.n_steps)

      self.optimizer=optimizer

      self.alpha=1

    def forward_prediction_correction(self):
      #print(f"prediction correction simulation")

      states_pred=[self.v_co]
      #print(self.v_co[0].shape)
      correction=self.st_model(
          self.Space2Tensor(self.v_co[0],self.v_co[0].geometry),
          self.Space2Tensor(self.v_co[1],self.v_co[1].geometry)
          )
      #print(correction[0].shape)
      states_corr=[(
        self.Tensor2Space(correction[0][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[1][0],self.v_co[0].geometry)
        )]

      states_pred=[
        (self.v_co[0]+states_corr[-1][0],
         self.v_co[1]+states_corr[-1][1],
         )
        ]

      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps):
        print("internal step")
        print(i)
        #print(states_pred[-1])

        # Step last in states_pred
        states_pred.append(self.ph_model.RK4(*states_pred[-1],self.co_dt))
        # Correct with model of last states_pred
        #print("forward prediction")
        #print(states_pred[-1][0].shape)
        correction=self.st_model(
          self.Space2Tensor(states_pred[-1][0],self.v_co[0].geometry),
          self.Space2Tensor(states_pred[-1][1],self.v_co[1].geometry)
          )
        states_corr.append((
          self.Tensor2Space(correction[0][0],self.v_co[0].geometry),
          self.Tensor2Space(correction[1][0],self.v_co[1].geometry)
          ))
        # Sum correction to last in states pred
        states_pred[-1]=(states_pred[-1][0]+states_corr[-1][0],
                         states_pred[-1][1]+states_corr[-1][1]
                         )

      #states_pred=list(map(lambda corr:self.Space2Tensor(corr,self.v_co[0].geometry),states_pred))
      states_pred=list(map(lambda corr:
                           (self.Space2Tensor(corr[0],self.v_co[0].geometry),
                                        self.Space2Tensor(corr[1],self.v_co[1].geometry))
                                        ,states_pred)) # b c x y

      return states_pred,states_corr

    def roll_to_batch(self,roll):
      """
      conver roll to batch
      roll list of tuples of tensors [H (T_space1 T_space1)]
      """
      return torch.concat(list(map( lambda ss:torch.concat(ss,axis=1),roll )),axis=1)
      
    def forward_fine_grained(self):
      states_gt=[(
        self.Space2Tensor(self.v_gt[0],self.v_co[0].geometry),
         self.Space2Tensor(self.v_gt[1],self.v_co[0].geometry)
         )]

      for i in range(int(self.n_steps/self.gt_factor)):
        self.v_gt=self.ph_model.RK4(*self.v_gt,self.gt_dt)
        if i%int(1/self.gt_factor)==0:
          states_gt.append((
            self.Space2Tensor(self.v_gt[0],self.v_co[0].geometry),
            self.Space2Tensor(self.v_gt[1],self.v_co[0].geometry)
            ))
      return states_gt # list [ fine_grained (Field[x y] Field[x y]) ]

    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        print(f"epoch {i}")
        gt_batch=[]
        co_batch=[]
        #list(range(len(self.init_states_gt)))
        batch_init_ind=sample(list(range(len(self.init_states_gt))),self.batch_size)
        #batch_init=sample(self.init_states_gt,self.batch_size)
        print(batch_init_ind)
        #for b in range(self.batch_size):
        #for b in batch_init:
        for i in batch_init_ind:
          #print(f"making batch {b}")
          print(f"making batch {i}")
          #self.v_gt=choice(self.init_states_gt) # (Field[x y] Field[x y])
          #self.v_gt=b
          self.v_gt=self.init_states_gt[i]
          states_gt=self.forward_fine_grained() # list [ fine_grained (Field[x y] Field[x y]) ]

          self.v_co=(
              self.Tensor2Space(states_gt[0][0][0,0].detach(),self.v_co[0].geometry),
              self.Tensor2Space(states_gt[0][1][0,0].detach(),self.v_co[0].geometry)
              )
          states_pred,states_corr=self.forward_prediction_correction()
          gt_batch=gt_batch+states_gt
          co_batch=co_batch+states_pred
        
        states_pred=self.roll_to_batch(states_pred)
        states_gt=self.roll_to_batch(states_gt)
        loss=self.loss(states_pred,states_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses.append(loss.cpu().detach().numpy())
      return losses

      def test(self,epochs):
        losses=[]
        for i in range(epochs):
          #self.alpha=self.alpha*(i/epochs)
          states_pred,states_corr=self.forward_prediction_correction()
          states_gt=self.forward_fine_grained()

          states_pred=torch.concat(states_pred,axis=0)
          states_gt=torch.concat(states_gt,axis=0)
          loss=self.loss(states_pred,states_gt)

          losses.append(loss.cpu().detach().numpy())
        return losses

# %%
from phi.torch.flow import *
import sympy

Sw_args={
    "SWR":0.01,
    "SOR":0.20
}
Pc_args=Sw_args.copy()
Pc_args.update({
    "lam":0.5,
    "Pi":1e3
    })
SWC=sympy.symbols("S_{wc}")
SOR=sympy.symbols("S_{or}")
Sw=sympy.symbols("S_w")
lam=sympy.symbols("\lambda")
Pi=sympy.symbols("P_i")
Sc=(Sw-SWC)/(1-SWC-SOR)
Pc=Pi*Sc**(-1/lam)
Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,*tuple(Pc_args.values()))
p_o=Pc_f(0.05)

geo=UniformGrid(x=240, y=240,bounds=Box(x=6e3, y=6e3))
phi_w=Field( geo,values=tensor(0.0),
      boundary= {
          'x-':2e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })

phi_o=Field( geo,values=tensor(p_o),
      boundary= {
          'x-': p_o + 2e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
         'y+': ZERO_GRADIENT
 })#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })

phy=two_phase_flow_RD_TBK(
   phi_w,
    phi_o,
    dtphi_w_1,
    dtphi_o_1,
    dt=0.01,
    por=0.1,
    mu_w=0.32,
    mu_o=1.295,
    Pc_args=Pc_args,
    #K_s=np.expand_dims(ndimage.zoom(K_s,(4.,4.)),axis=-1) * 9.869233e-13,
    K_s= ndimage.zoom(K_s,(4.,4.)) * 9.869233e-13,
    krwo=SKr
)

# %%
#from Differentiable_simulation import dK_w,K_w,K_o,grad_phi_dK
from Differentiable_simulation import two_phase_flow_RD_TBK
from phi.torch.flow import *

Sw_args={
    "SWR":0.01,
    "SOR":0.20
}
Pc_args=Sw_args.copy()
Pc_args.update({
    "lam":0.5,
    "Pi":1e3
    })
SWC=sympy.symbols("S_{wc}")
SOR=sympy.symbols("S_{or}")
Sw=sympy.symbols("S_w")
lam=sympy.symbols("\lambda")
Pi=sympy.symbols("P_i")
Sc=(Sw-SWC)/(1-SWC-SOR)
Pc=Pi*Sc**(-1/lam)
Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,*tuple(Pc_args.values()))
p_o=Pc_f(0.05)

geo=UniformGrid(x=240, y=240,bounds=Box(x=6e3, y=6e3))
phi_w=Field( geo,values=tensor(0.0),
      boundary= {
          'x-':2e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })

phi_o=Field( geo,values=tensor(p_o),
      boundary= {
          'x-': p_o + 2e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
         'y+': ZERO_GRADIENT
 })#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })




phy=two_phase_flow_RD_TBK(
#phy=two_phase_flow_RD_decoupled_DT(
   phi_w,
    phi_o,
    dtphi_w_1,
    dtphi_o_1,
    dt=1e3,
    por=0.1,
    mu_w=0.32,
    mu_o=1.295,
    Pc_args=Pc_args,
    K_s=ndimage.zoom(K_s,(4.,4.)) * 9.869233e-13,
    krwo=SKr
)

# %%
model=simple_dual_space_cnn_model().train()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)


T=SOL_trainer(ph_model=phy,
                coarse_model=(phi_w,phi_o),
                model=model,
                optimizer=optimizer,
                simulation_steps=1,
                time_step=1e5,
                gt_factor=0.5,
                batch_size=2,
                train_horizon=5)

# %%
T.train(3)

# %%
from phi.torch.flow import *
from Differentiable_simulation import two_phase_flow_RD_TBK
from phi.torch.flow import *
import sympy


Sw_args={
    "SWR":0.01,
    "SOR":0.20
}
Pc_args=Sw_args.copy()
Pc_args.update({
    "lam":0.5,
    "Pi":1e3
    })
SWC=sympy.symbols("S_{wc}")
SOR=sympy.symbols("S_{or}")
Sw=sympy.symbols("S_w")
lam=sympy.symbols("\lambda")
Pi=sympy.symbols("P_i")
Sc=(Sw-SWC)/(1-SWC-SOR)
Pc=Pi*Sc**(-1/lam)
Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,*tuple(Pc_args.values()))
p_o=Pc_f(0.05)

geo=UniformGrid(x=60, y=60,bounds=Box(x=6e3, y=6e3))
phi_w=Field( geo,values=tensor(0.0),
      boundary= {
          'x-':2e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {'x-': None,'x+': None,'y-': None,'y+': None})

phi_o=Field( geo,values=tensor(p_o),
      boundary= {'x-': p_o + 2e3,'x+': ZERO_GRADIENT,'y-': ZERO_GRADIENT,'y+': ZERO_GRADIENT,})#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {'x-': None,'x+': None,'y-': None,'y+': None})




phy=two_phase_flow_RD_TBK(
#phy=two_phase_flow_RD_decoupled_DT(
   phi_w,
    phi_o,
    dtphi_w_1,
    dtphi_o_1,
    dt=1e3,
    por=0.1,
    mu_w=0.32,
    mu_o=1.295,
    Pc_args=Pc_args,
    K_s=ndimage.zoom(K_s,(1.,1.)) * 9.869233e-13,
    #K_s=ndimage.zoom(K_s,(4.,4.)) * 9.869233e-20,
    krwo=SKr,
    max_dt=1e8,min_dt=-1e8
)

# %%
phi_t=[]
phi=(phi_w,phi_o,dtphi_w_1,dtphi_o_1)
phi_t.append(phi)
for i in range(240):
    print(i)
    
    phi=phy.RK4(*phi,60)
    #phi=phy.implicit_time_step(*phi,1e5)
    phi_t.append(phi)
    
    if np.isnan(np.sum(phy.compute_p_c(*phi[:2]).numpy("x,y"))):
        print("nan")
        print(i)
        break

# %%
for t,f in enumerate(phi_t[::-20]):
    fig,axs=plt.subplots(1,5,figsize=(15,5))
    #axs[0].imshow(Sw_Pc_f(phy.compute_p_c(*f)))
    pcm = axs[0].pcolormesh(phy.Sw_Pc_f(phy.compute_p_c(*f[:2])).numpy("y,x"))
    axs[0].title.set_text("Sw")
    fig.colorbar(pcm,ax=axs[0])
    pcm = axs[1].pcolormesh(phy.compute_p_c(*f[:2]).numpy("y,x"))
    axs[1].title.set_text("P_c")
    fig.colorbar(pcm,ax=axs[1])

    #pcm = axs[2].pcolormesh(1/dScdPc_f(phy.compute_p_c(*f[:2])).numpy("y,x"))
    #axs[2].title.set_text("1/dS")
    #fig.colorbar(pcm,ax=axs[2])

#    pcm = axs[2].pcolormesh(phi_sum(
#      phy.compute_convective_velocity(*f[:2],phy.dK_w,phy.dK_o)*f[1].gradient(),
#      "vector").numpy("y,x"))
#    axs[2].title.set_text("1/dS")
#    fig.colorbar(pcm,ax=axs[2])

    pcm = axs[3].pcolormesh(f[0].numpy("y,x"))
    axs[3].title.set_text("phi_w")
    fig.colorbar(pcm,ax=axs[3])

    pcm = axs[4].pcolormesh(f[1].numpy("y,x"))
    axs[4].title.set_text("phi_o")
    fig.colorbar(pcm,ax=axs[4])


    print(t)
    plt.show()

# %%

from DL_models.Models.CNN_models import simple_dual_space_cnn_model
from Train_differentiable_physics import SOL_trainer_darcyflow
import torch
model=simple_dual_space_cnn_model().train()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)


T=SOL_trainer_darcyflow(ph_model=phy,
                coarse_model=(phi_w,phi_o,dtphi_w_1,dtphi_o_1),
                model=model,
                optimizer=optimizer,
                simulation_steps=10,
                time_step=1e-1,
                gt_factor=0.25,
                batch_size=2,
                train_horizon=10)

# %%
LL=T.train(5)

# %%
from random import sample
from Differentiable_simulation import Space2Tensor,Tensor2Space
from Train_differentiable_physics import SOL_trainer_darcyflow
from einops import rearrange

class data_based_SOL(SOL_trainer_darcyflow):
    def __init__(self,ph_model,coarse_model,gt_data,model,optimizer,simulation_steps,time_step,gt_factor,batch_size):
        #super().__init__(**kwargs)
        self.Space2Tensor= lambda Space,geometry:Space2Tensor(Space,geometry,space_signature='x,y,vector',tensor_signature="b x y c ->b c x y")
        self.Tensor2Space= lambda Tensor,geometry:Tensor2Space(Tensor,geometry,space_signature="x:s,y:s",tensor_signature="x y -> x y")

        self.co_dt=time_step
        self.gt_dt=time_step*gt_factor
        self.gt_factor=gt_factor

        self.v_co=coarse_model  # (Field (x y) Field (x y))

        #self.v_gt=copy(coarse_model)

        self.init_states_gt=gt_data
        self.ph_model=ph_model

        #for i in range(train_horizon):
        #  self.init_states_gt.append(self.ph_model.RK4(*self.init_states_gt[-1],1e5)) # list [ horizon (Field[x y] Field[x y] dtF dtF) ]

        self.n_steps=simulation_steps
        self.st_model=model
        self.batch_size=batch_size

        #self.loss=(lambda y_,y: torch.sum((y[:,:2]-y_[:,:2])**2)/self.n_steps)
        self.loss=(lambda y_,y: torch.sum((y[:,:]-y_[:,:])**2)/self.n_steps)

        self.optimizer=optimizer

        self.gt_data_normalization()

    def transform_phase_pressure2Sw(self,Fields):
      P_w=Fields[:,0]
      P_o=Fields[:,1]
      return self.ph_model.Sw_Pc_f(self.ph_model.compute_p_c(P_w,P_o))

    def transform_phase_pressure2Pt(self,Fields):
        P_w=Fields[:,0]
        P_o=Fields[:,1]
        return P_w+P_o

    def transform_sw_pt2pa(self,Sw,pt):
      pc=Pc_f(Sw)
      p_o=(pt+pc)/2
      p_w=(pt-pc)/2
      return p_o,p_w
    
    def gt_data_normalization(self):
      self.init_states_gt.x=((self.init_states_gt.x.values/50-1)*0.5).astype(int)
      self.init_states_gt.y=(np.abs(self.init_states_gt.y.values/50-1)*0.5).astype(int)-1
      self.init_states_gt.t=(self.init_states_gt.t.values-1).astype(int)

    def prop_to_time_tensor_prop(self,t,prop):
      self.init_states_gt[self.init_states_gt.t==t]
      data_tensor=np.ndarray([self.init_states_gt.x.max()+1,self.init_states_gt.y.max()+1])
      data_tensor[self.init_states_gt.x.values.astype(int),self.init_states_gt.y.values.astype(int)]=self.init_states_gt[prop].values

      return data_tensor
    
    def forward_fine_grained(self,t,prop):
      v_gt=[]
      for i in range(self.n_steps+1):
        self.init_states_gt[self.init_states_gt.t==(t+i)]
        data_tensor=np.ndarray([int(self.init_states_gt.x.max()+1),int(self.init_states_gt.y.max()+1),len(prop)])
        data_tensor[self.init_states_gt.x.values.astype(int),self.init_states_gt.y.values.astype(int)]=self.init_states_gt[prop].values
        # TODO should return list of n Ws[Field], P[Field]
        if self.v_co[0].numpy("x,y").shape!=data_tensor.shape:
          data_tensor=ndimage.zoom(data_tensor,
                                   (self.v_co[0].numpy("x,y").shape[0]/data_tensor.shape[0],self.v_co[0].numpy("x,y").shape[1]/data_tensor.shape[1],1.0)
                                   )

        data_tensor=torch.from_numpy(data_tensor)
        v_gt.append(data_tensor)
      
      return v_gt # list [ fine_grained (Field[x y] Field[x y]) ]
    
    def forward_prediction_correction(self):

      states_pred=[self.v_co]
      
      correction=self.st_model(
          self.Space2Tensor(self.v_co[0],self.v_co[0].geometry),
          self.Space2Tensor(self.v_co[1],self.v_co[1].geometry),
          self.Space2Tensor(self.v_co[2],self.v_co[2].geometry),
          self.Space2Tensor(self.v_co[3],self.v_co[3].geometry)
          )

      states_corr=[(
        self.Tensor2Space(correction[0][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[1][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[2][0],self.v_co[0].geometry),
        self.Tensor2Space(correction[3][0],self.v_co[0].geometry),
        )]

      states_pred=[(
        self.v_co[0]+states_corr[-1][0],
        self.v_co[1]+states_corr[-1][1],
        self.v_co[2]+states_corr[-1][2],
        self.v_co[3]+states_corr[-1][3],
        )]

      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps*self.gt_factor):

        # Step last in states_pred
        states_pred.append(self.ph_model.RK4(*states_pred[-1],self.co_dt))
        # Correct with model of last states_pred
        correction=self.st_model(
          self.Space2Tensor(states_pred[-1][0],self.v_co[0].geometry),
          self.Space2Tensor(states_pred[-1][1],self.v_co[1].geometry),
          self.Space2Tensor(states_pred[-1][2],self.v_co[2].geometry),
          self.Space2Tensor(states_pred[-1][3],self.v_co[3].geometry)
          )
        states_corr.append((
          self.Tensor2Space(correction[0][0],self.v_co[0].geometry),
          self.Tensor2Space(correction[1][0],self.v_co[1].geometry),
          self.Tensor2Space(correction[2][0],self.v_co[2].geometry),
          self.Tensor2Space(correction[3][0],self.v_co[3].geometry)
          ))
        # Sum correction to last in states pred
        # TODO: apply data transform here recieving tuple of states the functions should reduce the states to the needed
        states_tuple=(states_pred[-1][0]+states_corr[-1][0],
                      states_pred[-1][1]+states_corr[-1][1],
                      states_pred[-1][2],
                      states_pred[-1][3],)

        states_pred[-1]=states_tuple

      states_pred=list(map(lambda corr:
                           (
                             self.Space2Tensor(corr[0],self.v_co[0].geometry),# TODO Implement transformation for fields in target
                             self.Space2Tensor(corr[1],self.v_co[1].geometry),
                             self.Space2Tensor(corr[2],self.v_co[2].geometry),
                             self.Space2Tensor(corr[3],self.v_co[3].geometry)
                             ),states_pred)) # b c x y

      return states_pred[::int(self.gt_factor)],states_corr[::int(self.gt_factor)] # Check

    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        gt_batch=[]
        co_batch=[]

        batch_init_ind=sample(list(range(len(np.unique(self.init_states_gt.t)))),self.batch_size)
        for i in batch_init_ind:
          #self.v_gt=self.init_states_gt[i]
          states_gt=self.forward_fine_grained(i,["P","Ws"]) # list [ fine_grained (Field[x y] Field[x y] dtF dtF) ]

          self.v_co=(
              self.Tensor2Space(T.transform_sw_pt2pa(states_gt[0][:,:,1]*1e3,states_gt[0][:,:,0])[1].detach(),self.v_co[0].geometry), # Field[x y]
              self.Tensor2Space(T.transform_sw_pt2pa(states_gt[0][:,:,1]*1e3,states_gt[0][:,:,0])[0].detach(),self.v_co[0].geometry), # Field[x y]
              self.Tensor2Space(torch.zeros_like(T.transform_sw_pt2pa(states_gt[0][:,:,1]*1e3,states_gt[0][:,:,0])[1].detach()),self.v_co[0].geometry), # dtF
              self.Tensor2Space(torch.zeros_like(T.transform_sw_pt2pa(states_gt[0][:,:,1]*1e3,states_gt[0][:,:,0])[1].detach()),self.v_co[0].geometry) # dtF
              )
          states_pred,states_corr=self.forward_prediction_correction()
          gt_batch=gt_batch+states_gt
          co_batch=co_batch+states_pred
        
        
        states_pred=self.roll_to_batch(states_pred) # [H T_space_1 T_space_2 T_space_3 T_space_4]
        
        Sw=self.ph_model.Sw_Pc_f(torch.clip(states_pred[:,1]-states_pred[:,0],min=self.ph_model.Pc_f(1-self.ph_model.Pc_args["SOR"]),max=self.ph_model.Pc_f(self.ph_model.Pc_args["SWR"])))
        Pt=states_pred[:,0]+states_pred[:,1]

        states_gt=rearrange(torch.stack(states_gt,axis=0),"b x y c ->b c x y")

        loss=self.loss(
          torch.stack((Pt,Sw),axis=1),
          states_gt
          )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(loss)

        losses.append(loss.cpu().detach().numpy())
      return losses

      def test(self,epochs):
        losses=[]
        for i in range(epochs):
          #self.alpha=self.alpha*(i/epochs)
          states_pred,states_corr=self.forward_prediction_correction()
          states_gt=self.forward_fine_grained()

          states_pred=torch.concat(states_pred,axis=0)
          states_gt=torch.concat(states_gt,axis=0)
          loss=self.loss(states_pred,states_gt)

          losses.append(loss.cpu().detach().numpy())
        return losses

# %%
from phi.torch.flow import *
from Differentiable_simulation import two_phase_flow_RD_TBK
from phi.torch.flow import *
import sympy


Sw_args={"SWR":0.01,"SOR":0.20}
Pc_args=Sw_args.copy()
Pc_args.update({"lam":0.5,"Pi":1e3})

SWC=sympy.symbols("S_{wc}")
SOR=sympy.symbols("S_{or}")
Sw=sympy.symbols("S_w")
lam=sympy.symbols("\lambda")
Pi=sympy.symbols("P_i")
Sc=(Sw-SWC)/(1-SWC-SOR)
Pc=Pi*Sc**(-1/lam)
Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,*tuple(Pc_args.values()))
#p_o=Pc_f(0.05)
p_o=151764.7184203469
p_w=144585.46546854207

geo=UniformGrid(x=60, y=60,bounds=Box(x=60*50, y=60*50))
phi_w=Field( geo,values=tensor(p_w),
      boundary= {
          'x-':None,
          'x+': None,
          'y-': None,
          'y+': None
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {'x-': None,'x+': None,'y-': None,'y+': None})

phi_o=Field( geo,values=tensor(p_o),
      boundary= {'x-': None,'x+': None,'y-': None,'y+': None,})#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {'x-': None,'x+': None,'y-': None,'y+': None})




phy=two_phase_flow_RD_TBK(
   phi_w,
    phi_o,
    dtphi_w_1,
    dtphi_o_1,
    dt=1e3,
    por=0.1,
    mu_w=0.32,
    mu_o=1.295,
    Pc_args=Pc_args,
    K_s=ndimage.zoom(K_s,(1.,1.)) * 9.869233e-13,
    krwo=SKr,
    max_dt=1.0,min_dt=-1.0
)

# %%

from DL_models.Models.CNN_models import simple_dual_space_with_time_derivative_cnn_model
from Train_differentiable_physics import SOL_trainer_darcyflow
import torch


import pandas as pd
gr_model=pd.read_csv(os.path.join(ROOT,"Data","60x60_T240_2D_2P_Darcy_flow.csv"))

model=simple_dual_space_with_time_derivative_cnn_model().train()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)


T=data_based_SOL(
    #optimizer,simulation_steps,time_step,gt_factor,batch_size
    ph_model=phy,
                coarse_model=(phi_w,phi_o,dtphi_w_1,dtphi_o_1),
                gt_data=gr_model,
                model=model,
                optimizer=optimizer,
                simulation_steps=2,
                time_step=60,
                gt_factor=30*24*60,
                batch_size=5)

# %%
PW=T.forward_fine_grained(0,["Ws","P"])[0]
p_o,p_w=T.transform_sw_pt2pa(PW[:,:,0],PW[:,:,1]*1e3)

# %%

phi_w=Field( geo,values=tensor(p_w,"x,y"),
      boundary= {
          'x-':None,
          'x+': None,
          'y-': None,
          'y+': None
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {'x-': None,'x+': None,'y-': None,'y+': None})

phi_o=Field( geo,values=tensor(p_o,"x,y"),
      boundary= {'x-': None,'x+': None,'y-': None,'y+': None,})#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {'x-': None,'x+': None,'y-': None,'y+': None})

# %%
30*7*24*60*60
60*60

# %%
phi_t=[]
phi=(phi_w,phi_o,dtphi_w_1,dtphi_o_1)
phi_t.append(phi)
for i in range(240):
    print(i)
    
    phi=phy.RK4(*phi,60)
    #phi=phy.implicit_time_step(*phi,1e5)
    phi_t.append(phi)
    
    if np.isnan(np.sum(phy.compute_p_c(*phi[:2]).numpy("x,y"))):
        print("nan")
        print(i)
        break

# %%
for t,f in enumerate(phi_t[::2]):
    fig,axs=plt.subplots(1,5,figsize=(15,5))
    #axs[0].imshow(Sw_Pc_f(phy.compute_p_c(*f)))
    pcm = axs[0].pcolormesh(phy.Sw_Pc_f(phy.compute_p_c(*f[:2])).numpy("y,x"))
    axs[0].title.set_text("Sw")
    fig.colorbar(pcm,ax=axs[0])
    pcm = axs[1].pcolormesh(phy.compute_p_c(*f[:2]).numpy("y,x"))
    axs[1].title.set_text("P_c")
    fig.colorbar(pcm,ax=axs[1])

    #pcm = axs[2].pcolormesh(1/dScdPc_f(phy.compute_p_c(*f[:2])).numpy("y,x"))
    #axs[2].title.set_text("1/dS")
    #fig.colorbar(pcm,ax=axs[2])

#    pcm = axs[2].pcolormesh(phi_sum(
#      phy.compute_convective_velocity(*f[:2],phy.dK_w,phy.dK_o)*f[1].gradient(),
#      "vector").numpy("y,x"))
#    axs[2].title.set_text("1/dS")
#    fig.colorbar(pcm,ax=axs[2])

    pcm = axs[3].pcolormesh(f[0].numpy("y,x"))
    axs[3].title.set_text("phi_w")
    fig.colorbar(pcm,ax=axs[3])

    pcm = axs[4].pcolormesh(f[1].numpy("y,x"))
    axs[4].title.set_text("phi_o")
    fig.colorbar(pcm,ax=axs[4])


    print(t)
    plt.show()

# %%
LL=T.train(5)

# %%
T.init_states_gt

# %%
import matplotlib.pyplot as plt
import numpy as np
import sympy
from copy import deepcopy
Sw_args={
    "SWR":0.05,
    "SOC":0.20
}
Pc_args=Sw_args.copy()
Pc_args.update({
    "lam":0.5,
    "Pi":1e3
    })

SWC=sympy.symbols("S_{wc}")
SOR=sympy.symbols("S_{or}")
Sw=sympy.symbols("S_w")
lam=sympy.symbols("\lambda")
Pi=sympy.symbols("P_i")
K_rw0=sympy.symbols("k_{rw0}")
K_ro0=sympy.symbols("k_{ro0}")

Pc_=sympy.symbols("P_c")

Sc=(Sw-SWC)/(1-SWC-SOR)
Pc=Pi*Sc**(-1/lam)

Sw_Pc=(1-SWC-SOR)*((Pc_/Pi)**(-1*lam))+SWC
dScdPc=sympy.diff(Sw_Pc,Pc_)


K_rw=K_rw0*Sc**((2+3*lam)/(lam))
K_ro=K_ro0*((1-Sc)**2)*(1-Sc**((2+lam)/(lam)))

#Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,0.05,0.45,0.5,1e5)
Pc_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi),Pc)(sw,*tuple(Pc_args.values()))
Se_f=lambda sw: sympy.lambdify((Sw,SOR,SWC),Sc)(sw,*tuple(Sw_args.values()))

K_w_f_t=lambda sw: 1000.0 * 9.869233e-13
K_o_f_t=lambda sw: 1000.0 * 9.869233e-13

#sympy.diff(K_rw,Sw)
#sympy.diff(K_ro,Sw)
#dK_rw_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi,K_rw0),sympy.diff(K_rw,Sw))(sw,*tuple(Pc_args.values()),0.3)
#dK_ro_f=lambda sw: sympy.lambdify((Sw,SOR,SWC,lam,Pi,K_ro0),sympy.diff(K_ro,Sw))(sw,*tuple(Pc_args.values()),0.5)

Sw_Pc_f=lambda sw: sympy.lambdify((Pc_,SOR,SWC,lam,Pi),Sw_Pc)(sw,*tuple(Pc_args.values()))
dScdPc_f=lambda sw: sympy.lambdify((Pc_,SOR,SWC,lam,Pi),dScdPc)(sw,*tuple(Pc_args.values()))

from phi.torch.flow import diffuse, advect, Solve, fluid, math,Field, unstack,stack,batch,field,vec
MUO=1.295
MUW=0.32
K_w=lambda p_c:stack(
    [stack([K_w_f_t(Sw_Pc_f(p_c))/(MUW*dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
    stack([math.zeros_like(p_c),K_w_f_t(Sw_Pc_f(p_c))/(MUW*dScdPc_f(p_c))],batch("k") )],batch("KK"))

K_o=lambda p_c:stack(
    [stack([K_o_f_t(Sw_Pc_f(p_c))/(MUO*dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
    stack([math.zeros_like(p_c),K_o_f_t(Sw_Pc_f(p_c))/(MUO*dScdPc_f(p_c))],batch("k") )],batch("KK"))

# %%
from phiml.math import sum as phi_sum



class two_phase_flow_RD(object):
  def __init__(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt):
    #self.v0=v0
    self.phi_w=phi_w
    self.phi_o=phi_o
    self.dtphi_o_1=dtphi_o_1
    self.dtphi_w_1=dtphi_w_1
    self.dt=dt
    self.p=None
    self.K_o=K_o
    self.K_w=K_w


  def compute_p_c(self,phi_w,phi_o):
    p_c=phi_o.sample(phi_o.geometry) -\
    phi_w.sample(phi_w.geometry)
    return p_c
    
  def phi_w_pde(self,phi_w,phi_o,dtphi_o):
    p_c=self.compute_p_c(phi_w,phi_o)
    x,y=unstack(phi_sum(self.K_o(p_c),"KK"),"k")
    spatial_diffusion=Field(phi_w.geometry,values=vec(x=x,y=y))
    w_diffusion_term=phi_w.with_values(phi_sum(phi_w.gradient(2)*spatial_diffusion,"vector"))
    pressure_chage_term = (dtphi_o.values)
    return phi_w.with_values(pressure_chage_term) - phi_w.with_values(w_diffusion_term)

  def phi_w_momentum_eq(self,phi_w,phi_o, dt):
    p_c=self.compute_p_c(phi_w,phi_o)

    #w_advection_term = phi_sum(self.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o)*phi_w.gradient(),"vector").sample(phi_w.geometry)

    x,y=unstack(phi_sum(self.K_o(p_c),"KK"),"k")
    spatial_diffusion=Field(phi_w.geometry,values=vec(x=x,y=y))
    w_diffusion_term=phi_w.with_values(phi_sum(phi_w.gradient(2)*spatial_diffusion,"vector"))
    pressure_chage_term = (self.dtphi_o_1.values)

    return phi_w + dt * (phi_w.with_values(pressure_chage_term) - phi_w.with_values(w_diffusion_term))
  
  def phi_o_pde(self,phi_o,phi_w,dtphi_w):
    p_c=self.compute_p_c(phi_w,phi_o)
    x,y=unstack(phi_sum(self.K_o(p_c),"KK"),"k")
    spatial_diffusion=Field(phi_o.geometry,values=vec(x=x,y=y))
    w_diffusion_term=phi_o.with_values(phi_sum(phi_o.gradient(2)*spatial_diffusion,"vector"))
    pressure_chage_term = (dtphi_w.values)
    return phi_o.with_values(pressure_chage_term) - phi_o.with_values(w_diffusion_term)
  
  def phi_o_momentum_eq(self,phi_o,phi_w, dt):
    #grad_phi_w=field.spatial_gradient(phi_w,phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)

    x,y=unstack(phi_sum(self.K_w(p_c),"KK"),"k")
    spatial_diffusion=Field(phi_o.geometry,values=vec(x=x,y=y))
    w_diffusion_term=phi_o.with_values(phi_sum(phi_o.gradient(2)*spatial_diffusion,"vector"))
    pressure_chage_term = (self.dtphi_w_1.values)

    return phi_o + dt * (phi_o.with_values(pressure_chage_term) - phi_o.with_values(w_diffusion_term))
  
  def RK4(self,phi_w,phi_o,dt):
    K_o1=self.phi_o_pde(phi_o,phi_w,self.dtphi_w_1)
    K_w1=self.phi_w_pde(phi_w,phi_o,self.dtphi_o_1)

    K_o2=self.phi_o_pde(phi_o,phi_w+0.5*K_o1.values*dt,K_w1)
    K_w2=self.phi_w_pde(phi_w,phi_o+0.5*K_w1.values*dt,K_o1)

    K_o3=self.phi_o_pde(phi_o,phi_w+0.5*K_o2.values*dt,K_w2)
    K_w3=self.phi_w_pde(phi_w,phi_o+0.5*K_w2.values*dt,K_o2)

    K_o4=self.phi_o_pde(phi_o,phi_w+K_o3.values*dt,K_w3)
    K_w4=self.phi_w_pde(phi_w,phi_o+K_w3.values*dt,K_o3)

    self.dtphi_o_1 = (1/6)  * (K_o1 + 2*K_o2 + 2*K_o3 + K_o4)
    self.dtphi_w_1 = (1/6)  * (K_w1 + 2*K_w2 + 2*K_w3 + K_w4)

    phi_o = phi_o + dt * (1/6)  * (K_o1 + 2*K_o2 + 2*K_o3 + K_o4)
    phi_w = phi_w + dt * (1/6)  * (K_w1 + 2*K_w2 + 2*K_w3 + K_w4)
    return phi_w,phi_o
  
  def compute_phi_k(self,phi_w,phi_o,phi_w_1,phi_o_1,dt):
    return (phi_w-phi_w_1)/dt,(phi_o-phi_o_1)/dt


  def implicit_time_step(self, phi_w,phi_o, dt):
    new_phi_w = self.phi_w_momentum_eq(phi_w,phi_o, dt)
    new_phi_o = self.phi_o_momentum_eq(phi_o,phi_w, dt)
    self.dtphi_w_1,self.dtphi_o_1=self.compute_phi_k(new_phi_w,new_phi_o,phi_w,phi_o, dt)
    return new_phi_w,new_phi_o

# %%
Sw_Pc

# %%
1/dScdPc

# %%
1/dScdPc_f(1e3)

# %%
#from Differentiable_simulation import dK_w,K_w,K_o,grad_phi_dK
from phi.torch.flow import *

p_o=1e4

geo=UniformGrid(x=30, y=30,bounds=Box(x=6e3, y=6e3))
phi_w=Field( geo,values=tensor(0.0),
      boundary= {
          'x-':5e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
          'y+': ZERO_GRADIENT
 })

dtphi_w_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })

phi_o=Field( geo,values=tensor(p_o),
      boundary= {
          'x-': p_o + 5e3,
          'x+': ZERO_GRADIENT,
          'y-': ZERO_GRADIENT,
         'y+': ZERO_GRADIENT
 })#
dtphi_o_1=Field( geo,values=tensor(0.0),
      boundary= {
          'x-': None,
          'x+': None,
          'y-': None,
          'y+': None
 })

phy=two_phase_flow_RD(
   phi_w,
    phi_o,
    dtphi_w_1,
    dtphi_o_1,
    dt=0.01,
)

print(phy.compute_p_c(phi_w,phi_o))
print(Sw_Pc_f(phy.compute_p_c(phi_w,phi_o)))
print(K_w_f_t(Sw_Pc_f(phy.compute_p_c(phi_w,phi_o))))
print(dScdPc_f(phy.compute_p_c(phi_w,phi_o)))
print(K_w(phy.compute_p_c(phi_w,phi_o)))
print(K_o(phy.compute_p_c(phi_w,phi_o)))

# %%
phi_t=[]
phi=(phi_w,phi_o)
phi_t.append(phi)
for i in range(500):
    phi=phy.RK4(*phi,1e3)
    #phi=phy.implicit_time_step(*phi,1e5)
    phi_t.append(phi)
    if np.isnan(np.sum(Sw_Pc_f(phy.compute_p_c(*phi)).numpy("x,y"))):
        print("nan")
        print(i)
        break

# %%
i

# %%

#for f in np.array(phi_t):
#for t,f in enumerate(phi_t[::10]):
for t,f in enumerate(phi_t[-10:]):
    fig,axs=plt.subplots(1,5,figsize=(15,5))
    #axs[0].imshow(Sw_Pc_f(phy.compute_p_c(*f)))
    pcm = axs[0].pcolormesh(Sw_Pc_f(phy.compute_p_c(*f)).numpy("y,x"))
    axs[0].title.set_text("Sw")
    fig.colorbar(pcm,ax=axs[0])
    pcm = axs[1].pcolormesh(phy.compute_p_c(*f).numpy("y,x"))
    axs[1].title.set_text("P_c")
    fig.colorbar(pcm,ax=axs[1])

    pcm = axs[2].pcolormesh(1/dScdPc_f(phy.compute_p_c(*f)).numpy("y,x"))
    axs[2].title.set_text("1/dS")
    fig.colorbar(pcm,ax=axs[2])

    pcm = axs[3].pcolormesh(f[0].numpy("y,x"))
    axs[3].title.set_text("phi_w")
    fig.colorbar(pcm,ax=axs[3])

    pcm = axs[4].pcolormesh(f[1].numpy("y,x"))
    axs[4].title.set_text("phi_o")
    fig.colorbar(pcm,ax=axs[4])


    print(t)
    plt.show()


# %%
Sw_Pc_f(phy.compute_p_c(*f)).numpy("x,y")
