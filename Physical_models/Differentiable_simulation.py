from phi import physics
from phi.torch.flow import diffuse, advect, Solve, fluid, math,Field, unstack,stack,batch,field,vec
from einops import rearrange
import anisotropic_diffusion
from phiml.math import sum as phi_sum
import sympy
from phiml import math as pmath
import numpy as np
from copy import copy
import torch

import sys
import os
sys.path.append(os.path.join("sympytorch","sympytorch"))
from sympy_module import SymPyPhiFlowModule,_reduce
from hide_floats_m import hide_floats
import operator


class Kr_LinearInterpolation(torch.autograd.Function):
    """
    custom implementation of linear interpolation in pytorch
    """
    @staticmethod
    def forward(ctx, input,Tf):
        ctx.save_for_backward(input,Tf)
        i1=np.argmin(np.abs(Tf[:,0]-input))
        x1=Tf[i1,0]
        y1=Tf[i1,1]
        Tf[i1,0]=1e6
        i2=np.argmin(np.abs(Tf[:,0]-input))
        x2=Tf[i2,0]
        y2=Tf[i2,1]
        dy=(y1-y2)
        dx=(x1-x2)
        y=y1+(dy/dx)*(input-x1)
        print(y)
        return np.clip(y,0.0,1.0)
        #return 0.5 * (5 * input ** 3 - 3 * input)
    

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input,Tf = ctx.saved_tensors
        i1=np.argmin(np.abs(Tf[:,0]-input))
        x1=Tf[i1,0]
        y1=Tf[i1,1]
        Tf[i1,0]=1e6
        i2=np.argmin(np.abs(Tf[:,0]-input))
        x2=Tf[i2,0]
        y2=Tf[i2,1]
        dy=(y1-y2)
        dx=(x1-x2)
        return grad_output * (dy/dx)
    
class dKr_LinearInterpolation(torch.autograd.Function):
    """
    custom implementation of linear interpolation in pytorch
    """
    @staticmethod
    def forward(ctx, input,Tf):
        ctx.save_for_backward(input,Tf)
        i1=np.argmin(np.abs(Tf[:,0]-input))
        x1=Tf[i1,0]
        y1=Tf[i1,1]
        Tf[i1,0]=1e6
        i2=np.argmin(np.abs(Tf[:,0]-input))
        x2=Tf[i2,0]
        y2=Tf[i2,1]
        dy=(y1-y2)
        dx=(x1-x2)
        y=y1+(dy/dx)*(input-x1)
        print(y)
        return dy/dx
        #return 0.5 * (5 * input ** 3 - 3 * input)
    

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input,Tf = ctx.saved_tensors
        i1=np.argmin(np.abs(Tf[:,0]-input))
        x1=Tf[i1,0]
        y1=Tf[i1,1]
        Tf[i1,0]=1e6
        i2=np.argmin(np.abs(Tf[:,0]-input))
        x2=Tf[i2,0]
        y2=Tf[i2,1]
        dy=(y1-y2)
        dx=(x1-x2)
        return grad_output # Obs: this assumes linear interpolation, and 2 order derivatives are not implemented yet

def Space2Tensor(Space,geometry,space_signature='x,y,vector,',tensor_signature="b x y c->b c x y"):
  return rearrange(Space.sample(geometry).native(space_signature).unsqueeze(0),tensor_signature)


def Tensor2Space(Tensor,geometry,tensor_signature='c x y->x y c',space_signature="x:s,y:s,vector:c"):
  return Field(geometry=geometry,values=math.wrap(rearrange(Tensor,tensor_signature),space_signature))


class physical_model(object):
	def __init__(self,domain,dt):
		#self.v0=v0
		self.domain=domain
		self.dt=dt
		self.p=None

	def momentum_eq(self,u, u_prev, dt, diffusivity=0.01):
		diffusion_term = dt * diffuse.implicit(u,diffusivity, dt=dt,correct_skew=False)
		advection_term = dt * advect.semi_lagrangian(u, u_prev,dt)
		return u + advection_term + diffusion_term


	def implicit_time_step(self, v, dt):
			v = math.solve_linear(self.momentum_eq, v, Solve('CG-adaptive',
																										1e-2,
																										1e-2,x0=v), u_prev=v, dt=-dt)
			v,p = fluid.make_incompressible(v,solve=Solve('CG-adaptive',
																										1e-2,
																										1e-2,
																													#x0=self.p
																													))
			return v

	def step(self,v):
		return self.implicit_time_step(v,self.dt)


class two_phase_flow_StableFluids(object):
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

    w_diffusion_term = dt * anisotropic_diffusion.implicit(phi_w,K_w(p_c), dt=dt,correct_skew=False).sample(phi_w.geometry)
    #o_diffusion_term = dt * anisotropic_diffusion.implicit(phi_o,K_o(p_c), dt=dt,correct_skew=False).sample(phi_w.geometry)

    pressure_chage_term = dt * (self.dtphi_o_1/(dsdpc(p_c)))

    return phi_w + phi_w.with_values(pressure_chage_term) + phi_w.with_values(w_advection_term) - phi_w.with_values(w_diffusion_term)
  
  def phi_o_momentum_eq(self,phi_o,phi_w, dt):
    #grad_phi_w=field.spatial_gradient(phi_w,phi_w.boundary)
    p_c=self.compute_p_c(phi_w,phi_o)

    o_advection_term = dt * advect.semi_lagrangian((phi_o),
                                                    self.compute_convective_velocity(phi_w,phi_o,dK_w,dK_o),
                                                    dt).sample(phi_o.geometry)

    o_diffusion_term = dt * anisotropic_diffusion.implicit(phi_o,K_o(p_c), dt=dt,correct_skew=False).sample(phi_o.geometry)

    pressure_chage_term = dt * (self.dtphi_w_1/(dsdpc(p_c)))

    return phi_o + phi_o.with_values(pressure_chage_term) + phi_o.with_values(o_advection_term) - phi_o.with_values(o_diffusion_term)
  
  def compute_phi_k(self,phi_w,phi_o,phi_w_1,phi_o_1,dt):
    return (phi_w-phi_w_1)/dt,(phi_o-phi_o_1)/dt


  def implicit_time_step(self, phi_w,phi_o, dt):
    new_phi_w = math.solve_linear(self.phi_w_momentum_eq, phi_w, self.w_advection_solver(phi_w),phi_o, dt=-dt)
    new_phi_o = math.solve_linear(self.phi_o_momentum_eq, phi_o, self.o_advection_solver(phi_o),phi_w, dt=-dt)
    self.dtphi_w_1,self.dtphi_o_1=self.compute_phi_k(new_phi_w,new_phi_o,phi_w,phi_o, dt)
    return new_phi_w,new_phi_o

class two_phase_flow_ReactionDiffusion(object):
  def __init__(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,por,mu_w,mu_o,K_s,kr_w,kr_o,Pc_args,max_dt=1e6,min_dt=-1e6):
    #self.v0=v0
    self.max_dt=max_dt
    self.min_dt=min_dt
    self.Pc_args=Pc_args
    self.phi_w=phi_w
    self.phi_o=phi_o
    self.dtphi_o_1=dtphi_o_1
    self.dtphi_w_1=dtphi_w_1
    self.dt=dt
    self.p=None
    self.K_l=K_s
    self.por=por
    self.mu_w=mu_w
    self.mu_o=mu_o

    self.SWR=sympy.symbols("S_{wc}")
    self.SOR=sympy.symbols("S_{or}")
    self.Sw=sympy.symbols("S_w")
    self.lam=sympy.symbols("\lambda")
    self.Pi=sympy.symbols("P_i")
    self.K_rw0=sympy.symbols("k_{rw0}")
    self.K_ro0=sympy.symbols("k_{ro0}")
    
    self.Pc_=sympy.symbols("P_c")
    
    self.Sc=(self.Sw-self.SWR)/(1-self.SWR-self.SOR)
    self.Pc=self.Pi*self.Sc**(-1/self.lam)
    
    self.Sw_Pc=(1-self.SWR-self.SOR)*((self.Pc_/self.Pi)**(-1*self.lam))+self.SWR
    dScdPc=sympy.diff(self.Sw_Pc,self.Pc_)
    
    K_rw=self.K_rw0*self.Sc**((2+3*self.lam)/(self.lam))
    K_ro=self.K_ro0*((1-self.Sc)**2)*(1-self.Sc**((2+self.lam)/(self.lam)))


    self.Pc_f_pyt=SymPyPhiFlowModule(expressions=[hide_floats(self.Pc.subs(list(map(lambda k:(getattr(self,k),Pc_args[k]) ,Pc_args))))])
    self.Pc_f=lambda x: self.Pc_f_pyt(S_w=x)[0]
    self.Sw_Pc_f_pyt=SymPyPhiFlowModule(expressions=[hide_floats(self.Sw_Pc.subs(list(map(lambda k:(getattr(self,k),Pc_args[k]) ,Pc_args))))],
      update_funcs={sympy.Pow: (lambda x,y: x**y),
                    sympy.Mul: _reduce(operator.mul),
                    sympy.Add: _reduce(lambda x,y:x+y)
                    }
      )
    self.Sw_Pc_f=lambda x: self.Sw_Pc_f_pyt(P_c=x)[0]

    self.K_rw_f_pyt=SymPyPhiFlowModule(expressions=[hide_floats(K_rw.subs(list(map(lambda k:(getattr(self,k),Pc_args[k]) ,Pc_args)) + [(self.K_rw0,kr_w)] ))])
    self.K_rw_f=lambda x: self.K_rw_f_pyt(S_w=x)[0]
    self.dK_rw_f_pyt=SymPyPhiFlowModule(expressions=[hide_floats(sympy.diff(K_rw,self.Sw).subs(list(map(lambda k:(getattr(self,k),Pc_args[k]) ,Pc_args)) + [(self.K_rw0,kr_w)] ))])
    self.dK_rw_f=lambda x: self.dK_rw_f_pyt(S_w=x)[0]
    self.K_ro_f_pyt=SymPyPhiFlowModule(expressions=[hide_floats(sympy.expand(K_ro.subs(list(map(lambda k:(getattr(self,k),Pc_args[k]) ,Pc_args)) + [(self.K_ro0,kr_o)] )))])
    self.K_ro_f=lambda x: self.K_ro_f_pyt(S_w=x)[0]
    self.dK_ro_f_pyt=SymPyPhiFlowModule(expressions=[hide_floats(sympy.diff(K_ro,self.Sw).subs(list(map(lambda k:(getattr(self,k),Pc_args[k]) ,Pc_args)) + [(self.K_ro0,kr_o)] ))])
    self.dK_ro_f=lambda x: self.dK_ro_f_pyt(S_w=x)[0]

    self.dScdPc_f_pyt=SymPyPhiFlowModule(expressions=[hide_floats(dScdPc.subs(list(map(lambda k:(getattr(self,k),Pc_args[k]) ,Pc_args))))])
    self.dScdPc_f=lambda x: self.dScdPc_f_pyt(P_c=x)[0]

    self.K_w=lambda K_l,p_c:stack(
          [stack([K_l*self.K_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w*self.dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
          stack([math.zeros_like(p_c),K_l*self.K_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w*self.dScdPc_f(p_c))],batch("k") )],batch("KK"))

    self.K_o=lambda K_l,p_c:stack(
        [stack([K_l*self.K_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o*self.dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
        stack([math.zeros_like(p_c),K_l*self.K_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o*self.dScdPc_f(p_c))],batch("k") )],batch("KK"))

    self.dK_w=lambda K_l,p_c:stack(
        [stack([K_l*self.dK_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w),math.zeros_like(p_c)],batch("dk") ),
        stack([math.zeros_like(p_c),K_l*self.dK_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w)],batch("dk") )],batch("dKK"))

    self.dK_o=lambda K_l,p_c:stack(
        [stack([K_l*self.dK_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o),math.zeros_like(p_c)],batch("dk") ),
        stack([math.zeros_like(p_c),K_l*self.dK_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o)],batch("dk") )],batch("dKK"))
    
    self.grad_phi_dK = lambda phi_a,dK_a:(math.dot(field.spatial_gradient(phi_a,phi_a.boundary).sample(phi_a.geometry),"vector",dK_a,"dKK"))


  def compute_p_c(self,phi_w,phi_o):
    p_c=phi_o.sample(phi_o.geometry) -\
    phi_w.sample(phi_w.geometry)
    p_c=pmath.clip(p_c,lower_limit=self.Pc_f(1-self.Pc_args["SOR"]),upper_limit=self.Pc_f(self.Pc_args["SWR"]))
    return p_c

  def compute_convective_velocity(self,phi_a,phi_b,dK_a,dK_b):
    p_c=self.compute_p_c(phi_a,phi_b)
    convective_velocity = self.grad_phi_dK(phi_a,dK_a(self.K_l,p_c))\
                         - self.grad_phi_dK(phi_b,dK_b(self.K_l,p_c))

    V=unstack(convective_velocity,"dk")
    convective_velocity=Field(self.phi_o.geometry,values=vec(x=V[0],y=V[1]))
    return convective_velocity
  
  def phi_w_pde(self,phi_w,phi_o,dtphi_o):
    p_c=self.compute_p_c(phi_w,phi_o)
    
    p_advection_term = phi_sum(
      self.compute_convective_velocity(phi_w,phi_o,self.dK_w,self.dK_w)*phi_w.gradient(),
      "vector").sample(phi_w.geometry)
    
    x,y=unstack(phi_sum(self.K_w(self.K_l,p_c),"KK"),"k")
    spatial_diffusion=Field(phi_w.geometry,values=vec(x=x,y=y))
    p_diffusion_term=phi_sum(phi_o.gradient(2)*spatial_diffusion,"vector").sample(phi_w.geometry)

    pressure_chage_term = (dtphi_o.values)
    return phi_w.with_values(pmath.clip(pressure_chage_term + p_advection_term - p_diffusion_term,
                                        lower_limit=self.min_dt,upper_limit=self.max_dt))
  
  def phi_o_pde(self,phi_o,phi_w,dtphi_w):
    p_c=self.compute_p_c(phi_w,phi_o)
    
    p_advection_term = phi_sum(
      self.compute_convective_velocity(phi_w,phi_o,self.dK_o,self.dK_o)*phi_o.gradient(),
      "vector").sample(phi_o.geometry)
    
    x,y=unstack(phi_sum(self.K_o(self.K_l,p_c),"KK"),"k")
    spatial_diffusion=Field(phi_o.geometry,values=vec(x=x,y=y))
    p_diffusion_term=phi_sum(phi_o.gradient(2)*spatial_diffusion,"vector").sample(phi_o.geometry)

    pressure_chage_term = (dtphi_w.values)
    return phi_o.with_values(pmath.clip(pressure_chage_term + p_advection_term - p_diffusion_term,
                                        lower_limit=self.min_dt,upper_limit=self.max_dt))
  
  def RK4(self,phi_w,phi_o,dt):

    K_o1=self.phi_o_pde(phi_o,phi_w,self.dtphi_w_1)
    K_w1=self.phi_w_pde(phi_w,phi_o,self.dtphi_o_1)

    K_o2=self.phi_o_pde(phi_o+0.5*K_o1.values*dt,phi_w+0.5*K_w1.values*dt,K_w1)
    K_w2=self.phi_w_pde(phi_w+0.5*K_w1.values*dt,phi_o+0.5*K_o1.values*dt,K_o1)

    K_o3=self.phi_o_pde(phi_o+0.5*K_o2.values*dt,phi_w+0.5*K_w2.values*dt,K_w2)
    K_w3=self.phi_w_pde(phi_w+0.5*K_w2.values*dt,phi_o+0.5*K_o2.values*dt,K_o2)

    K_o4=self.phi_o_pde(phi_o+K_o3.values*dt,phi_w+K_w3.values*dt,K_w3)
    K_w4=self.phi_w_pde(phi_w+K_w3.values*dt,phi_o+K_o3.values*dt,K_o3)

    self.dtphi_o_1 = (1/6)  * (K_o1 + 2*K_o2 + 2*K_o3 + K_o4)
    self.dtphi_w_1 = (1/6)  * (K_w1 + 2*K_w2 + 2*K_w3 + K_w4)

    phi_o = phi_o.with_values(pmath.finite_fill(phi_o.values + dt * self.dtphi_o_1.values))
    phi_w = phi_w.with_values(pmath.finite_fill(phi_w.values + dt * self.dtphi_w_1.values))
    return phi_w,phi_o
  
  def compute_phi_k(self,phi_w,phi_o,phi_w_1,phi_o_1,dt):
    return (phi_w-phi_w_1)/dt,(phi_o-phi_o_1)/dt


  def implicit_time_step(self, phi_w,phi_o, dt):
    new_phi_o = phi_o + dt * self.phi_o_pde(phi_o,phi_w,self.dtphi_w_1)
    new_phi_w = phi_w + dt * self.phi_w_pde(phi_w,phi_o,self.dtphi_o_1)
    self.dtphi_w_1,self.dtphi_o_1=self.compute_phi_k(new_phi_w,new_phi_o,phi_w,phi_o, dt)
    return new_phi_w,new_phi_o
  

class two_phase_flow_RD_decoupled_DT(two_phase_flow_ReactionDiffusion):
  def RK4(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt):

    K_o1=self.phi_o_pde(phi_o,phi_w,dtphi_w_1)
    K_w1=self.phi_w_pde(phi_w,phi_o,dtphi_o_1)

    K_o2=self.phi_o_pde(phi_o+0.5*K_o1.values*dt,phi_w+0.5*K_w1.values*dt,K_w1)
    K_w2=self.phi_w_pde(phi_w+0.5*K_w1.values*dt,phi_o+0.5*K_o1.values*dt,K_o1)

    K_o3=self.phi_o_pde(phi_o+0.5*K_o2.values*dt,phi_w+0.5*K_w2.values*dt,K_w2)
    K_w3=self.phi_w_pde(phi_w+0.5*K_w2.values*dt,phi_o+0.5*K_o2.values*dt,K_o2)

    K_o4=self.phi_o_pde(phi_o+K_o3.values*dt,phi_w+K_w3.values*dt,K_w3)
    K_w4=self.phi_w_pde(phi_w+K_w3.values*dt,phi_o+K_o3.values*dt,K_o3)

    dtphi_o_1 = (1/6)  * (K_o1 + 2*K_o2 + 2*K_o3 + K_o4)
    dtphi_w_1 = (1/6)  * (K_w1 + 2*K_w2 + 2*K_w3 + K_w4)

    phi_o = phi_o.with_values(pmath.finite_fill(phi_o.values + dt * dtphi_o_1.values))
    phi_w = phi_w.with_values(pmath.finite_fill(phi_w.values + dt * dtphi_w_1.values))
    return phi_w,phi_o,dtphi_w_1,dtphi_o_1
  
  def compute_phi_k(self,phi_w,phi_o,phi_w_1,phi_o_1,dt):
    return (phi_w-phi_w_1)/dt,(phi_o-phi_o_1)/dt


  def implicit_time_step(self, phi_w,phi_o,dtphi_w_1,dtphi_o_1, dt):
    new_phi_o = phi_o + dt * self.phi_o_pde(phi_o,phi_w,dtphi_w_1)
    new_phi_w = phi_w + dt * self.phi_w_pde(phi_w,phi_o,dtphi_o_1)
    dtphi_w_1,dtphi_o_1=self.compute_phi_k(new_phi_w,new_phi_o,phi_w,phi_o, dt)
    return new_phi_w,new_phi_o,dtphi_w_1,dtphi_o_1
  


class two_phase_flow_RD_TBK(two_phase_flow_RD_decoupled_DT):
    def __init__(self,phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,por,mu_w,mu_o,Pc_args,K_s,krwo,max_dt=1e6,min_dt=-1e6):
        super().__init__(phi_w,phi_o,dtphi_w_1,dtphi_o_1,dt,por,mu_w,mu_o,K_s=K_s,Pc_args=Pc_args,kr_w=0.3,kr_o=0.3,max_dt=1e6,min_dt=-1e6)
        self.krwo=krwo

        self.K_w=lambda K_l,p_c:stack(
              [stack([K_l*self.K_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w*self.dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
              stack([math.zeros_like(p_c),K_l*self.K_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w*self.dScdPc_f(p_c))],batch("k") )],batch("KK"))

        self.K_o=lambda K_l,p_c:stack(
            [stack([K_l*self.K_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o*self.dScdPc_f(p_c)),math.zeros_like(p_c)],batch("k") ),
            stack([math.zeros_like(p_c),K_l*self.K_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o*self.dScdPc_f(p_c))],batch("k") )],batch("KK"))

        self.dK_w=lambda K_l,p_c:stack(
            [stack([K_l*self.dK_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w),math.zeros_like(p_c)],batch("dk") ),
            stack([math.zeros_like(p_c),K_l*self.dK_rw_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_w)],batch("dk") )],batch("dKK"))

        self.dK_o=lambda K_l,p_c:stack(
            [stack([K_l*self.dK_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o),math.zeros_like(p_c)],batch("dk") ),
            stack([math.zeros_like(p_c),K_l*self.dK_ro_f(self.Sw_Pc_f(p_c))/(self.por*self.mu_o)],batch("dk") )],batch("dKK"))

        self.grad_phi_dK = lambda phi_a,dK_a:(math.dot(field.spatial_gradient(phi_a,phi_a.boundary).sample(phi_a.geometry),"vector",dK_a,"dKK"))

        # Autograd definitions
        #self.K_rw_f = lambda x: Kr_LinearInterpolation.apply(x,copy(self.krwo[:,[0,1]]))
        #self.dK_rw_f = lambda x: dKr_LinearInterpolation.apply(x,copy(self.krwo[:,[0,2]]))
        #self.K_ro_f = lambda x: Kr_LinearInterpolation.apply(x,copy(self.krwo[:,[0,1]]))
        #self.dK_ro_f = lambda x: dKr_LinearInterpolation.apply(x,copy(self.krwo[:,[0,2]]))

        # Exponential interpolation definition
        self.get_exp()
        self.K_rw_f = lambda x: self.A_rw*pmath.exp(self.k_rw*x)
        self.dK_rw_f = lambda x: self.k_rw*self.A_rw*pmath.exp(self.k_rw*x)
        self.K_ro_f = lambda x: self.A_ro*pmath.exp(self.k_ro*x)
        self.dK_ro_f = lambda x: self.k_ro*self.A_ro*pmath.exp(self.k_ro*x)


    def get_exp(self):
        n=self.krwo.shape[0]
        self.k_rw=(np.sum(np.prod(self.krwo[:,[0,1]],axis=1))-(1/n)*np.prod(np.sum(self.krwo[:,[0,1]],axis=0)))\
            /(np.sum(self.krwo[:,0]**2)-(1/n)*np.sum(self.krwo[:,0])**2)
        a=(1/n)*(np.sum(self.krwo[:,1],axis=0)-np.sum(self.krwo[:,0],axis=0))
        self.A_rw=np.exp(a)
        self.k_ro=(np.sum(np.prod(self.krwo[:,[0,2]],axis=1))-(1/n)*np.prod(np.sum(self.krwo[:,[0,2]],axis=0)))\
            /(np.sum(self.krwo[:,0]**2)-(1/n)*np.sum(self.krwo[:,0])**2)
        a=(1/n)*(np.sum(self.krwo[:,2],axis=0)-np.sum(self.krwo[:,0],axis=0))
        self.A_ro=np.exp(a)
