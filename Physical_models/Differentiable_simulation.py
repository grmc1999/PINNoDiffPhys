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
#sys.path.append(os.path.join("sympytorch","sympytorch"))
#from sympy_module import SymPyPhiFlowModule,_reduce
#from hide_floats_m import hide_floats
#import operator



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
  