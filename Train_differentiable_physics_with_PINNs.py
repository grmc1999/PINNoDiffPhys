import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from DL_models.Models.GAN import *
from DL_models.Models.PINN import *
from DL_models.PINNS.utils import derivatives

from Transforms.Data_transforms import *

import fire
import json

from phi.torch.flow import vec,UniformGrid, Field, tensor
from random import choice,sample
from Physical_models.Differentiable_simulation import physical_model,Space2Tensor,Tensor2Space
from copy import copy
from einops import rearrange
from scipy import ndimage
from random import randint
      
from phi.torch.flow import fluid,Solve
      

class PINNS_based_SOL_trainer(object):
    def __init__(self,field,physical_model,statistical_model,optimizer,simulation_steps,time_step,loss):

      self.dt=time_step
      self.v=field
      self.physical_model_constructor=physical_model

      self.ph_model=physical_model(self.v,dt=self.dt)

      self.init_states_gt=[self.v]
      self.T=[0.0]

      for i in range(50):
        self.init_states_gt.append(self.ph_model.step(self.init_states_gt[-1]))
        self.T.append(self.T[-1]+self.dt)

      self.n_steps=simulation_steps
      self.st_model=statistical_model
      self.loss=loss
      self.optimizer=optimizer

      self.geometry=self.v.geometry

    def generate_postion_time_code(self,field,t):  # Re implement if more dimensions are needed
      X=field.geometry.center.native("x,y")
      T=torch.ones(X.shape[:2]+(1,))*t
      XT=rearrange(torch.concat((X,T),axis=-1),"x y c -> c x y").unsqueeze(0)
      return XT

    def correct(self,states_pred):
      # TODO: implement a method to be re implemented for other architecures
      XTUp=rearrange(self.st_model(rearrange(states_pred," b (x y) u -> b u x y",x=self.os_[0],y=self.os_[1])),"b u x y  -> b (x y) u ")
      # GRAD HERE
      #x_grad(u,x.reshape(2,4,-1).transpose(1,-1),0,0)
      return states_pred,XTUp
    
    def forward_prediction_correction(self):

      # FIELD FORMAT
      self.os_=self.v.geometry.center.native("x,y").shape
      # SQUARED TENSOR FORMAT
      XT=self.generate_postion_time_code(self.v,self.t)
      XTU=torch.concat((XT,Space2Tensor(self.v,self.geometry)),axis=1)

      # TABULAR TENSOR FORMAT
      states_pred=[rearrange(XTU,"b u x y-> b (x y) u").requires_grad_(True)]  # START OF FLUX
      XTUp_1,XTUp=self.correct(states_pred[-1])
      
      # OBS: correction is made in field forrmat, for no aparent reason (TODO: try to train in field format)
      states_in=[XTUp_1]
      states_corr=[XTUp]
      states_pred=[XTUp_1[:,:,-1:]+XTUp]

      # For steps in correction run (4 in example) (incidencia nos iniciais)
      for i in range(self.n_steps):
        
        # Step last in states_pred
        U_space=Tensor2Space(rearrange(states_pred[-1]," b (x y) u -> b u x y",x=self.os_[0],y=self.os_[1])[0],self.geometry)
        states_pred.append(self.ph_model.step(
          U_space
          ))
        #U_tensor=Space2Tensor(U_space,self.geometry)
        self.t=self.t+self.dt
        XT=self.generate_postion_time_code(U_space,self.t)
        XTU=torch.concat((XT,Space2Tensor(U_space,self.geometry)),axis=1)
        # GRAD HERE
        
        states_pred[-1]=rearrange(XTU,"b u x y-> b (x y) u")
        # Correct with model of last states_pred
        XTUp_1,XTUp=self.correct(states_pred[-1])

        states_in.append(XTUp_1)
        states_corr.append(XTUp)
        states_pred[-1]=XTUp_1[:,:,-1:]+XTUp

      return states_pred,states_corr,states_in
    
    def detach_field(self,field):
      tensor=Space2Tensor(field,self.geometry).detach()
      return Tensor2Space(tensor[0],self.geometry)
    
    def train(self,epochs):
      losses=[]
      for i in range(epochs):
        #print(f"epoch {i}")
        co_batch=[]
        in_batch=[]
        for b in range(20):
          #print(f"batch {b}")

          random_idx=randint(0,len(self.init_states_gt)-1)
          self.v=self.detach_field(self.init_states_gt[random_idx])
          self.t=self.T[random_idx]
          states_pred,states_corr,states_in=self.forward_prediction_correction()
          co_batch=co_batch+states_pred
          in_batch=in_batch+states_in

        loss=0
        for i,(u,x) in enumerate(zip(co_batch,in_batch)):
          sample_loss=self.loss(u,x)
          loss=loss+sample_loss

        #states_in=torch.concat(states_in,axis=0) # [B X Y U]
        #states_pred=torch.concat(states_pred,axis=0) # [B X Y U]
        #print("LOSS")
        #loss=self.loss(states_pred,states_in)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses.append(loss.cpu().detach().numpy())
      return losses

      def test(self,epochs):
        losses=[]
        for i in range(epochs):
          gt_batch=[]
          co_batch=[]
          for b in range(5):
            self.v_gt=choice(self.init_states_gt)
            states_gt=self.forward_fine_grained()
            self.v_co=Field(values=Tensor2Space(states_gt[0].detach(),self.geo_co),geometry=self.geo_co,boundary=self.boundary)
            states_pred,states_corr=self.forward_prediction_correction()
            gt_batch=gt_batch+states_gt
            co_batch=co_batch+states_pred

          states_pred=torch.concat(states_pred,axis=0)
          states_gt=torch.concat(states_gt,axis=0)
          loss=self.loss(states_pred,states_gt)

          losses.append(loss.cpu().detach().numpy())
        return losses