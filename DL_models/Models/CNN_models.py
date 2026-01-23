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
    x=self.act3(self.conv3(x))*1
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
    self.act3=torch.nn.Identity()
  def forward(self,x1,x2,dtx1,dtx2):
    x=torch.concatenate((x1,x2),axis=1)
    x=self.act1(self.conv1(x))
    x=self.act2(self.conv2(x))
    x=self.act3(self.conv3(x))*1
    return x[:,0],x[:,1],dtx1[0],dtx2[0]


class simple_dual_space_with_time_derivative_cnn_model(torch.nn.Module):
  def __init__(self):
    super(simple_dual_space_with_time_derivative_cnn_model,self).__init__()

    self.conv1=torch.nn.Conv2d(4, 32, (9,9), padding=4)
    self.conv2=torch.nn.Conv2d(32, 64, (9,9), padding=4)
    self.conv3=torch.nn.Conv2d(64, 4, (9,9), padding=4)
    self.act1=torch.nn.ReLU()
    self.act2=torch.nn.ReLU()
    self.act3=torch.nn.Identity()
  def forward(self,x1,x2,dtx1,dtx2):
    x=torch.concatenate((x1,x2,dtx1,dtx2),axis=1)
    x=self.act1(self.conv1(x))
    x=self.act2(self.conv2(x))
    x=self.act3(self.conv3(x))*1
    return x[:,0],x[:,1],x[:,2],x[:,3]