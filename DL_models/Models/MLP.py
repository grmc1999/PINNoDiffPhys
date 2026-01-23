import torch
from torch import nn
from einops import rearrange,repeat



class MLP(torch.nn.Module):
    def __init__(self,layers,activators):
        super(MLP,self).__init__()
        self.layer_sizes=layers
        self.activators=activators
        #for layer,activator in zip(layers,activators):
        #    self.NN.appe
        #self.NN=torch.nn.Sequential()

        self.layers=nn.ModuleList(
            [
                nn.Sequential(*(torch.nn.Linear(in_size,out_size),act))
                for in_size,out_size,act in zip(
                    self.layer_sizes[:-1],
                    self.layer_sizes[1:],
                    self.activators,
                )
            ]
        )
    
    def forward(self,x):
        init_shape=x.shape
        #print(init_shape)
        if len(init_shape)==3:
            x=rearrange(x,"b p v -> b (p v)")
        for layer in self.layers:
            x=layer(x)
        if len(init_shape)==3:
            x=repeat(x,"b v -> b p v",p=1)
        return x