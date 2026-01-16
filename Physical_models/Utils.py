
import torch
import math


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