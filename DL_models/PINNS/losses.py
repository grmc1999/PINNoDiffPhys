import torch
from functools import reduce


# This losses depend on the type of PDE

#  

# 1. The error on the data
# 2. The error on the residual of the PDE
# 3. The error on the parameters of the PDE



class PDE_res(object):
    def __init__(self,R,f,norm):
        """
        R: function of the combination of derivatives and perturbation function:: input derivatives and output scalar
        f: perturbation funtion:: input derivatives?
        """
        self.R=R
        self.f=f
        self.norm=norm
    def __call__(self,U,X):
        """
        U: list of derivatives
        """

        return self.norm(self.R(U,X)-self.f(U,X))

class PDE_U(object):
    def __init__(self,norm):
        """
        R: function of the combination of derivatives and perturbation function:: input derivatives and output scalar
        f: perturbation funtion:: input derivatives?
        """
        self.norm=norm
    def __call__(self,U,U_):
        """
        U: list of derivatives
        """

        return self.norm(U-U_)


class PINN_loss(object):
    #def __init__(self,R,f,res_norm,supervised_norm,weights={"Residual_loss":1.,"Supervised_loss":1.}):
    def __init__(self,PDE_res_args,PDE_sup_args,weights={"PDE_residual_loss":1.,"PDE_supervised_loss":1.}):
        self.res_loss=PDE_res(**PDE_res_args)
        self.sup_loss=PDE_U(**PDE_sup_args)

        self.w=weights
        total_w=reduce(lambda x,y:x+y,list(self.w.values()))
        for k in self.w.keys():
            self.w[k]=self.w[k]/total_w
    def __call__(self,X,U,U_):
        self.total_loss={
            "PDE_residual_loss":self.res_loss(U_,X),
            "PDE_supervised_loss":self.sup_loss(U,U_)
            }
        self.total_loss.update({"total_loss":
                                self.total_loss["PDE_residual_loss"]*self.w["PDE_residual_loss"]+\
                                self.total_loss["PDE_supervised_loss"]*self.w["PDE_supervised_loss"]
                                })
        return self.total_loss

# Model specific losses

class Discriminator_loss(object):
    def __init__(self):
        pass
    def __call__(self,U,logits_G): # u : [b, t, x , y]
        # Mean of cross entropy loss
        real_loss=torch.mean(
            -1*torch.sum(torch.log(1 - torch.nn.functional.sigmoid(U)),axis=0) # u : [t, x , y]
            )
        fake_loss=torch.mean(
            -1*torch.sum(torch.log(torch.nn.functional.sigmoid(logits_G)),axis=0) # u : [t, x , y]
        )
        return real_loss + fake_loss

class Generator_loss(object):
    def __init__(self,W_posterior):
        self.W_posterior=W_posterior
    def __call__(self,logits_G,logits_P): # u : [b, t, x , y]
        # Mean of cross entropy loss
        gen_loss_entropy=torch.mean(logits_G)
        gen_loss_posterior=torch.mean(
            (self.W_posterior-1.0)*(-1*torch.sum(torch.log(torch.nn.functional.sigmoid(logits_P)),axis=0))
        )
        return {"generative_entropy_loss":gen_loss_entropy,"generative_posterior_loss": gen_loss_posterior}

class PDE_Generator_loss(object):
    def __init__(self,args_Gen,args_PDE_res,args_PDE_sup):
        self.G_loss=Generator_loss(**args_Gen)
        self.PDE_res_loss=PDE_res(**args_PDE_res)
        self.PDE_sup_loss=PDE_U(**args_PDE_sup)
        #super().__init__(**args)
    def __call__(self,logits_G,logits_P,X,U):
        loss=self.G_loss(logits_G,logits_P)
        loss.update({
            "PDE_residual_loss":self.PDE_res_loss(logits_G,X),
            "PDE_supervised_loss":self.PDE_sup_loss(logits_G,U),
            })
        return loss

class PDE_GAN_loss(object):
    def __init__(self,args_Gen,args_PDE_res,args_PDE_sup,weights={"generative_posterior_loss":1.,
                                                                "generative_entropy_loss":1.,
                                                                "PDE_residual_loss":1.,
                                                                "PDE_supervised_loss":1.}):
        self.G_loss=PDE_Generator_loss(args_Gen,args_PDE_res,args_PDE_sup)
        self.D_loss=Discriminator_loss()
        self.w=weights
        total_w=reduce(lambda x,y:x+y,list(self.w.values()))
        for k in self.w.keys():
            self.w[k]=self.w[k]/total_w
        
        #self.PDE_loss=PDE_res(**args_PDE)
        
    def __call__(self,logits_G,logits_P,logits_F,logits_R,X,U):
        self.total_loss=self.G_loss(logits_G,logits_P,X,U)
        self.total_loss.update({"Discriminator_loss":self.D_loss(U,logits_G)})
        self.total_loss.update({"Generator_loss":self.total_loss["generative_posterior_loss"]*self.w["generative_posterior_loss"]+\
                                                self.total_loss["generative_entropy_loss"]*self.w["generative_entropy_loss"]+\
                                                self.total_loss["PDE_residual_loss"]*self.w["PDE_residual_loss"]+\
                                                self.total_loss["PDE_supervised_loss"]*self.w["PDE_supervised_loss"]
        })
        self.total_loss.update({"total_loss":
        torch.sum(reduce(lambda x,y:x+y,list(self.total_loss.values())))
        #torch.sum(torch.Tensor(list(self.total_loss.values())))
        })
        return self.total_loss