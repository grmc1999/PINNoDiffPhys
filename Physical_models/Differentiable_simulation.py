from phi.torch.flow import Field, math
from einops import rearrange

from Physical_models.pde_models import NavierStokesModel



def Space2Tensor(Space, geometry, space_signature="x,y,vector,", tensor_signature="b x y c->b c x y"):
  return rearrange(Space.sample(geometry).native(space_signature).unsqueeze(0),tensor_signature)


def Tensor2Space(Tensor, geometry, tensor_signature="c x y->x y c", space_signature="x:s,y:s,vector:c"):
  return Field(geometry=geometry,values=math.wrap(rearrange(Tensor,tensor_signature),space_signature))


physical_model = NavierStokesModel
  
