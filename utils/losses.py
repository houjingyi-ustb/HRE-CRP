import torch
import torch.nn.functional as F
import math


def xe(y_true, y_pred):
    loss = F.cross_entropy(y_pred, y_true)
    return loss
    
    
def cl_pos(p, z):
    z = z.detach()
    z = F.normalize(z, dim=1)
    p = F.normalize(p, dim=1)
    return -(p*z).sum(dim=1).mean()


def cl_neg(z1,z2):
    batch_size = z1.shape[0]
    n_neg = z1.shape[0]*2 - 2
    z = F.normalize(torch.cat([z1,z2], dim=0), dim=-1)
    mask = 1-torch.eye(batch_size, dtype=z.dtype, device=z.device).repeat(2,2)
    out = torch.matmul(z, z.T) * mask
    return (out.div(self.temp).exp().sum(1)-2).div(n_neg).mean().log()

