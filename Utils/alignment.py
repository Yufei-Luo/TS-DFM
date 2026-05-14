import torch
from torch_scatter import scatter_mean, scatter_add

def Kabsch_alignment(pos_to_fit, pos, batch):
    v = pos.shape[-1]
    center = scatter_mean(pos, batch, dim = -2) # B * 3
    pos_to_fit_center = scatter_mean(pos_to_fit, batch, dim = -2) # B * 3
    pos_c = pos - center[batch]
    pos_to_fit_c = pos_to_fit - pos_to_fit_center[batch]
    pos_c = pos_c.repeat([1,v])
    pos_to_fit_c = pos_to_fit.repeat_interleave(v,dim=-1)
    H = scatter_add(pos_c * pos_to_fit_c, batch, dim = -2).reshape(-1,v,v) # B * 3 * 3
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V @ U.transpose(2,1)
    t = center - (pos_to_fit_center.unsqueeze(1) @ R.transpose(2,1)).squeeze(1)
    R = R[batch]
    t = t[batch]
    p_aligned = (pos_to_fit.unsqueeze(1) @ R.transpose(2,1)).squeeze(1) + t
    return p_aligned
