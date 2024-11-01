from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import normalize
from einops import rearrange, repeat
from ..flow_util import warp, nn_gather, get_gt_correspondence_mask, warp_grid, batch_project_to_other_img

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from ..geometry.projection import sample_image_grid
import torch.nn.functional as F
@dataclass
class LossposeCfg:
    weight_2d: float
    weight_3d: float


@dataclass
class LossposeCfgWrapper:
    pose: LossposeCfg

class Losspose(Loss[LossposeCfg, LossposeCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        c2w,
        depth,
        corr,
        xyz_h
    ) -> Float[Tensor, ""]:
        
        """Compute the correspondence loss
        """
        b, v, c ,h, w = batch['context']['image'].shape
        corr_loss = {}
       
        xyz = xyz_h
        index_lists = [(a, b) for a in range(v) for b in range(a + 1, v)]
        relpose, abspose = c2w[0], c2w[1]
        corre, conf, trans_conf = corr[0], corr[1], corr[2]
        xy_ray, _ = sample_image_grid((h, w), batch['context']['image'].device)
        xy_ray = repeat(rearrange(xy_ray, "h w xy -> (h w) () xy"), "hw () xy -> b hw xy", b=b) # 1 65536 2
        pred_depth = rearrange(depth[0], '(b v) 1 h w -> b v  h w', b=batch['target']['image'].shape[0])
        pts = {}
        pts_rel = {}
        
        for idx, (i_idx, j_idx) in enumerate(index_lists):
            abs_Rt_ij = c2w[1][:,[i_idx, j_idx]]
            rel_Rt_ij = c2w[0][:, [idx]]
            
            if i_idx == 0:
                abs_Rt = abs_Rt_ij[:, 1]
            else:
                abs_Rt = get_relative_Rt(abs_Rt_ij[:, 0], abs_Rt_ij[:, 1])
            rel_Rt = rel_Rt_ij.squeeze(1)
            pts[(i_idx, j_idx)] = batch_project_to_other_img(xy_ray, pred_depth[:, i_idx].flatten(-2,-1), batch["target"]["intrinsics"][:,i_idx], batch["target"]["intrinsics"][:,j_idx],abs_Rt)
            pts_rel[(i_idx, j_idx)] = batch_project_to_other_img(xy_ray, pred_depth[:, i_idx].flatten(-2,-1), batch["target"]["intrinsics"][:,i_idx], batch["target"]["intrinsics"][:,j_idx],rel_Rt)
    
        
        
        cnt = 0
        total_loss = 0
        loss_abs_list = []
        loss_rel_list = []
        loss_abs_2d_list = []
        loss_rel_2d_list = []
        for i in range(b):
            for idx, (i_idx, j_idx) in enumerate(index_lists):
                
                id_i, id_j, w_ij = corre[(i_idx, j_idx)][i]
                w_ij = w_ij
                w_ij_n = normalize(w_ij, p=1, dim=-1)
                conf_ij = trans_conf[(i_idx, j_idx)][i]
                
                corr_i = nn_gather(xyz[i, i_idx].flatten(-2,-1).unsqueeze(0).permute(0,2,1), id_i.unsqueeze(0))
                corr_j = nn_gather(xyz[i, j_idx].flatten(-2,-1).unsqueeze(0).permute(0,2,1), id_j.unsqueeze(0))
                
                corr_i_j_2d = nn_gather(pts[(i_idx, j_idx)][i].unsqueeze(0), id_i.unsqueeze(0))
                corr_i_j_2d_rel = nn_gather(pts_rel[(i_idx, j_idx)][i].unsqueeze(0), id_i.unsqueeze(0))
                
                corr_j_2d = nn_gather(xy_ray[i].unsqueeze(0), id_j.unsqueeze(0))
                
                abs_corr_2d_diff = F.huber_loss((corr_i_j_2d - corr_j_2d).norm(p=2, dim=-1), torch.zeros_like((corr_i_j_2d - corr_j_2d).norm(p=2, dim=-1)), reduction='none', delta=0.01) / 0.01
                rel_corr_2d_diff = F.huber_loss((corr_i_j_2d_rel - corr_j_2d).norm(p=2, dim=-1), torch.zeros_like((corr_i_j_2d_rel - corr_j_2d).norm(p=2, dim=-1)), reduction='none', delta=0.01) / 0.01

                
                loss_abs_2d = (abs_corr_2d_diff).sum(dim=-1)
                loss_rel_2d = ( rel_corr_2d_diff).sum(dim=-1)
                
                loss_abs_2d_list.append(loss_abs_2d )
                loss_rel_2d_list.append(loss_rel_2d  )
                
                abs_Rt_ij = abspose[i][[i_idx, j_idx]]
                rel_Rt_ij = relpose[i][[idx]]
                if i_idx == 0:
                    abs_Rt = abs_Rt_ij[1]
                else:
                    abs_Rt = get_relative_Rt(abs_Rt_ij[0], abs_Rt_ij[1])
                rel_Rt = rel_Rt_ij
            
                abs_corr_i = transform_points_Rt(corr_i, abs_Rt)
                rel_corr_i = transform_points_Rt(corr_i, rel_Rt)
      
                # loss is weighted sum over residuals; weights are L1 normalized first
        

                abs_corr_diff = (abs_corr_i - corr_j).norm(p=2, dim=-1)
                rel_corr_diff = (rel_corr_i - corr_j).norm(p=2, dim=-1)
    
                loss_abs = (w_ij_n * abs_corr_diff).sum(dim=-1)
                loss_rel = (w_ij_n * rel_corr_diff).sum(dim=-1)
                
                
                loss_abs_list.append(loss_abs* conf_ij )
                loss_rel_list.append(loss_rel  * conf_ij)
                
                cnt += 1
        
        return (torch.stack(loss_abs_list,dim=0).mean() ) * self.cfg.weight_3d   + \
             (torch.stack(loss_abs_2d_list,dim=0).mean()) * self.cfg.weight_2d 
     
def transform_points_Rt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    R = viewpoint[..., :3, :3]
    t = viewpoint[..., None, :3, 3]
    # N.B. points is (..., n, 3) not (..., 3, n) X R^T    R X
    if inverse:
        return (points - t) @ R
    else:
        return points @ R.transpose(-2, -1) + t
def get_relative_Rt(Rt_i, Rt_j):
    """Generates the relative Rt assuming that we have two world
    to camera Rts. Hence, Rt_ij = inverse(Rt_i) @ Rt_j.

    Args:
        Rt_i (FloatTensor): world_to_camera for camera i (batch, 4, 4)
        Rt_j (FloatTensor): world_to_camera for camera j (batch, 4, 4)

    Returns:
        Rt_ij (FloatTensor): transformation from i to j (batch, 4, 4)
    """
    assert Rt_i.shape == Rt_j.shape, "Shape mismatch"
    assert Rt_i.size(-2) == 4
    assert Rt_i.size(-1) == 4

    return Rt_j @ Rt_i.inverse()

