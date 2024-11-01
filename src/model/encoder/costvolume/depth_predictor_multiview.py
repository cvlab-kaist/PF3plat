import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .ldm_unet.unet import UNetModel



def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().inverse()
                    @ extrinsics[:, v0].clone()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)
    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0]
        pose_tgt = extrinsics[:, 1]
        pose = pose_tgt.inverse() @ pose_ref
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
        * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr


class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=256,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=3,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
      
       
        # Cost volume refinement: 2D U-Net
        input_channels =  (num_depth_candidates + feature_channels)
        channels = self.regressor_feat_dim
        
    
        modules = [
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=costvolume_unet_attn_res,
                channel_mult=costvolume_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
            ]
        self.corr_refine_net = nn.Sequential(*modules)
        # cost volume u-net skip connection
        self.regressor_residual = nn.Conv2d(
            input_channels, num_depth_candidates, 1, 1, 0
        )

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )
        self.mono_expand = nn.Sequential(
                nn.Conv2d(self.num_depth_candidates, self.num_depth_candidates, kernel_size=3, padding=1, stride=2),
                nn.GELU(),
                nn.Conv2d(self.num_depth_candidates, self.num_depth_candidates, kernel_size=3, padding=1, stride=2),
                nn.GELU(),
            )
        self.multi_expand = nn.Sequential(
            nn.Conv2d(self.num_depth_candidates, self.num_depth_candidates, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            nn.Conv2d(self.num_depth_candidates, self.num_depth_candidates, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
        )
        self.qk_channels = self.num_depth_candidates
        self.lin_multi_v = nn.Conv2d(self.num_depth_candidates, self.num_depth_candidates, kernel_size=1)
        self.lin_mono_k = nn.Conv2d(self.num_depth_candidates, self.qk_channels, kernel_size=1)
        self.lin_mono_q = nn.Conv2d(self.num_depth_candidates, self.qk_channels, kernel_size=1)
        self.multi_reg = nn.Sequential(
                nn.Conv2d(self.num_depth_candidates, self.num_depth_candidates, kernel_size=1, padding=0),
                nn.GELU(),
            )
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # CNN-based feature upsampler
        proj_in_channels = feature_channels 
        upsample_out_channels = feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        
    
        self.refine_unet = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1, 
                attention_resolutions=depth_unet_attn_res,
                channel_mult=depth_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
        )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
      
        channels = depth_unet_feat_dim
        disps_models = [
            nn.Conv2d(channels + 1 + upsample_out_channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
        ]
        self.to_disparity = nn.Sequential(*disps_models)
    def correlation(self, src_feat, trg_feat, eps=1e-5):
        '''src_feat = src_feat / (src_feat.norm(dim=1, p=2, keepdim=True) + eps)
        trg_feat = trg_feat / (trg_feat.norm(dim=1, p=2, keepdim=True) + eps)'''

        return torch.einsum("bchw, bcxy -> bhwxy", src_feat, trg_feat)
    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""

        # format the input
        b, v, c, h, w = features.shape
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )
        feat01 = feat_comb_lists[0]

        raw_correlation_in_lists = []
        for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
            # sample feat01 from feat10 via camera projection
            feat01_warped = warp_with_pose_depth_candidates(
                feat10,
                intr_curr,
                pose_curr,
                1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                warp_padding_mode="zeros",
            )  # [B, C, D, H, W]
            # calculate similarity
            raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                1
            ) / (
                c**0.5
            )  # [vB, D, H, W]
            raw_correlation_in_lists.append(raw_correlation_in)
        # average all cost volumes
        raw_correlation_in = torch.mean(
            torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
        )  # [vxb d, h, w]
        raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
       
        
        raw_correlation = self.corr_refine_net(raw_correlation_in)  # (vb d h w)
        # apply skip connection
        raw_correlation = raw_correlation + self.regressor_residual(
            raw_correlation_in
        )
        '''
        multi guided by mono
        '''
        mono_feat = self.mono_expand(extra_info['monocular_cue']) # 64 64
        multi_feat = self.multi_expand(raw_correlation) # 16 16
        b_v,_,h_down,w_down = multi_feat.shape

        # mono attention
        mono_q = self.lin_mono_q(mono_feat).view(b_v,-1,h_down*w_down).permute(0,2,1)
        mono_k = self.lin_mono_k(mono_feat).view(b_v,-1,h_down*w_down)
        mono_score = torch.bmm(mono_q, mono_k)
        mono_atten = torch.softmax(mono_score,dim=-1)
      
        multi_v = self.lin_multi_v(multi_feat).view(b_v,-1,h_down*w_down)
        multi_out = torch.bmm(multi_v, mono_atten.permute(0,2,1))
        multi_out = multi_out.view(b_v,self.num_depth_candidates, h_down,w_down)


        # upsample
        fused = torch.nn.functional.interpolate(multi_out, size=(h,w))

    
        multi_residual = self.multi_reg(raw_correlation)

        fused_cost_volume = multi_residual + self.gamma * fused
        
        # softmax to get coarse depth and density
        pdf = F.softmax(
            self.depth_head_lowres(fused_cost_volume), dim=1
        )  # [2xB, D, H, W]
        
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
        pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)
        
        # depth refinement
        proj_feat_in_fullres = self.upsampler(feat01)
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        refine_out = self.refine_unet(torch.cat(
            (extra_info["images"], proj_feature, extra_info['disparity'], pdf_max), dim=1
        ))
        
        # gaussians head
        raw_gaussians_in = [refine_out,
                            extra_info["images"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(
            raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        )


        delta_disps_density = self.to_disparity(torch.cat([refine_out,extra_info['disparity'], proj_feat_in_fullres], dim=1))
        _, raw_densities = delta_disps_density.split(
            gaussians_per_pixel, dim=1
        )

        # combine coarse and fine info and match shape
        densities = repeat(
            F.sigmoid(raw_densities),
            "(v b) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        )


        return densities, raw_gaussians