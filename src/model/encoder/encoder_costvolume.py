from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
from .costvolume.ldm_unet.unet import UNetModel
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid, unproject, get_world_rays
from ..types import Gaussians
import torch.nn.functional as F

from ..LightGlue.lightglue.lightglue import LearnableFourierPositionalEncoding, SelfBlock, CrossBlock
from ..LightGlue.lightglue import LightGlue, SuperPoint, match_pair
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
from ...global_cfg import get_cfg
from ..unidepth import UniDepthV2
from ..unidepth.layers import MLP, Mlp
from ...flow_util import * 
from .aggregation import LocalFeatureTransformer
from .multiview_transformer import MultiViewFeatureTransformer

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    fcgf_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
 


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)
        
        # intrinsic 
        self.in_features = nn.Linear(256, 128)
        self.confidence_min = 0.5
     
        # dino v2 feature aggregation
        self.dino_projector = nn.Linear(2048, 256)
        self.dino_aggregator = LocalFeatureTransformer()
        self.cross_view_aggregator = MultiViewFeatureTransformer() # while the name is cross_view_aggregator, it is actually used for self-attention
        self.scale_shift_predictor = MLP(128, expansion=2, dropout=0.0,  output_dim = 2)
        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        # matching
        self.extractor = SuperPoint(max_num_keypoints=None).eval()
        self.matcher = LightGlue(features='superpoint',depth_confidence=-1, width_confidence=-1).eval()
        # depth
        self.uni_depth = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        h, self.n_layers, d = 4, 6, 128
        head_dim = d // h
        self.depth_self_attn = nn.ModuleList(
            [SelfBlock(d, h, True) for _ in range(self.n_layers)]
        )
        self.posenc = LearnableFourierPositionalEncoding(
            2 , head_dim, head_dim
        )
        self.conv_proj = nn.Conv2d(256 + 6, 128, 3, 1, 1)
        # pose 
        self.pose_transformers = nn.ModuleList(
            [SelfBlock(d, h, False) for _ in range(self.n_layers)]
        )
        self.pose_self_attn = nn.ModuleList(
            [SelfBlock(d, h, False) for _ in range(self.n_layers)]
        )
        self.pose_cross_attn = nn.ModuleList(
            (CrossBlock(d, h, False) for _ in range(self.n_layers))
        )
        self.embed_pose = Mlp( in_features=9,
            hidden_features=64,
            out_features=128,
            drop=0,
        )
        self.pose_branch = Mlp(
            in_features=128,
            hidden_features=128 * 2,
            out_features=128 + 9 +2,
            drop=0,
        )
        self.ffeat_updater = nn.Sequential(
            nn.Linear(128, 128), nn.GELU()
        )
        self.pose_trunk = nn.Sequential(
            *[
                SelfBlock(d, h, True) 
                for _ in range(self.n_layers)
                ]
        )
        self.pose_cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.norm = nn.LayerNorm(
            128, elementwise_affine=False, eps=1e-6
        )
        self.pose_token = nn.Parameter(
            torch.zeros(1, 1, 1, 128)
        )  
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.pose_token, std=1e-6)


        
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views ,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        )
     
     
       
    def feature_add_position_list(self, features_list, attn_splits, feature_channels):
        pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
        if attn_splits > 1:  # add position in splited window
            features_splits = [
                split_feature(x, num_splits=attn_splits) for x in features_list
            ]
            position = pos_enc(features_splits[0])
            features_splits = [x + position for x in features_splits]

            out_features_list = [
                merge_splits(x, num_splits=attn_splits) for x in features_splits
            ]
        else:
            position = pos_enc(features_list[0])
            out_features_list = [x + position for x in features_list]
        return out_features_list

    def get_scaleshift(self, x, h, w):
        scale, shift = torch.chunk(x, 2, dim=-1)
        scale = scale.exp()
        shift = shift.clamp(min=-5, max = 5)
        # b v (h w) 1
        return rearrange(scale, 'b v (h w) 1 -> (b v) 1 h w', h = h, w = w), rearrange(shift, 'b v (h w) 1 -> (b v) 1 h w', h = h, w = w)
    
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 1
        
        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    
    def r6d2mat(self, d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalisation per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
                first two rows of the rotation matrix.
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row
    
    def plucker_embedding(self, H, W, cam_pos,ray_dirs, jitter=False):
        """Computes the plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
        c2w: (B, 4, 4)
        intrinsics: (B, 3, 3)
        """    
        
        cross = torch.cross(cam_pos, ray_dirs, dim=-1)
        plucker = torch.cat((ray_dirs, cross), dim=-1)

        plucker = plucker.view(-1, H, W, 6).permute(0, 3, 1, 2)
        return plucker  # (B, 6, H, W, ) 
    
    def matrix_to_rotation_6d(self, pose_mtx):
        return pose_mtx[..., :2, :].clone().reshape(*pose_mtx.size()[:-2], 6)

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
     
        
        # imagenet normalization
        img = context["image"].clone()
        unnorm_intrinsics = context["intrinsics"].clone()
        unnorm_intrinsics[:, :, :2, :] = unnorm_intrinsics[:, :, :2, :] *255

        '''
        unidepth prediction
        '''

        predictions = self.uni_depth.infer(rearrange(img *255., 'b v c h w -> (b v) c h w'), rearrange(unnorm_intrinsics, 'b v x y -> (b v) x y'))
        disparity = 1. /predictions["depth"]
        
        
        '''
        cross-view feature extraction
        '''
        
        predictions['feat'] = self.dino_projector(torch.cat(predictions['feat'], dim=-1))
        predictions['feat'] = rearrange(predictions['feat'], ' (b v) h w c -> b v c h w', b = b, v =v )
        _, _, _, h_down, w_down = predictions['feat'].shape
       
        pre_cross_view_features = rearrange(self.dino_aggregator(rearrange(predictions['feat'], 'b v c h w -> (v b) (h w) c')), '(v b) L c -> b v L c', b=b, v=v)
        list_features = [rearrange(pre_cross_view_features[:, vi, :, :], 'b (h w) c -> b c h w', h =h_down) for vi in range(v)]
        list_features = self.feature_add_position_list(list_features, self.cfg.multiview_trans_attn_split,256)
        per_view_depth_features = torch.stack(self.cross_view_aggregator(list_features, attn_num_splits=self.cfg.multiview_trans_attn_split), dim = 1 )
        per_view_depth_features = rearrange(F.interpolate(rearrange(per_view_depth_features, 'b v c h w -> (v b) c h w', v=v, b=b), (h//4, w//4), mode='bilinear'), '(v b) c h w -> b v c h w', v=v, b=b)
      
        predictions["depth"] = predictions["depth"].clamp(rearrange(context["near"], 'b v -> (b v) () () ()'), rearrange(context["far"], 'b v -> (b v) () () ()'))
        depths = predictions["depth"]

        '''
        scale and shift prediction
        Here, we use self and cross attention between feature maps, then we use the mlp to predict per patch shift
        we thus update the depth here
        '''
        
        _, _, P, c = pre_cross_view_features.shape
        cross_view_depth_features_scale_shift = rearrange(self.in_features(rearrange(pre_cross_view_features, 'b v N c -> (b v) N c')), '(b v) N c -> b v N c', b=b, v=v)
        for idx in range(self.n_layers):
            cross_view_depth_features_scale_shift = rearrange(cross_view_depth_features_scale_shift, 'b v p c -> (b v) p c')
            cross_view_depth_features_scale_shift = self.depth_self_attn[idx](cross_view_depth_features_scale_shift, None)
            cross_view_depth_features_scale_shift = rearrange(cross_view_depth_features_scale_shift, "(b s) p c -> b s p c", b=b, s=v)
            
            feat_0 = cross_view_depth_features_scale_shift[:, 0:1]
            feat_others = cross_view_depth_features_scale_shift[:, 1:]
            cross_view_depth_features_scale_shift = torch.cat([feat_0, feat_others], dim=1)

        scale_shift = rearrange(F.interpolate(rearrange(self.scale_shift_predictor(cross_view_depth_features_scale_shift), 'b v (h w) c -> (b v) c h w', v = v, h =h_down, w=w_down), (h, w), mode='bilinear'), '(b v) c h w -> b v (h w) c', v = v)
        _, shift = self.get_scaleshift(scale_shift, h, w)
        depths = (depths +  shift).clamp(rearrange(context["near"], 'b v -> (b v) () () ()'), rearrange(context["far"], 'b v -> (b v) () () ()'))
        
        '''
        mono-depth-onehotencoding 
        '''
        disparity_down = F.interpolate(depths, (h//4, w//4), mode='bilinear')
        depth_hypothesis = ((1.0/ context['far'])[0,0].item()+ torch.linspace(0.0, 1.0, 128).unsqueeze(0).to(depths.device)* ((1.0/context['near'])[0,0].item() - (1.0/ context['far'])[0,0].item())).view(1, -1, 1, 1).expand(b*v, -1, h//4, w//4).detach()
        abs_depth_diff = torch.abs(disparity_down - depth_hypothesis)
        min_diff_index = torch.argmin(abs_depth_diff, dim=1, keepdim=True)
        pseudo_cost = depth_hypothesis.new_zeros(depth_hypothesis.shape)
        ones = depth_hypothesis.new_ones(depth_hypothesis.shape)
        mono_cue = rearrange(pseudo_cost.scatter_(dim = 1, index = min_diff_index, src = ones).detach(), '(b v) c h w -> (v b) c h w',b=b,v=v)
      
        '''
        unproject points to 3D using unidepth prediction
        '''
        
        grid_up, _ = sample_image_grid((h, w), device)
        grid_up = repeat(grid_up, 'h w c -> b h w c', b = b).permute(0,3,1,2) 
        xyz_up_h = unproject(repeat(grid_up.flatten(-2,-1),'b c N -> (b v) c N', v=v).permute(0,2,1), rearrange(depths.flatten(-2,-1), '(b v) 1 N -> (b v) N', b=b,v=v), rearrange(context['intrinsics'],'b v c k -> (b v) 1 c k'))
        xyz_up_h = rearrange(xyz_up_h, '(b v) (h w) c -> b v c h w', b = b, v = v, h = depths.shape[-2], w = depths.shape[-1])
        
        '''
        estimate coarse camera poses using PnP + RANSAC
        '''
        relpose = {}
        conf_relpose = {}
        conf_transformation = {}
        corr_up_list = {}
        conf_up_list = {}
        pw_relpose_list = []
        pw_relpose= {}
        pw_conf_relpose = {}
        pw_relpose_dict = {}
        pw_relpose_2d3d = {}

        index_lists = [(a, b) for a in range(v) for b in range(a + 1, v)]
        
        for i_idx, j_idx in index_lists:
            corr_up_list[(i_idx, j_idx)] = []
            conf_up_list[(i_idx, j_idx)] = []
            pw_relpose[(i_idx, j_idx)] = []
            pw_relpose_dict[(i_idx, j_idx)] = []
            pw_conf_relpose[(i_idx, j_idx)] = []
            pw_relpose_2d3d[(i_idx, j_idx)] = []

  
        for i_idx, j_idx in index_lists:
            for i in range(b):
                with torch.no_grad():
                    feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, img[i,i_idx], img[i,j_idx])
                kpts0, kpts1, matches, scores = feats0["keypoints"], feats1["keypoints"], matches01["matches"], matches01["scores"].to(device)
                
                mkpts_s, mkpts_t = kpts0[matches[..., 0]].long(), kpts1[matches[..., 1]].long()
                mkpts_s = (mkpts_s[...,1] * depths.shape[-1] + mkpts_s[...,0]).long().to(device)
                mkpts_t =  (mkpts_t[...,1] * depths.shape[-1] + mkpts_t[...,0]).long().to(device)
               
                corr_ij_up = (mkpts_s, mkpts_t, scores)
                corr_up_list[(i_idx, j_idx)].append(corr_ij_up)
                _, corr_j_up_id, _ = corr_ij_up[:3]
               
                corr_j = nn_gather(xyz_up_h[i, j_idx].flatten(-2,-1).unsqueeze(0).permute(0,2,1), corr_j_up_id.unsqueeze(0))
                try:
                    success, r_pose, t_pose, _ = cv2.solvePnPRansac(corr_j.squeeze(0).cpu().detach().numpy().astype(np.float32),\
                                                                     kpts0[matches[..., 0]].cpu().detach().numpy().astype(np.float32),\
                                                                     unnorm_intrinsics[i,i_idx].cpu().detach().numpy(), None,\
                                                                     flags=cv2.SOLVEPNP_SQPNP,iterationsCount=1000,\
                                                                    reprojectionError=5.0,confidence=0.9999
                                                                    )
                    if success:
                        r_pose = cv2.Rodrigues(r_pose)[0]  
                        RT = np.r_[np.c_[r_pose, t_pose], [(0,0,0,1)]] 
                        relpose[(i_idx, j_idx)] = torch.from_numpy(np.linalg.inv(RT)).unsqueeze(0).to(device).float()
                    else:
                        relpose[(i_idx, j_idx)] = torch.eye(4).unsqueeze(0).to(device).float()
                except:
                        relpose[(i_idx, j_idx)] = torch.eye(4).unsqueeze(0).to(device).float()
              
                conf_relpose[(i_idx, j_idx)] = scores.mean()
                pw_relpose[(i_idx, j_idx)].append(relpose[(i_idx,j_idx)].squeeze())
                pw_conf_relpose[(i_idx, j_idx)].append(conf_relpose[(i_idx,j_idx)])
                conf_up_list[(i_idx, j_idx)].append(corr_ij_up[2])
        
            conf_ij = torch.stack(pw_conf_relpose[(i_idx,j_idx)])
            if abs(i_idx - j_idx) > 1:
                conf_ij = (conf_ij - self.confidence_min).relu()
                conf_ij = conf_ij / (1 - self.confidence_min)
            conf_transformation[(i_idx, j_idx)] = conf_ij
            pw_relpose_dict[(i_idx,j_idx)] = torch.stack(pw_relpose[(i_idx,j_idx)])
            pred_Rt = pw_relpose_dict[(i_idx,j_idx)]
            pw_relpose_list.append(pred_Rt)
            pw_relpose_2d3d[(i_idx, j_idx)] = pred_Rt

        pw_relpose_list = torch.stack(pw_relpose_list, dim = 1)
        sync_abspose = camera_synchronization(pw_relpose_2d3d, conf_transformation, v)
        
        '''
        Refine the camera poses using multi-view transformer
        '''
        
        xy_ray_pose, _ = sample_image_grid((h//4, w//4), context['image'].device)
        xy_ray_pose = repeat(rearrange(xy_ray_pose, "h w xy -> (h w) () xy"), "hw () xy -> b hw xy", b=b*v)
        encoding0 = self.posenc(torch.cat((xy_ray_pose, torch.zeros_like(xy_ray_pose[:,:1])), dim=1))
     
        origins, directions = get_world_rays(rearrange(xy_ray_pose, "(b v) hw xy -> b v () hw () xy", b=b, v=v), rearrange(sync_abspose.inverse(),  "b v i j -> b v () () () i j"), rearrange(context['intrinsics'], "b v i j -> b v () () () i j"))
        plucker_enc = self.plucker_embedding(h//4, w//4, rearrange(origins,'b v 1 hw srf xyz -> (b v) hw srf xyz').squeeze(-2), rearrange(directions,'b v 1 hw srf xyz -> (b v) hw srf xyz').squeeze(-2) )

        desc0 = rearrange(self.conv_proj(torch.cat((F.interpolate(rearrange(predictions['feat'], 'b v c h w -> (b v) c h w'), (h//4, w//4)), plucker_enc), dim=1)),  'b c h w -> b (h w) c')
        desc0 = torch.cat((self.pose_cls_token.expand(desc0.shape[0], -1, -1), desc0), dim=1)
        
        '''
        self-attention with plucker embedding to provide current pose information
        '''
        
        for i in range(self.n_layers):
            desc0 = self.pose_transformers[i](desc0, encoding0)
        desc0 = rearrange(desc0[:, 1:], '(b v) L c -> b v L c', b=b, v=v)
        _, _, P, c = desc0.shape
        patch_num = int(math.sqrt(P))

        # add embedding of 2D spaces
        pos_embed = get_2d_sincos_pos_embed(
            c, grid_size=(patch_num, patch_num)
        ).permute(0, 2, 3, 1)[None]
        pos_embed = pos_embed.reshape(1, 1, patch_num * patch_num, c).to(
            depths.device
        )
        rgb_feat = desc0 + pos_embed
        # register for pose
        pose_token = self.pose_token.expand(b, v, -1, -1)
        rgb_feat = torch.cat([pose_token, rgb_feat], dim=-2)
        _, _, P, c = rgb_feat.shape

        '''
        learning pose-cls token via self- and cross-view attentions
        '''
        for idx in range(self.n_layers):
            # self attention
            rgb_feat = rearrange(rgb_feat, "b s p c -> (b s) p c", b=b, s=v)
            rgb_feat = self.pose_self_attn[idx](rgb_feat, None)
            rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=b, s=v)
            
            feat_0 = rgb_feat[:, 0]
            feat_others = rgb_feat[:, 1:]
            feat_cross = torch.stack([torch.cat((rgb_feat[:,i+1:], rgb_feat[:, :i]), dim=1) for i in range(1, v)], dim=1)
            feat_cross = rearrange(feat_cross, 'b v t p c -> (b v) (t p) c')
            # cross attention
            feat_others = rearrange(
                feat_others, 'b m p c -> (b m) p c', m=v - 1, p=P
            )
            feat_others, _ = self.pose_cross_attn[idx](feat_others, feat_cross)
            feat_others = rearrange(
                feat_others, '(b m) p c -> b m p c', m=v - 1, p=P
            )
            rgb_feat = torch.cat([rgb_feat[:, 0:1], feat_others], dim=1)
        rgb_feat = rgb_feat[:, :, 0]

        '''
        residual connection between PnP + RANSAC pose and the estimated pose from transformer
        '''
        init_pose = sync_abspose
        raw_rot = self.matrix_to_rotation_6d(init_pose[:, :, :3, :3])
        raw_trans = init_pose[:, :, :3, 3]
        pred_pose_enc = torch.cat([raw_rot, raw_trans], dim=-1) 
       
    
        pose_embed = self.embed_pose(pred_pose_enc)
        rgb_feat = rgb_feat + pose_embed
        pose_cls_token = self.pose_trunk(rgb_feat)

        # Predict the delta feat and pose encoding at each iteration
        delta = self.pose_branch(pose_cls_token)
    
        delta_pred_pose_enc = delta[..., : 9] 
        delta_feat = delta[..., 9 : -2] 
        
        rgb_feat = self.ffeat_updater(self.norm(delta_feat)) + rgb_feat
    
        pred_pose = pred_pose_enc[:, 1:] + delta_pred_pose_enc[:, 1:] * self.gamma
        pred_concat_pose = torch.cat([pred_pose_enc[:, :1], pred_pose], dim=1)
        
        rot6d = self.r6d2mat(pred_concat_pose[:, :, :6])
        trans3d = pred_concat_pose[:, :, 6:]
        Rt = torch.cat([rot6d, trans3d[..., None]], dim=-1)
        pad = torch.zeros_like(Rt[..., 2:3, :])
        pad[..., -1] = 1.0
        refine_sync_abspose = torch.cat((Rt, pad), dim=-2)
  
        
        '''
        Gaussian params prediction
        Inputs : One hot encoding from monocular depth, rgb, disparity and per_view_depth_features(self-atteneded)
        '''
        
        in_feats = per_view_depth_features
        extra_info = {}
        extra_info['images'] = rearrange(context["image"][:, (0,-1)], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        
        extra_info['disparity'] = rearrange(rearrange(disparity, '(b v) c h w -> b v c h w', b=b, v=v)[:, (0,-1)], "b v c h w -> (v b) c h w")        
        extra_info['monocular_cue'] = rearrange(rearrange(mono_cue, '(v b) c h w -> b v c h w', b=b, v=v)[:, (0,-1)], 'b v c h w -> (v b) c h w')
        gpp = self.cfg.gaussians_per_pixel
        densities, raw_gaussians = self.depth_predictor(
            in_feats[:, (0,-1)],
            context['intrinsics'][:, (0,-1)],
            refine_sync_abspose.inverse()[:, (0,-1)],
            context["near"][:, (0,-1)],
            context["far"][:, (0,-1)],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=None,
        )
        
        '''
        visualize
        '''
        depth_mono_shift_visualize = shift.clone()
        depths_visualize = predictions['depth'].clone()
        depths_refine = (rearrange(depths, '(b v) 1 h w -> b v (h w) 1', b=b, v=v).unsqueeze(-1)).clamp(rearrange(context["near"], 'b v -> b v () () ()'), rearrange(context["far"], 'b v -> b v () () ()'))
        depths_combine_visualize = depths_refine.clone()

        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
       
        '''
        update the 3D point using the updated depth
        
        '''
        
        xyz_refine = unproject(repeat(grid_up.flatten(-2,-1),'b c N -> (b v) c N', v=v).permute(0,2,1), rearrange(depths_refine.squeeze(-1), 'b v N 1 -> (b v) N', b=b,v=v), rearrange(context['intrinsics'],'b v c k -> (b v) 1 c k'))
        xyz_refine = rearrange(xyz_refine, '(b v) (h w) c -> b v c h w', b = b, v = v, h = depths.shape[-2], w = depths.shape[-1])
       
        depths_final = rearrange(depths_refine, ' b v (h w) c 1 -> (b v) c h w', b=b, v=v, h=h, w=w)
        gaussians = self.gaussian_adapter.forward(
            rearrange(refine_sync_abspose.inverse(), "b v i j -> b v () () () i j")[:, (0,-1)],
            rearrange(context['intrinsics'], "b v i j -> b v () () () i j")[:, (0,-1)],
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy")[:, (0,-1)],
            rearrange(depths_final, '(b v) 1 h w -> b v (h w) 1 1', b=b, v=v)[:, (0,-1)],
            self.map_pdf_to_opacity(densities[:, (0,-1)], global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )
        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1
        
        return (Gaussians(
            rearrange(
                gaussians.means[:, (0, -1)],
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances[:, (0, -1)],
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics[:, (0, -1)],
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities[:, (0, -1)],
                "b v r srf spp -> b (v r srf spp)",
            ),
        ),
        (pw_relpose_list, refine_sync_abspose, pw_relpose_list, sync_abspose),
        (depths_final, predictions["depth"], depths_visualize, depths_combine_visualize,depth_mono_shift_visualize),
        xyz_refine,
        (corr_up_list, conf_up_list, conf_transformation))

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None