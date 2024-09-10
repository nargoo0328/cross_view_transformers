import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .sparsebev import sampling_4d
from .PointBEV_gridsample import PositionalEncodingMap

class SimpleBEVHead(nn.Module):
    """
    Predict bbox -> project to BEV -> sparse_conv2d/conv2d
    """
    def __init__(self,
                transformer=None,
                embed_dims=128,
                **kwargs):
        
        super().__init__()

        self.transformer = transformer
        self.embed_dims = embed_dims
        
        self._init_bev_layers(**kwargs)

    def _init_bev_layers(self, H=25, W=25, Z=8, num_points_in_pillar=4, **kwargs):
        
        # 3d
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
                                ).view(num_points_in_pillar, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W
                            ).flip(0).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H
                            ).flip(0).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((ys, xs, zs), -1)

        self.register_buffer('grid', ref_3d, persistent=False) # z h w 3
        self.h = H
        self.w = W

    def forward(self, mlvl_feats, lidar2img):
        
        bs = lidar2img.shape[0]
        # bev_pos = rearrange(self.grid, 'z h w d -> d h w', h=self.h, w=self.w)
        bev_pos = repeat(self.grid, '... -> b ...', b=bs)
        
        bev = self.transformer(
            mlvl_feats, 
            lidar2img,
            bev_pos, 
        )

        # bev = rearrange(bev, 'b (h w) d -> b d h w', h=self.h, w=self.w)
        return bev
    
class SimpleBEVDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, embed_dims, num_points=4, num_groups=1, pc_range=[], h=0, w=0, **kwargs):
        super().__init__()

        self.num_groups = num_groups
        self.layer = SimpleBEVDecoderLayer(embed_dims, num_points, pc_range, h, w)
        
    def forward(self,
                mlvl_feats, 
                lidar2img,
                bev_pos,
                ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        G = self.num_groups
        for lvl, feat in enumerate(mlvl_feats):
            GC = feat.shape[2]
            C = GC // G
            feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

            mlvl_feats[lvl] = feat.contiguous()
        
        bev_feats = self.layer(
                mlvl_feats, 
                lidar2img,
                bev_pos, 
        )

        return bev_feats, None, None

class SimpleBEVDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_points, pc_range, h, w, position_encoder=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = pc_range

        self.position_encoder = position_encoder if position_encoder is not None else PositionalEncodingMap(in_c=3, out_c=128, mid_c=128 * 2)
        self.mid_conv = nn.Sequential(
            nn.Conv2d(num_points * 128, embed_dims * 4, 1),
            nn.InstanceNorm2d(embed_dims * 4),
            nn.GELU(),
            nn.Conv2d(embed_dims * 4, embed_dims, 1),
            nn.InstanceNorm2d(embed_dims),
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 1),
        )
        self.sampling = SimpleBEVSampling(pc_range=pc_range, h=h, w=w)

    def forward(self,
                mlvl_feats, 
                lidar2img,
                bev_pos,
                scale=1.0
            ):
        if scale != 1.0:
            b, z = bev_pos.shape[:2]
            bev_pos = rearrange(bev_pos, 'b z h w d -> (b z) d h w')
            bev_pos = F.interpolate(bev_pos, scale_factor= 1 / scale, mode='bilinear')
            bev_pos = rearrange(bev_pos, '(b z) d h w -> b z h w d', b=b, z=z)

        h, w = bev_pos.shape[2:4]
        sampled_feat = self.sampling(
            mlvl_feats,
            bev_pos,
            lidar2img,
            self.position_encoder,
        )

        # sampled_feat = rearrange(sampled_feat, 'b q g p c -> b (p g c) q 1')
        sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b (p g c) h w', h=h, w=w)
        sampled_feat = self.mid_conv(sampled_feat)

        return sampled_feat
    
class SimpleBEVSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, pc_range=[], h=0, w=0):
        super().__init__()

        self.img_h = h
        self.img_w = w
        self.pc_range = pc_range

    def forward(self, mlvl_feats, reference_points, lidar2img, pos_encoder=None):

        reference_points = rearrange(reference_points, 'b p h w d -> b (h w) 1 p d').clone()

        reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        sampled_feats = sampling_4d(
            reference_points,
            mlvl_feats,
            None,
            lidar2img,
            self.img_h, self.img_w
        )  # [B, Q, G, P, C]

        if pos_encoder is not None:
            # normalized back
            reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2]) 
            sampled_feats = sampled_feats + pos_encoder(reference_points)

        return sampled_feats
    
class SimpleBEVDecoderLayer_pixel(nn.Module):
    def __init__(self, embed_dims, num_points, pc_range, h, w, position_encoder=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = pc_range

        self.position_encoder = position_encoder
        # self.mid_conv = nn.Sequential(
        #     nn.Conv2d(num_points * 128, embed_dims * 4, 1),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dims * 4, embed_dims, 1),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dims, embed_dims, 1),
        # )
        self.mid_conv = nn.Sequential(
                nn.Conv2d(128, 128 * 4, 3, padding=1, bias=False),
                nn.InstanceNorm2d(128 * 4),
                nn.Conv2d(128 * 4, 128 * 4, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(128 * 4, embed_dims, 3, padding=1, bias=False),
        )
        self.sampling = SimpleBEVSampling_pixel(pc_range=pc_range, h=h, w=w)
        height = torch.linspace(0, 1.0, num_points)
        self.register_buffer('height', height, persistent=False) # p

    def forward(self,
                mlvl_feats, 
                lidar2img,
                bev_pos,
                scale=1.0
            ):

        if scale != 1.0:
            # b, z = bev_pos.shape[:2]
            # bev_pos = rearrange(bev_pos, 'b z h w d -> (b z) d h w')
            # bev_pos = F.interpolate(bev_pos, scale_factor= 1 / scale, mode='bilinear')
            # bev_pos = rearrange(bev_pos, '(b z) d h w -> b z h w d', b=b, z=z)

            bev_pos = rearrange(bev_pos, 'b h w d -> b d h w')
            bev_pos = F.interpolate(bev_pos, scale_factor= 1 / scale, mode='bilinear')
            bev_pos = rearrange(bev_pos, 'b d h w -> b h w d')
            bev_pos = repeat(bev_pos, 'b ... -> b p ...', p=self.num_points) # b p h w d
        
        b, _, h, w = bev_pos.shape[:4]
        bev_pos[..., 2:] = self.height.view(1,-1, 1, 1, 1).expand(b, -1, h, w, 1).to(bev_pos.device)

        sampled_feat, mid_output = self.sampling(
            mlvl_feats,
            bev_pos,
            lidar2img,
            self.position_encoder,
        )

        sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b p (g c) h w', h=h, w=w, g=1)
        sampled_feat = sampled_feat.sum(1)
        # sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b (p g c) h w', h=h, w=w) # b p d h w & b d h w & b Q d b K d
        sampled_feat = self.mid_conv(sampled_feat)

        return sampled_feat, mid_output
    
class SimpleBEVSampling_pixel(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, pc_range=[], h=0, w=0):
        super().__init__()

        self.img_h = h
        self.img_w = w
        self.pc_range = pc_range
        self.alpha = 0.075

    def forward(self, mlvl_feats, reference_points, lidar2img, pos_encoder=None):

        reference_points = rearrange(reference_points, 'b p h w d -> b (h w) 1 p d').clone()

        reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        sampled_feats, sample_points_cam = sampling_4d(
            reference_points,
            mlvl_feats,
            None,
            lidar2img,
            self.img_h, self.img_w
        )  # [B, Q, G, P, C]

        if sampled_feats.shape[-1] != 128:
            sampled_feats, pos_3d = torch.split(sampled_feats, 128, dim=-1)
            # sampled_feats = self.projection(sampled_feats)
            norm = torch.norm((reference_points - pos_3d), dim=-1)
            weight = torch.exp(-self.alpha * norm**2).unsqueeze(-1)
            # weight = torch.where(weight < self.threshold, torch.tensor(0.0, device=weight.device), weight)
            sampled_feats = sampled_feats * weight

        mid_output = {}
        # mid_output.update({'sample_points_cam': sample_points_cam, 'reference_points': reference_points})
        mid_output.update({'sample_points_cam': sample_points_cam, 'pos_3d': pos_3d, 'reference_points': reference_points, 'weight': weight})
        if pos_encoder is not None:
            # normalized back
            # reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            # reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            # reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2]) 
            # sampled_feats = sampled_feats + pos_encoder(reference_points)

            pos_3d_normalized = pos_3d.clone()
            pos_3d_normalized[..., 0:1] = (pos_3d_normalized[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            pos_3d_normalized[..., 1:2] = (pos_3d_normalized[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            pos_3d_normalized[..., 2:3] = (pos_3d_normalized[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2]) 
            pos_3d_normalized = torch.clamp(pos_3d_normalized, min=0.0, max=1.0)
            
            sampled_feats = sampled_feats + pos_encoder(pos_3d_normalized)

        return sampled_feats, mid_output