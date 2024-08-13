import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint as cp

from .common import Normalize
from einops import rearrange, repeat
from .sparsebev import sampling_4d
from .PointBEV_gridsample import PositionalEncodingMap
from .decoder import DecoderBlock
from .simple_bev import SimpleBEVDecoderLayer

class SparseBEVSeg(nn.Module):
    def __init__(
            self,
            backbone,
            encoder=None,
            head=None,
            neck=None,
            decoder=nn.Identity(),
            box_encoder=None,
            scale: float = 1.0,
            box_encoder_type='',
            threshold=0.5,
            fusion=0.0,
            sparse=False,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        self.encoder = encoder
        self.head = head
        self.neck = neck
        self.box_encoder = box_encoder 
        self.box_encoder_type = box_encoder_type
        self.decoder = decoder
        self.threshold = threshold
        self.fusion = fusion
        self.sparse = sparse

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = [rearrange(self.down(y),'(b n) ... -> b n ...', b=b,n=n) for y in self.backbone(self.norm(image))] 

        if self.neck is not None:
            features = self.neck(features)
            features = [rearrange(y,'(b n) ... -> b n ...', b=b,n=n) for y in features]
        
        x = self.encoder(features, lidar2img)
        x = self.decoder(x)
        output = self.head(x)
        # output['height'] = height.squeeze(1) # squeeze group dimension

        return output
    
class TwoStageHead(nn.Module):
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

    def _init_bev_layers(self, H=25, W=25, Z=8, num_points_in_pillar=4, mode='pillar', **kwargs):
        
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
                                ).view(num_points_in_pillar, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W
                            ).flip(0).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H
                            ).flip(0).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((ys, xs, zs), -1)

        self.register_buffer('grid', ref_3d, persistent=False) # z h w 3

    def forward(self, mlvl_feats, lidar2img):
        
        bs = lidar2img.shape[0]
        bev_pos = repeat(self.grid, '... -> b ...', b=bs)
        
        bev, height = self.transformer(
            mlvl_feats, 
            lidar2img,
            bev_pos, 
        )

        # bev = rearrange(bev, 'b (h w) d -> b d h w', h=self.h, w=self.w)
        return bev, height
    
class TwoStageDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, embed_dims, num_points=4, num_groups=1, num_levels=4, pc_range=[], h=0, w=0, up_scale=2, **kwargs):
        super().__init__()

        self.num_groups = num_groups
        self.pc_range = pc_range
        position_encoding = PositionalEncodingMap(in_c=3, out_c=128, mid_c=128 * 2)
        self.stage1 = SimpleBEVDecoderLayer(4 * embed_dims, num_points, num_groups, num_levels, pc_range, h, w, position_encoding)
        self.stage2 = TwoStageDecoderLayer(embed_dims, num_points, num_groups, num_levels, pc_range, h, w, position_encoding)
        self.decoder = DecoderBlock(4 * embed_dims, 4 * embed_dims // up_scale, up_scale, 4 * embed_dims, True)

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
        
        bev_feats = self.stage1(
                mlvl_feats, 
                lidar2img,
                bev_pos, 
        )
        bev_feats = self.decoder(bev_feats, bev_feats)
        bev_feats, height = self.stage2(
                mlvl_feats, 
                lidar2img,
                bev_feats,
                bev_pos, 
        )

        return bev_feats, height

class TwoStageDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_points, num_groups, num_levels, pc_range, h, w, position_encoding):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = pc_range

        self.position_encoder = position_encoding
        # self.compressor = MLP(num_points * embed_dims, embed_dims * 4, embed_dims, 3, as_conv=True)

        self.in_conv = nn.Sequential(
            # nn.Conv2d(embed_dims, embed_dims, 5, padding=2),
            # nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 3, padding=1),
            # nn.GELU(),
            # nn.Conv2d(embed_dims * 4, embed_dims, 1),
        )
        self.mid_conv = nn.Sequential(
            nn.Conv2d(num_points * 128, embed_dims * 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims * 4, embed_dims, 1),
        )
        self.out_conv = nn.Sequential(
            # nn.Conv2d(embed_dims, embed_dims, 1),
            # nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 3, padding=1),
            # nn.GELU(),
            # nn.Conv2d(embed_dims, embed_dims, 1),
        )
        self.sampling = SegSampling(embed_dims, num_groups=num_groups, num_points=num_points, num_levels=num_levels, pc_range=pc_range, h=h, w=w)
        # self.ffn = MLP_sparse(embed_dims, embed_dims, embed_dims, 2)

        self.norm1 = nn.InstanceNorm2d(embed_dims)
        self.norm2 = nn.InstanceNorm2d(embed_dims)
        self.norm3 = nn.InstanceNorm2d(embed_dims)
        
        self.init_weights()

    def init_weights(self):
        self.sampling.init_weights()

    def forward(self,
                mlvl_feats, 
                lidar2img,
                bev_query, 
                bev_pos,
            ):
        """
            bev_pos: b z h w 3
            bev_query: b d h w
        """
        # if scale != 1.0:
        #     if mode == 'grid':
        #         bev_pos = rearrange(bev_pos, 'b h w d -> b d h w')
        #         bev_pos = F.interpolate(bev_pos, scale_factor= 1 / scale, mode='bilinear')
        #         bev_pos = rearrange(bev_pos, 'b d h w -> b h w d')
        #     elif mode == 'pillar':
        #         b, z = bev_pos.shape[:2]
        #         bev_pos = rearrange(bev_pos, 'b z h w d -> (b z) d h w')
        #         bev_pos = F.interpolate(bev_pos, scale_factor= 1 / scale, mode='bilinear')
        #         bev_pos = rearrange(bev_pos, '(b z) d h w -> b z h w d', b=b, z=z)
        bev_pos = bev_pos.mean(1)
        print(bev_pos.shape)
        h, w = bev_query.shape[2:]
        # bev_pos_embed = self.position_encoder(bev_pos[:, 3]) # b h w 2 -> b h w d
        # # bev_pos_embed = self.position_encoder(bev_pos).mean(1) # b z h w d -> b h w d
        # bev_pos_embed = rearrange(bev_pos_embed, 'b h w d -> b d h w')
        # bev_query = bev_query + bev_pos_embed
        bev_query = bev_query + self.in_conv(bev_query)
        bev_query = self.norm1(bev_query)

        sampled_feat, height = self.sampling(
            bev_query,
            mlvl_feats,
            bev_pos,
            lidar2img,
            self.position_encoder,
            1.0,
            'grid',
        )

        # sampled_feat = rearrange(sampled_feat, 'b q g p c -> b (p g c) q 1')
        sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b (p g c) h w', h=h, w=w)
        bev_query = bev_query + self.mid_conv(sampled_feat)
        bev_query = self.norm2(bev_query)
        bev_query = bev_query + self.out_conv(bev_query)
        bev_query = self.norm3(bev_query)

        height = rearrange(height, 'b (h w) g p -> b g p h w', h=h, w=w)
        return bev_query, height
    
class SegSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims, num_groups=1, num_points=8, num_levels=1, pc_range=[], h=0, w=0, eps=1e-6):
        super().__init__()

        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.img_h = h
        self.img_w = w
        self.pc_range = pc_range

        self.sampling_offset = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dims, num_groups * num_points * 3, 1),
        )
        self.scale_weights = nn.Conv2d(embed_dims, num_groups * num_points * num_levels, 1) if num_levels!= 1 else None
        self.eps = eps

    def init_weights(self):
        bias = self.sampling_offset[-1].bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset[-1].weight)
        nn.init.uniform_(bias[:, 0:3], -4.0, 4.0)

    def forward(self, query, mlvl_feats, reference_points, lidar2img, pos_encoder=None, scale=1.0, mode='grid'):

        # 2d sampling offset 
        if mode == 'grid': 
            sampling_offset = self.sampling_offset(query).sigmoid() # b (g p 3) h w
            sampling_offset = rearrange(sampling_offset, 'b (g p d) h w -> b (h w) g p d',
                                g=self.num_groups,
                                p=self.num_points,
                                d=3
                            ).clone()
            sampling_offset[..., :2] = (sampling_offset[..., :2] * (0.25 * scale + self.eps) * 2) \
                                        - (0.25 * scale + self.eps)
            sampling_offset[..., 2:3] = (sampling_offset[..., 2:3] * (4.0 + self.eps) * 2) \
                                        - (4.0 + self.eps)
            # sampling_offset = (sampling_offset * (0.25 * scale + self.eps) * 2) \
            #                             - (0.25 * scale + self.eps)
            reference_points = rearrange(reference_points, 'b h w d -> b (h w) 1 1 d', d=3).clone()

            reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            reference_points = reference_points + sampling_offset
        # 3d sampling offset 
        elif mode == 'pillar':
            num_points_pillar = reference_points.shape[1]
            sampling_offset = self.sampling_offset(query).sigmoid() # b (g p 3) h w
            sampling_offset = rearrange(sampling_offset, 'b (g p1 p2 d) h w -> b (h w) g p1 p2 d',
                                g=self.num_groups,
                                p1=num_points_pillar,
                                p2=self.num_points//num_points_pillar,
                                d=3
                            ).clone()
            sampling_offset[..., :2] = (sampling_offset[..., :2] * (0.25 * scale + self.eps) * 2) \
                                        - (0.25 * scale + self.eps)
            sampling_offset[..., 2:3] = (sampling_offset[..., 2:3] * (0.5 + self.eps) * 2) \
                                        - (0.5 + self.eps)
            
            reference_points = rearrange(reference_points, 'b p1 h w d -> b (h w) 1 p1 1 d').clone()

            reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            reference_points = reference_points + sampling_offset
            reference_points = rearrange(reference_points, 'b q g p1 p2 d -> b q g (p1 p2) d')

        if self.scale_weights is not None:
            # scale weights
            scale_weights = rearrange(self.scale_weights(query), 'b (g p l) h w -> b (h w) g 1 p l', p=self.num_points, g=self.num_groups, l=self.num_levels) # b q g 1 p l
            scale_weights = scale_weights.softmax(-1) 
        else:
            # no scale     
            scale_weights = None  
        # sampling
        sampled_feats = sampling_4d(
            reference_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            self.img_h, self.img_w
        )  # [B, Q, G, P, C]
        if pos_encoder is not None:
            # normalized back
            reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2]) 
            sampled_feats = sampled_feats + pos_encoder(reference_points)

        return sampled_feats, sampling_offset[..., -1]