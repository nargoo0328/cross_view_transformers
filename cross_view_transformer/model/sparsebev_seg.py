import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint as cp

from .common import Normalize
from .csrc.wrapper import msmv_sampling
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
# from .checkpoint import checkpoint as cp

import math
import spconv.pytorch as spconv
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy
from .sparsebev import sampling_4d, inverse_sigmoid
from .sparsebev import MLP as MLP_sparse
from .PointBEV_gridsample import PositionalEncodingMap, MLP
from .decoder import DecoderBlock

def pos2posemb3d(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.GELU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
    
class SparseBEVSeg(nn.Module):
    def __init__(
            self,
            backbone,
            encoder=None,
            head=None,
            neck=None,
            pos_encoder_2d=None,
            decoder=nn.Identity(),
            box_encoder=None,
            box_encoder_type='',
            threshold=0.5,
            fusion=0.0,
            sparse=False,
            pc_range=None,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone

        self.encoder = encoder
        self.head = head
        self.neck = neck
        self.pos_encoder_2d = pos_encoder_2d
        self.box_encoder = box_encoder 
        self.box_encoder_type = box_encoder_type
        self.decoder = decoder
        self.threshold = threshold
        self.fusion = fusion
        self.sparse = sparse

        self.pc_range = pc_range
        self.depth_num = 64
        self.depth_start = 1
        self.position_encoder = nn.Sequential(
                nn.Conv2d(self.depth_num * 3, 128 * 4, kernel_size=1, stride=1, padding=0),
                nn.GELU(), 
                nn.Conv2d(128 * 4, 128, kernel_size=1, stride=1, padding=0),
        )
        self.fpe = SELayer(128)
        self.positional_encoding = SinePositionalEncoding3D(128 // 2, normalize=True)
        self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(128*3//2, 128*4, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(128*4, 128, kernel_size=1, stride=1, padding=0),
        )

    def position_embeding(self, img_feats, lidar2img):
        eps = 1e-5
        
        B, N, C, H, W = img_feats[0].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * 224 / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * 480 / W

        index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
        index_1 = index + 1
        bin_size = (self.pc_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
        # else:
        #     index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
        #     bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
        #     coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        img2lidars = lidar2img.inverse() # b n 4 4

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]

        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        # coords_mask = (coords3d > 1.0) | (coords3d < 0.0) 
        # coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        # coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)
        
        return coords_position_embeding.view(B, N, 128, H, W)
    
    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = self.backbone(self.norm(image))

        if self.neck is not None:
            features = self.neck(features)
        
        if self.pos_encoder_2d is not None:
            features = self.pos_encoder_2d(features)
        
        features = [rearrange(y,'(b n) ... -> b n ...', b=b,n=n) for y in features]
        pos_embedding_2d = self.position_embeding(features, lidar2img)
        pos_embedding_2d = self.fpe(pos_embedding_2d.flatten(0,1), features[0].flatten(0,1)).view(features[0].size())

        masks = features[0].new_zeros(
            (b, n, features[0].shape[-2], features[0].shape[-1]))
        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(features[0].size())
        pos_embedding_2d = pos_embedding_2d + sin_embed

        features[0] = torch.cat((features[0], pos_embedding_2d), dim=2)
        
        x, mid_output, inter_output = self.encoder(features, lidar2img)
        x = self.decoder(x)
        output = self.head(x)
        # if len(inter_output) != 0:
        #     inter_output = [self.head(inter_output[i], aux=True) for i in range(len(inter_output))]
        #     output['aux'] = inter_output
        mid_output.update({'inter_output':inter_output})
        output['mid_output'] = mid_output#.squeeze() # squeeze group dimension

        return output
    
class SegHead(nn.Module):
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

    def _init_bev_layers(self, H=25, W=25, Z=8, num_points_in_pillar=4, mode='pillar', start_H=50, start_W=50, **kwargs):
        
        self.mode = mode
        if mode == 'pillar':
            # 3d
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
                                    ).view(num_points_in_pillar, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W
                                ).flip(0).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H
                                ).flip(0).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((ys, xs, zs), -1)
        elif mode == 'grid':
            # 2d
            xs = torch.linspace(0.5, W - 0.5, W
                                ).flip(0).view(1, W).expand(H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H
                                ).flip(0).view(H, 1).expand(H, W) / H
            ref_3d = torch.stack((ys, xs), -1)
            ref_3d = torch.cat([ref_3d, torch.zeros((H, W, 1)) + 0.5], dim=-1)

        self.register_buffer('grid', ref_3d, persistent=False) # z h w 3

        self.h = start_H
        self.w = start_W
        
        scale = H // start_H
        self.bev_query = nn.Embedding(start_H * start_W, self.embed_dims * scale) 
        # self.bev_query = nn.Embedding(200 * 200, self.embed_dims) 

    def forward(self, mlvl_feats, lidar2img):
        
        bs = lidar2img.shape[0]

        bev_query = rearrange(self.bev_query.weight, '(h w) d -> d h w', h=self.h, w=self.w)
        # bev_query = rearrange(self.bev_query.weight, '(h w) d -> d h w', h=200, w=200)
        bev_query = repeat(bev_query, '... -> b ...', b=bs)

        # bev_query = torch.zeros((bs, self.embed_dims * 4, self.h, self.w)).to(lidar2img.device)

        # bev_pos = rearrange(self.grid, 'z h w d -> d h w', h=self.h, w=self.w)
        bev_pos = repeat(self.grid, '... -> b ...', b=bs)
        
        bev, mid_output, inter_output = self.transformer(
            mlvl_feats, 
            lidar2img,
            bev_query, 
            bev_pos, 
            self.mode,
        )

        # bev = rearrange(bev, 'b (h w) d -> b d h w', h=self.h, w=self.w)
        return bev, mid_output, inter_output
    
class SegTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, embed_dims, num_points=4, num_groups=1, num_layers=[1], num_levels=4, pc_range=[], h=0, w=0, scales=[1.0], up_scales=[], return_inter=False, **kwargs):
        super().__init__()

        assert len(num_layers) == len(scales)
        self.num_layers = num_layers
        self.scale = scales
        self.num_groups = num_groups
        self.pc_range = pc_range
        # position_encoding = PositionalEncodingMap(in_c=3, out_c=128, mid_c=128 * 2)
        position_encoding = None
        position_encoding_bev = nn.Sequential(
                nn.Linear(128 * 3 // 2, 128 * 4),
                nn.GELU(),
                nn.Linear(128 * 4, 128),
        )
        self.layer = nn.ModuleList([SegTransformerDecoderLayer(int(scale) * embed_dims, num_points, num_groups, num_levels, scale, pc_range, h, w, position_encoding, position_encoding_bev) for scale in self.scale])
        # raise BaseException
        decoder_layers = list()
        for scale, up_scale in zip(self.scale, up_scales):
            decoder_layers.append(DecoderBlock(int(scale) * embed_dims, int(scale) * embed_dims // up_scale, up_scale, int(scale) * embed_dims, True))
        self.decoder = nn.ModuleList(decoder_layers)
        self.return_inter = return_inter

    def forward(self,
                mlvl_feats, 
                lidar2img,
                bev_query, 
                bev_pos,
                mode,
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
        scale = self.scale
        G = self.num_groups
        for lvl, feat in enumerate(mlvl_feats):
            GC = feat.shape[2]
            C = GC // G
            feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

            mlvl_feats[lvl] = feat.contiguous()
        
        inter_output = []
        for lid in range(len(self.num_layers)):
            bev_query, mid_output = self.layer[lid](
                mlvl_feats, 
                lidar2img,
                bev_query, 
                bev_pos, 
                scale[lid],
                mode,
            )
            if lid < len(self.decoder):
                bev_query = self.decoder[lid](bev_query, bev_query) # self.decoder[lid](bev_query, bev_query)
            
            if self.return_inter:
                inter_output.append(bev_query)

        return bev_query, mid_output, inter_output

class SegTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_points, num_groups, num_levels, scale, pc_range, h, w, position_encoding, position_encoding_bev):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = pc_range
        self.scale = scale

        self.position_encoder = position_encoding
        self.position_encoding_bev = position_encoding_bev
        # self.compressor = MLP(num_points * embed_dims, embed_dims * 4, embed_dims, 3, as_conv=True)

        self.in_conv = nn.Sequential(
            # nn.Conv2d(embed_dims, embed_dims, 5, padding=2),
            # nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 3, padding=1),
            # nn.GELU(),
            # nn.Conv2d(embed_dims * 4, embed_dims, 1),
        )

        # self.mid_conv = nn.Sequential(
        #         nn.Conv2d(num_points * 128, embed_dims * 4, 1),
        #         nn.GELU(),
        #         nn.Conv2d(embed_dims * 4, embed_dims * 4, 1),
        #         nn.GELU(),
        #         nn.Conv2d(embed_dims * 4, embed_dims, 1),
        # )

        self.cross_attn = PointCrossAttention(embed_dims, num_points)

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
                scale, 
                mode,
            ):
        """
            bev_pos: b z h w 3
            bev_query: b d h w
        """
        if scale != 1.0:
            if mode == 'grid':
                bev_pos = rearrange(bev_pos, 'b h w d -> b d h w')
                bev_pos = F.interpolate(bev_pos, scale_factor= 1 / scale, mode='bilinear')
                bev_pos = rearrange(bev_pos, 'b d h w -> b h w d')

            elif mode == 'pillar':
                b, z = bev_pos.shape[:2]
                bev_pos = rearrange(bev_pos, 'b z h w d -> (b z) d h w')
                bev_pos = F.interpolate(bev_pos, scale_factor= 1 / scale, mode='bilinear')
                bev_pos = rearrange(bev_pos, '(b z) d h w -> b z h w d', b=b, z=z)

        h, w = bev_query.shape[2:]
        # bev_pos_embed = self.position_encoder(bev_pos[:, 3]) # b h w 2 -> b h w d
        # # bev_pos_embed = self.position_encoder(bev_pos).mean(1) # b z h w d -> b h w d
        # bev_pos_embed = rearrange(bev_pos_embed, 'b h w d -> b d h w')
        # bev_query = bev_query + bev_pos_embed
        bev_query = bev_query + self.in_conv(bev_query)
        bev_query = self.norm1(bev_query)

        sampled_feat, pos_embed_2d, mid_output = self.sampling(
            bev_query,
            mlvl_feats,
            bev_pos,
            lidar2img,
            self.position_encoder,
            scale,
            mode,
        )
        # sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b (p g c) h w', h=h, w=w) # b p d h w & b d h w & b Q d b K d
        # bev_query = bev_query + self.mid_conv(sampled_feat)

        bev_pos_embed = self.position_encoding_bev(pos2posemb3d(bev_pos)) # b h w 2 -> b h w d
        bev_query = bev_query + self.cross_attn(bev_query, sampled_feat, bev_pos_embed, pos_embed_2d)
        
        # points_weight = self.points_weight(bev_query) # b p h w
        # points_weight = points_weight.softmax(1).unsqueeze(1) # b 1 p h w
        # sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b (g c) p h w', h=h, w=w) # b c p h w
        # sampled_feat = (sampled_feat * points_weight).sum(2) # b c h w
        # bev_query = bev_query + self.mid_conv(sampled_feat)

        bev_query = self.norm2(bev_query)
        bev_query = bev_query + self.out_conv(bev_query)
        bev_query = self.norm3(bev_query)

        # mid_output.update({'points_weight': points_weight})
        return bev_query, mid_output
    
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
        # self.pixel_positional_embedding = PositionalEncodingMap(in_c=2, out_c=128, mid_c=128, num_hidden_layers=0, camera_embedding=6)

        # self.sampling_offset = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dims, num_groups * num_points * 3, 1),
        # )
        self.sampling_offset = nn.Conv2d(embed_dims, num_groups * num_points * 3, 1)
        # self.z_pred = nn.Conv2d(embed_dims, 2, 1) # predict height & length
        self.scale_weights = nn.Conv2d(embed_dims, num_groups * num_points * num_levels, 1) if num_levels!= 1 else None
        self.eps = eps

    def init_weights(self):
        # original
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:2], -4.0, 4.0)
        height = torch.linspace(-0.25, 1.25, self.num_groups * self.num_points).unsqueeze(1)
        bias[:, 2:3] = height

        # height & length
        # nn.init.zeros_(self.z_pred.weight)
        # self.z_pred.bias.data = torch.tensor([0.0, 4])

        # 2d sampling
        # from torch.nn.init import constant_
        # constant_(self.sampling_offset.weight.data, 0.0)
        # thetas = torch.arange(self.num_points, dtype=torch.float32) * (
        #     2.0 * math.pi / self.num_points
        # )
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (
        #     (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        #     .view(1, self.num_points, 2)
        #     .repeat(self.num_groups, 1, 1)
        # )
        # # for i in range(self.n_points):
        # #     grid_init[:, :, i, :] *= i + 1
        # with torch.no_grad():
        #     self.sampling_offset.bias = nn.Parameter(grid_init.view(-1))

    def forward(self, query, mlvl_feats, reference_points, lidar2img, pos_encoder=None, scale=1.0, mode='grid'):

        # 2d sampling offset 
        if mode == 'grid': 
            # pred_z = self.z_pred(query)
            # pred_z = rearrange(pred_z, 'b d h w -> b (h w) 1 1 d')

            # sampling_offset = self.sampling_offset(query).sigmoid() # b (g p 3) h w
            # sampling_offset = rearrange(sampling_offset, 'b (g p d) h w -> b (h w) g p d',
            #                     g=self.num_groups,
            #                     p=self.num_points,
            #                     d=3
            #                 )
            # sampling_offset_new = sampling_offset.clone()
            # sampling_offset_new[..., :2] = (sampling_offset_new[..., :2] * (0.25 * scale + self.eps) * 2) \
            #                             - (0.25 * scale + self.eps)
            # z = (sampling_offset_new[..., 2:3] * (1.0 + self.eps) * 2) \
            #                             - (1.0 + self.eps) # -1 ~ 1
            # sampling_offset_new[..., 2:3] = pred_z[..., 0:1] + pred_z[..., 1:2] * z
            # sampling_offset = sampling_offset_new

            # reference_points = rearrange(reference_points, 'b h w d -> b (h w) 1 1 d', d=3).clone()
            # reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            # reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            # reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            # reference_points = reference_points + sampling_offset

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

            # if reference_points.shape[1] == 40000:
            #     y = 135
            #     x = 55
            #     index = x + y * 200
            #     print("3d coord", reference_points[0, index])

            #     x = x + 2
            #     index = x + y * 200
            #     print("3d coord", reference_points[0, index])

            #     x = x + 2
            #     index = x + y * 200
            #     print("3d coord", reference_points[0, index])
            
        # 3d sampling offset 
        elif mode == 'pillar':

            sampling_offset = self.sampling_offset(query) # b (g p 3) h w
            sampling_offset = rearrange(sampling_offset, 'b (g p d) h w -> b (h w) g p d',
                                g=self.num_groups,
                                p=self.num_points,
                                d=2
                            )
            # sampling_offset[..., :2] = (sampling_offset[..., :2] * (0.25 * scale + self.eps) * 2) \
            #                             - (0.25 * scale + self.eps)
            # sampling_offset[..., 2:3] = (sampling_offset[..., 2:3] * (0.5 + self.eps) * 2) \
            #                             - (0.5 + self.eps)
            
            reference_points = rearrange(reference_points, 'b p1 h w d -> b (h w) 1 p1 1 d').clone()

            reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # reference_points = reference_points + sampling_offset
            reference_points = rearrange(reference_points, 'b q g p1 p2 d -> b q g (p1 p2) d')

        if self.scale_weights is not None:
            # scale weights
            scale_weights = rearrange(self.scale_weights(query), 'b (g p l) h w -> b (h w) g 1 p l', p=self.num_points, g=self.num_groups, l=self.num_levels) # b q g 1 p l
            scale_weights = scale_weights.softmax(-1) 
        else:
            # no scale     
            scale_weights = None  

        # sampling
        sampled_feats, sample_points_cam = sampling_4d(
            reference_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            self.img_h, self.img_w,
            # pixel_positional_embedding=self.pixel_positional_embedding
            # sampling_offset,
        )  # [B, Q, G, P, C]
        sampled_feats, pos_embed_2d = torch.split(sampled_feats, 128, dim=-1)

        mid_output = {}
        # mid_output.update({'reference_points': reference_points})
        if pos_encoder is not None:
            # normalized back
            reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2]) 
            sampled_feats = sampled_feats + pos_encoder(reference_points)
        
        # mid_output.update({'sample_points_cam': sample_points_cam})
        return sampled_feats, pos_embed_2d, mid_output #sampling_offset[..., -1]
    
class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, out_dim, n_groups=4, n_points=8):
        super(AdaptiveMixing, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_groups = n_groups

        self.parameter_generator = nn.Conv2d(in_dim, n_groups * ((in_dim // n_groups) ** 2), 1)
        self.layer_norm = nn.LayerNorm(in_dim // n_groups)
        self.act = nn.ReLU(inplace=True)
        self.out_proj = nn.Linear(in_dim // n_groups * n_points, out_dim // n_groups)

    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def forward(self, bev_query, pts):
        """
        bev_query: b c h w
        pts: b (h w) p c
        """
        pts = pts.squeeze(2)
        b, _, h, w = bev_query.shape
        param = self.parameter_generator(bev_query) # b (gc gc) h w
        param = rearrange(param, 'b (g c1 c2) h w -> (b g) (h w) c1 c2', g=self.n_groups, c1=self.in_dim // self.n_groups, c2=self.in_dim // self.n_groups)

        pts = rearrange(pts, 'b q p (g c) -> (b g) q p c', g=self.n_groups)
        pts = torch.matmul(pts, param)
        pts = self.act(self.layer_norm(pts))

        pts = rearrange(pts, 'b q p c -> b q (p c)')
        pts = self.out_proj(pts)
        pts = rearrange(pts, '(b g) (h w) c -> b (g c) h w',h=h, w=w, b=b, g=self.n_groups)

        return pts
    
class PixelPositionalEncoding(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        dim: int,
        image_height: int,
        image_width: int,
    ):
        super().__init__()

        # 1 1 3 h w
        xs = torch.linspace(0, 1, feat_width)
        ys = torch.linspace(0, 1, feat_height)
        image_plane = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
        image_plane = F.pad(image_plane, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w

        image_plane[0] *= image_width
        image_plane[1] *= image_height
        image_plane = image_plane.view(1, 1, 3, feat_height, feat_width) # 1 1 3 h w
        self.register_buffer('image_plane', image_plane, persistent=False)

        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

    def forward(
        self,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w
        return img_embed
    
class SinePositionalEncoding3D(nn.Module):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            n_embed = (n_embed + self.offset) / \
                      (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, :, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_n = n_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, N, H, W = mask.size()
        pos_n = torch.stack(
            (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
        return pos
    
class PointCrossAttention(nn.Module):
    def __init__(self, embed_dims, num_points, num_heads=4, features_dim=128):
        super().__init__()

        self.q_proj = nn.Linear(embed_dims, features_dim)
        self.to_q = nn.Linear(features_dim, features_dim)

        self.k_proj = nn.Linear(features_dim, features_dim)
        self.to_k = nn.Linear(features_dim, features_dim)

        self.heads = num_heads
        self.dim_head = features_dim // num_heads
        self.scale = self.dim_head ** -0.5

        self.out_conv = nn.Sequential(
                nn.Conv2d(num_points * features_dim, embed_dims * 4, 1),
                nn.GELU(),
                nn.Conv2d(embed_dims * 4, embed_dims * 4, 1),
                nn.GELU(),
                nn.Conv2d(embed_dims * 4, embed_dims, 1),
        )

    def forward(self, bev_query, sampled_feat, bev_pos_embed, pos_embed_2d):
        
        b, d, h, w = bev_query.shape
        query = rearrange(bev_query, 'b d h w -> (b h w) 1 d')
        bev_pos_embed = rearrange(bev_pos_embed, 'b h w d -> (b h w) 1 d')
        key = rearrange(sampled_feat, 'b q g p d -> (b q g) p d')
        pos_embed_2d = rearrange(pos_embed_2d, 'b q g p d -> (b q g) p d')

        q = self.to_q(self.q_proj(query) + bev_pos_embed)
        k = self.to_k(self.k_proj(key) + pos_embed_2d)

        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # (b h w m) 1 d
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head) # (b h w m) p d

        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k) # (b h w m) 1 p
        att = dot.softmax(dim=-1)
        # if h == 200:
        #     att_ = rearrange(att, '(b h w m) 1 p -> b h w m 1 p', h=h, w=w, b=b, m=self.heads)
        #     y = 135
        #     x = 55
        #     print("attention score",att_[0, y, x])

        #     x = x+2
        #     print("attention score",att_[0, y, x])

        #     x = x+2
        #     print("attention score",att_[0, y, x])

        att = att.squeeze(1).unsqueeze(-1) # (b h w m) p 1

        sampled_feat = rearrange(sampled_feat, 'b q g p (m d) -> (b q g m) p d', m=self.heads, d=self.dim_head)
        sampled_feat = att * sampled_feat # (b h w m) p 1 * (b h w m) p d
        sampled_feat = rearrange(sampled_feat, '(b h w m) p d -> b (p m d) h w', h=h, w=w, b=b, m=self.heads)
        sampled_feat = self.out_conv(sampled_feat)
        
        return sampled_feat