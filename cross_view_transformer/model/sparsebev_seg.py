import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint as cp

from .common import Normalize
from .csrc.wrapper import msmv_sampling
from .layers import LayerNorm2d
from einops import rearrange, repeat
import copy
# from .checkpoint import checkpoint as cp

import math
import spconv.pytorch as spconv
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy
from .sparsebev import sampling_4d, inverse_sigmoid
from .sparsebev import MLP as MLP_sparse
from .PointBEV_gridsample import PositionalEncodingMap, MLP
from .decoder import BEVDecoder, DecoderBlock
# from .simple_bev import SimpleBEVDecoderLayer_pixel

class DepthAdjustment(nn.Module):
    def __init__(self, num_pieces=8, epsilon=1e-3):
        super().__init__()
        
        # Number of pieces (intervals)
        self.num_pieces = num_pieces
        self.epsilon = epsilon
        
        # Define learnable parameters for each piece
        self.scales = nn.Parameter(torch.linspace(1.0, 0.6, num_pieces))   # Learnable scale for each interval
        self.shifts = nn.Parameter(torch.zeros(num_pieces))  # Learnable shift for each interval
    
    def forward(self, depth_pred):
        # Define piece boundaries (you can adjust these based on your problem)
        boundaries = torch.linspace(1, 80, self.num_pieces + 1)
        
        # Piecewise transformation
        depth_transformed = torch.zeros_like(depth_pred)
        
        for i in range(self.num_pieces):
            # Apply different transformations based on the interval
            lower_bound = boundaries[i]
            upper_bound = boundaries[i + 1]
            
            # Different transformation for each piece (logarithmic, inverse, etc.)
            # if i == 0:
            #     # For small depths, use a linear transformation
            #     piecewise_value = self.scales[i] * depth_pred + self.shifts[i]
            # elif i == 1:
            #     # For moderate depths, use a logarithmic transformation
            #     piecewise_value = self.scales[i] * torch.log(depth_pred + self.epsilon) + self.shifts[i]
            # else:
            #     # For large depths, use an inverse transformation
            #     piecewise_value = self.scales[i] * (1 / (depth_pred + self.epsilon)) + self.shifts[i]
            
            piecewise_value = self.scales[i] * depth_pred + self.shifts[i]
            # Apply the transformation only within the current piece
            depth_transformed += torch.where((depth_pred >= lower_bound) & (depth_pred < upper_bound), piecewise_value, torch.zeros_like(depth_pred))
        
        return depth_transformed

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
            aux=False,
            input_depth=False,
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
        self.aux = aux
        self.input_depth = input_depth

        self.pc_range = pc_range
        self.depth_num = 64
        self.depth_start = 1
        # self.position_encoder = nn.Sequential(
        #         nn.Conv2d(self.depth_num * 3, 128 * 4, kernel_size=1, stride=1, padding=0),
        #         nn.GELU(), 
        #         nn.Conv2d(128 * 4, 128, kernel_size=1, stride=1, padding=0),
        # )
        # self.fpe = SELayer(128)
        # self.positional_encoding = SinePositionalEncoding3D(128 // 2, normalize=True)
        # self.adapt_pos3d = nn.Sequential(
        #         nn.Conv2d(128*3//2, 128*4, kernel_size=1, stride=1, padding=0),
        #         nn.GELU(),
        #         nn.Conv2d(128*4, 128, kernel_size=1, stride=1, padding=0),
        # )
        # self.depth = nn.Conv2d(128, self.depth_num, 1)
        self.LID = True
        # self.scale = nn.Parameter(torch.tensor([1.0]))
        # self.shift = nn.Parameter(torch.tensor([0.0]))
        if self.input_depth:
            self.depth_adjust = nn.Identity()# DepthAdjustment()

    def get_pixel_coords_3d(self, depth, lidar2img):
        eps = 1e-5
        
        B, N, C, H, W = depth.shape
        scale = 224 // H
        # coords_h = torch.arange(H, device=depth.device).float() * 224 / H
        # coords_w = torch.arange(W, device=depth.device).float() * 480 / W
        coords_h = torch.linspace(scale // 2, 224 - scale//2, H, device=depth.device).float()
        coords_w = torch.linspace(scale // 2, 480 - scale//2, W, device=depth.device).float()

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=depth.device).float()
            index_1 = index + 1
            bin_size = (self.pc_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=depth.device).float()
            bin_size = (self.pc_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        img2lidars = lidar2img.inverse() # b n 4 4

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H D 3

        return coords3d, coords_d
    
    def pred_depth(self, lidar2img, depth):
        b, n, c, h, w = depth.shape
        coords_3d, coords_d = self.get_pixel_coords_3d(depth, lidar2img) # b n w h d 3

        depth = rearrange(depth, 'b n ... -> (b n) ...')
        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_depth = (depth_prob * coords_d.view(1, self.depth_num, 1, 1)).sum(1)

        coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
        coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3
        coords_3d = rearrange(coords_3d, '(b n) h w d-> b n d h w', b=b, n=n)

        uncertainty_map = torch.sqrt((depth_prob * ((coords_d.view(1, -1, 1, 1) - pred_depth.unsqueeze(1).repeat(1, self.depth_num, 1, 1))**2)).sum(1))
        uncertainty_map = rearrange(uncertainty_map, '(b n) h w -> b n h w',b=b, n=n)

        pred_depth = rearrange(pred_depth, '(b n) h w -> b n h w',b=b, n=n)

        return coords_3d, pred_depth, uncertainty_map
    
    def get_pixel_depth(self, depth, img_feats, lidar2img, pred_depth):
        eps = 1e-6

        B, N, C, H, W = img_feats[0].shape
        scale = 224 // H
        # B, N, C, H, W = depth.shape

        # depth = self.scale * depth + self.shift
        depth = self.depth_adjust(depth)
        depth = depth.flatten(0,1)
        depth = F.interpolate(depth, size=[H,W], mode='bilinear')
        depth = rearrange(depth, '(b n) c h w -> b n w h c', b=B, n=N)
        pred_depth = rearrange(pred_depth, '(b n) c h w -> b n w h c', b=B, n=N)
        # depth = (depth / 61.2 + pred_depth) * 61.2
        depth = depth + pred_depth

        # coords_h = torch.arange(H, device=img_feats[0].device).float() * 224 / H
        # coords_w = torch.arange(W, device=img_feats[0].device).float() * 480 / W
        coords_h = torch.linspace(scale // 2, 224 - scale//2, H, device=img_feats[0].device).float()
        coords_w = torch.linspace(scale // 2, 480 - scale//2, W, device=img_feats[0].device).float()
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0)[None,None] # W, H, 2
        coords = coords.expand(B, N, W, H, 2)
        # coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        # coords = torch.cat((coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps), coords[..., 2:]), dim=-1)
        coords = coords * torch.maximum(depth, torch.ones_like(depth)*eps)
        coords = torch.cat((coords, depth), dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), dim=-1)
        
        img2lidars = lidar2img.inverse() # b n 4 4
        coords = coords.unsqueeze(-1)
        # coords = coords.view(1, 1, W, H, 4, 1).repeat(B, N, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 4, 4).repeat(1, 1, W, H, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H 3
        return coords3d.permute(0,1,4,3,2)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = self.backbone(self.norm(image))
        # import copy
        # features_ = copy.deepcopy(features)
        # features = features[2:]
        if self.neck is not None:
            # features, depth = self.neck(features, batch['depth'])
            features, depth = self.neck(features)
        
        if self.pos_encoder_2d is not None:
            features = self.pos_encoder_2d(features)
        
        features = [rearrange(y,'(b n) ... -> b n ...', b=b,n=n) for y in features]

        if depth is not None:
            depth = rearrange(depth,'(b n) ... -> b n ...', b=b,n=n)
            pred_coords_3d, pred_depth, uncertainty_map = self.pred_depth(lidar2img, depth)
            # features.append(pred_coords_3d)
            features[0] = torch.cat((features[0], pred_coords_3d), dim=2)
            features[0] = torch.cat((features[0], uncertainty_map.unsqueeze(2)), dim=2)

        if self.input_depth:
            gt_depth = self.get_pixel_depth(batch['depth'], features, lidar2img, depth)
            features[0] = torch.cat((features[0], gt_depth), dim=2)
            # features.append(gt_depth)
        # pos_embedding_2d = self.fpe(pos_embedding_2d.flatten(0,1), features[0].flatten(0,1)).view(features[0].size())

        # masks = features[0].new_zeros(
        #     (b, n, features[0].shape[-2], features[0].shape[-1]))
        # sin_embed = self.positional_encoding(masks)
        # sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(features[0].size())
        # pos_embedding_2d = pos_embedding_2d + sin_embed

        # features[0] = torch.cat((features[0], pred_depth), dim=2)
        
        x, mid_output, inter_output = self.encoder(features, lidar2img)
        x = self.decoder(x)
        output = self.head(x)

        if self.aux:
            inter_output_pred = [self.head(inter_output[i], aux=True) for i in range(len(inter_output))]
            output['aux'] = inter_output_pred

        mid_output['inter_output'] = inter_output
        mid_output['uncertainty_map'] = uncertainty_map
        output['depth'] = pred_depth
        # mid_output['features'] = features_

        output['mid_output'] = mid_output #.squeeze() # squeeze group dimension

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
        # self.bev_query = nn.Embedding(start_H * start_W, self.embed_dims) 
        # self.bev_query = nn.Embedding(200 * 200, self.embed_dims) 

    def forward(self, bev_query, mlvl_feats, lidar2img):
        
        bs = lidar2img.shape[0]

        # bev_query = rearrange(self.bev_query.weight, '(h w) d -> d h w', h=self.h, w=self.w)
        # bev_query = repeat(bev_query, '... -> b ...', b=bs)
        # bev_query = bev_query + x
        # bev_query = None

        # bev_query = torch.zeros((bs, self.embed_dims * 4, self.h, self.w)).to(lidar2img.device)

        # bev_pos = rearrange(self.grid, 'z h w d -> d h w', h=self.h, w=self.w)
        bev_pos = repeat(self.grid, '... -> b ...', b=bs)
        
        bev = self.transformer(
            mlvl_feats, 
            lidar2img,
            bev_query, 
            bev_pos, 
        )

        # bev = rearrange(bev, 'b (h w) d -> b d h w', h=self.h, w=self.w)
        return bev
    
class SegTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, embed_dims, num_points=4, num_groups=1, num_levels=1, pc_range=[], h=0, w=0, pos_encode=False, **kwargs):
        super().__init__()

        self.num_groups = num_groups
        self.pc_range = pc_range
        position_encoding = PositionalEncodingMap(in_c=3, out_c=128, mid_c=128 * 2) if pos_encode else None
        self.layer = SegTransformerDecoderLayer(embed_dims, num_points, num_groups, num_levels, pc_range, h, w, position_encoding)
        self.n = 6
        
    def forward(self,
                mlvl_feats, 
                lidar2img,
                bev_query,
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
        b = bev_pos.shape[0]
        G = self.num_groups
        for lvl, feat in enumerate(mlvl_feats):
            GC = feat.shape[1]
            C = GC // G
            feat = rearrange(feat, '(b n) (g c) h w -> (b g) n h w c', b=b, n=self.n, g=G,c=C)

            mlvl_feats[lvl] = feat.contiguous()
        
        return self.layer(mlvl_feats, lidar2img, bev_query, bev_pos)

class SegTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_points, num_groups, num_levels, pc_range, h, w, position_encoding):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = pc_range
        self.in_conv = nn.Conv2d(embed_dims, embed_dims, 3, padding=1)
        self.mid_conv = nn.Sequential(
                nn.Conv2d(128 * num_points, 128 * 4, 1),
                nn.GELU(),
                nn.Conv2d(128 * 4, 128 * 4, 1),
                nn.GELU(),
                nn.Conv2d(128 * 4, embed_dims, 1),
        )
        self.out_conv = nn.Conv2d(embed_dims, embed_dims, 3, padding=1)
        # self.conv = MLP(num_points*embed_dims, embed_dims, embed_dims, 4, as_conv=True)

        self.sampling = SegSampling(embed_dims, pos_encoder=position_encoding, num_groups=num_groups, num_points=num_points, num_levels=num_levels, pc_range=pc_range, h=h, w=w)
        self.norm1 = nn.InstanceNorm2d(embed_dims)
        self.norm2 = nn.InstanceNorm2d(embed_dims)
        self.norm3 = nn.InstanceNorm2d(embed_dims)
        
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

        h, w, _ = bev_pos.shape[1:]

        bev_query = bev_query + self.in_conv(bev_query)
        bev_query = self.norm1(bev_query)

        sampled_feat = self.sampling(
            mlvl_feats,
            bev_query,
            bev_pos,
            lidar2img,
        )
        sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b (g p c) h w', h=h, w=w)
        bev_query = bev_query + self.mid_conv(sampled_feat)
        bev_query = self.norm2(bev_query)
        bev_query = bev_query + self.out_conv(bev_query)
        bev_query = self.norm3(bev_query)

        return bev_query
    
class SegSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims, pos_encoder=None, num_groups=1, num_points=8, num_levels=1, pc_range=[], h=0, w=0, eps=1e-6):
        super().__init__()

        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.img_h = h
        self.img_w = w
        self.pc_range = pc_range
        self.pos_encoder = pos_encoder
        self.sampling_offset = nn.Conv2d(embed_dims, num_groups * num_points * 3, 1)
        self.eps = 1e-5
        self.init_weights()

    def init_weights(self):
        # original
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:2], -4.0, 4.0)
        height = torch.linspace(-0.25, 1.25, self.num_groups * self.num_points).unsqueeze(1)
        bias[:, 2:3] = height

    def forward(self, mlvl_feats, bev_query, reference_points, lidar2img):
        sampling_offset = self.sampling_offset(bev_query).sigmoid() # b (g p 3) h w
        sampling_offset = rearrange(sampling_offset, 'b (g p d) h w -> b (h w) g p d',
                            g=self.num_groups,
                            p=self.num_points,
                            d=3
                        )
        sampling_offset_new = sampling_offset.clone()
        sampling_offset_new[..., :2] = (sampling_offset_new[..., :2] * (0.25 + self.eps) * 2) \
                                    - (0.25 + self.eps)
        sampling_offset_new[..., 2:3] = (sampling_offset_new[..., 2:3] * (4.0 + self.eps) * 2) \
                                    - (4.0 + self.eps)
        sampling_offset = sampling_offset_new

        reference_points_new = rearrange(reference_points, 'b h w d -> b (h w) 1 1 d', d=3).clone()

        reference_points_new[..., 0:1] = (reference_points_new[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        reference_points_new[..., 1:2] = (reference_points_new[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        reference_points_new[..., 2:3] = (reference_points_new[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        reference_points = reference_points_new
        
        reference_points = reference_points + sampling_offset
        # reference_points = rearrange(reference_points, 'b p1 h w d -> b (h w) 1 p1 d').clone()
        # if self.pos_encoder is not None:
        #     pos_encode = self.pos_encoder(reference_points)

        # reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        # reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        # reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        scale_weights = None  
        # sampling
        sampled_feats = sampling_4d(
            reference_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            self.img_h, self.img_w,
            # pixel_positional_embedding=self.pixel_positional_embedding
            # sampling_offset,
        )  # [B, Q, G, P, C] 
        # if pos_encode is not None:
        #     sampled_feats = sampled_feats + pos_encode
            
        return sampled_feats