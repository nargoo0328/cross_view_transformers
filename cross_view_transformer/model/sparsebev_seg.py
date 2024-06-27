import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint as cp

from .common import Normalize
from .csrc.wrapper import msmv_sampling
from einops import rearrange, repeat
import copy
# from .checkpoint import checkpoint as cp

import math
import spconv.pytorch as spconv
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy
from .sparsebev import sampling_4d, grid_sample, SparseBEVSelfAttention, inverse_sigmoid, AdaptiveMixing
from .sparsebev import MLP as MLP_sparse
from .PointBEV_gridsample import PositionalEncodingMap, MLP

from torchvision.models.resnet import Bottleneck
from torchvision.models.swin_transformer import SwinTransformerBlock

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

class SparseBEVSeg(nn.Module):
    def __init__(
            self,
            backbone,
            encoder,
            head,
            neck=None,
            decoder=nn.Identity(),
            box_encoder=None,
            scale: float = 1.0,
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
        self.decoder = decoder

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
        
        if self.box_encoder is not None:
            # pred_box = self.box_encoder(x)
            # x = self.head(x)

            x = self.head(x)
            pred_box = self.box_encoder(features, lidar2img)

            x.update(pred_box)
        else:
            x = self.head(x)

        return x

class SparseHead(nn.Module):
    """
    Predict bbox -> project to BEV -> sparse_conv2d/conv2d
    """
    def __init__(self,
                transformer=None,
                embed_dims=128,
                query_type='bev',
                **kwargs):
        
        super().__init__()

        self.transformer = transformer
        self.embed_dims = embed_dims
        
        assert query_type in ['bev','box']
        self.query_type = query_type
        if query_type == 'bev':
            self._init_bev_layers(**kwargs)

        elif query_type == 'box':
            self._init_box_layers(**kwargs)

    def _init_bev_layers(self, H=25, W=25, Z=8, num_points_in_pillar=4, **kwargs):

        zs = torch.linspace(0, Z , num_points_in_pillar
                                ).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0, W, W
                            ).flip(0).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0, H , H
                            ).flip(0).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        
        # zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
        #                         ).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        # xs = torch.linspace(0.5, W - 0.5, W
        #                     ).flip(0).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        # ys = torch.linspace(0.5, H - 0.5, H
        #                     ).flip(0).view(1, H, 1).expand(num_points_in_pillar, H, W) / H

        ref_3d = torch.stack((ys, xs, zs), -1)#.flip(0, 1)
        self.register_buffer('grid', ref_3d, persistent=False)
        self.h = H
        self.w = W
        # self.query = nn.Parameter(0.1 * torch.rand((self.embed_dims, self.h, self.w)))

    def forward(self, mlvl_feats, lidar2img, seg_head=None):
        
        bs = lidar2img.shape[0]
        device = lidar2img.device

        # query = self.query.to(device)
        # query = repeat(query, '... -> b ...', b=bs) 
        query = None

        query_pos = self.grid.flatten(1,2)
        query_pos = repeat(query_pos, '... -> b ...', b=bs) # bev: b z n 3, box: b n 6
        
        x = self.transformer(
            mlvl_feats, 
            query=query, 
            query_pos=query_pos, 
            lidar2img=lidar2img,
            seg_head=seg_head,
        )

        return x # .flip(-2, -1)

class SparseBEVTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, n_layer, num_groups=1, return_intermediate=False, pc_range=[], **kwargs):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_layers = n_layer
        self.num_groups = num_groups
        self.pc_range = pc_range

        self._init_layers(n_layer, num_groups, **kwargs)

    def _init_layers(self, n_layer, num_groups, **kwargs):
        layer = SparseBEVTransformerDecoderLayer(num_groups=num_groups, pc_range=self.pc_range, **kwargs)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(n_layer)])

    def forward(self,
                mlvl_feats,
                query=None,
                key=None,
                value=None,
                query_pos=None,
                seg_head=None,
                **kwargs):
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
        output = query

        G = self.num_groups
        for lvl, feat in enumerate(mlvl_feats):
            GC = feat.shape[2]
            C = GC // G
            feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c', g=G,c=C)

            mlvl_feats[lvl] = feat.contiguous()

        pred = None
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                value=mlvl_feats,
                query_pos=query_pos,
                #mask_query=mask_query,
                **kwargs
            )
            # if seg_head is not None:
            #     tmp = seg_head(output)
            #     if pred is not None:
            #         for k in pred.keys():
            #             pred[k] = pred[k] + tmp[k]
            #     else:
            #         pred = tmp
        
        return output

class SparseBEVTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_groups=4, num_points_in_pillar=4, num_points=4, num_levels=4, pc_range=[], h=224, w=480, checkpointing=False, scale=1.0):
        super(SparseBEVTransformerDecoderLayer, self).__init__()

        self.embed_dims = embed_dims
        self.pc_range = pc_range

        # self.in_c = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims * 2, 3, padding=1),
        #     nn.InstanceNorm2d(embed_dims * 2),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dims * 2, embed_dims, 1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.GELU(),
        # )
        # self.in_c = ResNetBottleNeck(embed_dims)
        # 9:22 min, 12469 MiB
        # self.in_c = SwinTransformerBlock(embed_dims, 4, window_size=[2,2], shift_size=[1,1])

        # self.position_encoder = nn.Sequential(
        #     nn.Linear(3 * num_points_in_pillar, self.embed_dims), 
        #     nn.LayerNorm(self.embed_dims),
        #     nn.GELU(),
        #     nn.Linear(self.embed_dims, self.embed_dims),
        #     nn.LayerNorm(self.embed_dims),
        #     nn.GELU(),
        # )
        self.position_encoder = PositionalEncodingMap()

        self.sampling = SparseBEVSampling(embed_dims, num_groups=num_groups, num_points=num_points, num_levels=num_levels, pc_range=pc_range, h=h, w=w, checkpointing=checkpointing, num_points_in_pillar=num_points_in_pillar, scale=scale)   
        # self.ffn = MLP(num_points * embed_dims, 512, embed_dims, 2)       
        # self.norm = nn.LayerNorm(embed_dims)
        self.compressor = MLP(8*128, 512, 128, 4, as_conv=True)
        # self.out_c = nn.Sequential(
        #     nn.Conv2d(num_points * embed_dims, embed_dims, 3, padding=1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dims, embed_dims, 1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.GELU(),
        # )
        
    #     self.init_weights()

    # def init_weights(self):
    #     self.sampling.init_weights()

    def forward(self,
                query,
                key=None,
                value=None,
                pos=None,
                query_pos=None,
                key_pos=None,
                lidar2img=None,
                mask_query_pos=None,
                mask_query=None,
                pred_boxes=None,
                box_feats=None,
                **kwargs):
        # sample -> pos_embed -> forward z
        # h, w = query.shape[-2:]
        # x = query
        # query = self.in_c(query) # b c h w
        # query = rearrange(query, 'b d h w -> b (h w) d')

        query_pos = rearrange(query_pos, 'b z q d -> b q z d')
        pos_embedding = self.position_encoder(query_pos) # normalized=False
        # query = query + pos_embedding

        sampled_feat = self.sampling(
                query if query is not None else pos_embedding,
                value,
                query_pos,
                lidar2img,
            )# [B, Q, G, TP, C]
        
        # sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b (p g c) h w', h=h, w=w)
        # sampled_feat = self.out_c(sampled_feat)

        sampled_feat = rearrange(sampled_feat, 'b q g p c -> b q p (g c)')
        sampled_feat = sampled_feat + pos_embedding
        sampled_feat = sampled_feat.flatten(-2)
        sampled_feat = rearrange(sampled_feat, 'b q d -> b d q 1')
        sampled_feat = self.compressor(sampled_feat)
        sampled_feat = rearrange(sampled_feat, 'b d (h w) 1 -> b d h w', h=200, w=200)

        return sampled_feat #, mask_query
    
class SparseBEVSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, num_groups=4, num_points=8, num_levels=4, pc_range=[], h=0, w=0, checkpointing=False, num_points_in_pillar=4, scale=1.0):
        super().__init__()

        self.num_points = num_points
        self.num_points_in_pillar = num_points_in_pillar
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.img_h = h
        self.img_w = w
        self.pc_range = pc_range
        self.scale = scale

        # self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_levels) if num_levels != 1 else None

        self.checkpointing = checkpointing

    # def init_weights(self):
    #     bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
    #     nn.init.zeros_(self.sampling_offset.weight)
    #     nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

        # bias = self.scale_weights.bias.data.view(self.num_groups * self.num_points, self.num_levels)
        # nn.init.zeros_(self.scale_weights.weight)
        # nn.init.constant_(bias, 1.0)

    def forward(self, query, mlvl_feats, reference_points, lidar2img):

        # B, Q = query.shape[:2]
        # num_pillar = reference_points.shape[1]

        # sampling offset of all frames
        # sampling_offset = self.sampling_offset(query) # b q (g p 3)
        # sampling_offset = rearrange(sampling_offset, 'b q (g z p d) -> b q g z p d',
        #                     g=self.num_groups,
        #                     z=self.num_points_in_pillar,
        #                     p=self.num_points // self.num_points_in_pillar,
        #                     d=3
        #                 )
        # sampling_offset[..., :2] = sampling_offset[..., :2].sigmoid() * (0.5 / self.scale * 2) - (0.5 / self.scale)
        # sampling_offset[..., 2] = sampling_offset[..., 2].sigmoid() * 2 - 1.0

        reference_points = rearrange(reference_points, 'b q z d -> b q 1 z 1 d', d=3).clone()

        reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        # reference_points = reference_points + sampling_offset
        # no sampling
        reference_points = repeat(reference_points, 'b q 1 z 1 d -> b q g z p d', g=self.num_groups, p=self.num_points // self.num_points_in_pillar)
        reference_points = rearrange(reference_points, 'b q g z p d -> b q g (z p) d')

        # scale weights
        if self.scale_weights is not None:
            scale_weights = rearrange(self.scale_weights(query), 'b q p (g l) -> b q g 1 p l',p=self.num_points, g=self.num_groups, l=self.num_levels) # b q g 1 p l
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
        )  # [B, Q, G, FP, C]
        
        return sampled_feats
    
class SparseGridSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, num_points=8, num_levels=4, pc_range=[], h=0, w=0, checkpointing=False, num_points_in_pillar=4, query_type='', num_groups=None, scale=0.25):
        super().__init__()

        self.num_points = num_points
        self.num_points_in_pillar = num_points_in_pillar
        self.num_levels = num_levels
        self.img_h = h
        self.img_w = w
        self.pc_range = pc_range

        self.num_groups = num_groups
        # self.sampling_offset = nn.Linear(embed_dims, num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_points * num_levels) if num_levels != 1 else None
        # self.pointmlp = MLP(num_points * embed_dims, num_points * embed_dims, embed_dims, 2)

        self.scale = scale

    def init_weights(self):
        # bias = self.sampling_offset.bias.data.view(self.num_points, 3)
        # nn.init.zeros_(self.sampling_offset.weight)
        # nn.init.uniform_(bias[:, 0:3], -0.25, 0.25)
        pass

    def forward(self, query, mlvl_feats, reference_points, lidar2img, pos_embedding):
        
        B, Q = query.shape[:2]
        # num_pillar = reference_points.shape[1]

        # sampling offset of all frames
        # sampling_offset = self.sampling_offset(query) # b q (p 3)

        # sampling_offset = rearrange(sampling_offset, 'b q (p d) -> b q p d',
        #                         p=self.num_points,
        #                         d=3
        #                     )
        # sampling_offset[..., :2] = sampling_offset[..., :2].sigmoid() * (0.5 / self.scale * 2) - (0.5 / self.scale)
        # sampling_offset[..., 2] = sampling_offset[..., 2].sigmoid() * 2 - 1.0

        # reference_points = rearrange(reference_points, 'b q d -> b q 1 d').clone()
        reference_points = reference_points.clone()
        
        reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        
        if self.scale_weights is not None:
            scale_weights = self.scale_weights(query).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
            scale_weights = scale_weights.softmax(-1)
        else:
            scale_weights = None

        # sampling
        sampled_feats = grid_sample(
            reference_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            self.img_h, self.img_w
        )  # [B, Q, G, FP, C]

        # sampled_feats = self.pointmlp(sampled_feats.flatten(-2))
        return sampled_feats

class DetSeg(nn.Module):
    def __init__(
            self,
            backbone,
            encoder,
            head,
            neck=None,
            decoder=nn.Identity(),
            scale: float = 1.0,
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
        self.decoder = decoder

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = [rearrange(self.down(y),'(b n) ... -> b n ...', b=b,n=n) for y in self.backbone(self.norm(image))] 
        if self.neck is not None:
            features = self.neck(features)
            features = [rearrange(y,'(b n) ... -> b n ...', b=b,n=n) for y in features]
    
        x, pred_boxes = self.encoder(features, lidar2img)
        x = self.decoder(x)
        x = self.head(x)
        x.update(pred_boxes)

        return x

class DetSegHead(nn.Module):
    """
    Predict bbox -> project to BEV -> sparse_conv2d/conv2d
    """
    def __init__(self,
                transformer=None,
                embed_dims=128,
                _bev=False,
                _box=False,
                **kwargs):
        
        super().__init__()
        assert _bev or _box , 'At least specify one query mode.'

        self.transformer = transformer
        self.embed_dims = embed_dims
        
        self.bev = _bev
        self.box = _box
        if _bev:
            self._init_bev_layers(**kwargs)
        if _box:
            self._init_box_layers(**kwargs)
        else:
            self.cls_branches = None
            self.reg_branches = None

    def _init_bev_layers(self, H=25, W=25, **kwargs):

        # zs = torch.linspace(0, Z , num_points_in_pillar
        #                         ).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        # xs = torch.linspace(0, W, W
        #                     ).flip(0).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        # ys = torch.linspace(0, H , H
        #                     ).flip(0).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        
        xs = torch.linspace(0.5, W - 0.5, W
                            ).flip(0).view(1, W).expand(H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H
                            ).flip(0).view(H, 1).expand(H, W) / H

        ref_2d = torch.stack((ys, xs), -1)
        ref_2d = F.pad(ref_2d, (0, 1), value=0)

        self.register_buffer('grid', ref_2d, persistent=False)
        self.h = H
        self.w = W
        self.bev_query = nn.Embedding(self.h * self.w, self.embed_dims)

    def _init_box_layers(self, num_query=100, num_reg_fcs=2, num_classes=1, **kwargs):

        self.num_query = num_query
        self.init_query_bbox = nn.Embedding(num_query, 6)  # cx, cy, cz, w, h, l
        self.det_query = nn.Embedding(num_query, self.embed_dims)  # DAB-DETR

        nn.init.zeros_(self.init_query_bbox.weight[:, 2:3])
        nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 1.5)

        grid_size = int(math.sqrt(self.num_query))
        assert grid_size * grid_size == self.num_query
        x = y = torch.arange(grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # [0, grid_size - 1]
        xy = torch.cat([xx[..., None], yy[..., None]], dim=-1)
        xy = (xy + 0.5) / grid_size  # [0.5, grid_size - 0.5] / grid_size ~= (0, 1)
        with torch.no_grad():
            self.init_query_bbox.weight[:, :2] = xy.reshape(-1, 2)  # [Q, 2]

        cls_branch = []
        for _ in range(num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, num_classes + 1))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 6))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = fc_cls
        self.reg_branches = reg_branch

    def _prepare_queries(self, b):
        if self.bev:
            bev_query = repeat(self.bev_query.weight, '... -> b ...', b=b)
            bev_pos = self.grid.flatten(0, 1)
            bev_pos = repeat(bev_pos, '... -> b ...', b=b)
        else:
            bev_query = bev_pos = None
        if self.box:
            det_query = repeat(self.det_query.weight, '... -> b ...', b=b)
            det_pos = repeat(self.init_query_bbox.weight, '... -> b ...', b=b)
        else:
            det_query = det_pos = None

        return bev_query, bev_pos, det_query, det_pos

    def forward(self, mlvl_feats, lidar2img):
        
        bs = lidar2img.shape[0]
        
        bev_query, bev_pos, det_query, det_pos = self._prepare_queries(bs)
        
        bev, pred_boxes = self.transformer(
            mlvl_feats, 
            lidar2img,
            bev_query, 
            bev_pos, 
            det_query, 
            det_pos,
            self.cls_branches,
            self.reg_branches,
        )
        bev = rearrange(bev, 'b (h w) d -> b d h w', h=self.h, w=self.w)
        return bev, pred_boxes # .flip(-2, -1)
    
class DetSegTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, embed_dims, num_points=4, num_groups=1, num_layers=6, num_levels=4, pc_range=[], h=0, w=0, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.pc_range = pc_range
        self.layer = DetSegTransformerDecoderLayer(embed_dims, num_points, num_groups, num_levels, pc_range, h, w)

    def refine_bbox(self, bbox_proposal, bbox_delta):
        xyz = inverse_sigmoid(bbox_proposal[..., 0:3])
        xyz_delta = bbox_delta[..., 0:3]
        xyz_new = torch.sigmoid(xyz_delta + xyz)

        return torch.cat([xyz_new, bbox_delta[..., 3:]], dim=-1)

    def forward(self,
                mlvl_feats, 
                lidar2img,
                bev_query, 
                bev_pos, 
                det_query, 
                det_pos,
                cls_branches,
                reg_branches,
                **kwargs):
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
        cls_scores, bbox_preds = [], []

        G = self.num_groups
        for lvl, feat in enumerate(mlvl_feats):
            GC = feat.shape[2]
            C = GC // G
            feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

            mlvl_feats[lvl] = feat.contiguous()
        
        for lid in range(self.num_layers):
            # print("level:",lid)
            bev_query, det_query = self.layer(
                mlvl_feats, 
                lidar2img,
                bev_query, 
                bev_pos, 
                det_query, 
                det_pos,
            )

            if cls_branches is not None:
                cls_score = cls_branches(det_query)  # [B, Q, num_classes]
                bbox_pred = reg_branches(det_query)  # [B, Q, code_size]
                tmp = self.refine_bbox(det_pos, bbox_pred)

                det_pos = tmp.clone()

                cx = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                cy = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                cz = (tmp[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
                wl = tmp[..., 3:5]
                h = tmp[..., 5:6]
                bbox_pred = torch.cat([cx,cy,wl,cz,h], dim=-1) # cx cy cz w l h -> cx cy w l cz h

                cls_scores.append(cls_score)
                bbox_preds.append(bbox_pred)
        
        if cls_branches is not None:
            cls_scores = torch.stack(cls_scores)
            bbox_preds = torch.stack(bbox_preds)
            box_output = {
                'pred_logits': cls_scores[-1],
                'pred_boxes': bbox_preds[-1],
                'aux_outputs': [{'pred_logits': outputs_class, 'pred_boxes': outputs_coord} 
                                for outputs_class, outputs_coord in zip(cls_scores[:-1],bbox_preds[:-1])],
            }
        else:
            box_output = {}

        return bev_query, box_output

class DetSegTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_points, num_groups, num_levels, pc_range, h, w):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = pc_range

        self.position_encoder = PositionalEncodingMap(out_c=embed_dims, mid_c=embed_dims * 2)
        self.compressor = MLP(num_points * embed_dims, embed_dims * 4, embed_dims, 3, as_conv=True)

        self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=4, dropout=0.1, pc_range=pc_range, checkpointing=True)
        # self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points, n_groups=num_groups, out_points=embed_dims, checkpointing=True)
        self.sampling = DetSegSampling(embed_dims, num_groups=num_groups, num_points=num_points, num_levels=num_levels, pc_range=pc_range, h=h, w=w)
        # self.ffn = MLP_sparse(embed_dims, 512, embed_dims, 2)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        # self.norm3 = nn.LayerNorm(embed_dims)
        
        self.init_weights()

    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        # self.mixing.init_weights()

    def _parse_input(self, bev_query, bev_pos, det_query, det_pos):

        if bev_pos is None:
            query = det_query
            query_pos = det_pos
            det_query_index = 0

        elif det_pos is None:
            query = bev_query
            query_pos = bev_pos
            det_query_index = bev_query.shape[1]
        else:
            query = torch.cat([bev_query, det_query], dim=1) # b n1+n2 d
            query_pos = torch.cat([bev_pos, det_pos[..., :3]], dim=1) # b n1+n2 3
            det_query_index = bev_query.shape[1]

        return query, query_pos, det_query_index

    def forward(self,
                mlvl_feats, 
                lidar2img,
                bev_query, 
                bev_pos, 
                det_query, 
                det_pos, 
            ):
        
        query, query_pos, det_query_index = self._parse_input(bev_query, bev_pos, det_query, det_pos)
        res_q = query

        pos_embed = self.position_encoder(query_pos)
        query = query + pos_embed

        # self attention only performs on det queries
        query = self.norm1(self.self_attn(query_pos, query, None, query))

        sampled_feat = self.sampling(
            query,
            mlvl_feats,
            query_pos,
            lidar2img
        )

        # bev_query = query[:, :det_query_index]
        # det_query = query[:, det_query_index:]
        # bev_sampled_feat = sampled_feat[:, :det_query_index]
        # det_sampled_feat = sampled_feat[:, det_query_index:]

        # det_query = self.norm2(self.mixing(det_sampled_feat, det_query))
        # det_query = self.norm3(self.ffn(det_query))
        # # 
        # bev_sampled_feat = rearrange(bev_sampled_feat, 'b q g p c -> b (p g c) q 1')
        # bev_sampled_feat = self.compressor(bev_sampled_feat).squeeze(-1) # b d (q1+q2)
        # bev_sampled_feat = rearrange(bev_sampled_feat, 'b d q -> b q d')
        # bev_query = bev_query + bev_sampled_feat

        sampled_feat = rearrange(sampled_feat, 'b q g p c -> b (p g c) q 1')
        sampled_feat = self.compressor(sampled_feat).squeeze(-1) # b d (q1+q2)
        sampled_feat = self.norm2(rearrange(sampled_feat, 'b d q -> b q d'))
        query = res_q + sampled_feat

        bev_query = query[:, :det_query_index]
        det_query = query[:, det_query_index:]

        return bev_query, det_query
    
class DetSegSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims, num_groups=1, num_points=8, num_levels=1, pc_range=[], h=0, w=0):
        super().__init__()

        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.img_h = h
        self.img_w = w
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_points * num_groups * num_levels) if num_levels!= 1 else None

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def forward(self, query, mlvl_feats, reference_points, lidar2img):

        # B, Q = query.shape[:2]
        # sampling offset 
        sampling_offset = self.sampling_offset(query) # b q (g p 3)
        sampling_offset = rearrange(sampling_offset, 'b q (g p d) -> b q g p d',
                            g=self.num_groups,
                            p=self.num_points ,
                            d=3
                        )

        reference_points = rearrange(reference_points, 'b q d -> b q 1 1 d', d=3).clone()

        reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
 
        reference_points = reference_points + sampling_offset # b q g p d
        # print(reference_points[0,1240:1250,0,:4].reshape(-1,3))
        if self.scale_weights is not None:
            # scale weights
            scale_weights = rearrange(self.scale_weights(query), 'b q (p g l) -> b q g 1 p l',p=self.num_points, g=self.num_groups, l=self.num_levels) # b q g 1 p l
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
        
        return sampled_feats