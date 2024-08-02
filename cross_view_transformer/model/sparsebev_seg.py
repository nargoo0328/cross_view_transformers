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
from .sparsebev import sampling_4d, grid_sample, SparseBEVSelfAttention, inverse_sigmoid, AdaptiveMixing
from .sparsebev import MLP as MLP_sparse
from .PointBEV_gridsample import PositionalEncodingMap, MLP
from .decoder import DecoderBlock
from torchvision.models.resnet import Bottleneck
from torchvision.models.swin_transformer import SwinTransformerBlock

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

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

        output = {}
        
        if self.box_encoder_type == 'img':
            pred_box = self.box_encoder(features, lidar2img)
            output.update(pred_box)

        if self.encoder is not None:
            x = self.encoder(features, lidar2img)
            x = self.decoder(x)
        
        if self.box_encoder_type == 'bev':
            pred_box = self.box_encoder(x)

            output.update(pred_box)

            if self.fusion:
                masks = self.get_mask(pred_box)
                masks_features = x * masks
                alpha = self.fusion #.sigmoid()
                x = alpha * masks_features + (1 - alpha) * x
               
            elif self.sparse:
                masks = self.get_mask(pred_box)
                N, C, H, W = x.shape

                # Step 1: Flatten the mask and get the indices of non-zero elements
                mask_flat = masks.view(N, -1)
                indices = torch.nonzero(mask_flat, as_tuple=False).int()
                if not indices.any():
                    indices = torch.zeros((N,2)).to(x.device).int()
                
                # Step 2: Extract the valid features using the indices
                valid_features = x.view(N, C, -1).permute(0, 2, 1)[indices[:, 0], indices[:, 1]]
                
                # Step 3: Create the sparse tensor
                # The coordinates should be in the format (batch_index, z, y, x)
                coords = torch.cat([indices[:, 0:1], torch.div(indices[:, 1:], W, rounding_mode='floor'), indices[:, 1:] % W], dim=1)

                # Create the sparse tensor
                x = spconv.SparseConvTensor(features=valid_features, indices=coords, spatial_shape=(H, W), batch_size=N)
                # print(alpha)
                # import matplotlib.pyplot as plt
                # fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
                # # Generate some data and plot in each subplot
                # for i in range(4):
                #     for j in range(4):
                #         axes[i, j].imshow(masks_features[0,i*4+j].sigmoid().cpu().numpy())
                #         axes[i, j].set_title(f'channel: {i*4 + j + 1}')
                # # plt.imshow(masks_features[0,1].cpu().numpy())
                # plt.tight_layout()
                # plt.show()

        if self.head is not None:
            pred_bev = self.head(x)
            if self.fusion:
                for k in pred_bev:
                    x = pred_bev[k]
                    masks_features = x * masks
                    alpha = self.fusion
                    x = alpha * masks_features + (1 - alpha) * x
                    pred_bev[k] = x
                pred_bev['mask'] = masks.bool()

            elif self.sparse:
                for k in pred_bev:
                    if isinstance(pred_bev[k], spconv.SparseConvTensor):
                        pred_bev[k] = pred_bev[k].dense()
                pred_bev['mask'] = masks.bool()
            
            output.update(pred_bev)

        return output

    def get_mask(self, pred):
        pred_boxes = pred['pred_boxes'].clone()

        # filter with topk candidates region
        pred_logits = pred['pred_logits'].clone()
        scores, _ = pred_logits.softmax(-1)[..., :-1].max(-1)
        scores, filter_idx = torch.topk(scores, k=300, dim=-1)

        # Expand dimensions for filter_idx for matching with pred_boxes_coords
        filter_idx_expand = filter_idx.unsqueeze(-1).expand(*filter_idx.shape, pred_boxes.shape[-1])
        pred_boxes = torch.gather(pred_boxes, 1, filter_idx_expand)

        # project box from lidar to bev
        b, N = pred_boxes.shape[:2]
        device = pred_boxes.device

        pred_boxes = pred_boxes * 200
        pred_boxes_coords = box_cxcywh_to_xyxy(pred_boxes, transform=False)

        # pad with box
        yy, xx = torch.meshgrid(torch.arange(200, device=device), torch.arange(200, device=device))

        # Expand dimensions for xx and yy to match the dimensions of box
        # xx = xx[None, None, ...]
        # yy = yy[None, None, ...]
        xx = repeat(xx, '... -> b 1 ...', b=b)
        yy = repeat(yy, '... -> b 1 ...', b=b)
        masks = []
        for i in range(b):
            pred_boxes_coords_batch = pred_boxes_coords[i]
            pred_boxes_coords_batch = pred_boxes_coords_batch[scores[i] > self.threshold]
            # Check if the coordinates are inside the boxes
            mask_batch = (xx[i] >= pred_boxes_coords_batch[:, 0, None, None]) & (xx[i] <= pred_boxes_coords_batch[:, 2, None, None]) & \
                    (yy[i] >= pred_boxes_coords_batch[:, 1, None, None]) & (yy[i] <= pred_boxes_coords_batch[:, 3, None, None]) # N h w
            
            # Combine the masks for different boxes using logical OR
            mask_batch = mask_batch.any(dim=0) # h w
            masks.append(mask_batch)

        return torch.stack(masks).unsqueeze(1).float()

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
        self.position_encoder = PositionalEncodingMap(out_c=embed_dims, mid_c=embed_dims * 2)

        self.sampling = SparseBEVSampling(embed_dims, num_groups=num_groups, num_points=num_points, num_levels=num_levels, pc_range=pc_range, h=h, w=w, checkpointing=checkpointing, num_points_in_pillar=num_points_in_pillar, scale=scale)   
        # self.ffn = MLP(num_points * embed_dims, 512, embed_dims, 2)       
        # self.norm = nn.LayerNorm(embed_dims)
        self.compressor = MLP(8*embed_dims, embed_dims*4, embed_dims, 4, as_conv=True)
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

    def _init_bev_layers(self, H=25, W=25, Z=8, num_points_in_pillar=4, **kwargs):

        # zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
        #                         ).view(num_points_in_pillar, 1, 1).expand(num_points_in_pillar, H, W) / Z
        # xs = torch.linspace(0.5, W - 0.5, W
        #                     ).flip(0).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        # ys = torch.linspace(0.5, H - 0.5, H
        #                     ).flip(0).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        xs = torch.linspace(0.5, W - 0.5, W
                            ).flip(0).view(1, W).expand(H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H
                            ).flip(0).view(H, 1).expand(H, W) / H
        # ref_3d = torch.stack((ys, xs, zs), -1)
        ref_3d = torch.stack((ys, xs), -1)
        ref_3d = torch.cat([ref_3d, torch.zeros((H, W, 1)) + 0.5], dim=-1)
        self.register_buffer('grid', ref_3d, persistent=False) # z h w 3
        self.h = H
        self.w = W
        self.bev_query = nn.Embedding(50 * 50, 512)

    def forward(self, mlvl_feats, lidar2img):
        
        bs = lidar2img.shape[0]

        bev_query = rearrange(self.bev_query.weight, '(h w) d -> d h w', h=50, w=50)
        bev_query = repeat(bev_query, '... -> b ...', b=bs)

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

    def __init__(self, embed_dims, num_points=4, num_groups=1, num_layers=6, num_levels=4, pc_range=[], h=0, w=0, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.pc_range = pc_range
        self.scale = [4.0, 1.0]
        position_encoding = PositionalEncodingMap(in_c=3, out_c=embed_dims, mid_c=embed_dims * 2)
        self.layer = nn.ModuleList([SegTransformerDecoderLayer(int(scale) * embed_dims, num_points, num_groups, num_levels, pc_range, h, w, position_encoding) for scale in self.scale])
        self.decoder = DecoderBlock(512, 128, 4, 512, True)

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
        scale = self.scale
        G = self.num_groups
        for lvl, feat in enumerate(mlvl_feats):
            GC = feat.shape[2]
            C = GC // G
            feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

            mlvl_feats[lvl] = feat.contiguous()
        
        for lid in range(self.num_layers):
            # print("level:",lid)
            bev_query = self.layer[lid](
                mlvl_feats, 
                lidar2img,
                bev_query, 
                bev_pos, 
                scale[lid],
            )
            if lid < self.num_layers-1:
                bev_query = self.decoder(bev_query, bev_query)

        return bev_query

class SegTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_points, num_groups, num_levels, pc_range, h, w, position_encoding):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = pc_range

        self.position_encoder = position_encoding
        # self.compressor = MLP(num_points * embed_dims, embed_dims * 4, embed_dims, 3, as_conv=True)

        self.in_conv = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 1),
        )
        self.mid_conv = nn.Sequential(
            nn.Conv2d(num_points * 128, embed_dims * 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims * 4, embed_dims, 1),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 1),
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
            ):
        """
            bev_pos: b z h w 3
            bev_query: b d h w
        """
        if scale != 1.0:
            b, z = bev_pos.shape[:2]
            bev_pos = rearrange(bev_pos, 'b h w d -> b d h w')
            bev_pos = F.interpolate(bev_pos, scale_factor= 1 / scale, mode='bilinear')
            bev_pos = rearrange(bev_pos, 'b d h w -> b h w d')

        h, w = bev_query.shape[2:]
        # bev_pos_embed = self.position_encoder(bev_pos[:, 3]) # b h w 2 -> b h w d
        # # bev_pos_embed = self.position_encoder(bev_pos).mean(1) # b z h w d -> b h w d
        # bev_pos_embed = rearrange(bev_pos_embed, 'b h w d -> b d h w')
        # bev_query = bev_query + bev_pos_embed
        bev_query = bev_query + self.in_conv(bev_query)
        bev_query = self.norm1(bev_query)

        sampled_feat = self.sampling(
            bev_query,
            mlvl_feats,
            bev_pos,
            lidar2img,
            self.position_encoder,
            scale,
        )

        # sampled_feat = rearrange(sampled_feat, 'b q g p c -> b (p g c) q 1')
        sampled_feat = rearrange(sampled_feat, 'b (h w) g p c -> b (p g c) h w', h=h, w=w)
        bev_query = bev_query + self.mid_conv(sampled_feat)
        bev_query = self.norm2(bev_query)
        bev_query = bev_query + self.out_conv(bev_query)
        bev_query = self.norm3(bev_query)

        return bev_query
    
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

        # self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        # self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels) if num_levels!= 1 else None
        self.sampling_offset = nn.Conv2d(embed_dims, num_groups * num_points * 3, 1)
        self.scale_weights = nn.Conv2d(embed_dims, num_groups * num_points * num_levels, 1) if num_levels!= 1 else None
        self.eps = eps

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -3.5, 3.5)

    def forward(self, query, mlvl_feats, reference_points, lidar2img, pos_encoder=None, scale=1.0):

        # num_points_pillar = reference_points.shape[1]
        
        # sampling offset 
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
        # reference_points = rearrange(reference_points, 'b q g p1 p2 d -> b q g (p1 p2) d')

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

        return sampled_feats