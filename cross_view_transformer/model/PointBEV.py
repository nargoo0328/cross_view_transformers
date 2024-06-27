from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint as cp

from .common import Normalize
from .sparsebev import AdaptiveMixing, sampling_4d, MLP
from einops import rearrange, repeat
import copy

import spconv.pytorch as spconv
from .PointBEV_projector import CamProjector
from .PointBEV_gridsample import GridSampleVT, PositionalEncodingMap
from .PointBEV_coordselector import SampledCoordSelector

from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy

class PointBEV(nn.Module):
    def __init__(
            self,
            backbone,
            encoder,
            encoder_type,
            unet,
            neck=None,
            box_encoder=None,
            outputs: dict = {},
            dim_last: int = 64,
            sparse: bool = False,
            multi_head: bool = True,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        self.encoder = encoder
        self.encoder_type = encoder_type
        self.box_encoder = box_encoder

        self.unet = unet
        self.neck = neck if neck is not None else nn.Identity()

        dim_total = 0
        dim_max = 0
        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total
        
        self.outputs = outputs
        self.sparse = sparse
        if not sparse:
            if multi_head:
                layer_dict = {}
                for k, (start, stop) in outputs.items():
                    layer_dict[k] = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    nn.InstanceNorm2d(dim_last),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_last, stop-start, 1)
                )
                self.head = nn.ModuleDict(layer_dict)
            else:
                self.head = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    nn.InstanceNorm2d(dim_last),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_last, dim_max, 1)
                )
        else:
            algo = spconv.ConvAlgo.Native
            if multi_head:
                layer_dict = {}
                for k, (start, stop) in outputs.items():
                    layer_dict[k] = spconv.SparseSequential(
                        spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                        nn.InstanceNorm1d(dim_last, momentum=0.1),
                        nn.ReLU(inplace=False),
                        spconv.SubMConv2d(
                            dim_last, out_channels=stop-start, kernel_size=1, padding=0, algo=algo
                        )
                    )
                self.head = nn.ModuleDict(layer_dict)
            else:
                self.head = spconv.SparseSequential(
                    spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                    nn.BatchNorm1d(dim_last, momentum=0.1),
                    nn.ReLU(inplace=False),
                    spconv.SubMConv2d(dim_last, dim_max, 1, algo=algo),
                )
            
        self.multi_head = multi_head

    def forward_head(self, feats):
        if self.multi_head:
            return {k: v(feats)for k, v in self.head.items()}
        else:
            feats = self.head(feats).dense()
            return {k: feats[:, start:stop] for k, (start, stop) in self.outputs.items()}

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        extrinsics = batch['extrinsics']
        intrinsics = batch['intrinsics']
        intrinsics = update_intrinsics(intrinsics, 1 / 8)

        features = [y for y in self.backbone(self.norm(image))] 
        features = self.neck(features)

        # features = [rearrange(f, '(b n) ... -> b n ...',b=b,n=n) for f in features]
        if isinstance(features, list):
            # mlvl_feats = [rearrange(f, '(b n) ... -> b n ...',b=b,n=n) for f in features]
            # features = mlvl_feats[-2]
            # G = 4
            # for lvl, feat in enumerate(mlvl_feats):
            #     GC = feat.shape[2]
            #     C = GC // G
            #     feat = rearrange(feat.clone(), 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

            #     mlvl_feats[lvl] = feat.contiguous()
            features = [rearrange(f, '(b n) ... -> b n ...',b=b,n=n) for f in features]
        else:
            features = rearrange(features, '(b n) ... -> b n ...',b=b,n=n)
        
        if self.encoder_type == 'seg':
            pred_coarse, coarse_idx, coarse_mask = self.encoder(features, extrinsics, intrinsics)
            pred_coarse = self.unet(pred_coarse[:,0], grid_idx=coarse_idx, spatial_shape=[200,200], batch_size=b)
            pred_coarse = self.forward_head(pred_coarse)

            # if not self.training:
            return {k: v if isinstance(v, torch.Tensor) else v.dense() for k, v in pred_coarse.items()}
        else:
            pass

        if self.box_encoder is not None:
            pred_box = self.box_encoder(features, lidar2img)
        else:
            pred_box = {}
        
        pred_fine, fine_idx, fine_mask = self.encoder(features, extrinsics, intrinsics, pred_seg=pred_coarse, pred_mask=coarse_mask, pred_box=pred_box, view=batch['5d_view'])
        pred_fine = self.unet(pred_fine, grid_idx=fine_idx, spatial_shape=[200,200], batch_size=b)
        pred_fine = self.forward_head(pred_fine)

        sum_masks = coarse_mask.float() + fine_mask.float()
        union = sum_masks.bool()
        non_union = (~union).float()
        out = {}
        out['mask'] = union.squeeze(1)
        out.update(pred_box)

        # process output
        for k in pred_coarse.keys():
            if not isinstance(pred_coarse[k], torch.Tensor):
                pred_coarse[k] = pred_coarse[k].dense()
            if not isinstance(pred_fine[k], torch.Tensor):
                pred_fine[k] = pred_fine[k].dense()

            elem = pred_coarse[k] + pred_fine[k]
            out[k] = elem / (sum_masks + non_union)

        return out
    
class PointBEV_Encoder(nn.Module):
    def __init__(self,  
        mode='',
    ):
        super().__init__()
        in_shape = {'projector': [200, 200, 8], 'spatial_bounds': [-49.75, 49.75, -49.75, 49.75, -3.375, 5.375]}
        voxel_ref = "spatial"
        sampled_kwargs = {
            'N_coarse': 2500, 
            'mode': 'dense', # 'rnd_pillars', 
            'val_mode': 'dense', 
            'patch_size': 1, 
            'compress_height': False, 
            'with_fine': True,
            'valid_fine': False, 
            'N_fine': 2500, 
            'N_anchor': 100, 
            'fine_patch_size': 9, 
            'fine_thresh': 0.2, 
            'reinject_pts': True
        }
    
        self.coord_selector = SampledCoordSelector(in_shape, voxel_ref, sampled_kwargs)
        self.projector = CamProjector()
        self.view_transform = GridSampleVT(
            grid_sample_mode = 'sparse_optim',
            input_sparse = True,
            return_sparse = False,
        )


    def forward(self, img_feats, extrinsics, intrinsics, pred_seg=None, pred_mask=None, pred_box=None, view=None):
        
        b = extrinsics.shape[0]
        device = extrinsics.device

        Hfeats, Wfeats = img_feats.shape[-2:]
        dict_shape = {
            "b":b,
            "n":6,
            "t":1,
            "Hfeats":Hfeats,
            "Wfeats":Wfeats
        }

        if pred_seg is None and pred_box is None:
            dict_vox = {}
            dict_vox.update(self.coord_selector._get_vox_coords_and_idx(b, device))
        else:
            dict_vox = self.coord_selector._get_sampled_fine_coords(
                pred_seg,
                pred_mask,
                pred_box,
                view,
            )

        dict_vox.update(self.projector(dict_shape, dict_vox, extrinsics, intrinsics))
        bev_feats, mask, vox_idx = self.view_transform(
            img_feats,
            dict_vox,
        )
        return bev_feats, vox_idx, mask.squeeze(1)

class PointBEVDET(nn.Module):
    def __init__(
            self,
            backbone,
            bev_encoder,
            neck=None,
            box_encoder=None,
            unet=None,
            outputs: dict = {},
            dim_last: int = 64,
            sparse: bool = False,
            multi_head: bool = True,
            num_groups: int = 2,
            new: bool = False,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        self.neck = neck if neck is not None else nn.Identity()

        self.box_encoder = box_encoder
        self.bev_encoder = bev_encoder

        self.unet = unet if unet is not None else nn.Identity()
        self.num_groups = num_groups

        dim_total = 0
        dim_max = 0
        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total
        
        self.outputs = outputs
        self.sparse = sparse

        if not sparse:
            if multi_head:
                layer_dict = {}
                for k, (start, stop) in outputs.items():
                    layer_dict[k] = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    nn.InstanceNorm2d(dim_last),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_last, stop-start, 1)
                )
                self.head = nn.ModuleDict(layer_dict)
            else:
                self.head = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    nn.InstanceNorm2(dim_last),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_last, dim_max, 1)
                )
        else:
            algo = spconv.ConvAlgo.Native
            if multi_head:
                layer_dict = {}
                for k, (start, stop) in outputs.items():
                    layer_dict[k] = spconv.SparseSequential(
                        spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                        nn.InstanceNorm1d(dim_last, momentum=0.1),
                        nn.ReLU(inplace=False),
                        spconv.SubMConv2d(
                            dim_last, out_channels=stop-start, kernel_size=1, padding=0, algo=algo
                        )
                    )
                self.head = nn.ModuleDict(layer_dict)
            else:
                self.head = spconv.SparseSequential(
                    spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                    nn.BatchNorm1d(dim_last, momentum=0.1),
                    nn.ReLU(inplace=False),
                    spconv.SubMConv2d(dim_last, dim_max, 1, algo=algo),
                )
            
        self.multi_head = multi_head

    def _format_group_feats(self, features, grouped=False, b=None):
        if not grouped:
            G = self.num_groups
            for lvl, feat in enumerate(features):
                GC = feat.shape[2]
                C = GC // G
                feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

                features[lvl] = feat.contiguous()

        else:
            G = self.num_groups
            for lvl, feat in enumerate(features):
                feat = rearrange(feat, '(b g) n h w c -> b n h w (g c)',b=b, g=G)

                features[lvl] = feat.contiguous()

        return features

    def forward_head(self, feats):
        if self.multi_head:
            return {k: v(feats) for k, v in self.head.items()}
        else:
            feats = self.head(feats).dense()
            return {k: feats[:, start:stop] for k, (start, stop) in self.outputs.items()}

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        extrinsics = batch['extrinsics']
        intrinsics = batch['intrinsics']
        intrinsics = update_intrinsics(intrinsics, 1 / 8)

        features = [y for y in self.backbone(self.norm(image))]
        features = self.neck(features)
        features = rearrange(features, '(b n) ... -> b n ...',b=b,n=n) # [rearrange(f, '(b n) ... -> b n ...',b=b,n=n) for f in features]
        
        if self.training:
            pred_box = self.box_encoder(features, lidar2img)
        else:
            pred_box = {}

        pred_fine, fine_idx, fine_mask = self.bev_encoder(features, extrinsics, intrinsics, pred=pred_box, view=batch['5d_view'])
        pred_fine = self.unet(pred_fine, grid_idx=fine_idx, spatial_shape=[200,200], batch_size=b)
        pred_fine = self.forward_head(pred_fine)

        fine_mask = fine_mask[:,0]

        out = {}
        out['mask'] = fine_mask.bool()
        out.update(pred_box)

        # process output
        for k in pred_fine.keys():
            if not isinstance(pred_fine[k], torch.Tensor):
                pred_fine[k] = pred_fine[k].dense()
        
        out.update(pred_fine)
        return out

class SegDET(nn.Module):
    def __init__(
            self,
            backbone,
            box_encoder=None,
            unet=None,
            outputs: dict = {},
            dim_last: int = 64,
            sparse: bool = False,
            multi_head: bool = True,
            num_groups: int = 2,
            num_anchor: int = 100,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        self.box_encoder = box_encoder

        self.unet = unet if unet is not None else nn.Identity()
        self.num_groups = num_groups

        dim_total = 0
        dim_max = 0
        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total
        
        self.outputs = outputs
        self.sparse = sparse
        self.num_anchor = num_anchor
        self.h = 200
        self.w = 200

        if not sparse:
            if multi_head:
                layer_dict = {}
                for k, (start, stop) in outputs.items():
                    layer_dict[k] = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    nn.InstanceNorm2d(dim_last),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_last, stop-start, 1)
                )
                self.head = nn.ModuleDict(layer_dict)
            else:
                self.head = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    nn.InstanceNorm2(dim_last),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_last, dim_max, 1)
                )
        else:
            algo = spconv.ConvAlgo.Native
            if multi_head:
                layer_dict = {}
                for k, (start, stop) in outputs.items():
                    layer_dict[k] = spconv.SparseSequential(
                        spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                        nn.InstanceNorm1d(dim_last, momentum=0.1),
                        nn.ReLU(inplace=False),
                        spconv.SubMConv2d(
                            dim_last, out_channels=stop-start, kernel_size=1, padding=0, algo=algo
                        )
                    )
                self.head = nn.ModuleDict(layer_dict)
            else:
                self.head = spconv.SparseSequential(
                    spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                    nn.BatchNorm1d(dim_last, momentum=0.1),
                    nn.ReLU(inplace=False),
                    spconv.SubMConv2d(dim_last, dim_max, 1, algo=algo),
                )
            
        self.multi_head = multi_head

    def _format_group_feats(self, features, grouped=False, b=None):
        if not grouped:
            G = self.num_groups
            for lvl, feat in enumerate(features):
                GC = feat.shape[2]
                C = GC // G
                feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

                features[lvl] = feat.contiguous()

        else:
            G = self.num_groups
            for lvl, feat in enumerate(features):
                feat = rearrange(feat, '(b g) n h w c -> b n h w (g c)',b=b, g=G)

                features[lvl] = feat.contiguous()

        return features

    def forward_head(self, feats):
        if self.multi_head:
            return {k: v(feats) for k, v in self.head.items()}
        else:
            feats = self.head(feats).dense()
            return {k: feats[:, start:stop] for k, (start, stop) in self.outputs.items()}

    # @torch.no_grad()
    def box_to_bev(self, pred, view):

        pred_boxes = pred['pred_boxes'].clone()
        pred_logits = pred['pred_logits'].clone()
        box_feats = pred['box_feats'].clone()
        scores, _ = pred_logits.softmax(-1)[..., :-1].max(-1)

        filter_idx = torch.topk(scores, k=self.num_anchor, dim=-1).indices
        filter_idx_expand = filter_idx.unsqueeze(-1).expand(*filter_idx.shape, box_feats.shape[-1])
        pred_boxes = torch.gather(pred_boxes, 1, filter_idx_expand[..., :pred_boxes.shape[-1]])
        box_feats = torch.gather(box_feats, 1, filter_idx_expand)

        b, N, C = box_feats.shape
        device = box_feats.device

        # project box from lidar to bev
        pred_boxes = pred_boxes[..., :4]
        pred_boxes[..., 2:4] = pred_boxes[..., 2:4].exp()
        pred_boxes_coords = box_cxcywh_to_xyxy(pred_boxes, transform=True)

        pred_boxes_coords = nn.functional.pad(pred_boxes_coords,(0, 1), value=1) # b filter_N 3
        pred_boxes_coords = (torch.einsum('b i j, b N j -> b N i', view, pred_boxes_coords)[..., :4]).int()
        
        bev = torch.zeros((b, self.h, self.w, C)).to(device)
        mask = torch.zeros((b, self.h, self.w)).to(device).bool()

        for i in range(b):
            for j in range(N):
                box_coord = pred_boxes_coords[i,j]
                bev[i, box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]] += box_feats[i,j]
                mask[i, box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]] = 1
        # for i in range(b):
        #     mask = scores[i] > 0.5
        #     pred_boxes_coords_i = pred_boxes_coords[i, mask]
        #     box_feats_i = box_feats[i, mask]
        #     N = pred_boxes_coords_i.shape[0]
        #     for j in range(N):
        #         box_coord = pred_boxes_coords_i[j]
        #         bev[i, box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]] += box_feats_i[j]
        #         mask[i, box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]] = 1

        return bev, mask

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        # extrinsics = batch['extrinsics']
        # intrinsics = batch['intrinsics']
        # intrinsics = update_intrinsics(intrinsics, 1 / 8)

        features = [y for y in self.backbone(self.norm(image))]
        features = [rearrange(f, '(b n) ... -> b n ...',b=b,n=n) for f in features]
        features = self._format_group_feats(features)

        pred_box = self.box_encoder(features, lidar2img)
        bev, mask = self.box_to_bev(pred_box, batch['5d_view'])
        bev = self.unet(bev, from_dense=True)
        pred_fine = self.forward_head(bev)

        out = {}
        out['mask'] = mask.bool()
        out.update(pred_box)

        # process output
        for k in pred_fine.keys():
            if not isinstance(pred_fine[k], torch.Tensor):
                pred_fine[k] = pred_fine[k].dense()
        
        out.update(pred_fine)
        return out

class SparseMaskHead(nn.Module):

    def __init__(self,
                embed_dims=128,
                num_anchor=100,
                num_points=900,
                patch_size=5,
                mode='seg',
                num_groups=4,
                sampling_kwargs={},
                grid_kwargs={},
                **kwargs):
        
        super().__init__()

        self.embed_dims = embed_dims
        self.num_anchor = num_anchor 
        self.num_points = num_points
        self.patch_size = patch_size
        self.mode = mode
        self.sampling = PointBEVSampling(num_groups=num_groups, **sampling_kwargs)
        # self.mask_embedding = nn.Parameter(torch.rand((embed_dims)))
        self._init_bev_layers(**grid_kwargs)

    def _init_bev_layers(self, H=25, W=25, Z=8, num_points_in_pillar=4, **kwargs):

        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
                                ) / Z
        xs = torch.linspace(0.5, W - 0.5, W
                            ).view(1, W).expand(H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H
                            ).view(H, 1).expand(H, W) / H
        
        ref_2d = torch.stack((xs, ys), -1)
        
        self.register_buffer('grid', ref_2d, persistent=False)
        self.register_buffer('z', zs, persistent=False)

        self.h = H
        self.w = W
        self.num_points_in_pillar = num_points_in_pillar

    @torch.no_grad()
    def _get_anchor_seg(self, pred, pred_mask, b, device):

        mask = torch.zeros([b, self.h * self.w]).to(device)
        B = torch.arange(b)[:, None].to(device)

        if pred is not None:
            if isinstance(pred, spconv.SparseConvTensor):
                pred = pred.dense()
            
            pred = pred[:, 0].sigmoid() # b 2 H W -> b H W sum(1)
            pred = pred * pred_mask
            pred = rearrange(pred, 'b H W -> b (H W)')
            idx = torch.topk(pred, k=self.num_anchor, dim=-1).indices # b num_anchor
        else:
            if self.coarse_mode == 'rand':
                idx = []
                for _ in range(b):
                    tmp = torch.randperm(self.h * self.w).to(device)
                    idx.append(tmp[:self.num_anchor])
                idx = torch.stack(idx)

            elif self.coarse_mode == 'fixed':
                x,y = torch.meshgrid([torch.arange(0,self.h,4),torch.arange(0,self.w,4)])
                idx = torch.stack([x,y],dim=2)
                idx = (idx[..., 0]*200 + idx[..., 1]).view(-1).to(device)
                idx = repeat(idx, '... -> b ...', b=b)

        mask[B, idx] = 1
        mask = rearrange(mask, 'b (H W) -> b H W',H=self.h, W=self.w)
        
        # pad with kernel
        if pred is not None:
            patch_size = self.patch_size
            kernel = torch.ones(
                (1, 1, patch_size, patch_size), dtype=torch.float64, device=device
            )
            mask = F.conv2d(
                mask.to(torch.float64).unsqueeze(1), kernel, padding=(patch_size - 1) // 2
            )
            mask = mask.squeeze(1)
    
        return mask.bool()

    @torch.no_grad()
    def _get_anchor_box(self, pred, view, b, device):
        pred_boxes = pred['pred_boxes'].clone()

        # filter with topk candidates region
        pred_logits = pred['pred_logits'].clone()
        scores, _ = pred_logits.softmax(-1)[..., :-1].max(-1)
        filter_idx = torch.topk(scores, k=self.num_anchor, dim=-1).indices

        # Expand dimensions for filter_idx for matching with pred_boxes_coords
        filter_idx_expand = filter_idx.unsqueeze(-1).expand(*filter_idx.shape, pred_boxes.shape[-1])
        pred_boxes = torch.gather(pred_boxes, 1, filter_idx_expand)

        # project box from lidar to bev
        pred_boxes = pred_boxes[..., :4]
        pred_boxes[..., 2:4] = pred_boxes[..., 2:4].exp()
        pred_boxes_coords = box_cxcywh_to_xyxy(pred_boxes, transform=True)

        pred_boxes_coords = nn.functional.pad(pred_boxes_coords,(0, 1), value=1) # b filter_N 3
        pred_boxes_coords = (torch.einsum('b i j, b N j -> b N i', view, pred_boxes_coords)[..., :4]).int()
        # pad with box
        yy, xx = torch.meshgrid(torch.arange(self.h, device=device), torch.arange(self.w, device=device))

        # Expand dimensions for xx and yy to match the dimensions of box
        xx = xx[None, None, ...]
        yy = yy[None, None, ...]

        # Check if the coordinates are inside the boxes
        box_mask = (xx >= pred_boxes_coords[..., 0, None, None]) & (xx <= pred_boxes_coords[..., 2, None, None]) & \
                (yy >= pred_boxes_coords[..., 1, None, None]) & (yy <= pred_boxes_coords[..., 3, None, None])

        # Combine the masks for different boxes using logical OR
        mask = box_mask.any(dim=1)
        return mask

    def _select_idx_to_keep(self, mask):
        """Select final points to keep.
        Either we keep Nfine points ordered by their importance or we reinject random points when points are
        predicted as not important, otherwise we will have an artefact at the bottom due to the selection
        on uniform null points.
        """
        # Alias
        bt = mask.size(0)
        device = mask.device
        N_pts = self.num_points if self.training else "dyna"

        # covert to 1d for better later operation
        mask = mask.flatten(1,2)

        out_idx = []
        ignore_idx_list = []

        if N_pts == "dyna":
            for i in range(bt):
                # Numbers of activated elements
                activ_idx = torch.nonzero(mask[i]).squeeze(1)
                ignore_idx = torch.nonzero(mask[i]==0).squeeze(1)
                out_idx.append(activ_idx)
                ignore_idx_list.append(ignore_idx)
        else:
            # Reinject random points in batches
            for i in range(bt):
                # Numbers of activated elements
                activ_idx = torch.nonzero(mask[i]).squeeze(1)
                ignore_idx = torch.nonzero(mask[i]==0).squeeze(1)

                # How many points are not activated.
                n_activ = len(activ_idx)
                idle = N_pts - n_activ

                # Less detected points than N_pts
                if idle > 0:
                    # Random selection
                    perm = torch.randperm(len(ignore_idx)).to(device)
                    augm_idx = ignore_idx[perm[:idle]]
                    ignore_idx = ignore_idx[perm[idle:]]
                else:
                    augm_idx = torch.empty([0], device=device, dtype=torch.int64)
                    ignore_idx = torch.cat([activ_idx[N_pts:], ignore_idx])
                    activ_idx = activ_idx[:N_pts]

                out_idx.append(torch.cat([activ_idx, augm_idx]))
                ignore_idx_list.append(ignore_idx)

        out_idx = torch.stack(out_idx)
        out_ignore_idx = torch.stack(ignore_idx_list)
        xy_vox_idx = torch.stack([out_idx // self.h, out_idx % self.w], dim=-1)

        return out_idx, out_ignore_idx, xy_vox_idx

    def _select_pos(self, grid_idx):

        b, num_points = grid_idx.shape
        B1 = torch.arange(b)[:, None].expand(-1, num_points)

        grid = rearrange(self.grid, 'h w d -> (h w) d')
        grid = repeat(grid, '... -> b ...', b=b)
        selected_grid = grid[B1, grid_idx]

        new_mask = torch.zeros((b, self.h*self.w)).to(grid_idx.device)
        new_mask[B1, grid_idx] = 1
        new_mask = rearrange(new_mask, 'b (h w) -> b h w', h=self.h, w=self.w)

        return selected_grid, new_mask

    def get_grid(self, b, device, pred=None, pred_mask=None, view=None):

        mode = self.mode if (self.training or pred) else 'all'
        if mode == 'seg':
            mask = self._get_anchor_seg(pred, pred_mask, b, device)
        elif mode == 'box':
            mask = self._get_anchor_box(pred, view, b, device)
        elif mode == 'all':
            mask = torch.ones((b, self.h, self.w)).to(device)

        grid_idx, ignore_idx, xy_vox_idx = self._select_idx_to_keep(mask) # b n 2
        selected_grid, selected_mask = self._select_pos(grid_idx)
        selected_grid = repeat(selected_grid, 'b n d -> b p n d', p=self.num_points_in_pillar)
        pad_zs = repeat(self.z, 'p -> b p n 1', b=b, n=selected_grid.shape[2])

        return grid_idx, torch.cat([selected_grid, pad_zs],dim=-1), selected_mask, xy_vox_idx, ignore_idx # b p n 3

    def forward(self, mlvl_feats, lidar2img, **grid_selection):
        b = lidar2img.shape[0]
        device = mlvl_feats[0].device

        grid = rearrange(self.grid, 'h w d -> (h w) d')
        grid = repeat(grid, '... -> b p ...', b=b, p=self.num_points_in_pillar)
        pad_zs = repeat(self.z, 'p -> b p n 1', b=b, n=self.h * self.w)
        query_pos = torch.cat([grid, pad_zs],dim=-1).to(device)
        # grid_idx, query_pos, mask, xy_vox_idx, ignore_idx = self.get_grid(b, device, **grid_selection)

        # query = query_pos.new_zeros((self.num_points, self.embed_dims))

        x = self.sampling(
            mlvl_feats, 
            query_pos,
            lidar2img,
        )  
        # concat_ind = torch.cat([grid_idx, ignore_idx],dim=1).unsqueeze(-1).expand(-1, -1, self.embed_dims)
        
        # b,  N = ignore_idx.shape[:2]
        # device = ignore_idx.device

        # mask_embeddings = self.mask_embedding[None,None].expand(b, N, -1)
        # concat_feats = torch.cat([x, mask_embeddings], dim=1)

        # batch_ind = torch.arange(b, device=device).unsqueeze(1).unsqueeze(2)
        # channel_ind = torch.arange(self.embed_dims).unsqueeze(0).unsqueeze(1)

        # x = torch.empty_like(concat_feats, device=device)
        # x[batch_ind, concat_ind, channel_ind] = concat_feats
        # x = rearrange(x, 'b (h w) d -> b d h w ',h=self.h, w=self.w)

        return x, None, None
    
class PointBEVSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=128, num_points_in_pillar=4, num_groups=4, pc_range=[], img_h=0, img_w=0):
        super().__init__()

        self.embed_dims = embed_dims
        self.img_h = img_h
        self.img_w = img_w
        self.pc_range = pc_range
        self.num_groups = num_groups
        
        self.position_encoder = PositionalEncodingMap()
        self.height_mlp = MLP(num_points_in_pillar * self.embed_dims, num_points_in_pillar*self.embed_dims*2, self.embed_dims, 4)
        self.weighting = nn.Linear(self.embed_dims, 4)
        
    def forward(self, mlvl_feats, reference_points, lidar2img):
        
        pos = rearrange(reference_points, 'b z n d -> b n z d')
        pos_embedding = self.position_encoder(pos, normalized=False)

        reference_points = repeat(reference_points, 'b z n d -> b n g z d', g=self.num_groups).clone()
        reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        # scale weights
        # scale_weights = None
        scale_weights = self.weighting(pos_embedding).softmax(-1)

        # sampling
        sampled_feats = sampling_4d(
            reference_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            self.img_h, self.img_w
        )  # [B, Q, G, TP, C]
        
        sampled_feats = sampled_feats.squeeze(2) # squeeze G dim -> B Q P C
        sampled_feats = sampled_feats + pos_embedding
        sampled_feats = self.height_mlp(sampled_feats.flatten(2))
        # sampled_feats = self.norm(self.mixing(sampled_feats, pos_embedding))

        return sampled_feats
    
def update_intrinsics(intrins, ratio_scale):
    """
    Parameters
    ----------
        intrins: torch.Tensor (3, 3)

        | fx | 0  | cx |
        |  0 | fy | cy |
        |  0 | 0  | 1  |
    """
    new_intrins = intrins.clone()
    # Adjust intrinsics scale due to resizing
    new_intrins[..., 0, [0, 2]] *= ratio_scale
    new_intrins[..., 1, [1, 2]] *= ratio_scale
 
    return new_intrins    