import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint as cp

from .common import Normalize
from .csrc.wrapper import msmv_sampling
from einops import rearrange, repeat
import copy
from .checkpoint import checkpoint as cp

import math
import spconv.pytorch as spconv
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy

class PointBEVDET(nn.Module):
    def __init__(
            self,
            backbone,
            box_encoder,
            bev_encoder,
            bev,
            unet,
            decoder=None,
            scale: float = 1.0,
            outputs: dict = {},
            dim_last: int = 64,
            sparse: bool = False,
            num_groups: int = 4,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x
        self.num_groups = num_groups

        self.box_encoder = box_encoder
        self.bev_encoder = bev_encoder
        self.bev = bev

        self.unet = unet
        self.decoder = decoder

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
            self.to_logits = nn.Sequential(
                nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim_last),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_last, dim_max, 1)
            )
        else:
            algo = spconv.ConvAlgo.Native
            self.to_logits = spconv.SparseSequential(
                spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                nn.BatchNorm1d(dim_last, momentum=0.1),
                nn.ReLU(inplace=False),
                spconv.SubMConv2d(dim_last, dim_max, 1, algo=algo),
            )

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = [rearrange(self.down(y),'(b n) ... -> b n ...', b=b,n=n) for y in self.backbone(self.norm(image))] 

        G = self.num_groups
        for lvl, feat in enumerate(features):
            GC = feat.shape[2]
            C = GC // G
            feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

            features[lvl] = feat.contiguous()
    
        pred_boxes = self.box_encoder(features, lidar2img)
        x = self.bev_encoder(
            features, 
            lidar2img, 
            pred_boxes, 
            {
                '3d': batch['view'],
                '5d': batch['5d_view']})
        x = self.unet(x)
        bev = self.bev(features, lidar2img)
        y = self.decoder(bev)
        z = self.to_logits(y)

        z = z + self.to_logits(x)
        if self.sparse:
            z = z.dense()

        out = {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
        out.update(pred_boxes)
        
        return out

class BEVSD(nn.Module):
    def __init__(
            self,
            backbone,
            encoder,
            decoder=None,
            neck=None,
            scale: float = 1.0,
            outputs: dict = {},
            dim_last: int = 64,
            box_only: bool = False,
            sparse: bool = False,
            multi_head: bool = False,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        self.encoder = encoder

        self.box_only = box_only
        self.multi_head = multi_head
        self.neck = neck 

        if not box_only:
            self.decoder = decoder if decoder is not None else nn.Identity()

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
                    self.to_logits = nn.ModuleDict(layer_dict)
                else:
                    self.to_logits = nn.Sequential(
                        nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                        nn.InstanceNorm2d(dim_last),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim_last, dim_max, 1)
                    )
            else:
                algo = spconv.ConvAlgo.Native
                self.to_logits = spconv.SparseSequential(
                    spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                    nn.BatchNorm1d(dim_last, momentum=0.1),
                    nn.ReLU(inplace=False),
                    spconv.SubMConv2d(dim_last, dim_max, 1, algo=algo),
                )

    def forward_head(self, x):
        if self.multi_head:
            return {k: v(x) for k, v in self.to_logits.items()}
        else:
            x = self.to_logits(x)
            return {k: x[:, start:stop] for k, (start, stop) in self.outputs.items()}

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = [rearrange(self.down(y),'(b n) ... -> b n ...', b=b,n=n) for y in self.backbone(self.norm(image))] 
        if self.neck is not None:
            features = self.neck(features)
            features = [rearrange(y,'(b n) ... -> b n ...', b=b,n=n) for y in features]
    
        x = self.encoder(features, lidar2img)
        if self.box_only:
            return x
        
        y = self.decoder(x)
        z = self.forward_head(y)

        if isinstance(x, dict):
            assert 'pred_boxes' in x
            z.update(x)
        
        return z

class SparseMaskHead(nn.Module):

    def __init__(self,
                transformer=None,
                embed_dims=128,
                num_anchor=100,
                num_points=900,
                pad_mode='box',
                patch_size=5,
                **kwargs):
        
        super().__init__()
        assert pad_mode in ['box', 'kernel'], 'Please choose padding mode with "box" or "kernel"'
        
        self.transformer = transformer
        self.embed_dims = embed_dims

        self.num_anchor = num_anchor 
        self.num_points = num_points
        
        self.pad_mode = pad_mode
        self.patch_size = patch_size
        
        
        self._init_bev_layers(**kwargs)

    def _init_bev_layers(self, H=25, W=25, Z=8, num_points_in_pillar=4, **kwargs):

        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
                                ).view(-1, 1).expand(num_points_in_pillar, self.num_points) / Z
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

        self.query_embedding = nn.Embedding(H*W, self.embed_dims)
        self.mask_embedding = nn.Parameter(torch.rand((self.embed_dims)))

    def _get_anchor(self, pred, view):
        pred_boxes = pred['pred_boxes'].clone()

        # filter with topk candidates region
        pred_logits = pred['pred_logits'].clone()
        scores, _ = pred_logits.softmax(-1)[..., :-1].max(-1)
        filter_idx = torch.topk(scores, k=self.num_anchor, dim=-1).indices

        # Expand dimensions for filter_idx for matching with pred_boxes_coords
        filter_idx_expand = filter_idx.unsqueeze(-1).expand(*filter_idx.shape, pred_boxes.shape[-1])
        pred_boxes = torch.gather(pred_boxes, 1, filter_idx_expand)

        # project box from lidar to bev
        B, N = pred_boxes.shape[:2]
        device = pred_boxes.device
        view_3d = view['3d']
        if self.pad_mode == 'box':
            view = view['5d']

            pred_boxes = pred_boxes[..., :4]

            pred_boxes[..., 2:4] = pred_boxes[..., 2:4].exp()
            pred_boxes[:, 2:4] = pred_boxes[:, 2:4].exp()

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

        else:
            view = view['3d']
            mask = torch.zeros([B,self.h,self.w]).to(device)

            pred_boxes_coords = pred_boxes[..., :2]
            pred_boxes_coords = nn.functional.pad(pred_boxes_coords,(0, 1), value=1) # b filter_N 3
            pred_boxes_coords = (torch.einsum('b i j, b N j -> b N i', view, pred_boxes_coords)[..., :2]).int()

            b = torch.arange(B)[:, None].to(device)
            mask[b, pred_boxes_coords[..., 1], pred_boxes_coords[..., 0]] = 1

            # pad with kernel
            patch_size = self.patch_size
            kernel = torch.ones(
                (1, 1, patch_size, patch_size), dtype=torch.float64, device=device
            )
            mask = F.conv2d(
                mask.to(torch.float64).unsqueeze(1), kernel, padding=(patch_size - 1) // 2
            )
            mask = mask.bool().squeeze(1)

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
        N_pts = self.num_points

        # covert to 1d for better later operation
        mask = mask.flatten(1,2)

        out_idx = []
        ignore_idx_list = []
        if N_pts == "dyna":
            for i in range(bt):
                # Numbers of activated elements
                activ_idx = torch.nonzero(mask[i]).squeeze(1)
                out_idx.append(activ_idx)
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
                    rand_idx = torch.randperm(n_activ).to(device)
                    ignore_idx = torch.cat([activ_idx[rand_idx[N_pts:]], ignore_idx])
                    activ_idx = activ_idx[rand_idx[:N_pts]]

                out_idx.append(torch.cat([activ_idx, augm_idx]))
                ignore_idx_list.append(ignore_idx)

        out_idx = torch.stack(out_idx)
        out_ignore_idx = torch.stack(ignore_idx_list)
        # xy_vox_idx = torch.stack([((out_idx // Y) % X), out_idx % Y], dim=-1)
        return out_idx, out_ignore_idx

    def _select_query(self, grid_idx, ignore_idx):
        
        b = grid_idx.shape[0]
        B1 = torch.arange(b)[:, None].expand(-1, self.num_points)
        B2 = torch.arange(b)[:, None].expand(-1, self.h * self.w - self.num_points)

        grid = rearrange(self.grid, 'h w d -> (h w) d')
        grid = repeat(grid, '... -> b ...', b=b)

        query = self.query_embedding.weight
        query = repeat(query, '... -> b ...', b=b)

        selected_grid = grid[B1, grid_idx]
        selected_query = query[B1, grid_idx]

        ignored_grid = grid[B2, ignore_idx]
        # ignored_query = query[B2, ignore_idx]

        # selected_query = rearrange(selected_query, 'b h w d -> b (h w) d')
        # return selected_grid, selected_query, ignored_grid, ignored_query
        return selected_grid, selected_query, ignored_grid

    def get_grid(self, pred, view):
        
        b = pred['pred_boxes'].shape[0]

        mask = self._get_anchor(pred, view)
        grid_idx, ignore_idx = self._select_idx_to_keep(mask) # b n 2
        selected_grid, selected_query, ignored_grid = self._select_query(grid_idx, ignore_idx)

        selected_grid = repeat(selected_grid, 'b n d -> b p n d', p=self.num_points_in_pillar)
        pad_zs = repeat(self.z, 'p n -> b p n 1', b=b)

        return grid_idx, ignore_idx, torch.cat([selected_grid, pad_zs],dim=-1), selected_query, ignored_grid # b p n 3



    def forward(self, mlvl_feats, lidar2img, pred, view):

        grid_idx, ignore_idx, query_pos, query, ignored_query_pos = self.get_grid(pred, view)
        # query = query_pos.new_zeros((self.num_points, self.embed_dims))

        x = self.transformer(
            mlvl_feats, 
            query=query, 
            query_pos=query_pos, 
            lidar2img=lidar2img,
            reg_branches=None,
            cls_branches=None,
            pred_boxes=pred['pred_boxes'],
            box_feats=pred['box_feats'],

            # mask_query_pos=ignored_query_pos, 
            # mask_query=ignored_query,
        )  

        concat_ind = torch.cat([grid_idx, ignore_idx],dim=1).unsqueeze(-1).expand(-1, -1, self.embed_dims)
        
        b,  N = ignore_idx.shape[:2]
        device = ignore_idx.device

        mask_embeddings = self.mask_embedding[None,None].expand(b, N, -1)
        concat_queries = torch.cat([x, mask_embeddings], dim=1)

        batch_ind = torch.arange(b, device=device).unsqueeze(1).unsqueeze(2)
        channel_ind = torch.arange(self.embed_dims).unsqueeze(0).unsqueeze(1)

        out = torch.empty_like(concat_queries, device=device)
        out[batch_ind, concat_ind, channel_ind] = concat_queries
        out = rearrange(out, 'b (h w) d -> b d h w ',h=self.h, w=self.w)

        return out

class SparseHead(nn.Module):
    """
    Predict bbox -> project to BEV -> sparse_conv2d/conv2d
    """
    def __init__(self,
                with_box_refine=True,
                transformer=None,
                embed_dims=128,
                query_type='bev',
                **kwargs):
        
        super().__init__()

        self.with_box_refine = with_box_refine
        self.transformer = transformer
        self.embed_dims = embed_dims
        
        assert query_type in ['bev','box']
        self.query_type = query_type
        if query_type == 'bev':
            self._init_bev_layers(**kwargs)

        elif query_type == 'box':
            self._init_box_layers(**kwargs)

    def _init_bev_layers(self, H=25, W=25, Z=8, num_points_in_pillar=4, **kwargs):

        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
                                ).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W
                            ).flip(0).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H
                            ).flip(0).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((ys, xs, zs), -1)

        self.register_buffer('grid', ref_3d, persistent=False)

        self.h = H
        self.w = W
        self.query_embedding = nn.Embedding(H*W, self.embed_dims)
            
    def _init_box_layers(self, num_classes=0, pc_range=[], num_query=100, num_reg_fcs=2, **kwargs):
        
        self.num_classes = num_classes
        self.pc_range = pc_range 
        self.num_query = num_query

        self.init_query_bbox = nn.Embedding(self.num_query, 6)  # cx, cy, cz, w, h, l

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
        
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        cls_branch = []
        for _ in range(num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes + 1))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 6))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.

        self.cls_branches = fc_cls
        self.reg_branches = reg_branch

        # num_pred = self.transformer.num_layers

        # if self.with_box_refine:
        #     self.cls_branches = _get_clones(fc_cls, num_pred)
        #     self.reg_branches = _get_clones(reg_branch, num_pred)
        # else:
        #     self.cls_branches = nn.ModuleList(
        #         [fc_cls for _ in range(num_pred)])
        #     self.reg_branches = nn.ModuleList(
        #         [reg_branch for _ in range(num_pred)])

    def forward(self, mlvl_feats, lidar2img):

        query = self.query_embedding.weight
        if self.query_type == 'bev':
            query_pos = self.grid.flatten(1,2)
            reg_branches = None
            cls_branches = None

        elif self.query_type == 'box':
            query_pos = self.init_query_bbox.weight
            reg_branches = self.reg_branches
            cls_branches = self.cls_branches
        
        bs = lidar2img.shape[0]
        query_pos = repeat(query_pos, '... -> b ...', b=bs) # bev: b z n 3, box: b n 6
        query = repeat(query, '... -> b ...', b=bs) 
        
        x = self.transformer(
            mlvl_feats, 
            query=query, 
            query_pos=query_pos, 
            lidar2img=lidar2img,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
        )

        if self.query_type == 'bev':
            x = rearrange(x, 'b (h w) d -> b d h w', h=self.h, w=self.w)

        return x

class SparseBEVTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, n_layer, num_groups=4, return_intermediate=False, pc_range=[], **kwargs):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_layers = n_layer
        self.num_groups = num_groups
        self.pc_range = pc_range

        self._init_layers(n_layer, num_groups, **kwargs)

    def _init_layers(self, n_layer, num_groups, **kwargs):
        self.layer = SparseBEVTransformerDecoderLayer(num_groups=num_groups, pc_range=self.pc_range, **kwargs)

    def refine_bbox(self, bbox_proposal, bbox_delta):
        xyz = inverse_sigmoid(bbox_proposal[..., 0:3])
        xyz_delta = bbox_delta[..., 0:3]
        xyz_new = torch.sigmoid(xyz_delta + xyz)

        return torch.cat([xyz_new, bbox_delta[..., 3:]], dim=-1)

    def forward(self,
                mlvl_feats,
                query=None,
                key=None,
                value=None,
                query_pos=None,
                reference_points=None,
                reg_branches=None,
                cls_branches=None,
                query_type='',
                mask_query=None,
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
        
        cls_scores = []
        bbox_preds = []
        C = query.shape[-1]
        if C == mlvl_feats[0].shape[2]:
            G = self.num_groups
            for lvl, feat in enumerate(mlvl_feats):
                GC = feat.shape[2]
                C = GC // G
                feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

                mlvl_feats[lvl] = feat.contiguous()
            
        # if query_type == 'bev':
        #     query_pos = repeat(query_pos, '... -> b ...', b=bs) # z h w 3 -> b z h w 3
        #     query = repeat(query, '... -> b ...', b=bs)
        #     query = rearrange(query, 'b h w d -> b (h w) d')
        #     output = query

        for lid in range(self.num_layers):
            output = self.layer(
                output,
                value=mlvl_feats,
                query_pos=query_pos,
                #mask_query=mask_query,
                **kwargs
            )
            if reg_branches is not None:
                # cls_score = cls_branches[lid](output)  # [B, Q, num_classes]
                # bbox_pred = reg_branches[lid](output)  # [B, Q, code_size]
                cls_score = cls_branches(output)  # [B, Q, num_classes]
                bbox_pred = reg_branches(output)  # [B, Q, code_size]
                tmp = self.refine_bbox(query_pos, bbox_pred)

                query_pos = tmp.clone()

                cx = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                cy = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                cz = (tmp[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
                wl = tmp[..., 3:5]
                h = tmp[..., 5:6]
                bbox_pred = torch.cat([cx,cy,wl,cz,h], dim=-1) # cx cy cz w l h -> cx cy w l cz h

                cls_scores.append(cls_score)
                bbox_preds.append(bbox_pred)

        if reg_branches is not None:
            cls_scores = torch.stack(cls_scores)
            bbox_preds = torch.stack(bbox_preds)
            box_output = {
                'pred_logits': cls_scores[-1],
                'pred_boxes': bbox_preds[-1],
                'aux_outputs': [{'pred_logits': outputs_class, 'pred_boxes': outputs_coord} 
                                for outputs_class, outputs_coord in zip(cls_scores[:-1],bbox_preds[:-1])],
                'box_feats': output
            }
            return box_output

        return output

class SparseBEVTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dims, num_groups=4, num_points_in_pillar=4, num_points=4, num_levels=4, pc_range=[], h=224, w=480, checkpointing=False, query_type='', cross_attn=False):
        super(SparseBEVTransformerDecoderLayer, self).__init__()
        assert query_type in ['bev','box'], 'Query type should be BEV or BOX'

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.query_type = query_type

        if query_type == 'bev':
            self.position_encoder = nn.Sequential(
                nn.Linear(3*num_points_in_pillar, self.embed_dims), 
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )
        elif query_type == 'box':
            self.position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims), 
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )

        self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range, checkpointing=checkpointing)
        self.sampling = SparseBEVSampling(embed_dims, num_groups=num_groups, num_points=num_points, num_levels=num_levels, pc_range=pc_range, h=h, w=w, checkpointing=checkpointing, num_points_in_pillar=num_points_in_pillar, query_type=query_type) # SparseBEVSampling(embed_dims, num_groups=num_groups, num_points=num_points, num_levels=num_levels, pc_range=pc_range, h=h, w=w, checkpointing=checkpointing, num_points_in_pillar=num_points_in_pillar, query_type=query_type)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points, n_groups=num_groups, out_points=embed_dims, checkpointing=checkpointing)
        
        # self.sampling = SparseGridSampling(embed_dims, num_groups=num_groups, num_points=num_points, num_levels=num_levels, pc_range=pc_range, h=h, w=w, checkpointing=checkpointing, num_points_in_pillar=num_points_in_pillar, query_type=query_type)
        # self.mixing = None
        self.ffn = MLP(embed_dims, 512, embed_dims, 2)
            
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        self.cross_attn = SparseBEVSelfAttention(embed_dims, num_heads=4, dropout=0.1, pc_range=pc_range, checkpointing=checkpointing, custom_attention=False) if cross_attn else None

        self.init_weights()

    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        if self.cross_attn is not None:
            self.cross_attn.init_weights()

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
        
        if self.query_type == 'bev':
            query_pos = rearrange(query_pos, 'b z n d -> b n (z d)')
            query_pos_embed = self.position_encoder(query_pos)
        elif self.query_type == 'box':
            query_pos_embed = self.position_encoder(query_pos[..., :3])

        query_feat = query + query_pos_embed
        query_feat = self.norm1(self.self_attn(query_pos, query_feat, None, query))
        sampled_feat = self.sampling(
                query=query_feat,
                value=value,
                reference_points=query_pos,
                lidar2img=lidar2img,
            )
        if self.mixing is not None:
            query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        else:
            query_feat = self.norm2(sampled_feat)
        # sampled_feat = rearrange(sampled_feat, 'b q g p c -> b q (p g c)')
        query_feat = self.norm3(self.ffn(query_feat))
        # query_feat = query_feat + sampled_feat

        if self.cross_attn is not None:
            query_feat = self.cross_attn(query_pos, query_feat, box_feats, box_feats, key_pos=pred_boxes)
 
        return query_feat #, mask_query

class SparseBEVSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, num_groups=4, num_points=8, num_levels=4, pc_range=[], h=0, w=0, checkpointing=False, num_points_in_pillar=4, query_type='', scale=0.25):
        super().__init__()

        self.num_points = num_points
        self.num_points_in_pillar = num_points_in_pillar
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.img_h = h
        self.img_w = w
        self.pc_range = pc_range
        self.query_type = query_type

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels) if num_levels != 1 else None

        self.checkpointing = checkpointing
        self.scale = scale
        
    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query, mlvl_feats, reference_points, lidar2img):

        B, Q = query.shape[:2]
        # num_pillar = reference_points.shape[1]

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query) # b q (g p 3)

        # query type: bev
        if self.query_type == 'bev': 
            sampling_offset = rearrange(sampling_offset, 'b q (g z p d) -> b q g z p d',
                                g=self.num_groups,
                                z=self.num_points_in_pillar,
                                p=self.num_points // self.num_points_in_pillar,
                                d=3
                            )
            sampling_offset[..., :2] = sampling_offset[..., :2].sigmoid() * (0.5 / self.scale * 2) - (0.5 / self.scale)
            sampling_offset[..., 2] = sampling_offset[..., 2].sigmoid() * 2 - 1.0
            reference_points = rearrange(reference_points, 'b q (z d) -> b q 1 z 1 d', d=3).clone()

            reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            reference_points = reference_points + sampling_offset
            # import matplotlib.pyplot as plt
            # plt.scatter(reference_points[0,1225].reshape(-1,3)[:,0].cpu(), reference_points[0,1225].reshape(-1,3)[:,1].cpu())
            reference_points = rearrange(reference_points, 'b q g z p d -> b q g (z p) d')
        
        # query type: box
        elif self.query_type == 'box':
            sampling_offset = rearrange(sampling_offset, 'b q (g p d) -> b q g p d',
                                g=self.num_groups,
                                p=self.num_points,
                                d=3
                            )
            # xyz = reference_points[..., 0:3]  # [B, Q, 3]
            # wlh = reference_points[..., 3:6]  # [B, Q, 3]
            # delta_xyz = wlh[:, :, None, None, :] * sampling_offset  # [B, Q, G, P, 3]
            # reference_points = xyz[:, :, None, None, :] + delta_xyz  # [B, Q, P, 3]

            reference_points = rearrange(reference_points, 'b q d -> b q 1 1 d').clone()
            
            reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            reference_points = reference_points[..., :3] + sampling_offset

        # scale weights
        if self.scale_weights is not None:
            scale_weights = self.scale_weights(query).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
            scale_weights = scale_weights.softmax(-1)
        else:
            scale_weights = None
        # scale_weights = scale_weights.expand(B, Q, self.num_groups, 1, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            reference_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            self.img_h, self.img_w
        )  # [B, Q, G, FP, C]

        return sampled_feats
    
    def forward(self,
                query,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                lidar2img=None,
                **kwargs):
        
        if self.training and query.requires_grad and self.checkpointing:
            return cp(self.inner_forward, query, value, reference_points, lidar2img, use_reentrant=False)
        else:
            return self.inner_forward(query, value, reference_points, lidar2img)

class SparseBEVSelfAttention(nn.Module):
    """Scale-adaptive Self Attention"""
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], checkpointing=False, custom_attention=False, downsample=None):
        super().__init__()
        self.pc_range = pc_range

        if downsample is not None:
            in_dims = embed_dims
            embed_dims = embed_dims * 4
            self.down = nn.Conv2d(in_dims, embed_dims, kernel_size=3, stride=downsample, padding=1)
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=downsample, mode='bilinear', align_corners=True),
                nn.Conv2d(embed_dims, in_dims, 3, padding=1, bias=False),
            )
            self.downsample_scale = downsample
        else:
            self.down = None

        if custom_attention:
            self.attention = MultiheadAttention(embed_dims, num_heads, embed_dims//num_heads, True, skip=False)
        else:
            self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        
        self.gen_tau = nn.Linear(embed_dims, num_heads)

        self.checkpointing = checkpointing

    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def downsample(self, query_pos, q, v):

        query_pos = rearrange(query_pos, 'b (h w) d -> b h w d', h=200, w=200)
        query_pos = query_pos[:, 1::self.downsample_scale, 1::self.downsample_scale]
        query_pos = rearrange(query_pos, 'b h w d -> b (h w) d')

        q = rearrange(q, 'b (h w) d -> b d h w', h=200, w=200)
        q = self.down(q)
        q = rearrange(q, 'b d h w -> b (h w) d')

        v = rearrange(v, 'b (h w) d -> b d h w', h=200, w=200)
        v = self.down(v)
        v = rearrange(v, 'b d h w -> b (h w) d')

        return query_pos, q, v

    def inner_forward(self, query_pos, q, k, v, pre_attn_mask, key_pos):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        if self.down is not None:
            query_pos, q, v = self.downsample(query_pos, q, v)

        dist = self.calc_bbox_dists(query_pos, key_pos)
        tau = self.gen_tau(q)  # [B, Q, num_heads]

        tau = tau.permute(0, 2, 1)  # [B, num_heads, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, num_heads, Q, K]

        if pre_attn_mask is not None:  # for query denoising
            attn_mask[:, :, pre_attn_mask] = float('-inf')
            
        key = k if k is not None else q

        attn_mask = attn_mask.flatten(0, 1)  # [B x num_heads, Q, K]

        # print(attn_mask[0,1225,2500:].sum(),attn_mask[0,1225,:2500].sum())
        # print(attn_mask[0,1275,2500:].sum(), attn_mask[0,1275,:2500].sum())
        # import numpy as np
        # att_map = np.zeros((50,50))
        # for i, att_value in enumerate(attn_mask[0,1225,:2500]):
        #     att_map[i//50, i%50] = att_value
        # import matplotlib.pyplot as plt
        # plt.imshow(att_map)

        q = self.attention(
            query=q,
            key=key,
            value=v,
            attn_mask=attn_mask,
        )[0]
        # q = tmp[0]
        # v_t, index_t = torch.topk(tmp[1][0,1225], 50)
        # print(v_t, index_t)

        if self.down is not None:
            q = rearrange(q, 'b (h w) d -> b d h w', h=50, w=50)
            q = self.up(q)
            q = rearrange(q, 'b d h w -> b (h w) d')

        return q

    def forward(self, query_pos, q, k, v, pre_attn_mask=None, key_pos=None):
        if self.training and q.requires_grad and self.checkpointing:
            return cp(self.inner_forward, query_pos, q, k, v, pre_attn_mask, key_pos, use_reentrant=False)
        else:
            return self.inner_forward(query_pos, q, k, v, pre_attn_mask, key_pos)

    @torch.no_grad()
    def calc_bbox_dists(self, query_pos, key_pos):
        B = query_pos.shape[0]

        if query_pos.ndim == 5:
            centers_q = query_pos[:,0][..., :2].clone()
        else:
            centers_q = query_pos[..., :2].clone()

        centers_q[..., 0:1] = centers_q[..., 0:1]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        centers_q[..., 1:2] = centers_q[..., 1:2]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]

        if key_pos is not None:
            if key_pos.ndim == 5:
                centers_k = key_pos[:,0][..., :2].clone()
            else:
                centers_k = key_pos[..., :2].clone()
            
            # centers_k[..., 0:1] = centers_k[..., 0:1]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            # centers_k[..., 1:2] = centers_k[..., 1:2]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]

        else:
            centers_k = centers_q

        dist = []
        for b in range(B):
            dist_b = torch.norm(centers_q[b].reshape(-1, 1, 2) - centers_k[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, K]
        dist = -dist

        return dist

class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None, checkpointing=False):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

        self.checkpointing = checkpointing

    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):

        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad and self.checkpointing:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)

def sampling_4d(sample_points, mlvl_feats, scale_weights, lidar2img, image_h, image_w, sampling_offset=None, eps=1e-5, pixel_positional_embedding=None):
    """
    Args:
        sample_points: 3D sampling points in shape [B, Q, T, G, P, 3]
        mlvl_feats: list of multi-scale features from neck, each in shape [B*T*G, C, N, H, W]
        scale_weights: weights for multi-scale aggregation, [B, Q, G, T, P, L]
        lidar2img: 4x4 projection matrix in shape [B, TN, 4, 4]
    Symbol meaning:
        B: batch size
        Q: num of queries
        T: num of frames
        G: num of groups (we follow the group sampling mechanism of AdaMixer)
        P: num of sampling points per frame per group
        N: num of views (six for nuScenes)
        L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5)
    """
    B, Q, G, P, _ = sample_points.shape  # [B, Q, G, P, 3]
    N = 6
    L = len(mlvl_feats)

    sample_points = sample_points.reshape(B, Q, G * P, 3)

    # get the projection matrix
    lidar2img = lidar2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    lidar2img = lidar2img.expand(B, N, Q, G * P, 4, 4)
    lidar2img = lidar2img.reshape(B, N, Q, G*P, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[:, :, None, ..., None]     # [B, Q, GP, 4]
    sample_points = sample_points.expand(B, Q, N, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 2)   # [B, N, Q, GP, 4, 1]

    # project 3d sampling points to N views
    sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, N, Q, GP, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, N, Q, GP, 2]

    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    # check if out of image
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()  # [B, N, Q, GP]

    valid_mask = valid_mask.permute(0, 2, 3, 1)  # [B, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 2, 3, 1, 4)  # [B, Q, GP, N, 2]

    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1).expand(B, Q, G * P, 1)
    i_query = i_query.view(1, Q, 1, 1).expand(B, Q, G * P, 1)
    i_point = i_point.view(1, 1, G * P, 1).expand(B, Q, G * P, 1)
    
    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_cam = sample_points_cam[i_batch, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    
    if sampling_offset is not None:
        sampling_offset = rearrange(sampling_offset, 'b q g p d -> b q (g p) 1 d')
        sampling_offset = sampling_offset / torch.tensor(
                [mlvl_feats[0].shape[-2], mlvl_feats[0].shape[-1]]
            ).to(sampling_offset.device)[None, None, None, None]
        sample_points_cam = sample_points_cam + sampling_offset

    valid_mask = valid_mask[i_batch, i_query, i_point, i_view]  # [B, Q, GP, 1]

    # if Q == 40000:
    #     y = 135
    #     x = 55
    #     index = x + y * 200
    #     print(i_view[0, index])
    #     print(sample_points_cam[0,index])

    #     # y = 4
    #     x = x+2
    #     index = x + y * 200
    #     print(i_view[0, index])
    #     print(sample_points_cam[0,index])
    #     x = x+2
    #     index = x + y * 200
    #     print(i_view[0, index])
    #     print(sample_points_cam[0,index])

    # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    sample_points_cam = sample_points_cam.reshape(B, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 2, 1, 3, 4, 5)  # [B, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B*G, Q, P, 3)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    if scale_weights is not None:
        assert scale_weights.shape[-1] == L
        if scale_weights.ndim != 4:
            scale_weights = scale_weights.reshape(B, Q, G, P, -1)
            scale_weights = scale_weights.permute(0, 2, 1, 3, 4)
            scale_weights = scale_weights.reshape(B*G, Q, P, -1)
            
    else:
        scale_weights = torch.ones((B*G, Q, P, L)).to(sample_points.device)
        # scale_weights[..., 2] = scale_weights[..., 2] + 1.0
        # scale_weights = scale_weights + 1.0

    # multi-scale multi-view grid sample
    final = msmv_sampling(mlvl_feats, sample_points_cam.contiguous(), scale_weights.contiguous())
    # reorganize the sampled features
    C = final.shape[2]  # [BG, Q, C, P]
    final = final.reshape(B, G, Q, C, P)
    final = final.permute(0, 2, 1, 4, 3) # [B, Q, G, P, C]

    if pixel_positional_embedding is not None:
        sample_points_cam = rearrange(sample_points_cam, '(b g) q p d -> b q g p d',b=B, g=G)[..., :2]
        i_view = rearrange(i_view, 'b q (g p) d -> b q g p d', g=G, p=P).squeeze(-1)
        final = final + pixel_positional_embedding(sample_points_cam, camera_index=i_view)

    return final, sample_points_cam[..., :2]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm, skip=False):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        # self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        # self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
        self.skip = skip

    def forward(self, 
                query=None,
                key=None,
                value=None,
                attn_mask=None
        ):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        b = query.shape[0]
        # Project with multiple heads
        # q = self.to_q(query)                                # b (n H W) (heads dim_head)
        v = self.to_v(value)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        # q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        if attn_mask is not None:
            att = attn_mask.softmax(dim=-1)
        else:
            dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)
            att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if self.skip:
            z = z + query

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)

        return [z]
    
def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

from .ops.gs.functions import sparsed_grid_sample
def grid_sample(sample_points, mlvl_feats, scale_weights, lidar2img, image_h, image_w, eps=1e-5):
    """
        mlvl_feats: List[features], features: (b n) c 1 h w
    """
    B, Q, P, _ = sample_points.shape  # [B, Q, P, 3]
    N = 6
    device = sample_points.device

    # get the projection matrix
    lidar2img = lidar2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    lidar2img = lidar2img.expand(B, N, Q, P, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[:, :, None, ..., None]     # [B, Q, T, GP, 4]
    sample_points = sample_points.expand(B, Q, N, P, 4, 1)
    sample_points = sample_points.transpose(1, 2)   # [B, T, N, Q, GP, 4, 1]

    # project 3d sampling points to N views
    sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, T, N, Q, GP, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]

    # normalize
    # sample_points_cam[..., 0] /= image_w
    # sample_points_cam[..., 1] /= image_h

    # # check if out of image
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()  # [B, T, N, Q, GP]

    # normalize
    W = image_w
    H = image_h
    device = sample_points_cam.device
    denom = rearrange(
        torch.tensor([W - 1, H - 1], device=device), "i -> 1 1 1 1 i", i=2
    )
    add = rearrange(
        torch.tensor([(1 - W) / 2, (1 - H) / 2], device=device),
        "i -> 1 1 1 1 i",
        i=2,
    )
    sub = rearrange(
        torch.tensor([1 / (W - 1), 1 / (H - 1)], device=device),
        "i -> 1 1 1 1 i",
        i=2,
    )
    sample_points_cam = 2.0 * ((sample_points_cam + add) / denom) - sub
    sample_points_cam = sample_points_cam.clamp(-2,2)

    valid_mask = valid_mask.permute(0, 2, 3, 1)  # [B, T, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 2, 3, 1, 4)  # [B, T, Q, GP, N, 2]

    valid_mask = valid_mask.reshape(-1)
    sample_points_cam = sample_points_cam.reshape(-1, 2)

    sample_points_cam[torch.nonzero(valid_mask==0)] = -2
    valid_mask = valid_mask.reshape(B,Q,P,N)
    sample_points_cam = sample_points_cam.reshape(B,Q,P,N,2)

    # prepare batched indexing
    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1).expand(B, Q, P, 1)
    i_query = i_query.view(1, Q, 1, 1).expand(B, Q, P, 1)
    i_point = i_point.view(1, 1, P, 1).expand(B, Q, P, 1)
    
    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_cam = sample_points_cam[i_batch, i_query, i_point, i_view]  # [B, Q, GP, 1, 2]
    
    valid_mask = valid_mask[i_batch, i_query, i_point, i_view]  # [B, Q, GP, 1]

    i_view = i_view.reshape(-1)
    sample_points_cam = sample_points_cam.reshape(-1, 2)
    sample_points_cam = torch.clamp(sample_points_cam, min=0.0, max=1.0)
    sample_points_cam = torch.nan_to_num(sample_points_cam)
    sample_points_cam = sample_points_cam * 2 - 1

    index = torch.arange(B*Q*P).to(device)
    index_batch = (index // (Q*P)) % B

    if isinstance(mlvl_feats, list):
        out_feats = []
        for l in range(len(mlvl_feats)):
            img_feats = mlvl_feats[l]
            img_feats = rearrange(img_feats, 'b n h w c -> (b n) c 1 h w')
            feat = sparsed_grid_sample(
                img_feats, sample_points_cam, (index_batch * N + i_view).to(torch.int16)
            )  
            feat = feat.reshape(B, Q, 1, P, -1) # B Q P C
        #     out_feats.append(feat)
        # out_feats = torch.stack(out_feats, -1)
        return feat # out_feats * scale_weights
    else:
        mlvl_feats = rearrange(mlvl_feats, 'b n c h w -> (b n) c 1 h w')
        feat = sparsed_grid_sample(mlvl_feats, sample_points_cam, (index_batch * N + i_view).to(torch.int16))
        feat = feat.reshape(B, Q, P, -1)
        return feat

