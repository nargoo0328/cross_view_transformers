import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
import math
from .common import Normalize, BEVEmbedding, generate_grid, get_view_matrix
import copy
from mmengine.model import xavier_init, constant_init

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm, mask=False):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None,mask=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        b, _, H, W = q.shapev
        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b d H W -> b (H W) d')
        k = rearrange(k, 'b n d h w -> b (n h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # print(q.shape,k.shape) #torch.Size([1, 6, 625, 128]) torch.Size([1, 6, 6720, 128])
        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)
        # print(q.shape,k.shape) # torch.Size([1, 6, 625, 128]) torch.Size([1, 6, 6720, 128])
        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        # print(q.shape,k.shape) # torch.Size([4, 6, 625, 32]) torch.Size([4, 6, 6720, 32])
        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)
        att = dot.softmax(dim=-1)
        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)
        return z    
        
class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)

        self.skip = skip 

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
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
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
    
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        img_embed = d_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        # if self.mask:
        #     world = bev.grid[:2,:,:,8]
        # else:
        #     world = bev.grid[:2] 

        world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        bev_embed = w_embed                                                     # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        query_pos = repeat(bev_embed, '1 ... -> b ...', b=b)  
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x                                         # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        return self.cross_attend(query, key, val, skip=x if self.skip else None)
    

class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
            down_feature: bool = False,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)
        self.backbone.output_shapes = reversed(self.backbone.output_shapes)
            
        cross_views = list()
        layers = list()
        down_layers = list()
        self.down_feature = down_feature
        
        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            if down_feature:
                down_layers.append(nn.Sequential(
                    nn.Conv2d(feat_dim,feat_dim//2,1),
                    nn.BatchNorm2d(feat_dim//2),
                    nn.ReLU(),
                    nn.Conv2d(feat_dim//2,feat_dim//2,1),
                    nn.BatchNorm2d(feat_dim//2),
                    nn.ReLU(),
                ))
                feat_dim = feat_dim//2
            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        if down_feature:
            self.down_layer = nn.ModuleList(down_layers)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch,inspect=False):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))] 
        features = reversed(features)
        if self.down_feature:
            features = [l(f) for l,f in zip(self.down_layer,features)]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W
        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)
        
        return x
    
class Detr3D(nn.Module):
    def __init__(
            self,
            backbone,
            head,
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        self.head = head
        # self.backbone.output_shapes = reversed(self.backbone.output_shapes)


    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = [rearrange(self.down(y),'(b n) ... -> b n ...', b=b, n=n) for y in self.backbone(self.norm(image))] 
        # features = reversed(features)
        pred = self.head(features, lidar2img)
        return pred
    
class BEVSD(nn.Module):
    def __init__(
            self,
            backbone,
            head,
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        self.head = head
        # self.backbone.output_shapes = reversed(self.backbone.output_shapes)


    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = [rearrange(self.down(y),'(b n) ... -> b n ...', b=b,n=n) for y in self.backbone(self.norm(image))] 
        # features = reversed(features)
        pred = self.head(features, lidar2img)
        return pred 
    
class BEVSDHead(nn.Module):
    def __init__(self,
                with_box_refine=False,
                transformer=None,
                decoder=None,
                H=25,
                W=25,
                Z=4,
                num_points_in_pillar=4,
                embed_dims=128,
                num_classes=0,
                num_reg_fcs=1,
                pc_range=None,
                orientation=False,
                outputs=None,
                dim_last=0,
                num_levels=4,
                **kwargs):
        super().__init__()
        self.with_box_refine = with_box_refine
        self.transformer = transformer
        self.decoder = decoder
        self.h = H
        self.w = W
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_reg_fcs = num_reg_fcs
        self.pc_range = pc_range

        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
                                ).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W
                            ).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H
                            ).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        # ref_3d = ref_3d.permute(0, 3, 1, 2)
        self.register_buffer('grid', ref_3d, persistent=False)       

        dim_total = 0
        dim_max = 0
        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.outputs = outputs
        self._init_layers(dim_last, dim_max)

        self.in_projection = nn.ModuleList([nn.Conv2d(256, embed_dims, 1) for _ in range(num_levels)])

    def _init_layers(self, dim_last, dim_max):
        sigma = 1.0
        self.query_embedding = nn.Parameter(sigma * torch.randn(self.h, self.w, self.embed_dims))
        # self.init_query_pos = nn.Embedding(self.h, self.w, 4)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1)
        )

    def forward(self, mlvl_feats, lidar2img):
        lvl = len(mlvl_feats)
        b, n = mlvl_feats[0].shape[:2]
        mlvl_feats = [rearrange(feat, 'b n ... -> (b n) ...') for feat in mlvl_feats]
        mlvl_feats = [rearrange(self.in_projection[i](mlvl_feats[i]), '(b n) ... -> b n ...',b=b,n=n) for i in range(lvl)]
        bev = self.transformer(mlvl_feats, query=self.query_embedding, query_pos=self.grid, lidar2img=lidar2img)
        bev = self.decoder(bev)
        pred = self.reg_branch(bev)
        return {k: pred[:, start:stop] for k, (start, stop) in self.outputs.items()}

class BEVSDTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, n_layer, return_intermediate=False, **kwargs):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_layers = n_layer
        self._init_layers(n_layer, **kwargs)

    def _init_layers(self, n_layer, **kwargs):
        layer = BEVSDTransformerDecoderLayer(**kwargs)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(n_layer)])

    def forward(self,
                mlvl_feats,
                query=None,
                key=None,
                value=None,
                query_pos=None,
                reference_points=None,
                reg_branches=None,
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
        bs = mlvl_feats[0].size(0)
        h, w = query.shape[:2]
        query_pos = repeat(query_pos, '... -> b ...', b=bs) # z h w 3 -> b z h w 3
        query = repeat(query, '... -> b ...', b=bs)
        query = rearrange(query, 'b h w d -> b (h w) d')
        output = query
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                value=mlvl_feats,
                query_pos=query_pos,
                **kwargs
            )
        output = rearrange(output, 'b (h w) d -> b d h w', h=h, w=w)
        return output
    
class BEVSDTransformerDecoderLayer(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, num_points_in_pillar=8,
                 activation="relu", normalize_before=False, pc_range=None, **kwargs):
        super().__init__()
        self.position_encoder = nn.Sequential(
            nn.Linear(3*num_points_in_pillar, d_model), 
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = Detr3DCrossAtten(d_model, dropout=dropout, pc_range=pc_range, num_points_in_pillar=num_points_in_pillar, **kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, query, query_pos):
        return query + self.position_encoder(query_pos)
    
    def forward(self,
                query,
                key=None,
                value=None,
                pos=None,
                query_pos=None,
                key_pos=None,
                lidar2img=None,
                **kwargs):
        query_pos_embed = rearrange(query_pos, 'b z h w d -> b (h w) (z d)')
        query_pos_embed = self.position_encoder(query_pos_embed)
        q = k = query + query_pos_embed
        tgt2 = self.self_attn(q, k, value=query)[0]
        tgt = query + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(query=tgt,
                                key=key,
                                value=value,
                                query_pos=query_pos_embed,
                                reference_points=query_pos,
                                lidar2img=lidar2img,
                                **kwargs)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Detr3DCrossAtten(nn.Module):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points_in_pillar=8,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 batch_first=False,
                 h=0,
                 w=0,):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.img_h = h
        self.img_w = w
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0
        if not _is_power_of_2(dim_per_head):
            print(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')
            
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points_in_pillar * num_points
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads*num_levels*self.num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self.sampling_offsets = nn.Linear(embed_dims, num_heads*num_levels*self.num_points * 2)

        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)

        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                lidar2img=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        # query = query.permute(1, 0, 2)
        bs, num_query, _ = query.size()

        offsets = rearrange(self.sampling_offsets(query), 'b q (h l p d) -> b q h l p d', h=self.num_heads, l=self.num_levels, p=self.num_points, d=2)
        reference_points = rearrange(reference_points, 'b p h w d -> b (h w) p d')
        # reference_points = rearrange(reference_points, 'b q p d -> b (q p) d', q=num_query, p=self.num_points)
        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_heads, self.num_points*self.num_levels)
        output, mask = feature_sampling(
            value, reference_points, offsets, self.pc_range, lidar2img, self.img_h, self.img_w) # b d q n_head n_pillar*n_points*mlvl
        output = torch.nan_to_num(output)
        # mask = torch.nan_to_num(mask)
        # mask = rearrange(mask, 'b n q p d -> b q n p d', q=num_query, p=self.num_points).unsqueeze(1)
        output = output * attention_weights
        output = output.sum(-1).sum(-1)
        output = output.permute(0, 2, 1)
        
        count = mask.sum(-1) > 0 # b 1 q p
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        output = output / count[..., None]

        output = self.output_proj(output)

        return self.dropout(output) + inp_residual #+ pos_feat


def feature_sampling(mlvl_feats, reference_points, offsets, pc_range, lidar2img, h, w):
    """
        mlvl_feats: 
        reference_points: b q num_points_in_pillar 3
        offsets: b q num_heads num_levels num_points_in_pillar*num_points 2
    """
    bs, num_query, num_points_in_pillar = reference_points.shape[:-1]
    num_heads, num_levels, num_points = offsets.shape[2:-1]
    num_points = num_points // num_points_in_pillar
    offsets = rearrange(offsets, 'b q h l (n_pillar n_points) d-> b q n_pillar h l n_points d',n_pillar=num_points_in_pillar,n_points=num_points)
    # reference_points = reference_points.clone()
    # project to lidar coordinates
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

    # pad 1 for projection
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(bs, 1, num_query, num_points_in_pillar, 4).repeat(1, num_cam, 1, 1, 1).unsqueeze(-1) # b n q p 4 1
    lidar2img = lidar2img.view(bs, num_cam, 1, 1, 4, 4).repeat(1, 1, num_query, num_points_in_pillar, 1, 1) # b n q p 4 4
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)

    # check if hit
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps) # b n q p 1
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= w
    reference_points_cam[..., 1] /= h
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    # _, proj_index = mask.sum(-2).max(1) # b q
    # batch_grid = torch.arange(bs).view(-1, 1).expand(-1, num_query).long().to(device)
    # print(reference_points_cam.shape)
    # reference_points_cam = reference_points_cam
    reference_points_cam = reference_points_cam.permute(1,0,2,3,4) # n b q p 2

    mask = mask.squeeze(-1).permute(1,0,2,3)
    indexes = []
    for i, mask_per_img in enumerate(mask): # n b q p
        index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        indexes.append(index_query_per_img)
    max_len = max([len(each) for each in indexes])
    reference_points_rebatch = reference_points_cam.new_zeros(
        [bs, num_cam, max_len, num_points_in_pillar, num_heads, num_levels, num_points, 2])
    for j in range(bs):
        for i, reference_points_per_img in enumerate(reference_points_cam):   
            index_query_per_img = indexes[i]
            reference_points_rebatch[j, i, :len(index_query_per_img)] = \
                reference_points_per_img[j, index_query_per_img].unsqueeze(-2).unsqueeze(-2).unsqueeze(-2) \
                    + offsets[j, index_query_per_img]
    reference_points_rebatch = rearrange(reference_points_rebatch, \
                                    'b n_cam l n_pillar n_head n_lvl n_p d -> b n_cam l n_head (n_pillar n_p) d n_lvl')
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        C, H, W = feat.shape[-3:]
        feat = feat.view(bs*num_cam, C, H, W)
        reference_points_cam_lvl = reference_points_rebatch[..., lvl].reshape(bs*num_cam, max_len, num_heads*num_points_in_pillar*num_points, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl) # bs*n d max_len num_points
        sampled_feat = sampled_feat.reshape(bs, num_cam, C, max_len, num_heads, num_points_in_pillar*num_points).permute(0,1,3,2,4,5)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1) # b n q_max d p mlvl
    slots = sampled_feats.new_zeros(
        [bs, num_query, C, num_heads, num_points_in_pillar*num_points, lvl+1]
    )
    for j in range(bs):
        for i, index_query_per_img in enumerate(indexes):
            slots[j, index_query_per_img] += sampled_feats[j, i, :len(index_query_per_img)]
    slots = slots.permute(0, 2, 1, 3, 4, 5).flatten(-2) # b d q n_head n_pillar*n_points mlvl
    return slots, mask
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

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