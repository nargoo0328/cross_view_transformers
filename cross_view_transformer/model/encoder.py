import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
import math
from .common import Normalize, BEVEmbedding, generate_grid

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

        self.attention = []
        self.viz = False

        self.mask = mask

    def forward(self, q, k, v, skip=None,mask=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        b, n, _, H, W = q.shape
        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
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
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)

        if self.mask:
            # mask 25x25x6 -> n Q -> expand b n Q K
            mask = rearrange(mask, 'h w n -> n (h w)').unsqueeze(0).unsqueeze(-1)
            mask = mask.expand(b*self.heads,-1,-1,dot.shape[-1])
            dot.masked_fill_(~mask, float('-inf'))

        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
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
        if self.viz:
            self.attention.append(att)
        return z    
    
    def _set_visualize(self):
        self.viz = True
    
    def _pop(self):
        self.attention.pop()

class CrossAttentionMaskView(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
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
        self.attention = []

    def forward(self, q, k, v, skip=None):
        """
        q: (b d q)
        k: (b d h w)
        v: (b d h w)
        """
        b, n, _ = q.shape
        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b d q -> b q d')
        k = rearrange(k, 'b d h w -> b (h w) d')
        v = rearrange(v, 'b d h w -> b (h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

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
            z = z + rearrange(skip, 'b d q -> b q d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b q d -> b d q')
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
        mask: bool = False,
        tri_view: str = '',
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
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        if mask == 1:
            self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias,mask=mask)
        elif mask == 2:
            self.cross_attend = CrossAttentionMaskView(dim, heads, dim_head, qkv_bias)
        else:
            self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)

        self.skip = skip 
        self.mask = mask
        self.grid_index = None
        if tri_view:
            if tri_view == 'bev':
                self.grid_index = [0,1]
                self.compressed_dim = -1
            elif tri_view == 'side':
                self.grid_index = [0,2]
                self.compressed_dim = -2
            else:
                self.grid_index = [1,2]
                self.compressed_dim = -3
    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        lvl_embed = None,
        ignore_index = None,
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

        c = E_inv[..., -1:]                                                     # b n 4 1
        # E_inv[..., -1:] = E_inv[..., -1:]-c
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

        # get view's grid
        if self.grid_index is not None:
            grid = bev.grid.mean(dim=self.compressed_dim) # 3 H W Z -> 3 X Y
            world = grid[self.grid_index]
        else:
            world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        bev_embed = w_embed - c_embed                                           # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]                                          # b n d H W
        if lvl_embed is not None:
            key_flat = key_flat + lvl_embed.reshape(1,-1,1,1)
            query = query + lvl_embed.reshape(1,1,-1,1,1)
            
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        if self.mask == 1:
            grid_view = bev.grid_view
            return self.cross_attend(query, key, val, skip=x if self.skip else None, mask=grid_view)
        
        elif self.mask == 2:
            h, w = x.shape[-2:]
            x = rearrange(x,'b d h w-> b d (h w)')
            query = rearrange(query,'b n d h w-> b n d (h w)')
            out_x = x.clone()
            for i in range(6):
                index = (bev.grid_view_index == i) # h*w
                new_q = self.cross_attend(query[:,i,:,index], key[:,i], val[:,i], skip=x[...,index] if self.skip else None) # b, d, q
                if ignore_index is not None:
                    # ignore_index: b h w
                    ignore_index_new = ignore_index & index.reshape(h,w).unsqueeze(0) # b, h, w & 1, h, w
                    ignore_index_new = ignore_index_new.reshape(b,h*w)
                    mask_A = ignore_index_new.unsqueeze(1).expand_as(out_x)
                    mask_B = ignore_index[:,index.reshape(h,w)].unsqueeze(1).expand_as(new_q)
                    out_x[mask_A] = new_q[mask_B]
                else:
                    out_x[...,index] = new_q
            return rearrange(out_x,'b d (h w)-> b d h w',h=h,w=w)
            
        else:
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
            fpn: bool = False,
            reversed_feat: bool = True,
            lvl_embedding: bool = False,
            tri_view: bool = False,
    ):
        super().__init__()

        self.norm = Normalize()
        self.reversed_feat = reversed_feat
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)
        if reversed_feat:
            self.backbone.output_shapes = reversed(self.backbone.output_shapes)
            
        cross_views = list()
        layers = list()
        down_layers = list()
        self.down_feature = down_feature
        self.tri_view = tri_view
        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            if fpn:
                feat_dim = 256
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
            if tri_view:
                s = ['bev', 'side', 'front']
                cva = nn.ModuleDict({k: CrossViewAttention(feat_height, feat_width, feat_dim, dim, tri_view=k,**cross_view) for k in s})
                cross_views.append(cva)
            else:
                cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
                cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        if down_feature:
            self.down_layer = nn.ModuleList(down_layers)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        if lvl_embedding:
            self.level_embedding = nn.Parameter(torch.Tensor(len(middle), dim))
            self.level_embedding = nn.init.normal_(self.level_embedding)
        else:
            self.level_embedding = None
        
        if tri_view:
            self.fusion = nn.Conv2d(dim*2, dim, 1)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        if self.reversed_feat: 
            features = reversed(features)
        if self.down_feature:
            features = [l(f) for l,f in zip(self.down_layer,features)]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        if self.tri_view:
            side = self.bev_embedding.get_prior('side')            
            side = repeat(side, '... -> b ...', b=b)             

            front = self.bev_embedding.get_prior('front')              
            front = repeat(front, '... -> b ...', b=b)              

        for lvl,(cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            if self.level_embedding is not None:
                lvl_embed = self.level_embedding[lvl]
            else:
                lvl_embed = None
            if self.tri_view:
                x = cross_view['bev'](x, self.bev_embedding, feature, I_inv, E_inv,lvl_embed) # b c h w
                side = cross_view['side'](side, self.bev_embedding, feature, I_inv, E_inv,lvl_embed) # b c h z
                front = cross_view['front'](front, self.bev_embedding, feature, I_inv, E_inv,lvl_embed) # b c w z
                z = torch.einsum('b d h z, b d w z -> b d h w', side, front)
                x = self.fusion(torch.cat((x,z),dim=1))
            else:
                x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv,lvl_embed)
            
            x = layer(x)
        return x
    

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(1,dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        ) 

        self.block1 = Block(dim, dim * 2)
        self.block2 = Block(dim * 2, dim)

    def forward(self, x, time_emb):

        scale_shift = None

        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h

class DiffEncoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            reversed_feat: bool = True,
            scale: int = 1.0,
            middle: List[int] = [2, 2],
    ):
        super().__init__()
        self.dim = dim
        
        self.norm = Normalize()
        self.reversed_feat = reversed_feat
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)
        if reversed_feat:
            self.backbone.output_shapes = reversed(self.backbone.output_shapes)

        cross_views = list()
        pre_layers = list()
        
        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            # pre_layer = nn.Sequential(*[ResnetBlock(dim) for _ in range(num_layers)])
            # pre_layers.append(pre_layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)

        # self.pre_layers = nn.ModuleList(pre_layers)
        self.pre_layers = nn.ModuleList([ResnetBlock(dim) for _ in range(len(middle))])
        self.post_layers = nn.ModuleList([ResnetBlock(dim) for _ in range(len(middle))])

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, batch, x, t):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        if self.reversed_feat: 
            features = reversed(features)

        time_embedding = self.time_mlp(t)
        for i, (cross_view, feature, pre_layer, post_layer) in enumerate(zip(self.cross_views, features, self.pre_layers, self.post_layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = pre_layer(x,time_embedding)
            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = post_layer(x,time_embedding)
        
        return x