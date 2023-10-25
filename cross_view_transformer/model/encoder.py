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

        # self.attention = []
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
        # self.attention.append(att)
        return z    

    def forward_radar(self,q, k, v, skip=None):
        b, _, H, W = q.shape
        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b d H W -> b (H W) d')
        k = rearrange(k, 'b d h w -> b (h w) d')
        v = rearrange(v, 'b d h w -> b (h w) d')

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
        # self.attention = []

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        b, n, _ = q.shape
        # Move feature dim to last for multi-head proj
        q = q.unsqueeze(-1)
        q = rearrange(q, 'b n d q -> b n q d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip.unsqueeze(-1), 'b d q -> b q d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = z.squeeze(1)# rearrange(z, 'b q d -> b d q')

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
        if mask:
            self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias,mask=mask)
        else:
            self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)

        self.skip = skip 
        self.mask = mask

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

        # if self.mask:
        #     world = bev.grid[:2,:,:,8]
        # else:
        #     world = bev.grid[:2] 

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
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        if self.mask:
            grid_view = bev.grid_view
            # for i in range(grid_view.shape[0]):
            #     for j in range(grid_view.shape[1]):
            #         print(i,j)
            #         m = grid_view[i,j]
            #         q, k, v, = query[:,m,:,i,j], key[:,m], val[:,m]
            #         x[:,:,i,j] = self.cross_attend(q, k, v, skip=x[:,:,i,j] if self.skip else None)
            return self.cross_attend(query, key, val, skip=x if self.skip else None, mask=grid_view)
        else:
            return self.cross_attend(query, key, val, skip=x if self.skip else None)
        
    def mask_view(self, bev, query, key, val, x):
        pass
    def cal_mask(self, bev, E, I):
        """
            E: (b, n, 4, 4)
            I: (b, n, 3, 3)
        """
        pass
        def image_(pts,depths):
            mask = torch.ones_like(pts[0], dtype=bool)
            mask = mask & (pts[0, :] < 480)
            mask = mask & (pts[0, :] >= 0) 
            mask = mask & (pts[1, :] >= 0)
            mask = mask & (pts[1, :] < 224)
            mask = mask & (depths > 0)
            return pts[:,mask]

        def view_points(points, view):
            
            viewpad = torch.eye(4)
            viewpad[:view.shape[0], :view.shape[1]] = view

            points = viewpad @ points
            points = points[:3, :]
            points = points / points[2, :]

            return points
        
        E = E.inverse()
        I = I.inverse()
        grid = bev.grid                                                         # 3 h w z
        out_l_2_tmp = E @ grid
        depths = out_l_2_tmp[2, :]
        points_tmp = view_points(out_l_2_tmp, I)
        points_tmp = image_(points_tmp,depths)

class Radar_stream(nn.Module):
    def __init__(
        self,
        feat_dim,
        dim,
        heads = 4,
        dim_head = 32,
        qkv_bias = True,
        skip = True,
        **kwargs
    ):
        super().__init__()

        self.skip = skip
        self.dim = dim
        self.bev_embed = nn.Conv2d(2, dim, 1)
        # self.k_embed = nn.Conv2d(2, dim, 1, bias=False)
        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)

        self.feature_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        # pos = positionalencoding2d(dim,h,w)

        # self.register_buffer('pos', pos, persistent=False)
        

    def forward(self,x,bev_embedding,feature):
        """
        x: (b, d, H, W)
        feature: (b, d, h, w)
        bev: (b, d, H, W)
        Returns: (b, d, H, W)
        """
        b = x.shape[0]
        # pos = self.pos[None]
        # pos = repeat(pos, '1 ... -> b ...', b=b)

        pos_bev = bev_embedding.grid[:2]                                                    # 2 H W
        pos_bev = self.bev_embed(pos_bev[None])
        # pos_bev = pos_bev / (pos_bev.norm(dim=1, keepdim=True) + 1e-7)
        pos_bev = repeat(pos_bev,'1 ... -> b ...', b=b)

        pos_radar = bev_embedding.grid_radar[:2]                                            # 2 H W
        pos_radar = self.bev_embed(pos_radar[None])
        # pos_radar = pos_radar / (pos_radar.norm(dim=1, keepdim=True) + 1e-7)
        pos_radar = repeat(pos_radar,'1 ... -> b ...', b=b)

        query = pos_bev + x
        key =  pos_radar + self.feature_proj(feature)
        val = self.feature_linear(feature)
        return self.cross_attend.forward_radar(query, key, val, skip=x if self.skip else None)


class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
            radar: bool = False,
            down_feature: bool = False,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        self.radar = radar
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

        if radar:
            # self.radar_conv = PointNetEncoder(18)
            self.radar_conv = nn.Sequential(
                nn.Conv2d(16,dim,3,padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                # nn.Conv2d(dim,dim*2,3,padding=1),
                # nn.BatchNorm2d(dim*2),
                # nn.ReLU(),
                nn.Conv2d(dim,dim,3,padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )
            self.cross_views_radar = Radar_stream(dim, dim, **cross_view)
            ###
            # self.pos_fusion = nn.Conv2d(dim*2,dim,1),
            self.fusion = nn.Sequential(
                nn.Conv2d(dim*2,dim,3,padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )

    def forward(self, batch,inspect=False):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))] 
        features = reversed(features)
        if self.down_feature:
            features = [l(f) for l,f in zip(self.down_layer,features)]
        # torch.Size([b, 32, 56, 120]) swinT: torch.Size([b, 96, 64, 176])
        # torch.Size([b, 112, 14, 30]) swinT: torch.Size([b, 384, 16, 44])

        mid_features = None # for debugging if inspect = True return with mid features

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W
        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)
        
        if self.radar:
            radar_bev, mid_radar_features = self.forward_radar(x,batch['radar'],inspect=inspect)

        if self.radar:
            bev = self.fusion(torch.cat((x,radar_bev),dim=1))
            if inspect:
                mid_features = {'img_bev':x,'radar_bev':radar_bev,'fusion_bev':bev,'radar_features':mid_radar_features}
            return bev, mid_features
        return x, mid_features
    
    def forward_radar(self,x,radar,inspect=False):
        radar = self.radar_conv(radar)
        radar_map = self.cross_views_radar(x,self.bev_embedding,radar)
        # radar = self.pos_fusion(torch.cat((radar,radar_map),dim=1))
        mid_radar_features = radar if inspect else None
        return radar_map, mid_radar_features
