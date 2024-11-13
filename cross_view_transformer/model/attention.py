import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from typing import Optional

from .PointBEV_gridsample import MLP

class PositionEmbeddingSine(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dims, heads, dim_head, slot, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head
        self.slot = slot

        self.to_q = nn.Sequential(
            norm(embed_dims), 
            nn.Linear(embed_dims, heads * dim_head, bias=True)
        )
        self.to_k = nn.Sequential(
            norm(embed_dims), 
            nn.Linear(embed_dims, heads * dim_head, bias=True)
        )
        self.to_v = nn.Sequential(
            norm(embed_dims), 
            nn.Linear(embed_dims, heads * dim_head, bias=True)
        )

        self.proj = nn.Linear(heads * dim_head, embed_dims)
        self.prenorm = norm(embed_dims)
        self.mlp = nn.Sequential(nn.Linear(embed_dims, 2 * embed_dims), nn.GELU(), nn.Linear(2 * embed_dims, embed_dims))
        self.postnorm = norm(embed_dims)
        self.eps = 1e-6

    def forward(self, q, k, v):

        # Project with multiple heads
        q = self.to_q(q)                                # b l1 (heads dim_head)
        k = self.to_k(k)                                # b l2 (heads dim_head)
        v = self.to_v(v)                                # b l2 (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)
        if self.slot:
            att = dot.softmax(dim=-2) + self.eps
            att = att / att.sum(dim=-1, keepdim=True)
        else:
            att = dot.softmax(dim=-1)
        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)

        return z, att
    
class SlotAttention(nn.Module):
    def __init__(self, embed_dims, feats_dim, num_layers, num_slots):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_slots = num_slots

        self.feats_to_k = nn.Sequential(
            nn.BatchNorm2d(feats_dim),
            nn.ReLU(),
            nn.Conv2d(feats_dim, embed_dims, 1, bias=False))
        
        self.feats_to_v = nn.Sequential(
            nn.BatchNorm2d(feats_dim),
            nn.ReLU(),
            nn.Conv2d(feats_dim, embed_dims, 1, bias=False))
        
        self.cross_attns = CrossAttentionLayer(embed_dims, 4, 32, True)
        self.mlps = MLP(embed_dims, embed_dims*2, embed_dims, num_layers=2)
        self.slots = nn.Parameter(torch.randn(self.num_slots, embed_dims))
        self.pixel_pe = PositionEmbeddingSine(embed_dims // 2, normalize=True)

    def forward(self, feats):
        b = feats.shape[0]
        pos_embed = self.pixel_pe(feats)

        feats_k = self.feats_to_k(feats) + pos_embed
        feats_v = self.feats_to_v(feats)

        slots = repeat(self.slots, '... -> b ...', b=b)
        feats_k = rearrange(feats_k, 'b d h w -> b (h w) d')
        feats_v = rearrange(feats_v, 'b d h w -> b (h w) d')
        for i in range(self.num_layers):
            slots_tmp, att = self.cross_attns(slots, feats_k, feats_v)
            slots = slots + slots_tmp
            slots = slots + self.mlps(slots)

        return slots, att
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dims, num_layers, norm=nn.LayerNorm):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.cross_attns = nn.ModuleList([CrossAttentionLayer(embed_dims, 4, 32, False) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([MLP(embed_dims, embed_dims*2, embed_dims, num_layers=2) for _ in range(num_layers)])
        self.pixel_pe = PositionEmbeddingSine(embed_dims // 2, normalize=True)
        self.out_proj = nn.Sequential(norm(embed_dims), nn.Linear(embed_dims, 2 * embed_dims), norm(embed_dims*2), nn.Linear(2 * embed_dims, embed_dims))

    def forward(self, context, slots):
        h, w = context.shape[-2:]

        context = context + self.pixel_pe(context)
        context = rearrange(context, 'b d h w -> b (h w) d')

        for i in range(self.num_layers):
            context = context + self.cross_attns[i](context, slots, slots)[0]
            context = context + self.mlps[i](context)

        context = self.out_proj(context)
        context = rearrange(context, 'b (h w) d -> b d h w', h=h, w=w)

        return context