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
from .simple_bev import SimpleBEVDecoderLayer_pixel

class SparseBEVSeg(nn.Module):
    def __init__(
            self,
            backbone,
            encoder,
            neck=None,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone
        self.neck = neck
        self.encoder = encoder

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = self.backbone(self.norm(image))

        if self.neck is not None:
            features, depth = self.neck(features)
        
        features = [rearrange(y,'(b n) ... -> b n ...', b=b,n=n) for y in features]
        bev_pred_list, mid_outputs = self.encoder(features, lidar2img)
        output = {}
        output.update(bev_pred_list[-1])
        output['mid_output'] = mid_outputs
        output['aux'] = bev_pred_list[:-1]

        return output
    
class BasicBEVUpdateEncoder(nn.Module):
    """
    Predict bbox -> project to BEV -> sparse_conv2d/conv2d
    """
    def __init__(self,
                head,
                sampling,
                bev_decoder,
                embed_dims=128,
                num_iterations=1,
                seg_dims=1,
                **kwargs):
        
        super().__init__()

        self.embed_dims = embed_dims
        self.num_iterations = num_iterations
        self.seg_dims = seg_dims
        self._init_bev_layers(**kwargs)
        self.sampling = sampling
        self.update_block = BasicUpdateBlockBEV(head, embed_dims, self.scale, seg_dims)
        self.bev_decoder = bev_decoder

    def _init_bev_layers(self, h=200, w=200, bev_h=200, bev_w=200, Z=8, num_points_in_pillar=8):
        
        # xs = torch.linspace(w / bev_w / 2, w - w / bev_w / 2, bev_w
        #                     ).flip(0).view(1, bev_w).expand(bev_h, bev_w) / w
        # ys = torch.linspace(h / bev_h / 2, h - h / bev_h / 2, bev_h
        #                     ).flip(0).view(bev_h, 1).expand(bev_h, bev_w) / h
        # ref_3d = torch.stack((ys, xs), -1)
        # ref_3d = torch.cat([ref_3d, torch.zeros((bev_h, bev_w, 1)) + 0.5], dim=-1)
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar
                                    ).view(num_points_in_pillar, 1, 1).expand(num_points_in_pillar, bev_h, bev_w) / Z
        xs = torch.linspace(0.5, w - 0.5, w
                            ).flip(0).view(1, 1, bev_w).expand(num_points_in_pillar, bev_h, bev_w) / w
        ys = torch.linspace(0.5, h - 0.5, h
                            ).flip(0).view(1, bev_h, 1).expand(num_points_in_pillar, bev_h, bev_w) / h
        ref_3d = torch.stack((ys, xs, zs), -1)

        self.register_buffer('grid', ref_3d, persistent=False) # z h w 3

        self.h = bev_h
        self.w = bev_w
        self.scale = h // bev_h
        # self.init_bev_feats = nn.Embedding(bev_h * bev_w, self.embed_dims) 
    
    def upsample_bev(self, bev_pred, mask):
        """ Upsample flow field [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, d, H, W = bev_pred.shape
        mask = mask.view(N, 1, 9, self.scale, self.scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_bev = F.unfold(self.scale * bev_pred, [3,3], padding=1)
        up_bev = up_bev.view(N, d, 9, 1, 1, H, W)

        up_bev = torch.sum(mask * up_bev, dim=2)
        up_bev = up_bev.permute(0, 1, 4, 2, 5, 3)
        
        return up_bev.reshape(N, d, self.scale*H, self.scale*W)

    def forward(self, mlvl_feats, lidar2img):
        
        bs = lidar2img.shape[0]
        device = lidar2img.device
        bev_pos = repeat(self.grid, '... -> b ...', b=bs)
        bev_pred = torch.zeros((bs, self.seg_dims, self.h, self.w)).to(device)
        # h = torch.zeros((bs, self.embed_dims, self.h, self.w)).to(device)
        bev_pred = {'VEHICLE': bev_pred[:, 0:1], 'center': bev_pred[:, 1:2], 'offset': bev_pred[:, 2:]}

        for lvl, feat in enumerate(mlvl_feats):
            GC = feat.shape[2]
            C = GC // 1
            feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=1,c=C)

            mlvl_feats[lvl] = feat.contiguous()

        bev_pred_list = []
        mid_outputs = []
        sampled_feats, mid_output = self.sampling(mlvl_feats, bev_pos, lidar2img, self.scale, mode='pillar')
        h = self.bev_decoder(sampled_feats.sum(2))
        mid_outputs.append(mid_output)

        for i in range(self.num_iterations):
            # sampled_feats, mid_output = self.sampling(h, mlvl_feats, bev_pos, lidar2img, self.scale)
            # for k in bev_pred:
            #     bev_pred[k] = bev_pred[k].detach()
            h, bev_pred_iter, weights = self.update_block(h, sampled_feats, bev_pred, i)
            for k in bev_pred_iter:
                bev_pred[k] = bev_pred[k] + bev_pred_iter[k]    

            bev_pred_out = {k: v.clone() for k, v in bev_pred.items()}
            # bev_pred_up = {}
            # for k, v in bev_pred_dict.items():
            #     bev_pred_up[k] = self.upsample_bev(v, mask)
            mid_outputs.append(weights)
            bev_pred_list.append(bev_pred_out)

        return bev_pred_list, mid_outputs

class BasicUpdateBlockBEV(nn.Module):
    def __init__(self, head, embed_dims, up_scale, seg_dims):
        super().__init__()
                
        self.encoder = BEVContextEncoder(embed_dims, seg_dims)
        self.gru = SepConvGRU(embed_dims, embed_dims)
        self.head = head
        # self.mask = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims * 2, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(embed_dims * 2, (up_scale ** 2) * 9, 1, padding=0)
        # )
        # self.points_weight = nn.Linear(embed_dims, 8)

    def forward(self, h, sampled_feats, bev_pred, iter):
        # dict to tensor
        bev_pred_tmp = []
        for _, v in bev_pred.items():
            bev_pred_tmp.append(v)
        bev_pred = torch.cat(bev_pred_tmp, dim=1)

        bev_feats, cosine_similarity = self.encoder(sampled_feats, bev_pred, iter)
        h = self.gru(h, bev_feats)
        bev_pred = self.head(h)
        # mask = self.mask(h)
        return h, bev_pred, cosine_similarity #, mask

class BEVContextEncoder(nn.Module):
    def __init__(self, embed_dims, seg_dims):
        super().__init__()
        self.conv_feats = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims * 2, embed_dims, 1),
            nn.GELU(),
        )
        self.conv_pred = nn.Sequential(
            nn.Conv2d(seg_dims, embed_dims, 1),
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.GELU(),
        )
        self.conv = nn.Conv2d(embed_dims * 2, embed_dims - seg_dims, 3, padding=1)

    def forward(self, sampled_feats, bev_pred, iter):

        tmp_bev_pred = self.conv_pred(bev_pred)

        if iter > 0:
            weights = F.cosine_similarity(tmp_bev_pred.unsqueeze(2), sampled_feats, dim=1).unsqueeze(1) # b 1 p h w
            sampled_feats = sampled_feats * weights # b d p h w * b 1 p h w
        else:
            weights = None
        sampled_feats = sampled_feats.sum(2)
        # sampled_feats = rearrange(sampled_feats, 'b (h w) d -> b d h w', h=200, w=200)

        tmp_sampled_feats = self.conv_feats(sampled_feats)

        bev = torch.cat([tmp_sampled_feats, tmp_bev_pred], dim=1)
        out = self.conv(bev)
        return torch.cat([out, bev_pred], dim=1), weights

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128, kernel_size=3):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,kernel_size), padding=(0,kernel_size//2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,kernel_size), padding=(0,kernel_size//2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,kernel_size), padding=(0,kernel_size//2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (kernel_size,1), padding=(kernel_size//2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (kernel_size,1), padding=(kernel_size//2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (kernel_size,1), padding=(kernel_size//2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1))) 
        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class BEVSampling(nn.Module):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims, num_points=8, pc_range=[], img_h=0, img_w=0, eps=1e-6):
        super().__init__()

        self.num_points = num_points
        self.img_h = img_h
        self.img_w = img_w
        self.pc_range = pc_range
        # self.sampling_offset = nn.Conv2d(embed_dims, num_points * 3, 1)
        self.eps = eps
        self.pos_encoder = PositionalEncodingMap(in_c=3, out_c=embed_dims, mid_c=embed_dims * 2)
        self.num_groups = 1
        # self.points_conv = nn.Sequential(
        #         nn.Conv2d(num_points * embed_dims, embed_dims * 4, 1),
        #         nn.GELU(),
        #         nn.Conv2d(embed_dims * 4, embed_dims * 4, 1),
        #         nn.GELU(),
        #         nn.Conv2d(embed_dims * 4, embed_dims, 1),
        # )
        # self.init_weights()

    def init_weights(self):
        # original
        bias = self.sampling_offset.bias.data.view(self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:2], -4.0, 4.0)
        height = torch.linspace(-0.5, 1.5, self.num_points).unsqueeze(1)
        bias[:, 2:3] = height

    def forward(self, mlvl_feats, reference_points, lidar2img, scale=1.0, mode='grid'):
        b, p, h, w, _ = reference_points.shape
        device = reference_points.device
        # 2d sampling offset 
        if mode == 'grid': 
            # sampling_offset = self.sampling_offset(query).sigmoid() # b (g p 3) h w
            # sampling_offset = rearrange(sampling_offset, 'b (g p d) h w -> b (h w) g p d',
            #                     g=self.num_groups,
            #                     p=self.num_points,
            #                     d=3
            #                 )
            # sampling_offset_new = sampling_offset.clone()
            # sampling_offset_new[..., :2] = (sampling_offset_new[..., :2] * (0.25 * scale + self.eps) * 2) \
            #                             - (0.25 * scale + self.eps)
            # sampling_offset_new[..., 2:3] = (sampling_offset_new[..., 2:3] * (4.0 + self.eps) * 2) \
            #                             - (4.0 + self.eps)
            # sampling_offset = sampling_offset_new

            sampling_offset_xy = torch.rand((b, h*w, 1, self.num_points, 2), device=device) * 2 - 1
            sampling_offset_z = torch.rand((b, h*w, 1, self.num_points, 1), device=device) * 8 - 4
            sampling_offset = torch.cat((sampling_offset_xy, sampling_offset_z), dim=-1)
            reference_points_new = rearrange(reference_points, 'b h w d -> b (h w) 1 1 d', d=3).clone()

            reference_points_new[..., 0:1] = (reference_points_new[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            reference_points_new[..., 1:2] = (reference_points_new[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            reference_points_new[..., 2:3] = (reference_points_new[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            reference_points = reference_points_new
            
            reference_points = reference_points + sampling_offset
            
        # 3d sampling offset 
        elif mode == 'pillar':

            # sampling_offset = self.sampling_offset(query) # b (g p 3) h w
            # sampling_offset = rearrange(sampling_offset, 'b (g p d) h w -> b (h w) g p d',
            #                     g=self.num_groups,
            #                     p=self.num_points,
            #                     d=2
            #                 )
            # sampling_offset[..., :2] = (sampling_offset[..., :2] * (0.25 * scale + self.eps) * 2) \
            #                             - (0.25 * scale + self.eps)
            # sampling_offset[..., 2:3] = (sampling_offset[..., 2:3] * (0.5 + self.eps) * 2) \
            #                             - (0.5 + self.eps)
            
            reference_points = rearrange(reference_points, 'b p h w d -> b (h w) 1 p d').clone()
            pos_embed = self.pos_encoder(reference_points)
            reference_points[..., 0:1] = (reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            reference_points[..., 1:2] = (reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            reference_points[..., 2:3] = (reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # reference_points = reference_points + sampling_offset
            # reference_points = rearrange(reference_points, 'b q g p1 p2 d -> b q g (p1 p2) d')

        scale_weights = None  

        # sampling
        sampled_feats, pos_3d, sample_points_cam = sampling_4d(
            reference_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            self.img_h, self.img_w,
            # pixel_positional_embedding=self.pixel_positional_embedding
            # sampling_offset,
        )  # [B, Q, G, P, C]
        pos_3d = None
        weight = None
            
        mid_output = {}
        mid_output.update({'sample_points_cam': sample_points_cam, 'pos_3d': pos_3d, 'reference_points': reference_points.clone(), 'weight': weight})

        # reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        # reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        # reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2]) 
        sampled_feats = sampled_feats + pos_embed # self.pos_encoder(reference_points)
        sampled_feats = rearrange(sampled_feats.squeeze(2), 'b (h w) p d -> b d p h w', h=h, w=w)
        # sampled_feats = rearrange(sampled_feats, 'b (h w) g p d -> b g (p d) h w', h=h, w=w).squeeze(1) # b p d h w
        # sampled_feats = self.points_conv(sampled_feats)
        return sampled_feats, mid_output 