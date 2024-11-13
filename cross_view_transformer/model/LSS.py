import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Normalize
from einops import rearrange, repeat

class LSS(nn.Module):
    def __init__(
            self,
            embed_dims,
            backbone,
            head,
            neck,
            grid_conf,
            decoder=nn.Identity(),
            depth_num=64,            
            img_h=224,
            img_w=480,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone

        self.head = head
        self.neck = neck
        self.decoder = decoder

        self.depth_num = depth_num
        self.depth_start = 1
        self.depth_max = 61
        
        # LSS
        self.grid_conf = grid_conf
        self.img_h = img_h
        self.img_w = img_w

        dx, bx, nx = gen_dx_bx(
            self.grid_conf['xbound'],
            self.grid_conf['ybound'],
            self.grid_conf['zbound'],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 8
        self.camC = 64
        self.frustum = self.create_frustum()
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.img_h, self.img_w
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)
    
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        return final
    
    def get_geometry(self, lidar2img):
        points = self.frustum
        eps = 1e-5
        
        b, n = lidar2img.shape[:2]
        d, h, w = points.shape[:-1]
        
        points = repeat(points, '... -> b n ...', b=b, n=n) # b n d h w 3
        points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
        points[..., :2] = points[..., :2] * torch.maximum(points[..., 2:3], torch.ones_like(points[..., 2:3])*eps)
        points = points.unsqueeze(-1) # b n d h w 4 1
        img2lidars = lidar2img.inverse() # b n 4 4

        img2lidars = img2lidars.view(b, n, 1, 1, 1, 4, 4).repeat(1, 1, d, h, w, 1, 1)
        points = torch.matmul(img2lidars, points).squeeze(-1)[..., :3] # b n d h w 3
        
        return points
    
    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        
        lidar2img = batch['lidar2img']
        features = self.backbone(self.norm(image))
        features, depth, _ = self.neck(features)
        features = depth.softmax(1).unsqueeze(1) * features.unsqueeze(2)
        features = rearrange(features, '(b n) c d h w -> b n d h w c', b=b, n=n)
        
        geom = self.get_geometry(lidar2img)
        x = self.voxel_pooling(geom, features)
        y = self.decoder(x)
        output = self.head(y)
        
        return output
        
def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None