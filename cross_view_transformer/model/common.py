import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import math


def generate_grid(height: int, width: int, z: int = 0):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    if z > 0 :
        zs = torch.linspace(0, 1, z)
        indices = torch.stack(torch.meshgrid((xs, ys, zs),indexing='xy'), 0)   
    else:
        indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
        indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0,offset=0.0, z=0, z_meters=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters
    if z > 0 :
        sz = z / z_meters
        return [
            [ 0., -sw,          0,          w/2.],
            [-sh,  0.,          0, h*offset+h/2.],
            [ 0.,  0.,          sz,         z/2.],
            [ 0.,  0.,          0,            1.]
        ]
    else:
        return [
            [ 0., -sw,          w/2.],
            [-sh,  0., h*offset+h/2.],
            [ 0.,  0.,            1.]
        ]

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    V = get_view_matrix(flip=True)
    V_inv = torch.FloatTensor(V).inverse()
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_w = F.pad(pos_w, (0, 2,0,0), value=0)
    pos_w = V_inv @ pos_w.T
    pos_w = pos_w[0,:].unsqueeze(1)

    pos_h = torch.arange(0., height).unsqueeze(1)
    pos_h = F.pad(pos_h, (1, 0,0,0), value=0)
    pos_h = F.pad(pos_h, (0, 1,0,0), value=0)
    pos_h = V_inv @ pos_h.T
    pos_h = pos_h[1,:].unsqueeze(1)
    
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))
    
class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
        bev_z: int = 0,
        z_meters: int = 0,
        pre_grid_view: int = 0,
        no_query: bool = False,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))
        z = bev_z // (2 ** len(decoder_blocks))
        # bev coordinates
        grid = generate_grid(h, w, z).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]
        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset,z=bev_z, z_meters=z_meters)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        if z != 0:
            grid = F.pad(grid,(0, 0, 0, 0, 0, 0, 0, 1), value=1) 
            grid = V_inv @ rearrange(grid, 'd h w z-> d (h w z)')                   # 4 (h w z)
            grid = rearrange(grid, 'd (h w z) -> d h w z', h=h, w=w, z=z)           # 4 h w z
            grid = grid[:3]                                                         # 3 h w z
        else:
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w

        if not no_query:
            self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w
            if z != 0:
                self.side_features = nn.Parameter(sigma * torch.randn(dim, h, z))
                self.front_features = nn.Parameter(sigma * torch.randn(dim, w, z))

        if pre_grid_view > 0:
            if pre_grid_view == 2:
                grid_view = torch.load('/media/user/data/nuscenes/trainval/grid_view_index.pt').to(float)
            else:
                grid_view = torch.load('/media/user/data/nuscenes/trainval/grid_view.pt').to(float)
            grid_view = grid_view.permute(2,0,1).unsqueeze(0)
            grid_view = torch.nn.functional.max_pool2d(grid_view,2 ** len(decoder_blocks))[0].permute(1,2,0).to(bool)
            self.register_buffer('grid_view', grid_view, persistent=False)
        else:
            self.grid_view = None
        
        if pre_grid_view == 2:
            h, w = self.grid_view.shape[:2]
            grid_view_index = torch.zeros((h,w))-1
            for i in range(h):
                for j in range(w):
                    tmp = (self.grid_view[i,j] == 1).nonzero()
                    if tmp.shape[0] == 1:
                        grid_view_index[i,j] = tmp[0]
            grid_view_index = grid_view_index.reshape(-1)
            self.register_buffer('grid_view_index', grid_view_index, persistent=False)
        else:
            self.grid_view_index = None

    def get_prior(self, s='bev'):
        if s == 'bev':
            return self.learned_features
        elif s == 'side':
            return self.side_features 
        elif s == 'front':
            return self.front_features