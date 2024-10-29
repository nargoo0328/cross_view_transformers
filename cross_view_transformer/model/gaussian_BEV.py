import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint as cp

from .common import Normalize
from einops import rearrange, repeat
import copy
# from .checkpoint import checkpoint as cp

import math
from .sparsebev import sampling_4d, inverse_sigmoid
from .PointBEV_gridsample import PositionalEncodingMap, MLP
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

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

class GaussianBEV(nn.Module):
    def __init__(
            self,
            backbone,
            gaussian_encoder,
            head,
            bev_decoder=nn.Identity(),
            neck=None,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone
        self.neck = neck
        self.gaussian_encoder = gaussian_encoder
        self.bev_decoder = bev_decoder
        self.head = head

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = self.backbone(self.norm(image))

        if self.neck is not None:
            features, _ = self.neck(features)
        
        features = [rearrange(y,'(b n) ... -> b n ...', b=b,n=n) for y in features]
        bev_preds, sampled_points, out_gaussians = self.gaussian_encoder(features, lidar2img)
        preds = []
        for bev_pred in bev_preds:
            bev_pred = self.bev_decoder(bev_pred)
            pred = self.head(bev_pred)
            preds.append(pred)

        output = {}
        output.update(preds[-1])
        output['aux'] = preds[:-1]
        output['sampled_points'] = sampled_points
        output['out_gaussians'] = out_gaussians

        return output
    
class GaussianEncoder(nn.Module):
    def __init__(self,
                gaussian_layer,
                gaussians_renderer,
                embed_dims=128,
                num_iterations=6,
                **kwargs):
        
        super().__init__()

        self.gaussian_layer = gaussian_layer
        self.embed_dims = embed_dims
        self.num_iterations = num_iterations
        self.gaussian_head = MLP(128, 256, 11-3)
        self.gaussians_renderer = gaussians_renderer
        
        self.init_gaussians(**kwargs)

    def init_gaussians(self, bev_h, bev_w, z, gaussian_h, gaussian_w):
        offset = bev_h // gaussian_h / 2
        xs = torch.linspace(offset, bev_w - offset, gaussian_w
                            ).flip(0).view(1, gaussian_w).expand(gaussian_h, gaussian_w) / bev_w
        ys = torch.linspace(offset, bev_h - offset, gaussian_h
                            ).flip(0).view(gaussian_h, 1).expand(gaussian_h, gaussian_w) / bev_h
        xyz = torch.stack((ys, xs), -1)
        xyz = torch.cat([xyz, torch.zeros((gaussian_h, gaussian_w, 1)) + 0.5], dim=-1).flatten(0,1)

        scale = torch.ones_like(xyz)
        scale = inverse_sigmoid(scale)

        rots = torch.zeros(gaussian_h * gaussian_w, 4, dtype=torch.float)
        rots[:, 0] = 1
        opacity = inverse_sigmoid(0.1 * torch.ones((gaussian_h * gaussian_w, 1), dtype=torch.float))
        self.gaussians = nn.Parameter(torch.cat([xyz, scale, rots, opacity], dim=1)) # xyz, scale, rots, opacity, 3+3+4+1
        self.gaussians_feat = nn.Parameter(torch.zeros((gaussian_h * gaussian_w, self.embed_dims))) # xyz, scale, rots, opacity, embed
        self.gaussians.register_hook(self.zero_grad_hook)

    def zero_grad_hook(self, grad):
        # Zero out the gradients for the first N elements
        grad[..., 0:3] = 0 # z
        # grad[..., 5] = 0 # z_scale
        return grad

    def refine_gaussians(self, gaussians, delta_gaussians):
        # xyz = inverse_sigmoid(gaussians[..., 0:3])
        # xyz_delta = delta_gaussians[..., 0:3]
        # xyz_new = torch.sigmoid(xyz_delta + xyz)
        # xy = inverse_sigmoid(gaussians[..., 0:2])
        # xy_delta = delta_gaussians[..., 0:2]
        # xy_new = torch.sigmoid(xy_delta + xy)

        # return torch.cat([xyz_new, delta_gaussians[..., 3:]], dim=-1)
        # return torch.cat([xy_new, gaussians[..., 2:3], delta_gaussians[..., 2:4], gaussians[..., 5:6], delta_gaussians[..., 4:]], dim=-1)

        return torch.cat([gaussians[..., 0:3], delta_gaussians], dim=-1)

    def forward(self, mlvl_feats, lidar2img):
        b = lidar2img.shape[0]

        G = 1
        for lvl, feat in enumerate(mlvl_feats):
            GC = feat.shape[2]
            C = GC // G
            feat = rearrange(feat, 'b n (g c) h w -> (b g) n h w c',g=G,c=C)

            mlvl_feats[lvl] = feat.contiguous()

        gaussians = repeat(self.gaussians, '... -> b ...', b=b)
        gaussians_feat = repeat(self.gaussians_feat, '... -> b ...', b=b)
        
        out_bev_feats = []
        out_sampled_points = []
        out_gaussians = [gaussians]
        for i in range(self.num_iterations):
            # if gaussians.device
            # print("Level:", i, gaussians[0,10:15,:3])
            gaussians_feat, sampled_points = self.gaussian_layer(mlvl_feats, lidar2img, gaussians, gaussians_feat)
            bev_feats = self.gaussians_renderer(gaussians, gaussians_feat)
            delta_gaussians = self.gaussian_head(gaussians_feat)
            gaussians = self.refine_gaussians(gaussians, delta_gaussians)
            out_bev_feats.append(bev_feats.clone())
            out_sampled_points.append(sampled_points)
            out_gaussians.append(gaussians)

        return out_bev_feats, out_sampled_points, out_gaussians

class GaussianEncoderLayer(nn.Module):
    def __init__(self, embed_dims, num_points, img_h, img_w, pc_range=[], scale_range=[]):
        super().__init__()

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.pos_encoder = PositionalEncodingMap(in_c=2, out_c=embed_dims, mid_c=embed_dims * 2)
        # self.self_attn = GaussianSelfAttention(embed_dims, num_heads=4, dropout=0.1, pc_range=pc_range)
        self.cross_attn = GaussianCrossAttention(embed_dims, num_points, img_h, img_w, pc_range=pc_range, scale_range=scale_range)
        # self.ffn = MLP(embed_dims, embed_dims * 2, embed_dims)
        
        # self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        # self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, mlvl_feats, lidar2img, gaussians, gaussians_feat):
        gaussians_pos_embed = self.pos_encoder(gaussians[..., :2])
        gaussians_feat = gaussians_feat + gaussians_pos_embed

        # gaussians_feat = self.norm1(self.self_attn(gaussians, gaussians_feat))

        sampled_feat, sampled_points = self.cross_attn(mlvl_feats, lidar2img, gaussians)
        gaussians_feat = gaussians_feat + sampled_feat.sum(2)
        gaussians_feat = self.norm2(gaussians_feat)

        return gaussians_feat, sampled_points

class GaussianSelfAttention(nn.Module):
    def __init__(self, embed_dims, num_heads=8, dropout=0.1, pc_range=[]):
        super().__init__()
        self.pc_range = pc_range
        self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def forward(self, gaussians, gaussians_feat):
        """
        gaussians: [B, G, 11]
        gaussians_feat: [B, G, C]
        """

        dist = self.calc_bbox_dists(gaussians[..., :2])
        tau = self.gen_tau(gaussians_feat)  # [B, Q, num_heads]

        tau = tau.permute(0, 2, 1)  # [B, num_heads, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, num_heads, Q, K]
        attn_mask = attn_mask.flatten(0, 1)  # [B x num_heads, Q, K]
        gaussians_feat = self.attention(
            query=gaussians_feat,
            key=gaussians_feat,
            value=gaussians_feat,
            attn_mask=attn_mask,
        )[0]
        return gaussians_feat

    @torch.no_grad()
    def calc_bbox_dists(self, xyz):
        bs = xyz.shape[0]
        xy = xyz.clone()
        xy[..., 0:1] = (xy[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        xy[..., 1:2] = (xy[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        dist = []
        for b in range(bs):
            dist_b = torch.norm(xy[b].reshape(-1, 1, 2) - xy[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist
    
class GaussianCrossAttention(nn.Module):
    def __init__(self, embed_dims, num_sampled_points, img_h, img_w, pc_range=[], scale_range=[]):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_sampled_points = num_sampled_points
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.img_h = img_h
        self.img_w = img_w

    def sampled_points(self, gaussians):

        mean = gaussians[..., :3].clone()
        mean[..., 0:1] = (mean[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        mean[..., 1:2] = (mean[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        mean[..., 2:3] = (mean[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        scales = gaussians[..., 3:6].sigmoid().clone()
        scales[..., 0:1] = (scales[..., 0:1] * (self.scale_range[3] - self.scale_range[0]) + self.scale_range[0])
        scales[..., 1:2] = (scales[..., 1:2] * (self.scale_range[4] - self.scale_range[1]) + self.scale_range[1])
        scales[..., 2:3] = (scales[..., 2:3] * (self.scale_range[5] - self.scale_range[2]) + self.scale_range[2])

        rotations = F.normalize(gaussians[..., 6:10], dim=-1)
        covariance = compute_covariance_matrix_batch(rotations, scales)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            mean, covariance_matrix=covariance)
        sampled_points = distribution.sample((self.num_sampled_points,)) # p b g 3
        sampled_points = rearrange(sampled_points, 'p b g d -> b g p d')
        return sampled_points

    def forward(self, mlvl_feats, lidar2img, gaussians):
        sampled_points = self.sampled_points(gaussians).unsqueeze(2)
        scale_weights = None
        # print(sampled_points[0])
        sampled_feats, pos_3d, sample_points_cam = sampling_4d(
            sampled_points,
            mlvl_feats,
            scale_weights,
            lidar2img,
            self.img_h, self.img_w
        )  # [B, G, P, C]

        # sampled_points[..., 0:1] = (sampled_points[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        # sampled_points[..., 1:2] = (sampled_points[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        # sampled_points[..., 2:3] = (sampled_points[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2]) 
        # sampled_feats = sampled_feats + self.pos_encoder(sampled_points)

        return sampled_feats.squeeze(2), sampled_points

class BEVCamera:
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), image_size=200):
        # Orthographic projection parameters
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.image_width = image_size
        self.image_height = image_size

        # Set up FoV to cover the range [-50, 50] for both X and Y
        self.FoVx = (self.x_max - self.x_min)  # Width of the scene in world coordinates
        self.FoVy = (self.y_max - self.y_min)  # Height of the scene in world coordinates

        # Camera position: placed above the scene, looking down along Z-axis
        self.camera_center = torch.tensor([0, 0, 100], dtype=torch.float32)  # High above Z-axis
        
        # View matrix: Identity matrix (no rotation, just top-down view)
        self.world_view_transform = torch.tensor([
            [ 0.,  2.,  0.,         0.],
            [ 2.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
        ], dtype=torch.float32)

        # Orthographic projection matrix for BEV
        self.full_proj_transform = self.orthographic_projection()
    
    def orthographic_projection(self):
        """ Set up an orthographic projection matrix for BEV. """
        # Create an orthographic projection matrix
        proj_matrix = torch.tensor([
            [ 0., -2.,  0.,         100.],
            [-2,   0.,  0.,         100.],
            [ 0.,  0.,  0.,           1.],
            [ 0.,  0.,  0.,           1.],
        ], dtype=torch.float32)
        return proj_matrix
    
    # def to(self, device):
    #     self.camera_center = self.camera_center.to(device)
    #     self.world_view_transform = self.world_view_transform.to(device)
    #     self.full_proj_transform = self.full_proj_transform.to(device)

class GaussianRenderer:
    def __init__(self, embed_dims, scaling_modifier, pc_range, scale_range):
        self.viewpoint_camera = BEVCamera()
        self.rasterizer = GaussianRasterizer()
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.embed_dims = embed_dims
        self.scaling_modifier = scaling_modifier
    
    def __call__(self, gaussians, gaussians_feat):
        b = gaussians.shape[0]
        device = gaussians.device
        self.set_Rasterizer(device)

        means3D = gaussians[..., :3].clone()
        means3D[..., 0:1] = (means3D[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        means3D[..., 1:2] = (means3D[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        means3D[..., 2:3] = (means3D[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        scales = gaussians[..., 3:6].sigmoid().clone()
        scales[..., 0:1] = (scales[..., 0:1] * (self.scale_range[3] - self.scale_range[0]) + self.scale_range[0])
        scales[..., 1:2] = (scales[..., 1:2] * (self.scale_range[4] - self.scale_range[1]) + self.scale_range[1])
        scales[..., 2:3] = (scales[..., 2:3] * (self.scale_range[5] - self.scale_range[2]) + self.scale_range[2])

        rotations = F.normalize(gaussians[..., 6:10], dim=-1)
        opacities = gaussians[..., 10:11].sigmoid()
        
        bev_out = []
        for i in range(b):
            rendered_bev, _ = self.rasterizer(
                means3D=means3D[i],
                means2D=None,
                shs=None,  # No SHs used
                colors_precomp=gaussians_feat[i],
                opacities=opacities[i],
                scales=scales[i],
                rotations=rotations[i],
                cov3D_precomp=None
            )
            bev_out.append(rendered_bev)

        return torch.stack(bev_out, dim=0)

    def set_Rasterizer(self, device):
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        bg_color = torch.zeros((self.embed_dims)).to(device)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.image_height),
            image_width=int(self.viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=self.viewpoint_camera.world_view_transform.to(device),
            projmatrix=self.viewpoint_camera.full_proj_transform.to(device),
            sh_degree=0,  # No SHs used for random Gaussians
            campos=self.viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=False
        )
        self.rasterizer.set_raster_settings(raster_settings)

def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]

def quaternion_to_rotation_matrix_batch(quaternions):
    """
    Converts a batch of quaternions to a batch of 3x3 rotation matrices.
    quaternions: Tensor of shape (b, G, 4) representing the quaternion (q_w, q_x, q_y, q_z)
    
    Output: Tensor of shape (b, G, 3, 3) representing the rotation matrices
    """
    q_w, q_x, q_y, q_z = torch.split(quaternions, 1, dim=-1)
    q_w = quaternions[..., 0]
    q_x = quaternions[..., 1]
    q_y = quaternions[..., 2]
    q_z = quaternions[..., 3]

    # Rotation matrix elements
    R = torch.zeros(quaternions.shape[:-1] + (3, 3), device=quaternions.device)
    
    R[..., 0, 0] = 1 - 2 * (q_y ** 2 + q_z ** 2)
    R[..., 0, 1] = 2 * (q_x * q_y - q_z * q_w)
    R[..., 0, 2] = 2 * (q_x * q_z + q_y * q_w)

    R[..., 1, 0] = 2 * (q_x * q_y + q_z * q_w)
    R[..., 1, 1] = 1 - 2 * (q_x ** 2 + q_z ** 2)
    R[..., 1, 2] = 2 * (q_y * q_z - q_x * q_w)

    R[..., 2, 0] = 2 * (q_x * q_z - q_y * q_w)
    R[..., 2, 1] = 2 * (q_y * q_z + q_x * q_w)
    R[..., 2, 2] = 1 - 2 * (q_x ** 2 + q_y ** 2)
    
    return R

def compute_covariance_matrix_batch(quaternions, scales):
    """
    Computes a batch of covariance matrices from quaternions and scales.
    quaternions: Tensor of shape (b, G, 4) representing the quaternions (q_w, q_x, q_y, q_z)
    scales: Tensor of shape (b, G, 3) representing the scale (variance) along x, y, z axes
    
    Output: Tensor of shape (b, G, 3, 3) representing the covariance matrices
    """
    # Convert quaternion to a batch of rotation matrices
    R = quaternion_to_rotation_matrix_batch(quaternions)
    
    # Create a diagonal scale matrix for each Gaussian
    S = torch.zeros(scales.shape[:-1] + (3, 3), device=scales.device)
    S[..., 0, 0] = scales[..., 0]  # Scale for x
    S[..., 1, 1] = scales[..., 1]  # Scale for y
    S[..., 2, 2] = scales[..., 2]  # Scale for z

    # Compute the covariance matrix: R * S * R^T
    # Use batch matrix multiplication: bmm for batched matrices
    L = R @ S 
    covariance_matrices = L @ L.transpose(-1, -2) # R S ST RT
    return covariance_matrices