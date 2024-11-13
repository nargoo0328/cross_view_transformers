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
from .sht import rsh_cart_8
# from .simple_bev import SimpleBEVDecoderLayer_pixel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class GaussianLSS(nn.Module):
    def __init__(
            self,
            embed_dims,
            backbone,
            head,
            neck,
            encoder=None,
            decoder=nn.Identity(),
            pc_range=None,
            error_tolerance=1.0,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone

        self.encoder = encoder
        self.head = head
        self.neck = neck
        self.decoder = decoder

        self.depth_num = 64
        self.depth_start = 1
        self.depth_max = 61
        self.pc_range = pc_range
        self.LID = False
        self.gs_render = GaussianRenderer(embed_dims, 1)

        self.camera_embed = MLP(81, embed_dims, embed_dims)
        # self.feats_fusion = MLPConv2D(embed_dims*2, embed_dims)
        # self.depth_head = MLPConv2D(embed_dims, self.depth_num)
        # self.opacity_head = MLPConv2D(embed_dims, 1)
        self.error_tolerance = error_tolerance
    
    def get_coordinates3D(self, extrinsics, angles, depth):
        # b, n, c, h, w = depth.shape
        coords_3d = get_pixel_coords_3d(extrinsics, angles) # b n w h d 3
        coords_3d = rearrange(coords_3d, 'b h w d c -> b d h w c')
        pred_coords_3d = (depth.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3

        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale

        return pred_coords_3d, cov, coords_3d

    def get_camera_embedding(self, intrinsics, img_h, img_w, feats_h, feats_w):
        rays = generate_rays(intrinsics.flatten(0,1), (img_h, img_w))
        rays = rearrange(rays, 'b (h w) d -> b d h w', h=img_h, w=img_w)
        rays = F.normalize(F.interpolate(rays, size=(feats_h, feats_w), mode='bilinear', align_corners=False, antialias=True), dim=1)
        rays = rearrange(rays, 'b d h w -> b h w d')

        theta = torch.atan2(rays[..., 0], rays[..., 1])
        phi = torch.acos(rays[..., 2])
        # pitch = torch.asin(ray_directions[..., 1])
        # roll = torch.atan2(ray_directions[..., 0], - ray_directions[..., 1])
        angles = torch.stack([theta, phi], dim=-1)

        rays_embed = rsh_cart_8(rays)
        camera_embedding = self.camera_embed(rays_embed)
        camera_embedding = rearrange(camera_embedding, 'b h w d -> b d h w')
        return camera_embedding, angles

    def forward(self, batch):
        b, n, _, h, w = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']

        features = self.backbone(self.norm(image))
        feats_h, feats_w = features[0].shape[-2:]
        camera_embedding, angles = self.get_camera_embedding(batch['intrinsics'], h, w, feats_h, feats_w) # (b n) 81 h w

        features, pred_depth, pred_opacity = self.neck(features, camera_embedding)

        means3D, cov3D, coord3d = self.get_coordinates3D(batch['extrinsics'], angles, pred_depth.softmax(1))
        cov3D = cov3D.flatten(-2, -1)
        cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        features = rearrange(features, '(b n) d h w -> b (n h w) d', b=b, n=n)
        means3D = rearrange(means3D, '(b n) h w d-> b (n h w) d', b=b, n=n)
        cov3D = rearrange(cov3D, '(b n) h w d -> b (n h w) d',b=b, n=n)
        pred_opacity = rearrange(pred_opacity, '(b n) d h w -> b (n h w) d', b=b, n=n)
        
        x, num_gaussians = self.gs_render(features, means3D, cov3D, pred_opacity)

        y = self.decoder(x)
        output = self.head(y)

        output['mid_output'] = {
            'features':features, 
            'mean': means3D, 
            'uncertainty':cov3D, 
            'opacities':pred_opacity, 
            # 'coord3d':coord3d, 
        }
        output['num_gaussians'] = num_gaussians
        return output
    
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
        self.camera_center = torch.tensor([0, 0, 0], dtype=torch.float32)  # High above Z-axis

        # Orthographic projection matrix for BEV
        self.set_transform()
    
    def set_transform(self, h=200, w=200, h_meters=100, w_meters=100):
        """ Set up an orthographic projection matrix for BEV. """
        # Create an orthographic projection matrix
        sh = h / h_meters
        sw = w / w_meters
        self.world_view_transform = torch.tensor([
            [ 0.,  sh,  0.,         0.],
            [ sw,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
        ], dtype=torch.float32)

        self.full_proj_transform = torch.tensor([
            [ 0., -sh,  0.,          h/2.],
            [-sw,   0.,  0.,         w/2.],
            [ 0.,  0.,  0.,           1.],
            [ 0.,  0.,  0.,           1.],
        ], dtype=torch.float32)

    def set_size(self, h, w):
        self.image_height = h
        self.image_width = w

class GaussianRenderer(nn.Module):
    def __init__(self, embed_dims, scaling_modifier):
        super().__init__()
        self.viewpoint_camera = BEVCamera()
        self.rasterizer = GaussianRasterizer()
        self.embed_dims = embed_dims
        self.scaling_modifier = scaling_modifier
        self.epsilon = 1e-4

        self.threshold = 0.05

    def forward(self, features, means3D, cov3D, opacities):#, orth_features, orth_means):
        """
        features: (b n) d h w
        means3D: (b n) h w 3
        uncertainty: (b n) h w
        direction_vector: (b n) h w 3
        opacities: (b n) h w 1
        """ 
        b = features.shape[0]
        device = means3D.device
        mask = (opacities > self.threshold)
        # opacities = mask.float()
        mask = mask.squeeze(-1)

        bev_out = []
        self.set_render_scale(int(200), int(200))
        self.set_Rasterizer(device)
        for i in range(b):
            rendered_bev, _ = self.rasterizer(
                means3D=means3D[i][mask[i]],
                means2D=None,
                shs=None,  # No SHs used
                colors_precomp=features[i][mask[i]],
                opacities=opacities[i][mask[i]],
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D[i][mask[i]]
            )
            bev_out.append(rendered_bev)

        return torch.stack(bev_out, dim=0), (mask.detach().float().sum(1)).mean().cpu()

    @torch.no_grad()
    def set_Rasterizer(self, device):
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        bg_color = torch.zeros((self.embed_dims)).to(device)
        # bg_color[-1] = -4
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

    @torch.no_grad()
    def set_render_scale(self, h, w):
        self.viewpoint_camera.set_size(h, w)
        self.viewpoint_camera.set_transform(h, w)

    @torch.no_grad()
    def forward_part_gaussians(self, 
            features, 
            means3D, 
            cov3D, 
            # direction_vector, 
            opacities, 
            shape, 
            cam_index, 
            y_range=None, 
            x_range=None, 
            # orth_features, 
            # orth_means
        ):
        
        """
        features: (b n) d h w
        means3D: (b n) h w 3
        uncertainty: (b n) h w
        direction_vector: (b n) h w 3
        opacities: (b n) h w 1
        """ 

        b, n = shape
        device = means3D.device
        self.set_Rasterizer(device)
        
        if y_range is not None:
            features = rearrange(features[cam_index, :, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'd h w -> 1 (h w) d')
            means3D = rearrange(means3D[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w d-> 1 (h w) d')
            # uncertainty = rearrange(uncertainty[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w -> 1 (h w)')
            cov3D = rearrange(cov3D[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w d-> 1 (h w) d')
            # direction_vector = rearrange(direction_vector[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w d-> 1 (h w) d')
            opacities = rearrange(opacities[cam_index, :, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'd h w -> 1 (h w) d')
        else:
            mask = (opacities[cam_index] > self.threshold).view(1, -1)
            features = rearrange(features[cam_index], 'd h w -> 1 (h w) d')[mask][None]
            means3D = rearrange(means3D[cam_index], 'h w d -> 1 (h w) d')[mask][None]
            cov3D = rearrange(cov3D[cam_index], 'h w d-> 1 (h w) d')[mask][None]
            opacities = rearrange(opacities[cam_index], 'd h w -> 1 (h w) d')[mask][None]

        bev_out = []
        for i in range(b):
            rendered_bev, _ = self.rasterizer(
                means3D=means3D[i],
                means2D=None,
                shs=None,  # No SHs used
                colors_precomp=features[i],
                opacities=opacities[i],
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D[i],
            )
            bev_out.append(rendered_bev)

        return torch.stack(bev_out, dim=0), None # orthogonal_uncertainty
    
class MLPConv2D(nn.Module):
    def __init__(self, embed_dims, out_dims):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
            nn.InstanceNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
            nn.InstanceNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, out_dims, kernel_size=1, padding=0)
        )
    
    def forward(self, x):
        return self.layer(x)
    
def generate_rays(
    camera_intrinsics, image_shape, noisy = False
):
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (3, H*W)
    ray_directions = F.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)
    return ray_directions

def get_pixel_coords_3d(extrinsics, angles, depth_num=64, depth_start=1, depth_max=61):
    """
    angles: (b n) h w 2
    """
    device = angles.device
    b, h, w = angles.shape[:-1]

    index  = torch.arange(start=0, end=depth_num, step=1, device=device).float()
    bin_size = (depth_max - depth_start) / depth_num
    d = depth_start + bin_size * index
    d = repeat(d, 'd -> b h w d', b=b, h=h, w=w)
    theta = angles[..., 0:1]  # Extract polar angle
    phi = angles[..., 1:2]  # Extract azimuthal angle
    y = d * torch.sin(phi) * torch.cos(theta)
    x = d * torch.sin(phi) * torch.sin(theta)
    z = d * torch.cos(phi)
    cam_3d = torch.stack([x, y, z, torch.ones_like(x)], dim=-1).unsqueeze(-1) # (b n) h w d 4 1

    extrinsics_inv = extrinsics.inverse().flatten(0,1) # (b n) 4 4
    extrinsics_inv = repeat(extrinsics_inv, 'b i j -> b h w d i j', h=h, w=w, d=depth_num)   
    coords3d = (extrinsics_inv @ cam_3d).squeeze(-1)[..., :3] #

    return coords3d