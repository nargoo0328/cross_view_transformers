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
from .decoder import BEVDecoder, DecoderBlock, UpsamplingAdd
# from .simple_bev import SimpleBEVDecoderLayer_pixel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from typing import Optional
from torchvision.models.resnet import BasicBlock

class GaussianLSS(nn.Module):
    def __init__(
            self,
            embed_dims,
            backbone,
            head,
            neck,
            # slot_attention,
            # cross_attention,
            encoder=None,
            decoder=nn.Identity(),
            input_depth=False,
            depth_update=None,
            pc_range=None,
            num_stages=1,
            error_tolerance=1.0,
            orth_scale=0.05,
            depth_num=64,
            opacity_filter=0.05,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone

        self.head = head
        self.neck = neck
        self.decoder = decoder
        self.input_depth = input_depth

        self.depth_num = depth_num
        self.depth_start = 1
        self.depth_max = 61
        self.pc_range = pc_range
        self.LID = False
        self.gs_render = GaussianRenderer(embed_dims, num_stages, opacity_filter)
        self.bev_refine = encoder
    
        self.error_tolerance = error_tolerance
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(embed_dims*2, embed_dims, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=1),
        # )
        # self.depth_embed = nn.Sequential(*[
        #     nn.Conv2d(depth_num, depth_num, 1),
        #     nn.BatchNorm2d(depth_num),
        #     nn.ReLU(),
        #     nn.Conv2d(depth_num, embed_dims, 1),
        # ])
        # self.se_layer = SELayer(embed_dims)
    
    def pred_depth(self, lidar2img, depth, coords_3d=None):
        # b, n, c, h, w = depth.shape
        if coords_3d is None:
            coords_3d, coords_d = get_pixel_coords_3d(depth, lidar2img, depth_num=self.depth_num) # b n w h d 3
            coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
            
        direction_vector = F.normalize((coords_3d[:, 1] - coords_3d[:, 0]), dim=-1) # (b n) h w c

        # depth = rearrange(depth, 'b n ... -> (b n) ...')
        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3
        
        # error_tolerance = self.error_tolerance * 2 if self.training else self.error_tolerance
            

        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale

        return pred_coords_3d, cov

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        # intrinsics = batch['intrinsics'].inverse()
        # extrinsics = batch['extrinsics'].inverse()
        
        lidar2img = batch['lidar2img']
        features = self.backbone(self.norm(image))
        
        # coords_3d, coords_d = get_pixel_coords_3d(features[0], lidar2img)
        # coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
        # direction_vector = F.normalize(coords_3d[:, 0], dim=-1)
        # direction_vector_embedding = self.ray_embed(direction_vector)
        # direction_vector_embedding = rearrange(direction_vector_embedding, 'b h w d -> b d h w')
        
        features, depth, opacities = self.neck(features)
        # depth_embed = self.depth_embed(depth)
        # bev = self.bev_refine([features], lidar2img)
        # features, depth, opacities = self.neck(features, direction_vector_embedding)
        means3D, cov3D = self.pred_depth(lidar2img, depth)

        cov3D = cov3D.flatten(-2, -1)
        cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        features = rearrange(features, '(b n) d h w -> b (n h w) d', b=b, n=n)
        # features = rearrange(features, '(b n) l d -> b (n l) d', b=b, n=n)
        # features_3d = rearrange(features_3d, '(b n) l d -> b (n l) d', b=b, n=n)

        # means3D = rearrange(means3D, '(b n) l d-> b (n l) d', b=b, n=n)
        means3D = rearrange(means3D, '(b n) h w d-> b (n h w) d', b=b, n=n)

        cov3D = rearrange(cov3D, '(b n) h w d -> b (n h w) d',b=b, n=n)
        opacities = rearrange(opacities, '(b n) d h w -> b (n h w) d', b=b, n=n)
        
        x, num_gaussians = self.gs_render(features, means3D, cov3D, opacities)
        # x = self.bev_refine(x, [features], lidar2img)
        # x = self.se_layer(x, bev)
        # x = self.fusion(torch.cat((x, bev), dim=1))
        # x = x + bev

        # 50 -> 100 -> 200
        # x = multi_scale[0]        
        # for i in range(len(multi_scale)-1):
        #     x = self.decoder_block[i](x, multi_scale[i+1])

        # stage_outputs = {'stage': []}
        # for bev in x:
        #     bev = self.decoder(bev)
        #     output = self.head(bev)
        #     stage_outputs['stage'].append(output)
        #     stage_outputs.update(output)

        y = self.decoder(x)
        # y = self.decoder(x, from_dense=True)
        output = self.head(y)

        # mask = x[:, 0:1] != 0
        # for k in output:
        #     if isinstance(output[k], spconv.SparseConvTensor):
        #         output[k] = output[k].dense()
        # output['mask'] = mask

        # output['VEHICLE'] += short_cut
        output['mid_output'] = {
            'features':features, 
            'mean': means3D, 
            'uncertainty':cov3D, 
            'opacities':opacities, 
            # 'depth':depth, 
            # 'sampling_points_cam_out':sampling_points_cam_out,
            # 'sampling_points': sampling_points,
            # 'direction_vector': direction_vector,
            # 'orth_features': orth_features
            # 'attn': attn,
            # 'means': means,
            # 'covs': covs,
            # 'x':x,
            # 'x_':x_,
            # 'x_3d':x_3d,
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
    def __init__(self, embed_dims, num_stages, threshold=0.05):
        super().__init__()
        self.viewpoint_camera = BEVCamera()
        self.rasterizer = GaussianRasterizer()
        self.embed_dims = embed_dims
        self.epsilon = 1e-4

        self.threshold = threshold
        self.gamma = 1.0
        self.num_stages = num_stages

        h = w = 200 // 2 ** (num_stages-1)
        self.h = h
        self.w = w
        # self.conv = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=5, padding=2, bias=False),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=1),
        # )
        self.bev_embedding =  nn.Embedding(h * w, self.embed_dims)
        self.conv1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            ) for i in range(num_stages)
            ]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Sequential(
                    BasicBlock(embed_dims, embed_dims),
                    BasicBlock(embed_dims, embed_dims),
            ) for i in range(num_stages)
            ]
        )
        self.decoder = nn.ModuleList(
            [UpsamplingAdd(embed_dims, embed_dims) for i in range(num_stages-1)]
        )
        self.pc_range = [-50, -50, -4, 50, 50, 4]

    def forward(self, features, means3D, cov3D, opacities):#, orth_features, orth_means):
        """
        features: b G d*stages
        means3D: b G 3
        uncertainty: b G 6
        opacities: b G 1*stages
        """ 
        b = features.shape[0]
        device = means3D.device
        
        # filter pc range
        mask_pos = \
                (means3D[:, :, 0] >= self.pc_range[0]) & (means3D[:, :, 0] <= self.pc_range[3]) & \
                (means3D[:, :, 1] >= self.pc_range[1]) & (means3D[:, :, 1] <= self.pc_range[4]) & \
                (means3D[:, :, 2] >= self.pc_range[2]) & (means3D[:, :, 2] <= self.pc_range[5])
        # mask = (opacities > self.threshold)
        # mask = mask.squeeze(-1)
        # mask = mask | mask_pos
        # bev_out = []
        # mask = (opacities > self.threshold)
        # mask = mask.squeeze(-1)
        # self.set_render_scale(200, 200)
        # self.set_Rasterizer(device)
        # for i in range(b):
        #     rendered_bev, _ = self.rasterizer(
        #         means3D=means3D[i][mask[i]],
        #         means2D=None,
        #         shs=None,  # No SHs used
        #         colors_precomp=features[i][mask[i]],
        #         opacities=opacities[i][mask[i]],
        #         scales=None,
        #         rotations=None,
        #         cov3D_precomp=cov3D[i][mask[i]]
        #     )
        #     bev_out.append(rendered_bev)
            
        bev_emdding = repeat(self.bev_embedding.weight, '... -> b ...', b=b)
        bev_emdding = rearrange(bev_emdding, 'b (h w) d -> b d h w', h=self.h, w=self.w)
        # features = torch.split(features, self.embed_dims, dim=-1)
        num_gaussians = 0.0
        multi_scale_bev = []
        for stage in range(self.num_stages):
            mask = (opacities[..., stage:stage+1] > self.threshold)
            # mask = (opacities > self.threshold)
            mask = mask.squeeze(-1)
            mask = mask & mask_pos

            bev_out = []
            self.set_render_scale(self.h * 2 ** stage, self.w * 2 ** stage)
            self.set_Rasterizer(device)
            for i in range(b):
                rendered_bev, _ = self.rasterizer(
                    means3D=means3D[i][mask[i]],
                    means2D=None,
                    shs=None,  # No SHs used
                    # colors_precomp=features[stage][i][mask[i]],
                    colors_precomp=features[i][mask[i]],
                    opacities=opacities[..., stage:stage+1][i][mask[i]],
                    # opacities=opacities[i][mask[i]],
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D[i][mask[i]]
                )
                bev_out.append(rendered_bev)

            multi_scale_bev.append(torch.stack(bev_out, dim=0))
            num_gaussians += (mask.detach().float().sum(1)).mean().cpu()

        x = bev_emdding
        for i, bev in enumerate(multi_scale_bev):
            bev = self.conv1[i](bev)
            if i-1 >=0:
                x = self.decoder[i-1](x, bev)
            else:
                x = x + bev
            x = self.conv2[i](x)

        return x, num_gaussians
        
        # bev = torch.stack(bev_out, dim=0)
        # bev_emdding = repeat(self.bev_embedding.weight, '... -> b ...', b=b)
        # bev_emdding = rearrange(bev_emdding, 'b (h w) d -> b d h w', h=self.h, w=self.w)
        # bev = bev + bev_emdding
        # bev_mask = bev[:, 0:1] == 0
        # mask_embedding = repeat(self.mask_embedding.weight, '1 d -> b d h w', b=b, h=200, w=200)
        # mask_embedding = mask_embedding * bev_mask
        # bev = bev + mask_embedding
        # bev = self.conv(bev) 
        # return bev, (mask.detach().float().sum(1)).mean().cpu()


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
            scale_modifier=1,
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

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class GaussianUpdate(nn.Module):
    def __init__(self, embed_dims, num_points, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.threshold = 0.05
        self.update = GaussianUpdateLayer(embed_dims, num_points)

    def forward(self, img_features, means3D, features, cov3D, direction_vector, opacities, lidar2img):
        # b, n = lidar2img.shape[:2]
        # features = rearrange(features, '(b n) d h w -> b n (h w) d', b=b, n=n)
        # means3D = rearrange(means3D, '(b n) h w d-> b n (h w) d', b=b, n=n)
        # cov3D = rearrange(cov3D, '(b n) h w d -> b n (h w) d',b=b, n=n)
        # opacities = rearrange(opacities, '(b n) d h w -> b n (h w) d', b=b, n=n)

        # mask = (opacities > self.threshold).squeeze(-1)

        # # aggregate input tensors
        # mask_features = []
        # mask_means3D = []
        # mask_cov3D = []
        
        # for i in range(b):
        #     mask_features_view = []
        #     mask_means3D_view = []
        #     mask_cov3D_view = []
        #     for j in range(n):
        #         mask_features_view.append(features[i, j][mask[i, j]])
        #         mask_means3D_view.append(means3D[i, j][mask[i, j]])
        #         mask_cov3D_view.append(cov3D[i, j][mask[i, j]])

        b, n = lidar2img.shape[:2]
        features = rearrange(features, '(b n) d h w -> b n (h w) d', b=b, n=n)
        means3D = rearrange(means3D, '(b n) h w d-> b n (h w) d', b=b, n=n)
        cov3D = rearrange(cov3D, '(b n) h w j k -> b n (h w) j k', b=b, n=n)
        direction_vector = rearrange(direction_vector, '(b n) h w d ->  b n (h w) d', b=b, n=n)

        sampling_points_cam_list = []
        gaussians_list = []
        for i in range(self.num_layers):
            features, means3D, cov3D, sampling_points_cam = self.update(img_features, means3D, features, cov3D, direction_vector, lidar2img)
            sampling_points_cam_list.append(sampling_points_cam)
            gaussians_list.append([features, means3D, cov3D])

        
        features = features.flatten(1,2)
        means3D = means3D.flatten(1,2)
        cov3D = cov3D.flatten(1,2)

        return features, means3D, cov3D, sampling_points_cam_list, gaussians_list
    
class GaussianUpdateLayer(nn.Module):
    def __init__(self, embed_dims, num_points):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_points = num_points
        self.pc_range = [-50.0, -50.0, -10.0, 50.0, 50.0, 10.0]
        self.aggregate = nn.Sequential(
            nn.Linear(embed_dims * num_points, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims)
        )
        self.update_mean = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 3),
        )
        self.update_variance = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 1),
        )
        # self.update_rotation = nn.Sequential(
        #     nn.Linear(embed_dims, embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(embed_dims, 4),
        # )
        
    def sampling(self, img_features, sampling_points, lidar2img):
        """
        sampling_offsets: b n l 3 p
        lidar2img: b n 4 4
        """
        b, n = sampling_points.shape[:2]
        sampling_points = rearrange(sampling_points, 'b n l d p -> b n l p d')
        lidar2img = rearrange(lidar2img, 'b n i j -> b n 1 1 i j')

        # expand
        ones = torch.ones_like(sampling_points[..., :1])
        sampling_points = torch.cat([sampling_points, ones], dim=-1).unsqueeze(-1)  # b n l p 4 1

        # project 2d
        sampling_points_cam = torch.matmul(lidar2img, sampling_points).squeeze(-1) # b n l p 4
        homo = sampling_points_cam[..., 2:3]
        homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + 1e-6)
        sampling_points_cam = sampling_points_cam[..., 0:2] / homo_nonzero  # b n l p 2
        sampling_points_cam_out = sampling_points_cam.clone()

        # normalize
        sampling_points_cam[..., 0] /= 480
        sampling_points_cam[..., 1] /= 224
        sampling_points_cam = sampling_points_cam * 2 - 1

        sampling_points_cam = sampling_points_cam.flatten(0, 1) # (b n) l p 2
        sampled_features = F.grid_sample(img_features, sampling_points_cam, mode='bilinear', align_corners=False)

        return rearrange(sampled_features, '(b n) d l p -> b n l (p d)', b=b, n=n), sampling_points_cam_out

    def update_mean3D(self, mean, delta):

        # normalize 0~1
        mean = mean.clone()
        mean[..., 0:1] = (mean[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        mean[..., 1:2] = (mean[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        mean[..., 2:3] = (mean[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        mean = torch.clamp(mean, min=0.0, max=1.0)

        mean = inverse_sigmoid(mean)

        mean = (mean + delta).sigmoid().clone()

        # unnormalized
        mean[..., 0:1] = (mean[..., 0:1] * (self.pc_range[3] - self.pc_range[0])) + self.pc_range[0]
        mean[..., 1:2] = (mean[..., 1:2] * (self.pc_range[4] - self.pc_range[1])) + self.pc_range[1]
        mean[..., 2:3] = (mean[..., 2:3] * (self.pc_range[5] - self.pc_range[2])) + self.pc_range[2]

        return mean

    def forward(self, img_features, means3D, features, cov3D, direction_vector, lidar2img):    
        b, n, l, _ = means3D.shape    
        device = means3D.device

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            means3D, covariance_matrix=cov3D)
        sampling_points = distribution.sample([self.num_points])
        sampling_points = sampling_points.permute(1,2,3,4,0)

        # sampling_offsets = torch.linspace(-3, 3, self.num_points, device=device) # num_points
        # sampling_offsets = repeat(sampling_offsets, '... -> b n l 1 ...', b=b, n=n, l=l)

        # sigma = (cov3D[..., [0, 3, 5]]).sqrt().unsqueeze(-1) # b n l 3 1
        # sampling_offsets = sigma * sampling_offsets # b n l 3 1 * b n l 1 p -> b n l 3 p

        # sampling_points = sampling_offsets + means3D.unsqueeze(-1)

        sampled_features, sampling_points_cam = self.sampling(img_features, sampling_points, lidar2img)

        # aggregate
        sampled_features = self.aggregate(sampled_features) # b n l (p d) -> b n l d

        # # update gaussian
        features = features + sampled_features
        means3D = self.update_mean3D(means3D, self.update_mean(sampled_features))
        var_weight = self.update_variance(sampled_features).sigmoid().unsqueeze(-1)
        e = torch.zeros_like(direction_vector)
        e[..., 2] = 1
        v = torch.cross(direction_vector, e, dim=-1)
        v = F.normalize(v, dim=-1)
        cov3D = cov3D * var_weight + v.unsqueeze(-1) @ v.unsqueeze(-2) * (1 - var_weight)
        # scales_delta = self.update_scale(sampled_features).sigmoid()
        # rotation_delta = F.normalize(self.update_rotation(sampled_features), dim=-1)
        # cov3D = cov3D + compute_covariance_matrix_batch(rotation_delta, scales_delta)

        return features, means3D, cov3D, sampling_points_cam

class MLPConv2D(nn.Module):
    def __init__(self, embed_dims, out_dims):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(embed_dims, out_dims, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size=1, padding=0)
        )
    
    def forward(self, x):
        return self.layer(x)
    
class GaussianNeck(nn.Module):
    def __init__(self, in_dims, scale, embed_dims, num_scales=2):
        super().__init__()

        self.scale_projs = nn.ModuleList(
            [nn.Sequential(
                    nn.Conv2d(dim, embed_dims, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(embed_dims),
                    nn.GELU(),
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
            ) 
                for dim in in_dims]
        )
        self.proj = nn.Linear(embed_dims * num_scales, embed_dims)
        self.pos_embed = PositionEmbeddingSine(embed_dims//2, normalize=True)
        self.ray_embed = PositionalEncodingMap(in_c=3, out_c=embed_dims, mid_c=embed_dims * 2)

        self.self_attn = AttentionBlock(embed_dims, 4, 0.1)
        self.cross_attn = AttentionBlock(embed_dims, 4, 0.1)
        
        self.feats_head = MLPConv2D(embed_dims, embed_dims)
        self.depth_head = MLPConv2D(embed_dims, 64)
        self.opacity_head = MLPConv2D(embed_dims, 1)

        self.error_tolerance = 0.5
        self.scale = scale

    def get_means_cov(self, depth_prob, coords_3d):
        coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')

        means3D = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3

        delta_3d = means3D.unsqueeze(1) - coords_3d
        cov3D = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov3D = cov3D * scale

        return means3D, cov3D

    def forward(self, feats, lidar2img):
        b, _, h, w = feats[0].shape
        device = feats[0].device

        coords3d, _ = get_pixel_coords_3d(feats[0], lidar2img) # B N W H D 3
        feats = [self.scale_projs[i](F.interpolate(feats[i], scale_factor=self.scale[i])) for i in range(len(feats))]
        feats = [rearrange(f, 'b d h w -> b (h w) d') for f in feats]

        pos_embed = self.pos_embed(
            torch.zeros(
                (b,1,h,w),
                device=device,
                requires_grad=False,
            )
        )
        pos_embed = rearrange(pos_embed, 'b d h w -> b (h w) d') 
        pos_embed = pos_embed.repeat(1, len(feats), 1) # b (h w l) d
        features_tokens = torch.cat(feats, dim=1) # b (h w l) d
        features_channels = torch.cat(feats, dim=-1) # b (h w) (d l)
        features = self.proj(features_channels)
        features = self.self_attn(features, features_tokens, features_tokens, k_pos_embed=pos_embed)

        rays = F.normalize(coords3d[..., 0, :], dim=-1)
        rays = rearrange(rays, 'b n w h d -> (b n) (h w) d')
        rays_embed = self.ray_embed(rays) # b (h w) d

        features = self.cross_attn(features, rays_embed, rays_embed) # b (h w) d
        features = rearrange(features, 'b (h w) d -> b d h w', h=h, w=w)

        out_features = self.feats_head(features)
        depth = self.depth_head(features).softmax(1)
        opacity = self.opacity_head(features).sigmoid()

        means3D, cov3D = self.get_means_cov(depth, coords3d)

        return out_features, means3D, cov3D, opacity
    
class AttentionBlock(nn.Module):
    def __init__(self, embed_dims, num_heads, dropout):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.mlp = MLP(embed_dims, embed_dims*4, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        
    def forward(self, src, k, v, q_pos_embed=None, k_pos_embed=None):
        if q_pos_embed is not None:
            q = src + q_pos_embed
        else:
            q = src

        if k_pos_embed is not None:
            k = k + k_pos_embed
        src2 = self.attn(q, k, v)[0]
        src = src + src2
        src = self.norm1(src)
        src = src + self.mlp(src)
        src = self.norm2(src)

        return src

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
    
class CameraAwareEmbedding(nn.Module):
    def __init__(self, img_h, img_w, feat_height, feat_width, embed_dims):
        super().__init__()    
        
        scale = img_h // feat_height
        coords_h = torch.linspace(scale // 2, img_h - scale//2, feat_height).float()
        coords_w = torch.linspace(scale // 2, img_w - scale//2, feat_width).float()
        image_plane = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(0, 2, 1) # 2 h w
        image_plane = torch.cat((image_plane, torch.ones_like(image_plane[0:1])), dim=0)[None, None] # 1 1 3 h w
        self.register_buffer('image_plane', image_plane, persistent=False)
        
        self.img_embed = nn.Conv2d(4, embed_dims, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, embed_dims, 1, bias=False)
        
    def forward(self, I_inv, E_inv):
        pixel = self.image_plane
        h, w = pixel.shape[-2:]
        
        c = E_inv[..., -1:]                                                     # b n 4 1
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
        
        return img_embed

def generate_rays(
    camera_intrinsics, image_shape, feat_shape
):
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    feat_h, feat_w = feat_shape
    scale = height // feat_h
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(scale // 2, width - scale//2, feat_w, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(scale // 2, height - scale//2, feat_h, device=device, dtype=dtype)

    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)

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

@torch.no_grad()
def get_pixel_coords_3d(depth, lidar2img, img_h=224, img_w=480, depth_num=64, depth_start=1, depth_max=61):
    eps = 1e-5
    
    B, N = lidar2img.shape[:2]
    H, W = depth.shape[-2:]
    scale = img_h // H
    # coords_h = torch.linspace(scale // 2, img_h - scale//2, H, device=depth.device).float()
    # coords_w = torch.linspace(scale // 2, img_w - scale//2, W, device=depth.device).float()
    coords_h = torch.linspace(0, 1, H, device=depth.device).float() * img_h
    coords_w = torch.linspace(0, 1, W, device=depth.device).float() * img_w
    coords_d = get_bin_centers(depth_max, depth_start, depth_num).to(depth.device)

    D = coords_d.shape[0]
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
    img2lidars = lidar2img.inverse() # b n 4 4

    coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
    img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
    coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H D 3

    return coords3d, coords_d

@torch.no_grad()
def get_bin_centers(max_depth, min_depth, depth_num):
    """
    depth: b d h w
    """
    depth_range = max_depth - min_depth
    interval = depth_range / depth_num
    interval = interval * torch.ones((depth_num+1))
    interval[0] = min_depth
    # interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

    bin_edges = torch.cumsum(interval, 0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

# class GaussianRendererOCC(nn.Module):
#     def __init__(self, embed_dims):
#         super().__init__()
#         self.rasterizer = LocalAggregator(1, 200, 200, 8, [-50.0, -50.0, -4.0], torch.tensor([0.5, 0.5, 1.0]))
#         self.embed_dims = embed_dims
#         self.epsilon = 1e-4
#         self.pc_range = [-50, -50, -4.0, 50, 50, 4.0]
#         self.xyz = self.get_meshgrid(self.pc_range, [200, 200, 8], [0.5, 0.5, 1])
#         self.flat = MLPConv2D(embed_dims*8, embed_dims)

#     def get_meshgrid(self, ranges, grid, reso):
#         xxx = torch.arange(grid[0], dtype=torch.float) * reso[0] + 0.5 * reso[0] + ranges[0]
#         yyy = torch.arange(grid[1], dtype=torch.float) * reso[1] + 0.5 * reso[1] + ranges[1]
#         zzz = torch.arange(grid[2], dtype=torch.float) * reso[2] + 0.5 * reso[2] + ranges[2]

#         xxx = xxx[:, None, None].expand(*grid)
#         yyy = yyy[None, :, None].expand(*grid)
#         zzz = zzz[None, None, :].expand(*grid)

#         xyz = torch.stack([
#             xxx, yyy, zzz
#         ], dim=-1)
#         return xyz # x, y, z, 3

#     def forward(self, features, means3D, cov3D, opacities):#, orth_features, orth_means):
#         """
#         features: b G d*stages
#         means3D: b G 3
#         uncertainty: b G 6
#         opacities: b G 1*stages
#         """ 
#         b = features.shape[0]
#         device = means3D.device
#         opacities = opacities.squeeze(-1)
        
#         sampled_xyz = self.xyz.flatten(0, 2).to(device)
#         scales = torch.cat((cov3D[..., 0:1], cov3D[..., 3:4], cov3D[..., 5:6]), dim=-1)
#         bev_out = []
#         for i in range(b):
            
#             mask = \
#                 (means3D[i, :, 0] >= self.pc_range[0]) & (means3D[i, :, 0] < self.pc_range[3]) & \
#                 (means3D[i, :, 1] >= self.pc_range[1]) & (means3D[i, :, 1] < self.pc_range[4]) & \
#                 (means3D[i, :, 2] >= self.pc_range[2]) & (means3D[i, :, 2] < self.pc_range[5])

#             rendered = self.rasterizer(
#                 sampled_xyz.clone().float(), 
#                 means3D[i][mask], 
#                 opacities[i][mask].squeeze(-1),
#                 features[i][mask],
#                 scales[i][mask],
#                 cov3D[i][mask]
#             )
#             # print("rendered", rendered.min(), rendered.max(), rendered.mean())
#             rendered = torch.nan_to_num(rendered, posinf=0.0, neginf=0.0)
#             rendered = rendered.reshape(200, 200, 8, 128).permute(2,3,1,0).flatten(0,1).flip(1).flip(2)
#             bev_out.append(rendered)
#         bev = torch.stack(bev_out, dim=0)
#         print(bev.min(), bev.max(), bev.mean())
#         bev = self.flat(bev)
#         return bev, 1 # orthogonal_uncertainty