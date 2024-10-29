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
            # decoder=nn.Identity(),
            input_depth=False,
            depth_update=None,
            pc_range=None,
            num_iters=1,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone

        self.encoder = encoder
        self.head = head
        self.neck = neck
        self.decoder = Decoder(embed_dims)
        self.gs_render = GaussianRenderer(embed_dims, 1)
        self.depth_update = depth_update

    def forward(self, batch):
        b, n, _, h, w = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        context = self.backbone(self.norm(image))
        features, _, _ = self.neck(context)
        mean, uncertainty, features, opacities = self.depth_update(context[0], features, lidar2img)
        
    
        x, num_gaussians = self.gs_render(features, mean, uncertainty, opacities, [b, n])
        y = self.decoder(*x)
        output = self.head(y)
        direction_vector = None
        output['mid_output'] = {'features':features, 'mean':mean, 'uncertainty':uncertainty, 'direction_vector':direction_vector, 'opacities':opacities}
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

        self.threshold = 0.01
        self.gamma = 1.0

    def forward(self, features, means3D, uncertainty, opacities, shape):
        """
        features: (b n) i d h w
        means3D: (b n) i h w 3
        uncertainty: (b n) i h w 3 3
        opacities: (b n) i 1 h w
        """ 

        b, n = shape
        device = means3D[0].device

        multi_scale_output = []
        # self.set_render_scale(int(200 * 1), int(200 * 1))
        # self.set_Rasterizer(device)
        num_gaussians = 0
        for scale in range(len(means3D)):
            bev_out = []    
            self.set_render_scale(int(200 * (2 **(scale - len(means3D) + 1))), int(200 * (2 **(scale - len(means3D) + 1))))
            self.set_Rasterizer(device)

            cov3D_scale = uncertainty[scale]
            cov3D_scale = cov3D_scale.flatten(-2, -1)
            cov3D_scale = torch.cat((cov3D_scale[..., 0:3], cov3D_scale[..., 4:6], cov3D_scale[..., 8:9]), dim=-1)

            features_scale = rearrange(features[scale], '(b n) d h w -> b (n h w) d', b=b, n=n)
            means3D_scale = rearrange(means3D[scale], '(b n) h w d-> b (n h w) d', b=b, n=n)
            cov3D_scale = rearrange(cov3D_scale, '(b n) h w d -> b (n h w) d',b=b, n=n)
            opacities_scale = rearrange(opacities[scale], '(b n) d h w -> b (n h w) d', b=b, n=n)

            cov3D_scale = torch.clamp(cov3D_scale, min=1)
            # num_gaussians = features.shape[1]
            mask = (opacities_scale > self.threshold).squeeze(-1)

            for i in range(b):
                rendered_bev, _ = self.rasterizer(
                    means3D=means3D_scale[i][mask[i]],
                    means2D=None,
                    shs=None,  # No SHs used
                    colors_precomp=features_scale[i][mask[i]],
                    opacities=opacities_scale[i][mask[i]],
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D_scale[i][mask[i]]
                )
                bev_out.append(rendered_bev)
            multi_scale_output.append(torch.stack(bev_out, dim=0))
            num_gaussians += (mask.detach().float().sum(1)).mean().cpu()

        return multi_scale_output, num_gaussians

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
            uncertainty,  
            opacities, 
            stage,
            shape, 
            cam_index, 
            y_range, 
            x_range
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
        
        features = rearrange(features[cam_index, stage, :, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'd h w -> 1 (h w) d')
        means3D = rearrange(means3D[cam_index, stage, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w d-> 1 (h w) d')
        # uncertainty = rearrange(uncertainty[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w -> 1 (h w)')
        uncertainty = rearrange(uncertainty[cam_index, stage, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w j k-> (h w) j k')
        # direction_vector = rearrange(direction_vector[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w d-> 1 (h w) d')
        opacities = rearrange(opacities[cam_index, stage, :, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'd h w -> 1 (h w) d')
        # gaussians = self.PruneAndDenify(features, means3D, uncertainty, direction_vector, opacities)

        # cov3D = uncertainty[..., None, None] * (direction_vector.unsqueeze(-1) @ direction_vector.unsqueeze(-2))
        # e = torch.zeros_like(direction_vector)
        # e[..., 2] = 1
        # v = torch.cross(direction_vector, e, dim=-1)
        # v = F.normalize(v, dim=-1)
        # w = torch.cross(direction_vector, v, dim=-1)
        # w = F.normalize(w, dim=-1)
        # # orthogonal_uncertainty = self.get_orthogonal_variance(features, means3D, uncertainty, v)
        # # cov3D = cov3D + orthogonal_uncertainty * (v.unsqueeze(-1) @ v.unsqueeze(-2)) + self.epsilon * (w.unsqueeze(-1) @ w.unsqueeze(-2))
        # cov3D = cov3D + self.epsilon * (v.unsqueeze(-1) @ v.unsqueeze(-2) + w.unsqueeze(-1) @ w.unsqueeze(-2))
        # print(cov3D)
        cov3D = uncertainty.flatten(-2, -1)
        cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)
    
        # features = rearrange(features, '(b n) d h w -> b (n h w) d', b=b, n=n)
        # means3D = rearrange(means3D, '(b n) h w d-> b (n h w) d', b=b, n=n)
        # cov3D = rearrange(cov3D, '(b n) h w d -> b (n h w) d',b=b, n=n)
        # opacities = rearrange(opacities, '(b n) d h w -> b (n h w) d', b=b, n=n)

        num_gaussians = features.shape[1]
        # mask = (opacities > self.threshold).squeeze(-1)

        # bev_out_list = []
        # for j in range(num_iter):
        #     bev_out = []
        #     for i in range(b):
        #         rendered_bev, _ = self.rasterizer(
        #             means3D=means3D[j, i],
        #             means2D=None,
        #             shs=None,  # No SHs used
        #             colors_precomp=features[j, i],
        #             opacities=opacities[j, i],
        #             scales=None,
        #             rotations=None,
        #             cov3D_precomp=cov3D[j, i]
        #         )
        #         bev_out.append(rendered_bev)
        #     bev_out_list.append(torch.stack(bev_out, dim=0))
        # return bev_out_list

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
                cov3D_precomp=cov3D[i]
            )
            bev_out.append(rendered_bev)

        return torch.stack(bev_out, dim=0), None # orthogonal_uncertainty

class DepthUpdateHead(nn.Module):
    def __init__(self, embed_dims, context_in_dim, depth_range, depth_num, num_iter, error_tolerance):
        super().__init__()
        self.depth_range = depth_range
        self.depth_num = depth_num
        self.num_iter = num_iter

        self.project = nn.Conv2d(embed_dims, embed_dims, 3, padding=1)
        self.encoder = ProjectionInputDepth(depth_num, hidden_dim=embed_dims, out_chs=embed_dims * 2)
        self.gru = SepConvGRU(embed_dims=embed_dims, in_dim=embed_dims * 2 + context_in_dim)
        self.depth_head = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims, depth_num, kernel_size=3, padding=1),
        )
        self.opacity_head = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims, 1, kernel_size=3, padding=1),
        )
        self.error_tolerance = error_tolerance

    def get_pixel_coords_3d(self, depth, lidar2img):
        eps = 1e-5
        
        B, N = lidar2img.shape[:2]
        D, H, W = depth.shape[1:]

        img2lidar = lidar2img.inverse()

        scale = 224 // H
        coords_h = torch.linspace(scale // 2, 224 - scale//2, H, device=depth.device).float()
        coords_w = torch.linspace(scale // 2, 480 - scale//2, W, device=depth.device).float()
        coords_wh = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0) # W, H, 2
        coords_wh = repeat(coords_wh, '... -> b d ... ', b=B*N, d=D) # b d w h 2

        depth = rearrange(depth, 'b d h w -> b d w h 1')
        coords_whdhomo = torch.cat((coords_wh * depth, depth, torch.ones_like(depth)), dim=-1).unsqueeze(-1) # b d w h 4 1

        img2lidar = repeat(img2lidar, 'b n i j -> (b n) 1 1 1 i j') # b d w h 4 4
        coords3d = torch.matmul(img2lidar, coords_whdhomo).squeeze(-1)[..., :3] # b d w h 3
        coords3d = rearrange(coords3d, 'b d w h i ->  b d h w i')

        return coords3d

    def get_gaussian(self, depth_prob, depths, lidar2img):

        coords_3d = self.get_pixel_coords_3d(depths, lidar2img) # (b n) d h w 3
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3

        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale
        return pred_coords_3d, cov

    def forward(self, context, feats, lidar2img):
        
        bn, _, h, w = context.shape
        device = context.device
        depth = torch.zeros([bn, 1, h, w]).to(device)
        pred_opacities = torch.ones([bn, 1, h, w]).to(device)
        gru_hidden = torch.tanh(self.project(feats))

        means = []
        features = []
        covs = []
        opacities = []
      
        min_depth = self.depth_range[0]
        max_depth = self.depth_range[1]
        depth_range = max_depth - min_depth
        interval = depth_range / self.depth_num
        interval = interval * torch.ones_like(depth)
        interval = interval.repeat(1, self.depth_num, 1, 1)
        interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        current_depths = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        index_iter = 0

        for i in range(self.num_iter):
            input_features = self.encoder(current_depths.detach(), pred_opacities.detach())
            input_c = torch.cat([input_features, context], dim=1)

            gru_hidden = self.gru(gru_hidden, input_c)
            pred_prob = self.depth_head(gru_hidden).softmax(dim=1)
            pred_opacities = self.opacity_head(gru_hidden).sigmoid()

            mean, cov = self.get_gaussian(pred_prob, current_depths.detach(), lidar2img)
            means.append(mean)
            covs.append(cov)
            features.append(gru_hidden)
            opacities.append(pred_opacities)

            depth_r = (pred_prob * current_depths.detach()).sum(1, keepdim=True)
            depth_uncertainty_map = torch.sqrt((pred_prob * ((current_depths.detach() - depth_r.repeat(1, self.depth_num, 1, 1))**2)).sum(1, keepdim=True))
        
            index_iter = index_iter + 1

            pred_label = get_label(torch.squeeze(depth_r, 1), bin_edges, self.depth_num).unsqueeze(1)

            label_target_bin_left = pred_label
            target_bin_left = torch.gather(bin_edges, 1, label_target_bin_left)
            label_target_bin_right = (pred_label.float() + 1).long()
            target_bin_right = torch.gather(bin_edges, 1, label_target_bin_right)

            bin_edges, current_depths = update_sample(bin_edges, target_bin_left, target_bin_right, depth_r.detach(), pred_label.detach(), self.depth_num, min_depth, max_depth, depth_uncertainty_map, self.error_tolerance)

        return means, covs, features, opacities # torch.stack(means, dim=1), torch.stack(covs, dim=1), torch.stack(features, dim=1), torch.stack(opacities, dim=1)

class SepConvGRU(nn.Module):
    def __init__(self, embed_dims=128, in_dim=128+192):
        super().__init__()

        self.convz1 = nn.Conv2d(embed_dims + in_dim, embed_dims, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(embed_dims + in_dim, embed_dims, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(embed_dims + in_dim, embed_dims, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(embed_dims + in_dim, embed_dims, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(embed_dims + in_dim, embed_dims, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(embed_dims + in_dim, embed_dims, (5,1), padding=(2,0))

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

class ProjectionInputDepth(nn.Module):
    def __init__(self, depth_num, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs 
        self.convd1 = nn.Conv2d(depth_num+1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd4 = nn.Conv2d(hidden_dim, out_chs, 3, padding=1)
        
    def forward(self, depth, opacity, depth_max=61, depth_min=1):
        depth = (depth - depth_min) / depth_max
        d = torch.cat((depth, opacity), dim=1)
        d = F.relu(self.convd1(d))
        d = F.relu(self.convd2(d))
        d = F.relu(self.convd3(d))
        d = F.relu(self.convd4(d))
                
        return d

class Decoder(nn.Module):
    def __init__(self, embed_dims):
        super(Decoder, self).__init__()
        
        self.project1 = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1)
        self.project2 = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1)
        self.project3 = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1)
        self.project4 = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1)

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(embed_dims * 2, embed_dims * 1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embed_dims * 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims * 1, embed_dims * 1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embed_dims * 1),
            nn.ReLU(inplace=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(embed_dims * 2, embed_dims * 1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embed_dims * 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims * 1, embed_dims * 1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embed_dims * 1),
            nn.ReLU(inplace=True)
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(embed_dims),
            # nn.ReLU(inplace=True)
        )
        
        # Final convolution to reduce the channel count after aggregation
        # self.out_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3, x4):
        """
        :param feat_1_8: BEV feature map at 1/8 resolution
        :param feat_1_4: BEV feature map at 1/4 resolution
        :param feat_1_2: BEV feature map at 1/2 resolution
        :param feat_1_1: BEV feature map at full resolution (1/1)
        :return: Aggregated feature map at 1/1 resolution
        """

        x = self.project1(x1)
        x2 = self.project2(x2)
        x3 = self.project3(x3)
        x4 = self.project4(x4)
        # Upsample from 1/8 to 1/4 and concatenate
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)  # Concatenate along the channel dimension
        x = self.up_conv1(x)  # Apply convolution after concatenation

        # Upsample from 1/4 to 1/2 and concatenate
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)

        # Upsample from 1/2 to 1/1 and concatenate
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv3(x)

        # # Final convolution
        # x = self.final_conv(x)

        return x

def update_sample(bin_edges, target_bin_left, target_bin_right, depth_r, pred_label, depth_num, min_depth, max_depth, uncertainty_range, error_tolerance):
    
    with torch.no_grad():    
        b, _, h, w = bin_edges.shape

        mode = 'direct'
        if mode == 'direct':
            depth_range = uncertainty_range
            depth_start_update = torch.clamp_min(depth_r - error_tolerance * depth_range, min_depth)
            # depth_end_update = torch.clamp_min(depth_r + 1.5 * depth_range, max_depth)
        else:
            depth_range = uncertainty_range + (target_bin_right - target_bin_left).abs()
            depth_start_update = torch.clamp_min(target_bin_left - error_tolerance * uncertainty_range, min_depth)

        interval = (depth_range * error_tolerance * 2) / depth_num
        interval = interval.repeat(1, depth_num, 1, 1)
        interval = torch.cat([torch.ones([b, 1, h, w], device=bin_edges.device) * depth_start_update, interval], 1)

        bin_edges = torch.cumsum(interval, 1).clamp(min_depth, max_depth)
        curr_depth = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

    return bin_edges.detach(), curr_depth.detach()

def get_label(gt_depth_img, bin_edges, depth_num):

    with torch.no_grad():
        gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device)
        for i in range(depth_num):
            bin_mask = torch.ge(gt_depth_img, bin_edges[:, i])
            bin_mask = torch.logical_and(bin_mask, 
                torch.lt(gt_depth_img, bin_edges[:, i + 1]))
            gt_label[bin_mask] = i
        
        return gt_label
    
