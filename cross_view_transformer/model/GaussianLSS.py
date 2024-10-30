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
            decoder=nn.Identity(),
            input_depth=False,
            depth_update=None,
            pc_range=None,
            num_iters=1,
            error_tolerance=1.0,
            orth_scale=0.05,
    ):
        super().__init__()
    
        self.norm = Normalize()
        self.backbone = backbone

        self.encoder = encoder
        self.head = head
        self.neck = neck
        self.decoder = decoder
        self.input_depth = input_depth

        self.depth_num = 64
        self.depth_start = 1
        self.depth_max = 61
        self.pc_range = pc_range
        self.LID = True
        self.gs_render = GaussianRenderer(embed_dims, 1)

        self.error_tolerance = error_tolerance
        self.scale = 2
        # self.orth_layer = GaussianOrthLayer(embed_dims, 2, extent=orth_scale)
        self.orth_scale = orth_scale
        
        # self.feats_layers = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=1, padding=0)
        # )
        # self.depth_layers = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims, self.depth_num, kernel_size=1, padding=0)
        # )
        # self.opacity_layers = nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
        #     nn.InstanceNorm2d(embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims, 1, kernel_size=1, padding=0)
        # )
        # self.bev_conv = nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1)

    def get_pixel_coords_3d(self, depth, lidar2img):
        eps = 1e-5
        
        B, N = lidar2img.shape[:2]
        H, W = depth.shape[2:]
        scale = 224 // H
        # coords_h = torch.arange(H, device=depth.device).float() * 224 / H
        # coords_w = torch.arange(W, device=depth.device).float() * 480 / W
        coords_h = torch.linspace(scale // 2, 224 - scale//2, H, device=depth.device).float()
        coords_w = torch.linspace(scale // 2, 480 - scale//2, W, device=depth.device).float()
        # coords_h = torch.linspace(0, 1, H, device=depth.device).float() * 224
        # coords_w = torch.linspace(0, 1, W, device=depth.device).float() * 480

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=depth.device).float()
            index_1 = index + 1
            bin_size = (self.depth_max - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=depth.device).float()
            bin_size = (self.depth_max - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        img2lidars = lidar2img.inverse() # b n 4 4

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # B N W H D 3

        return coords3d, coords_d
    
    def pred_depth(self, lidar2img, depth):
        b, n, c, h, w = depth.shape
        coords_3d, coords_d = self.get_pixel_coords_3d(depth, lidar2img) # b n w h d 3
        coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
        direction_vector = (coords_3d[:, 1] - coords_3d[:, 0])# F.normalize((coords_3d[:, 1] - coords_3d[:, 0]), dim=-1)

        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_depth = (depth_prob * coords_d.view(1, self.depth_num, 1, 1)).sum(1)
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3

        # print(depth_prob.shape, coords_d.shape, pred_depth.shape)
        # print(coords_d)
        # print(pred_depth[0,:5,0])
        # print(depth_prob[0, :,:5,0])
        uncertainty_map = (depth_prob * ((coords_d.view(1, -1, 1, 1) - pred_depth.unsqueeze(1))**2)).sum(1)
        print(pred_coords_3d[0,:5,0])
        print((pred_depth.unsqueeze(-1) * direction_vector + coords_3d[:,0])[0,:5,0])

        # pred_depth = rearrange(pred_depth, '(b n) h w -> b n h w',b=b, n=n)

        return pred_coords_3d, uncertainty_map, direction_vector
    
    def pred_depth_2(self, lidar2img, depth):
        # b, n, c, h, w = depth.shape
        coords_3d, coords_d = self.get_pixel_coords_3d(depth, lidar2img) # b n w h d 3
        coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
        direction_vector = F.normalize((coords_3d[:, 1] - coords_3d[:, 0]), dim=-1)

        # depth = rearrange(depth, 'b n ... -> (b n) ...')
        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3

        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale

        # pred_depth = (depth_prob * coords_d.view(1, self.depth_num, 1, 1)).sum(1)
        # uncertainty_map = (depth_prob * ((coords_d.view(1, -1, 1, 1) - pred_depth.unsqueeze(1))**2)).sum(1)
        # print(uncertainty_map)

        # e = torch.zeros_like(direction_vector)
        # e[..., 2] = 1
        # v = torch.cross(direction_vector, e, dim=-1)
        # v = F.normalize(v, dim=-1)
        # w = torch.cross(direction_vector, v, dim=-1)
        # w = F.normalize(w, dim=-1)
        # cov = cov + scale * self.orth_scale * v.unsqueeze(-1) @ v.unsqueeze(-2) # 0.1 * (v.unsqueeze(-1) @ v.unsqueeze(-2) + w.unsqueeze(-1) @ w.unsqueeze(-2))

        return pred_coords_3d, cov, direction_vector

    def get_coords_3d(self, depth, lidar2img):
        eps = 1e-5
        img2lidars = lidar2img.inverse() # b n 4 4
        B, N = lidar2img.shape[:2]
        I, _, C, H, W = depth.shape
        depth = rearrange(depth, 'i (b n) c h w -> i b n w h c', b=B, n=N)
        scale = 224 // H

        coords_h = torch.linspace(scale // 2, 224 - scale//2, H, device=depth.device).float()
        coords_w = torch.linspace(scale // 2, 480 - scale//2, W, device=depth.device).float()

        coords_wh = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0) # W, H, 2
        coords_wh = repeat(coords_wh, '... -> i b n ...', i=I, b=B, n=N)
        coords_whdhomo = torch.cat((coords_wh * depth, depth, torch.ones_like(depth)), dim=-1)[..., None] # i b n w h 4 1
        # used for calculating direction vector
        coords_whdhomo_right = torch.cat((coords_wh * (depth+1), depth+1, torch.ones_like(depth)), dim=-1)[..., None] # i b n w h 4 1

        img2lidars = repeat(img2lidars, 'b n j k -> i b n w h j k', i=I, w=W, h=H) # i b n w h 4 4
        coords3d = torch.matmul(img2lidars, coords_whdhomo).squeeze(-1)[..., :3] # i b n w h 3
        coords3d = rearrange(coords3d, 'i b n w h d -> i b n h w d')

        coords3d_right = torch.matmul(img2lidars, coords_whdhomo_right).squeeze(-1)[..., :3] # i b n w h 3
        coords3d_right = rearrange(coords3d_right, 'i b n w h d -> i b n h w d')
        direction_vector = F.normalize((coords3d_right - coords3d), dim=-1)
        return coords3d, direction_vector

    def densify(self, features, depth, opacities, lidar2img):
        
        features = F.interpolate(features, scale_factor=self.scale, mode='bilinear')
        depth = F.interpolate(depth, scale_factor=self.scale, mode='bilinear')
        opacities = F.interpolate(opacities, scale_factor=self.scale, mode='bilinear')

        coords_3d, coords_d = self.get_pixel_coords_3d(depth, lidar2img) # b n w h d 3
        coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')

        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1)  # (b n) h w 3

        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale

        return features, pred_coords_3d, cov, opacities
    
    def forward(self, batch):
        b, n, _, h, w = batch['image'].shape
        image = batch['image'].flatten(0, 1).contiguous()            # b n c h w
        lidar2img = batch['lidar2img']
        features = self.backbone(self.norm(image))
        features, depth, opacities = self.neck(features)
        # depth = self.depth_layers(features)
        # opacities = self.opacity_layers(features).sigmoid()
        # features = self.feats_layers(features)
        # opacities = torch.ones_like(features[:, 0:1])
        # opacities.requires_grad_ = False
        # 1
        # if depth is not None:
            # depth = rearrange(depth,'(b n) ... -> b n ...', b=b,n=n)
            # mean, uncertainty, direction_vector = self.pred_depth(lidar2img, depth)
        means3D, cov3D, direction_vector = self.pred_depth_2(lidar2img, depth)
            # means3D = self.pred_depth_2(lidar2img, depth)
        
        # elif self.input_depth:
        #     gt_depth = self.get_pixel_depth(batch['depth'], features, lidar2img, depth)
        #     features[0] = torch.cat((features[0], gt_depth), dim=2)
    
        # features_densified, means3D_densified, uncertainty_densified, opacities_densified = self.densify(features, depth, opacities, lidar2img)

        # orth_features, sampling_points_cam_out, sampling_points = self.orth_layer(means3D, features, direction_vector, cov3D, lidar2img)
        # scales = rearrange(scales, '(b n) d h w -> b (n h w) d', b=b, n=n)
        # rotations = rearrange(rotations, '(b n) d h w -> b (n h w) d', b=b, n=n)
        # cov3D = compute_covariance_matrix_batch(rotations, scales)
        cov3D = cov3D.flatten(-2, -1)
        cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        features = rearrange(features, '(b n) d h w -> b (n h w) d', b=b, n=n)
        # features = features + orth_features
        means3D = rearrange(means3D, '(b n) h w d-> b (n h w) d', b=b, n=n)
        cov3D = rearrange(cov3D, '(b n) h w d -> b (n h w) d',b=b, n=n)

        # features, means3D, cov3D, sampling_points_cam_list, gaussians_list = self.update(features, means3D, features.clone(), cov3D, direction_vector, opacities, lidar2img)
        # cov3D = cov3D.flatten(-2, -1)
        # cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)
        opacities = rearrange(opacities, '(b n) d h w -> b (n h w) d', b=b, n=n)

        # cov3D_densified = uncertainty_densified.flatten(-2, -1)
        # cov3D_densified = torch.cat((cov3D_densified[..., 0:3], cov3D_densified[..., 4:6], cov3D_densified[..., 8:9]), dim=-1)

        # features_densified = rearrange(features_densified, '(b n) d h w -> b (n h w) d', b=b, n=n)
        # means3D_densified = rearrange(means3D_densified, '(b n) h w d-> b (n h w) d', b=b, n=n)
        # cov3D_densified = rearrange(cov3D_densified, '(b n) h w d -> b (n h w) d',b=b, n=n)
        # opacities_densified = rearrange(opacities_densified, '(b n) d h w -> b (n h w) d', b=b, n=n)

        # features = torch.cat((features, features_densified), dim=1)
        # means3D = torch.cat((means3D, means3D_densified), dim=1)
        # cov3D = torch.cat((cov3D, cov3D_densified), dim=1)
        # opacities = torch.cat((opacities, opacities_densified), dim=1)

        # x, orthogonal_uncertainty = self.gs_render(features, mean, uncertainty, direction_vector, opacities, [b, n])
        
        x, num_gaussians = self.gs_render(features, means3D, cov3D, opacities)
        # x, num_gaussians = self.gs_render(features, means3D, cov3D, opacities, orth_features, sampling_points)
        # x = self.bev_conv(x)

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

        # y = self.decoder(x)
        y = self.decoder(x, from_dense=True)
        output = self.head(y)

        mask = x[:, 0:1] != 0
        for k in output:
            if isinstance(output[k], spconv.SparseConvTensor):
                output[k] = output[k].dense()
        output['mask'] = mask

        # output['VEHICLE'] += short_cut
        output['mid_output'] = {
            'features':features, 
            'mean':means3D, 
            'uncertainty':cov3D, 
            'opacities':opacities, 
            'depth':depth, 
            # 'sampling_points_cam_out':sampling_points_cam_out,
            # 'sampling_points': sampling_points,
            'direction_vector': direction_vector,
            # 'orth_features': orth_features
        }
        output['num_gaussians'] = num_gaussians
        # output['short_cut'] = short_cut
        # stage_outputs['mid_output'] = {'mean':mean, 'uncertainty':uncertainty, 'opacities':opacities}

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
        self.gamma = 1.0
        # self.pos_encoder = PositionalEncodingMap(in_c=3, out_c=embed_dims, mid_c=embed_dims * 2)
        # self.pc_range = [-61.0, -61.0, -10.0, 61.0, 61.0, 10.0] 
        self.densify_num = 1

    def get_orthogonal_variance(self, features, means3D, uncertainty, v):
        # means3D = rearrange(means3D, 'b h w d-> b d h w')
        uncertainty = rearrange(uncertainty, 'b h w -> b 1 h w')
        v = rearrange(v, 'b h w d -> b d h w')
        in_features = torch.cat((features, v), dim=1)
        orthogonal_uncertainty = uncertainty * (self.orthogonalVarHead(in_features).sigmoid()) + self.epsilon
        orthogonal_uncertainty = rearrange(orthogonal_uncertainty, 'b 1 h w -> b h w 1 1')

        return orthogonal_uncertainty
    
    def PruneAndDenify(self, features, means3D, uncertainty, direction_vector, opacities):
        b = features.shape[0]
        device = features.device

        mask = (opacities > self.threshold).squeeze(-1)
        gaissuans = []
        gammas = torch.linspace(-self.gamma, self.gamma, self.densify_num, device=device).view(1,self.densify_num) # here self.gamma = 1
        coeff = 1 - ((gammas ** 2).sum() / self.densify_num)

        for i in range(b):

            features_pruned = features[i][mask[i]] # N, 128
            means3D_pruned = means3D[i][mask[i]] # N, 3
            uncertainty_pruned = uncertainty[i][mask[i]] # N
            direction_vector_pruned = direction_vector[i][mask[i]] # N 3
            opacities_pruned = opacities[i][mask[i]]

            # each attributes has shape N, d
            sigma = gammas * uncertainty_pruned.sqrt().unsqueeze(-1) # 1 5 * N 1 -> N 5
            means3D_densified = means3D_pruned.unsqueeze(1) + direction_vector_pruned.unsqueeze(1) * sigma.unsqueeze(-1)
            means3D_densified = means3D_densified.flatten(0,1)

            uncertainty_densified = uncertainty_pruned.view(-1,1).repeat(1,self.densify_num).flatten(0,1) * coeff
            uncertainty_densified = torch.clamp(uncertainty_densified, min=1e-4)

            direction_vector_densified = direction_vector_pruned.unsqueeze(1).repeat(1,self.densify_num,1).flatten(0,1)
            features_densified = features_pruned.unsqueeze(1).repeat(1,self.densify_num,1).flatten(0,1)
            opacities_densified = opacities_pruned.unsqueeze(1).repeat(1,self.densify_num,1).flatten(0,1) * (1 / self.densify_num)
            
            gaissuans.append([features_densified, means3D_densified, uncertainty_densified, direction_vector_densified, opacities_densified])
        
        return gaissuans

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
        # num_sample_points = orth_features.shape[-2]
        # features = rearrange(features, '(b n) d h w -> b (n h w) d', b=b, n=n)
        # means3D = rearrange(means3D, '(b n) h w d-> b (n h w) d', b=b, n=n)
        # uncertainty = rearrange(uncertainty, '(b n) h w -> b (n h w)',b=b, n=n)
        # direction_vector = rearrange(direction_vector, '(b n) h w d-> b (n h w) d', b=b, n=n)
        # opacities = rearrange(opacities, '(b n) d h w -> b (n h w) d', b=b, n=n)
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

        mask = (opacities > self.threshold)
        # opacities = mask.float()
        mask = mask.squeeze(-1)

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
        self.set_render_scale(int(200), int(200))
        self.set_Rasterizer(device)
        for i in range(b):
            # means3D_input = means3D[i][mask[i]]
            # orth_opacities = repeat(opacities[i][mask[i]], '... d -> ... p d', p=num_sample_points)
            # num_gaussians = means3D_input.shape[0]
            # means3D_input = torch.cat((means3D_input, orth_means[i][mask[i]].flatten(0,1)), dim=0)
            # features_input = torch.cat((features[i][mask[i]], orth_features[i][mask[i]].flatten(0,1)), dim=0)
            # opacities_input = torch.cat((opacities[i][mask[i]], orth_opacities.flatten(0,1)), dim=0)
            # cov3D_input = torch.cat((cov3D[i][mask[i]], torch.zeros((num_gaussians*num_sample_points, 6), device=device)), dim=0)
            # rendered_bev, _ = self.rasterizer(
            #     means3D=means3D_input,
            #     means2D=None,
            #     shs=None,  # No SHs used
            #     colors_precomp=features_input,
            #     opacities=opacities_input,
            #     scales=None,
            #     rotations=None,
            #     cov3D_precomp=cov3D_input
            # )
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
            # orth_opacities = repeat(opacities[i][mask[i]], '... d -> ... p d', p=num_sample_points)
            # orth_bev, _ = self.rasterizer(
            #     means3D=orth_means[i][mask[i]].flatten(0,1),
            #     means2D=None,
            #     shs=None,  # No SHs used
            #     colors_precomp=orth_features[i][mask[i]].flatten(0,1),
            #     opacities=orth_opacities.flatten(0,1),
            #     scales=None,
            #     rotations=None,
            #     cov3D_precomp=torch.zeros((num_gaussians*num_sample_points, 6), device=device)
            # )
            # bev_out.append(torch.cat((rendered_bev, orth_bev), dim=0))
            bev_out.append(rendered_bev)

        # multi_scale_output = []

        # for scale in [0.25, 0.5, 1]:
        #     bev_out = []
        #     self.set_render_scale(int(200 * scale), int(200 * scale))
        #     self.set_Rasterizer(device)
        #     for i in range(b):
        #         rendered_bev, _ = self.rasterizer(
        #             means3D=means3D[i][mask[i]],
        #             means2D=None,
        #             shs=None,  # No SHs used
        #             colors_precomp=features[i][mask[i]],
        #             opacities=opacities[i][mask[i]],
        #             scales=None,
        #             rotations=None,
        #             cov3D_precomp=cov3D[i][mask[i]]
        #         )
        #         bev_out.append(rendered_bev)
        #     multi_scale_output.append(torch.stack(bev_out, dim=0))
    
        # bev_out = []
        # num_gaussians = []
        # for i in range(b):
        #     features, means3D, uncertainty, direction_vector, opacities = gaussians[i]
        #     num_gaussians.append(features.shape[0])
        #     cov3D = uncertainty[..., None, None] * (direction_vector.unsqueeze(-1) @ direction_vector.unsqueeze(-2))
        #     e = torch.zeros_like(direction_vector)
        #     e[..., 2] = 1
        #     v = torch.cross(direction_vector, e, dim=-1)
        #     v = F.normalize(v, dim=-1)
        #     w = torch.cross(direction_vector, v, dim=-1)
        #     # orthogonal_uncertainty = self.get_orthogonal_variance(features, means3D, uncertainty, v)
        #     # cov3D = cov3D + orthogonal_uncertainty * (v.unsqueeze(-1) @ v.unsqueeze(-2)) + self.epsilon * (w.unsqueeze(-1) @ w.unsqueeze(-2))
        #     cov3D = cov3D + self.epsilon * (v.unsqueeze(-1) @ v.unsqueeze(-2) + w.unsqueeze(-1) @ w.unsqueeze(-2))
        #     cov3D = cov3D.flatten(-2, -1)
        #     cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        #     # mean_normalized = means3D.clone()
        #     # mean_normalized[..., 0] = (mean_normalized[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        #     # mean_normalized[..., 1] = (mean_normalized[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        #     # mean_normalized[..., 2] = (mean_normalized[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        #     # mean_normalized = torch.clamp(mean_normalized, min=0.0, max=1.0)
        #     # features = features + self.pos_encoder(mean_normalized)

        #     rendered_bev, _ = self.rasterizer(
        #         means3D=means3D,
        #         means2D=None,
        #         shs=None,  # No SHs used
        #         colors_precomp=features,
        #         opacities=opacities,
        #         scales=None,
        #         rotations=None,
        #         cov3D_precomp=cov3D
        #     )
        #     bev_out.append(rendered_bev)
        # print("Total points:", (mask.detach().float().sum(-1)).mean().cpu().numpy())
        # print("Total points:", sum(num_gaussians) / len(num_gaussians))
        # return torch.stack(bev_out, dim=0), None
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
            y_range, 
            x_range, 
            orth_features, 
            orth_means
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
        
        # features = rearrange(features[:, :6*28*60], 'b (n h w) d -> (b n) d h w', n=6, h=28, w=60)
        # means3D = rearrange(means3D[:, :6*28*60], 'b (n h w) d -> (b n) h w d', n=6, h=28, w=60)
        # uncertainty = rearrange(uncertainty[:, :6*28*60], 'b (n h w) j k -> (b n) h w j k', n=6, h=28, w=60)
        # opacities = rearrange(opacities[:, :6*28*60], 'b (n h w) d -> (b n) h w d', n=6, h=28, w=60)

        
        features = rearrange(features[cam_index, :, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'd h w -> 1 (h w) d')
        means3D = rearrange(means3D[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w d-> 1 (h w) d')
        # uncertainty = rearrange(uncertainty[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w -> 1 (h w)')
        cov3D = rearrange(cov3D[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w d-> 1 (h w) d')
        # direction_vector = rearrange(direction_vector[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w d-> 1 (h w) d')
        opacities = rearrange(opacities[cam_index, :, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'd h w -> 1 (h w) d')
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
        # cov3D = uncertainty.flatten(-2, -1)
        # cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)
        num_points = orth_features.shape[-2]
        orth_features = rearrange(orth_features[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w p d -> 1 (h w p) d')
        orth_means = rearrange(orth_means[cam_index, y_range[0]:y_range[1], x_range[0]:x_range[1]], 'h w p d -> 1 (h w p) d')

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
            opacities_orth = repeat(opacities[i], '... d -> ... p d', p=num_points)
            rendered_bev, _ = self.rasterizer(
                means3D=torch.cat((means3D[i], orth_means[i]), dim=0),
                means2D=None,
                shs=None,  # No SHs used
                colors_precomp=torch.cat((features[i], orth_features[i]), dim=0),
                opacities=torch.cat((opacities[i], opacities_orth.flatten(0,1)), dim=0),
                scales=None,
                rotations=None,
                cov3D_precomp=torch.cat((cov3D[i], torch.zeros((num_gaussians*num_points, 6), device=device)), dim=0),
            )
            bev_out.append(rendered_bev)

        # bev_out = []
        # num_gaussians = []
        # for i in range(b):
        #     features, means3D, uncertainty, direction_vector, opacities = gaussians[i]
        #     num_gaussians.append(features.shape[0])
        #     cov3D = uncertainty[..., None, None] * (direction_vector.unsqueeze(-1) @ direction_vector.unsqueeze(-2))
        #     e = torch.zeros_like(direction_vector)
        #     e[..., 2] = 1
        #     v = torch.cross(direction_vector, e, dim=-1)
        #     v = F.normalize(v, dim=-1)
        #     w = torch.cross(direction_vector, v, dim=-1)
        #     # orthogonal_uncertainty = self.get_orthogonal_variance(features, means3D, uncertainty, v)
        #     # cov3D = cov3D + orthogonal_uncertainty * (v.unsqueeze(-1) @ v.unsqueeze(-2)) + self.epsilon * (w.unsqueeze(-1) @ w.unsqueeze(-2))
        #     cov3D = cov3D + self.epsilon * (v.unsqueeze(-1) @ v.unsqueeze(-2) + w.unsqueeze(-1) @ w.unsqueeze(-2))
        #     cov3D = cov3D.flatten(-2, -1)
        #     cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        #     # mean_normalized = means3D.clone()
        #     # mean_normalized[..., 0] = (mean_normalized[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        #     # mean_normalized[..., 1] = (mean_normalized[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        #     # mean_normalized[..., 2] = (mean_normalized[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        #     # mean_normalized = torch.clamp(mean_normalized, min=0.0, max=1.0)
        #     # features = features + self.pos_encoder(mean_normalized)

        #     rendered_bev, _ = self.rasterizer(
        #         means3D=means3D,
        #         means2D=None,
        #         shs=None,  # No SHs used
        #         colors_precomp=features,
        #         opacities=opacities,
        #         scales=None,
        #         rotations=None,
        #         cov3D_precomp=cov3D
        #     )
        #     bev_out.append(rendered_bev)
        # print("Filterd points:", num_gaussians - (mask.detach().float().sum(-1)).mean().cpu().numpy())
        # print("Total points:", sum(num_gaussians) / len(num_gaussians))
        return torch.stack(bev_out, dim=0), None # orthogonal_uncertainty

class Neck(nn.Module):
    def __init__(self, in_channels, embed_dims, depth_num):
        super().__init__()

        self.proj1 = nn.Conv2d(in_channels[-1], embed_dims, 1) # stage 4
        self.proj2 = nn.Conv2d(in_channels[-2]+embed_dims, embed_dims, 1) # stage 3 4
        self.proj3 = nn.Conv2d(in_channels[-3]+embed_dims, embed_dims, 1) # stage 2 3 4
        self.depth_head = MLP(embed_dims, embed_dims * 2, depth_num)
        self.opacities_head = MLP(embed_dims, embed_dims * 2, 1)

        self.depth_num = 64
        self.depth_start = 1
        self.depth_max = 61
        self.LID = True

    def get_pixel_coords_3d(self, img_feats):
        eps = 1e-5
        
        B, N, _, H, W = img_feats.shape
        device = img_feats.device
        scale = 224 // H
        # coords_h = torch.arange(H, device=depth.device).float() * 224 / H
        # coords_w = torch.arange(W, device=depth.device).float() * 480 / W
        coords_h = torch.linspace(scale // 2, 224 - scale//2, H, device=device).float()
        coords_w = torch.linspace(scale // 2, 480 - scale//2, W, device=device).float()

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=device).float()
            index_1 = index + 1
            bin_size = (self.depth_max - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=device).float()
            bin_size = (self.depth_max - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0) # W, H, 2
        coords = repeat(coords, '... -> b n ...', b=B, n=N) # b n w h 2

        return coords, coords_d

    def forward(self, img_feats, lidar2img):
        b, n = lidar2img.shape[:2]
        p4 = self.proj1(img_feats[-1])
        p3 = self.proj2(torch.cat((img_feats[-2], F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)), dim=1))
        p2 = self.proj3(torch.cat((img_feats[-3], F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)), dim=1))

        p4 = rearrange(p4, '(b n) d h w -> b n h w d')
        p3 = rearrange(p3, '(b n) d h w -> b n h w d')
        p2 = rearrange(p2, '(b n) d h w -> b n h w d')
        
        p4_depth, p4_opacities = self.depth_head(p4).softmax(), self.opacities_head(p4).sigmoid()
        p3_depth, p3_opacities = self.depth_head(p3).softmax(), self.opacities_head(p3).sigmoid()
        p2_depth, p2_opacities = self.depth_head(p2).softmax(), self.opacities_head(p2).sigmoid()

        pass

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
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

class GaussianOrthLayer(nn.Module):
    def __init__(self, embed_dims, num_points, extent=0.5):
        super().__init__()
        self.extent = extent
        self.num_points = num_points
        self.positiona_encoding = PositionalEncodingMap(in_c=3, out_c=embed_dims, mid_c=embed_dims * 2)
        # self.mlp = MLP(num_points * embed_dims, embed_dims*2, embed_dims)
        self.pc_range = [-50.0, -50.0, -4.0, 50.0, 50.0, 4.0]

    def sampling(self, img_features, sampling_points, lidar2img):
        """
        sampling_offsets: b n l 3 p
        lidar2img: b n 4 4
        """
        b, n = sampling_points.shape[:2]
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

        return rearrange(sampled_features, '(b n) d l p -> b n l p d', b=b, n=n), sampling_points_cam_out

    def forward(self, means3D, img_features, direction_vector, cov3D, lidar2img):  
        b, n = lidar2img.shape[:2]
        device = means3D.device  

        means3D = rearrange(means3D, '(b n) h w d-> b n (h w) d', b=b, n=n)
        direction_vector = rearrange(direction_vector, '(b n) h w d-> b n (h w) d', b=b, n=n)
        cov3D = rearrange(cov3D, '(b n) h w i j -> b n (h w) i j', b=b, n=n)

        # get uncertain mean
        uncertainty = torch.linspace(-1.5, 1.5, 5, device=device).view(1, 1, 1, 1, 5) * (direction_vector.unsqueeze(-2) @ cov3D @ direction_vector.unsqueeze(-1)).sqrt() * direction_vector.unsqueeze(-1) # b n l 3 p1
        means3D = means3D.unsqueeze(-1) + uncertainty

        # get orthogonal direction vector
        e = torch.zeros_like(direction_vector)
        e[..., -1] = 1
        orth_vector = torch.cross(direction_vector, e)

        # get sampling points
        sampling_offsets = torch.linspace(-self.extent, self.extent, self.num_points, device=device).view(1, 1, 1, 1, self.num_points) # num_points
        sampling_offsets = orth_vector.unsqueeze(-1) * sampling_offsets # b n l 3 p2
        sampling_points = sampling_offsets.unsqueeze(-2) + means3D.unsqueeze(-1)
        sampling_points = rearrange(sampling_points, 'b n l d p1 p2 -> b n l (p1 p2) d')
        sampling_features, sampling_points_cam_out = self.sampling(img_features, sampling_points, lidar2img)

        # normalize
        sampling_points_norm = sampling_points.clone()
        sampling_points_norm[..., 0:1] = (sampling_points_norm[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        sampling_points_norm[..., 1:2] = (sampling_points_norm[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        sampling_points_norm[..., 2:3] = (sampling_points_norm[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        sampling_points_norm = torch.clamp(sampling_points_norm, min=0.0, max=1.0)
        pos_embed = self.positiona_encoding(sampling_points_norm)
        sampling_features = sampling_features + pos_embed

        # sampling_features = self.mlp(sampling_features.flatten(-2,-1)) # b n l d
        return sampling_features.flatten(1,2), sampling_points_cam_out, sampling_points.flatten(1,2)

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