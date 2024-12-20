import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import numpy as np
from fvcore.nn import sigmoid_focal_loss
from einops import rearrange


logger = logging.getLogger(__name__)

class SpatialRegressionLoss(torch.nn.Module):
    def __init__(self, norm, min_visibility=0, ignore_index=None, key=''):
        super(SpatialRegressionLoss, self).__init__()
        # center:2, offset: 1
        self.norm = norm
        self.min_visibility = min_visibility
        self.ignore_index = ignore_index
        self.key = key

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, batch, eps=1e-6):

        pred_mask = prediction['mask'] if 'mask' in prediction else None
        prediction = prediction[self.key]
        if self.key == 'center':
            prediction = prediction.sigmoid()
        target = batch[self.key]

        assert len(prediction.shape) == 4, 'Must be a 4D tensor'
        # ignore_index is the same across all channels

        loss = self.loss_fn(prediction, target, reduction='none')

        # if self.norm == 1:
        #     loss = loss.sum(dim=1)[:,None]

        # # Sum channel dimension
        # mask = batch['visibility'] >= self.min_visibility
        
        # loss = loss[mask[:, None]]
        # return loss.mean()

        mask = torch.ones_like(loss, dtype=torch.bool)
        if self.min_visibility>0:
            vis_mask = batch['visibility'] >= self.min_visibility
            vis_mask = vis_mask[:, None]
            mask = mask * vis_mask

        if self.ignore_index is not None:
            mask = mask * (target != self.ignore_index)

        if pred_mask is not None:
            mask = mask * pred_mask

        return (loss * mask).sum() / (mask.sum() + eps)

class BCELoss(torch.nn.Module):
    def __init__(
        self,
        pos_weight,
        label_indices,
        key,
        min_visibility=None,
    ):
        """
        BCE(p) = -(y * log(p) + (1 - y) * log(1 - p))

        if y=0:
            BCE(p) = -log(1 - p)
            - if p ~ 0:
                well classified and BCE(p) ~ 0
            - if p ~ 1:
                badly classified and BCE(p) ~ inf

        if y=1:
            BCE(p) = -log(p)
            - if p ~ 0:
                badly classified and BCE(p) ~ inf
            - if p ~ 1:
                well classified and BCE(p) ~ 0
        """

        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]), reduction="none"
        )
        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.key = key

    def forward(self, pred_dict, batch):
        if isinstance(pred_dict, dict):            
            pred_mask = pred_dict['mask'] if 'mask' in pred_dict else None
            pred = pred_dict[self.key]

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        loss = self.loss_fn(pred, label)
        if self.min_visibility is not None:
            if self.key == 'ped':
                mask = batch['visibility_ped'] >= self.min_visibility
            else:
                mask = batch['visibility'] >= self.min_visibility

            mask = mask[:, None]
            if pred_mask is not None:
                mask = mask & pred_mask
            loss = loss[mask]

        elif pred_mask is not None:
            loss = loss[pred_mask]
            loss = torch.nan_to_num(loss)

        loss = loss.mean()

        return loss
    
class DepthConsistencyLoss(torch.nn.Module):
    def __init__(
        self,
        mode,
    ):
        super().__init__()
        if mode == 'L1':
            self.loss_fn = nn.SmoothL1Loss()
        elif mode == 'KL':
            self.loss_fn = nn.KLDivLoss(log_target=True)
        
        self.mode = mode
        self.adjacent_view = [1, 2, 5, 0, 3, 4]

        self.depth_num = 64
        self.depth_max = 61
        self.depth_start = 1
        self.smooth_loss = 1e-3

    def project(self, points, lidar2img, h, w):
        points = torch.cat((points, torch.ones_like(points[..., 0:1])), dim=-1) # b h w 4
        points = (lidar2img.view(-1, 1, 1, 4, 4) @ points.unsqueeze(-1)).squeeze(-1) # b h w 4 4 @ b h w 4 1 -> b h w 4
        # points = points.flatten(1,2) # (b h w) 4
        mask1 = points[..., 2] > 1e-5
        points[..., 2:3] = torch.maximum(points[..., 2:3], torch.zeros_like(points[..., 2:3]) + 1e-5)
        points[..., :2] /= points[..., 2:3]
        mask2 = (points[..., 0] <= w) & (points[..., 0] >= 0) & (points[..., 1] <= h) & (points[..., 1] >= 0)
        mask = mask1 & mask2
        points = points[..., :2] # b h w 2
        return points, mask.view(-1)
    
    def forward(self, pred, batch):
        images = batch['image']
        lidar2img = batch['lidar2img'] # b n 4 4
        b, n, _, h, w = images.shape
        scale = 8
        mean = pred['mean']
        mean = rearrange(mean, 'b (n h w) d -> b n h w d', n=n, h=h//scale, w=w//scale)
        depth = pred['depth'] # (b n) h w
        
    
        # apply mask for augmented images
        img_mask = (images == 0).all(2) # if all channels are all zero
        img_mask = F.interpolate(img_mask.float(), scale_factor=1/scale, mode='nearest').bool()
        img_mask = ~img_mask.unsqueeze(2)

        images = F.interpolate(images.flatten(0,1), scale_factor=1/scale, mode='bilinear')
        images = rearrange(images, '(b n) d h w -> b n d h w', b=b, n=n)
        images = images * img_mask

        target = []
        source = []
        for i in range(n):
            points, mask = self.project(mean[:, i].detach(), lidar2img[:, self.adjacent_view[i]].detach(), h, w)

            # normalized to -1 ~ 1 for sampling
            points[..., 0] /= w
            points[..., 1] /= h
            points = points * 2 - 1

            target_pix = images[:, i] # b d h w
            target_pix = rearrange(target_pix, 'b d h w -> (b h w) d')
            target_pix = target_pix[mask]
            zero_mask = ~((target_pix == 0).all(1))
            target_pix = target_pix[zero_mask]
            target.append(target_pix)

            source_pix = images[:, self.adjacent_view[i]]
            source_pix = F.grid_sample(source_pix, points)
            source_pix = rearrange(source_pix, 'b d h w -> (b h w) d')
            source_pix = source_pix[mask]
            source_pix = source_pix[zero_mask]
            source.append(source_pix)

        if self.smooth_loss > 0.0:
            img_mask = img_mask.flatten(0,1)
            images = images.flatten(0,1)
            disp = 1 / (depth * img_mask + 1e-6)
            return self.loss_fn(torch.cat(target), torch.cat(source)) + self.smooth_loss * get_smooth_loss(disp, images)
    
        return self.loss_fn(torch.cat(target), torch.cat(source))
    
class DepthSmoothLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, batch):
        images = batch['image']
        lidar2img = batch['lidar2img'] # b n 4 4
        b, n, _, h, w = images.shape
        scale = 8
        mean = pred['mean']
        mean = rearrange(mean, 'b (n h w) d -> b n h w d', n=n, h=h//scale, w=w//scale)
        depth = pred['depth'] # (b n) h w
        
    
        # apply mask for augmented images
        # img_mask = (images == 0).all(2) # if all channels are all zero
        # img_mask = F.interpolate(img_mask.float(), scale_factor=1/scale, mode='nearest').bool()
        # img_mask = ~img_mask.unsqueeze(2)

        images = F.interpolate(images.flatten(0,1), scale_factor=1/scale, mode='bilinear')
        # images = rearrange(images, '(b n) d h w -> b n d h w', b=b, n=n)
        # images = images * img_mask

        return get_smooth_loss(depth, images)
    
class DepthLoss(torch.nn.Module):
    def __init__(
        self,
        norm,
    ):
        super().__init__()

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss

    def forward(self, pred, label):
        pred_depth = pred['depth']
        gt_depth = label['depth']

        # re-scale gt depth to pred resolution
        h, w = pred_depth.shape[-2:]
        gt_depth = gt_depth.flatten(0,1)
        mask = gt_depth != 0
        # gt_depth = F.interpolate(gt_depth, size=[h,w], mode='bilinear').squeeze(1)
        # mask = F.interpolate(mask, size=[h,w], mode='nearest').squeeze(1)

        # masking
        gt_depth = gt_depth[mask]
        pred_depth = pred_depth[mask]

        return self.loss_fn(pred_depth, gt_depth, reduction='mean')

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label, reduction='none'):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)
    
class Features_Loss(torch.nn.Module):
    def __init__(
        self,
        loss,
    ):
        super().__init__()
        if loss == 'L1':
            self.loss = nn.L1Loss()
        elif loss == 'L2':
            self.loss = nn.MSELoss()

    def forward(self, pred, label):
        # label not used
        return self.loss(pred['pred_features'], pred['features'])

class diceLoss(torch.nn.Module):
    def __init__(
        self,
        eps=1e-7,
        key='bev',
        min_visibility=None,
        label_indices=None,
    ):
        super().__init__()
        self.eps = eps
        self.key = key
        self.min_visibility = min_visibility
        self.label_indices = label_indices

    def forward(self,pred, batch):
        if isinstance(pred, dict):
            pred = pred[self.key].sigmoid()

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        if self.min_visibility is not None:
            if self.key == 'ped':
                mask = batch['visibility_ped'] >= self.min_visibility
            else:
                mask = batch['visibility'] >= self.min_visibility
        label = label[:,0] * mask
        pred = pred[:,0] * mask
        intersection = 2 * torch.sum(pred * label) + self.eps
        union = torch.sum(pred) + torch.sum(label) + self.eps
        loss = 1 - intersection / union
        return loss

class BinarySegmentationLoss(SigmoidFocalLoss):
    def __init__(
        self,
        label_indices=None,
        min_visibility=0,
        alpha=-1.0,
        gamma=2.0,
        key='bev',
        seq_loss_gamma=0.8,
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')
        
        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.key = key
        self.seq_loss_gamma = seq_loss_gamma

    def forward(self, pred_dict, batch):
        if isinstance(pred_dict, dict):            
            pred_mask = pred_dict['mask'] if 'mask' in pred_dict else None
            pred = pred_dict[self.key]

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        loss = super().forward(pred, label)

        if self.min_visibility>0:
            if self.key == 'ped':
                mask = batch['visibility_ped'] >= self.min_visibility
            else:
                mask = batch['visibility'] >= self.min_visibility

            mask = mask[:, None]
            if pred_mask is not None:
                mask = mask & pred_mask
            loss = loss[mask]
        elif pred_mask is not None:
            loss = loss[pred_mask]
            loss = torch.nan_to_num(loss)

        loss = loss.mean()

        if 'aux' in pred_dict:
            N_iter = len(pred_dict['aux'])
            for i, aux_pred in enumerate(pred_dict['aux']):
                aux_pred = aux_pred[self.key]
                aux_loss = super().forward(aux_pred, label)

                if self.min_visibility is not None:
                    aux_loss = aux_loss[mask]

                loss += aux_loss.mean() * self.seq_loss_gamma ** (N_iter - i)

        return loss

class SeqSegmentationLoss(torch.nn.Module):
    def __init__(
        self,
        mode,
        label_indices=None,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0,
        key='bev',
        seq_loss_gamma=0.8,
    ):
        super().__init__()
        
        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.key = key
        self.seq_loss_gamma = seq_loss_gamma

        self.mode = mode
        if mode == "l1":
            self.loss_fn = F.l1_loss
        elif mode == "l2":
            self.loss_fn = F.mse_loss
        elif mode == "sigmoid":
            self.loss_fn = SigmoidFocalLoss(alpha=alpha, gamma=gamma, reduction='none')

    def forward(self, pred_dict, batch):
        pred = pred_dict[self.key]

        if self.mode == "sigmoid":
            label = batch['bev']

            if self.label_indices is not None:
                label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
                label = torch.cat(label, 1)
        else:
            label = batch[self.key]
        
        if self.min_visibility is not None:
            if self.key == 'PED':
                mask = batch['visibility_ped'] >= self.min_visibility
            elif self.key == 'VEHICLE':
                mask = batch['visibility'] >= self.min_visibility

            mask = mask[:, None]
        else:
            mask = None

        loss = self.loss_fn(pred, label, reduction='none')

        if mask is not None:
            loss = loss[mask]

        loss = loss.mean()

        N_iter = len(pred_dict['aux'])
        for i, aux_pred in enumerate(pred_dict['aux']):
            aux_pred = aux_pred[self.key]
            aux_loss = self.loss_fn(aux_pred, label, reduction='none')

            if self.min_visibility is not None:
                aux_loss = aux_loss[mask]

            loss += aux_loss.mean() * self.seq_loss_gamma ** (N_iter - i)

        return loss

class CenterLoss(SigmoidFocalLoss):
    def __init__(
        self,
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')

        self.min_visibility = min_visibility

    def forward(self, pred, batch):
        pred = pred['center']
        label = batch['center']
        loss = super().forward(pred, label)

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            loss = loss[mask[:, None]]

        return loss.mean()

class MultipleLoss(torch.nn.ModuleDict):
    """
    losses = MultipleLoss({'bce': torch.nn.BCEWithLogitsLoss(), 'bce_weight': 1.0})
    loss, unweighted_outputs = losses(pred, label)
    """
    def __init__(self, modules_or_weights):
        modules = dict()
        weights = dict()
        learnable_weights = dict()

        # Parse only the weights
        for key, v in modules_or_weights.items():
            if isinstance(v, float):
                k = key.replace('_weight', '')
                if v == -1:
                    weights[k] =  0.5 if k not in ['visible', 'ped'] else 10.0 # 0.5 if k not in ['visible', 'ped'] else 10.0 0.5 if k not in ['loss_bbox'] else 1.0
                    learnable_weights[k] = nn.Parameter(torch.tensor(0.0), requires_grad=True)
                else:
                    weights[k] = v

        # Parse the loss functions
        for key, v in modules_or_weights.items():
            if not isinstance(v, float):
                modules[key] = v

        super().__init__(modules)

        self._weights = weights
        self.learnable_weights = torch.nn.ParameterDict(learnable_weights)

    def forward(self, pred, batch):
        outputs = dict()
        weights = dict()

        for k, v in self.items():

            if k =='learnable_weights':
                continue
            elif k != 'Set':
                outputs[k] = v(pred, batch)
            else:
                if 'pred_logits' not in pred:
                    continue
                out = v(pred, batch)
                for k2, v2 in out.items():
                    outputs[k2] = v2
        # outputs = {k: v(pred, batch) for k, v in self.items()}
        loss = []
        for k, o in outputs.items():
            loss_weight = self._weights[k]
            if k in self.learnable_weights:
                loss_weight = (1 / torch.exp(self.learnable_weights[k])) * loss_weight
                weights[k] = loss_weight
                uncertainty = self.learnable_weights[k] * 0.5
            else:
                uncertainty = 0.0
            single_loss = loss_weight * o + uncertainty
            outputs[k] = single_loss
            loss.append(single_loss)

        return sum(loss), outputs, weights
    
def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()