import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from fvcore.nn import sigmoid_focal_loss


logger = logging.getLogger(__name__)

class SpatialRegressionLoss(torch.nn.Module):
    def __init__(self, norm, min_visibility=None, ignore_index=None, key=''):
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
        if self.min_visibility is not None:
            vis_mask = batch['visibility'] >= self.min_visibility
            vis_mask = vis_mask[:, None]
            mask = mask * vis_mask

        if self.ignore_index is not None:
            mask = mask * (target != self.ignore_index)
            
        return (loss * mask).sum() / (mask.sum() + eps)

class HeightRegressionLoss(torch.nn.Module):
    def __init__(self, norm, min_visibility=None, key='height', radius=0.5, ignore_index=None):
        super(HeightRegressionLoss, self).__init__()
        # center:2, offset: 1
        self.norm = norm
        self.min_visibility = min_visibility
        self.key = key
        self.radius = radius
        self.ignore_index = ignore_index

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, batch, eps=1e-6):

        prediction = prediction[self.key]
        target = batch[self.key]
        assert len(prediction.shape) == 4, 'Must be a 4D tensor'

        num_points = prediction.shape[1]
        target = target.expand(-1, num_points, -1, -1) # b 1 h w -> b p h w
        loss = self.loss_fn(prediction, target, reduction='none')
        loss -= self.radius ** self.norm
        loss = torch.clamp(loss, min=0.0)

        mask = torch.ones_like(loss, dtype=torch.bool)
        if self.ignore_index is not None:
            mask = mask * (target != 0.0)

        if self.min_visibility is not None:
            vis_mask = batch['visibility'] >= self.min_visibility
            vis_mask = vis_mask[:, None]
            mask = mask * vis_mask

        return (loss * mask).sum() / (mask.sum() + eps)

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

    def forward(self, pred, label):
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
        min_visibility=None,
        alpha=-1.0,
        gamma=2.0,
        key='bev',
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')
        
        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.key = key

    def forward(self, pred, batch):
        if isinstance(pred, dict):            
            pred_mask = pred['mask'] if 'mask' in pred else None
            pred = pred[self.key]

        label = batch['bev']

        if self.label_indices is not None:
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)

        loss = super().forward(pred, label)

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

        return loss.mean()
    
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
                    weights[k] =  0.001 if k not in ['visible', 'ped'] else 10.0 # 0.5 if k not in ['visible', 'ped'] else 10.0 0.5 if k not in ['loss_bbox'] else 1.0
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

            single_loss = loss_weight * o
            outputs[k] = single_loss
            loss.append(single_loss)

        return sum(loss), outputs, weights