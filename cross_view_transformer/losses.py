import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import logging
import numpy as np
from fvcore.nn import sigmoid_focal_loss


logger = logging.getLogger(__name__)

class SpatialRegressionLoss(torch.nn.Module):
    def __init__(self, norm, min_visibility=2):
        super(SpatialRegressionLoss, self).__init__()
        # center:2, offset: 1
        self.norm = norm
        self.min_visibility = min_visibility

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, batch):
        if self.norm == 2:
            prediction = prediction['center']
            target = batch['center']
            prediction = prediction
        else:
            prediction = prediction['offset']
            target = batch['offset']

        assert len(prediction.shape) == 4, 'Must be a 4D tensor'
        # ignore_index is the same across all channels

        loss = self.loss_fn(prediction, target, reduction='none')
        if self.norm == 1:
            loss = loss.sum(dim=1)[:,None]
        # Sum channel dimension
        mask = batch['visibility'] >= self.min_visibility
        loss = loss[mask[:, None]]
        return loss.mean()

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
        key='bev'
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')
        
        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.key = key

    def forward(self, pred, batch):
        if isinstance(pred, dict):
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
            loss = loss[mask[:, None]]

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

        # Parse only the weights
        for key, v in modules_or_weights.items():
            if isinstance(v, float):
                weights[key.replace('_weight', '')] = v

        # Parse the loss functions
        for key, v in modules_or_weights.items():
            if not isinstance(v, float):
                modules[key] = v

                # Assign weight to 1.0 if not explicitly set.
                if key not in weights:
                    logger.warn(f'Weight for {key} was not specified.')
                    weights[key] = 1.0
        assert modules.keys() == weights.keys()

        super().__init__(modules)

        self._weights = weights

    def forward(self, pred, batch):
        outputs = {k: v(pred, batch) for k, v in self.items()}
        total = sum(self._weights[k] * o for k, o in outputs.items())
        return total, outputs
