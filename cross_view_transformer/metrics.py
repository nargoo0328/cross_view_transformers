import torch

from torchmetrics import Metric
from typing import List, Optional
from torch.nn.functional import softmax
import numpy as np

class BaseIoUMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, thresholds=[0.4, 0.5, 0.6], is_ce=False): # [0.4, 0.5, 0.6] np.linspace(0.2,0.5,13)
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        if is_ce:
            thresholds = torch.FloatTensor([0.0])
        else:
            thresholds = torch.FloatTensor(thresholds)
        self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
        self.add_state('tp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')

    def update(self, pred, label):
        label = label.detach().bool().reshape(-1)

        if self.is_ce:
            pred = (pred == self.index)
        else:
            pred = pred[:, None] >= self.thresholds[None]
            label = label[:, None]
        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)
        #self.tn += (~pred & ~label).sum(0)

    def compute(self):
        thresholds = self.thresholds.squeeze(0)
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        if self.is_ce:
            return ious
        else:
            return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, ious)}

    def show_result(self):
        thresholds = self.thresholds.squeeze(0)
        return {f'@{t.item():.2f}': {'tp:':self.tp, 'fp:':self.tp,'fn:':self.fn} for i, t in enumerate(thresholds)}
    


class IoUMetric(BaseIoUMetric):
    def __init__(self, label_indices: List[List[int]], min_visibility: Optional[int] = None, key= 'bev',is_ce=False):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__(is_ce=is_ce)
        index_dict = {'NONE':0,'ped':1 , 'bev':2,'DIVIDER':3,'STATIC':4}
        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.key = key
        self.is_ce = is_ce
        if is_ce:
            self.index = index_dict[key]

    def update(self, pred, batch):
        if isinstance(pred, dict):
            if self.is_ce:
                pred = torch.cat((pred['NONE'],pred['ped'],pred['bev'],pred['DIVIDER'],pred['STATIC']),dim=1)  
            else:
                pred = pred[self.key]                                                              # b c h w
        if isinstance(batch, dict):
            label = batch['bev']                                                                # b n h w
            
        label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
        label = torch.cat(label, 1)                                                         # b c h w
        if self.is_ce:
            pred = softmax(pred.detach(),dim=1).argmax(dim=1)
        else:
            pred = pred.detach().sigmoid()
        if self.min_visibility is not None:
            if self.is_ce:
                pred = pred[:,None]
            if self.key == 'ped':
                mask = batch['visibility_ped'] >= self.min_visibility
            else:
                mask = batch['visibility'] >= self.min_visibility
            mask = mask[:, None].expand_as(pred)                                            # b c h w
            pred = pred[mask]                                                               # m
            label = label[mask]                                                             # m
        else:
            pred = pred.reshape(-1)
        return super().update(pred, label)
