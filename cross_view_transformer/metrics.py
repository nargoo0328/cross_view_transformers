import torch
from torchmetrics import Metric
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP
    MeanAveragePrecision = MAP

from typing import List, Optional
import numpy as np

from .util.box_ops import lidar_to_bev, box_cxcywh_to_xyxy, sincos2quaternion

import os
from typing import Any, Dict, List, Tuple

import json
import torch.distributed as dist
from pyquaternion import Quaternion
from pytorch_lightning.utilities import rank_zero_only
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox

get_attribute = dict(
    car = 'vehicle.moving', 
    truck = 'vehicle.parked', 
    bus = 'vehicle.moving',
    trailer = 'vehicle.parked', 
    construction_vehicle = 'vehicle.parked',
    pedestrian = 'pedestrian.standing',
    motorcycle = 'cycle.with_rider', 
    bicycle = 'cycle.without_rider'
)
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction_vehicle',
    'pedestrian',
    'motorcycle', 'bicycle',
    # 'emergency',
]

class BaseIoUMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, thresholds=[0.37, 0.39, 0.4, 0.41, 0.42, 0.43, 0.45, 0.5]): # np.linspace(0.3,0.6,31)
        super().__init__(dist_sync_on_step=False)
        # super().__init__()
        thresholds = torch.FloatTensor(thresholds)
        self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
        self.add_state('tp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')

    def update(self, pred, label):
        label = label.detach().bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]
        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)
        #self.tn += (~pred & ~label).sum(0)

    def compute(self):
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        return ious

    def show_result(self):
        thresholds = self.thresholds.squeeze(0)
        return {f'@{t.item():.2f}': {'tp:':self.tp, 'fp:':self.tp,'fn:':self.fn} for i, t in enumerate(thresholds)}
    
    def compute_recall(self):
        thresholds = self.thresholds.squeeze(0)
        recalls = self.tp / (self.tp + self.fn + 1e-7)
        
        return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, recalls)}

class IoUMetric(BaseIoUMetric):
    def __init__(self, label_indices: List[List[int]], min_visibility: Optional[int] = None, key= 'bev', sparse=False):
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
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.key = key
        self.sparse = sparse

    def update(self, pred, batch):

        if isinstance(pred, dict):
            pred = pred[self.key]                                                              # b c h w
        if isinstance(batch, dict):
            label = batch['bev']                                                                # b n h w
            
        label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
        label = torch.cat(label, 1)       
                                                          # b c h w
        pred = pred.clone().detach().sigmoid()
        if self.sparse:
            pred[pred == 0.5] = 0
        
        if self.min_visibility is not None:
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
    
    def compute(self):
        ious = super().compute()
        max_iou = torch.round(ious.max(), decimals=4)
        return {f"IoU_{self.key}": max_iou}

class BoxMAPMetric(MeanAveragePrecision):
    def __init__(self, box_3d, output_format, metrics_setting=None):
        self.output_format = output_format
        self.box_3d = box_3d
        super().__init__(**metrics_setting)
        # super().__init__()
    
    def update(self, pred, batch):
        # parse format
        if self.box_3d:
            boxes = pred['pred_boxes'][...,:4].clone().detach()
            boxes[...,2:4] = boxes[...,2:4].exp()
            if boxes.ndim == 4:
                boxes = boxes[-1]
            boxes = box_cxcywh_to_xyxy(boxes, transform=True)
            boxes = lidar_to_bev(boxes, batch['view'].detach())

            boxes_gt = []
            for i in range(len(batch['boxes'])):
                batch_boxes_gt = batch['boxes'][i][:,:4].clone().detach()
                batch_boxes_gt[...,2:4] = batch_boxes_gt[...,2:4].exp()
                boxes_gt.append(box_cxcywh_to_xyxy(batch_boxes_gt, transform=True))
            boxes_gt = [lidar_to_bev(batch_boxes_gt.unsqueeze(0), batch['view'][:1].detach())[0] for batch_boxes_gt in boxes_gt]
        else:
            boxes = pred['pred_boxes'].clone().detach()
            boxes = boxes * 200
            if boxes.ndim == 4:
                boxes = boxes[-1]
            boxes = box_cxcywh_to_xyxy(boxes, transform=False)

            boxes_gt = []
            for i in range(len(batch['boxes'])):
                batch_boxes_gt = batch['boxes'][i].clone().detach()
                batch_boxes_gt = batch_boxes_gt * 200
                boxes_gt.append(box_cxcywh_to_xyxy(batch_boxes_gt, transform=False))

        pred_logits = pred['pred_logits'].clone().detach()
        scores, labels = pred_logits.softmax(-1)[:, :, :-1].max(-1)
        preds = [{'boxes': box, 'scores': score, 'labels': label} for box, score, label in zip(boxes,scores,labels)]
        target = [{'boxes': box, 'labels': logit} for box, logit in zip(boxes_gt, batch['labels'])]
        super().update(preds, target)

    def compute(self, **kwargs):
        results = super().compute()
        return {k:v for k,v in results.items() if k in self.output_format}

class CustomNuscMetric:
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, nusc_path, pc_range, verbose=True, version='v1.0-trainval'): 
        _cfg = config_factory('detection_cvpr_2019')
        _nusc = NuScenes(version=version, verbose=False, dataroot=nusc_path)
        self.nusc_eval = NuScenesEvalCustom(_nusc, config=_cfg, verbose=verbose)

        self.pc_range = pc_range
        self.current_state = None
        self.reset()

    def reset(self):
        self.results_dict = dict(
            meta={
                'use_lidar': False,
                'use_camera': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False
            },
            results=dict()
        )
    
    def _set_current_state(self, state):
        self.current_state = state

    def update(self, pred, batch):
        pred_boxes = pred['pred_boxes'].clone().detach()

        cx = pred_boxes[..., 0:1]
        cy = pred_boxes[..., 1:2]
        cz = pred_boxes[..., 4:5]
        # size
        w = pred_boxes[..., 2:3]
        l = pred_boxes[..., 3:4]
        h = pred_boxes[..., 5:6]
        w = w.exp() 
        l = l.exp() 
        h = h.exp() 
        # transform to world
        ones = torch.ones_like(w)
        coor = torch.cat((cx,cy,cz,ones),dim=-1)
        coor = torch.einsum('b i j, b q j -> b q i', batch['pose'], coor)
        cx = coor[...,0:1]
        cy = coor[...,1:2]
        cz = coor[...,2:3]
        if pred_boxes.shape[-1]>6:
            rot_sine = pred_boxes[..., 6:7]
            rot_cosine = pred_boxes[..., 7:8]
            rots = torch.atan2(rot_sine, rot_cosine)
            pred_boxes = torch.cat([cx, cy, w, l, cz, h, rots], dim=-1).cpu().numpy()
        else:
            pred_boxes = torch.cat([cx, cy, w, l, cz, h], dim=-1).cpu().numpy()
        
        pred_scores, pred_labels = pred['pred_logits'].clone().detach().softmax(-1)[:, :, :-1].max(-1)
        pred_scores, pred_labels = pred_scores.cpu().numpy(), pred_labels.cpu().numpy()
        for token, boxes, labels, scores in zip(batch['token'], pred_boxes, pred_labels, pred_scores):
            # num_queries 6/num_classes
            tmp = []
            for box, label, score in zip(boxes, labels, scores):
                pred_class = DYNAMIC[label]
                if pred_boxes.shape[-1]>6:
                    rots = box[6]
                    x, i, j, k = list(sincos2quaternion(np.sin(rots), np.cos(rots)))
                else:
                    x, i, j, k = 0.0, 0.0, 0.0, 0.0
                box = [float(i) for i in box]
                tmp.append(
                    dict(
                        sample_token=token,
                        translation=[box[0],box[1],box[4]],
                        size=[box[2],box[3],box[5]],
                        rotation=[x, i, j, k],
                        velocity=[0.0,0.0],
                        detection_name=pred_class,
                        detection_score=float(score),
                        attribute_name=get_attribute[pred_class]
                    )
                )
            self.results_dict['results'][token] = tmp

    def compute(self, verbose=False):
        # rank = dist.get_rank()
        # world_size = dist.get_world_size()
        # if rank != 0:
        #     print("\n\nGathering at rank:",rank)
        #     dist.gather_object(obj=self.results_dict, object_gather_list=None, dst=0, group=_group.WORLD)
        #     print("\n\nFinish gathering at rank:",rank)
        #     return {}
        # else:
        #     list_gather_obj = [None] * world_size   # the container of gathered objects.
        #     print("\n\nGathering at rank:",rank)
        #     dist.gather_object(obj=self.results_dict, object_gather_list=list_gather_obj, dst=0, group=_group.WORLD)
        #     print("\n\nFinish gathering at rank:",rank)

        # results_dict = list_gather_obj[0]
        # for d in list_gather_obj[1:]:
        #     results_dict['results'].update(d['results'])
        path = './tmp_result.json'
        with open(path, 'w') as f:
            json.dump(self.results_dict, f)

        self.nusc_eval._set_current_epoch(self.current_state)
        metrics, _ = self.nusc_eval.evaluate()
        metrics_summary = metrics.serialize()
        if verbose:
            self.print_result(metrics_summary)
        result = {'mAP':metrics_summary['mean_ap']}
        class_aps = metrics_summary['mean_dist_aps']
        result.update({f"{class_name}_mAP": class_aps[class_name] for class_name in class_aps.keys()})
        return result

    def print_result(self, metrics_summary):
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'))
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err'],
                    class_tps[class_name]['attr_err']))

class NuScenesEvalCustom(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 output_dir: str = None,
                 verbose: bool = True,
                 result_path: str = './tmp_result.json'
                 ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.result_path = result_path

    def _set_current_epoch(self, eval_set):
        result_path = self.result_path
        self.eval_set = eval_set
        verbose = self.verbose

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'
        # Make dirs.
        if self.output_dir is not None:
            self.plot_dir = os.path.join(self.output_dir, 'plots')
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)
            if not os.path.isdir(self.plot_dir):
                os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')

        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."
        # return self.pred_boxes, self.gt_boxes
        # Add center distances.
        self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(self.nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(self.nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens
    