from hydra import core, initialize, compose
from omegaconf import OmegaConf
from cross_view_transformer.common import setup_experiment, load_backbone
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy
import cv2
from cross_view_transformer.metrics import CustomNuscMetric

def prepare_val(cfg,device,CHECKPOINT_PATH=None,batch_size=1,mode='split'):
    core.global_hydra.GlobalHydra.instance().clear()        # required for Hydra in notebooks
    initialize(config_path='./config')
    OmegaConf.resolve(cfg)
    model, data, viz = setup_experiment(cfg)


    # load dataset
    if mode == 'split':
        SPLIT = 'val_qualitative_000'
        SUBSAMPLE = 5
        dataset = data.get_split(SPLIT, loader=False)
        dataset = torch.utils.data.ConcatDataset(dataset)
        dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else :
        dataset = data.get_split(mode, loader=False)
        dataset = torch.utils.data.ConcatDataset(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    print("Dataset length:",len(dataset))
    if CHECKPOINT_PATH is not None:
        try:
        # if Path(CHECKPOINT_PATH).exists():
            network = load_backbone(CHECKPOINT_PATH,device=device)
            print("Loaded checkpoint.")
        except:
        # else:
            print("Checkpoint loading failed.")
            network = model.backbone
        #     network = model.backbone
    else:
        network = model.backbone

    return model, network, loader, viz, dataset

def get_cfg(DATASET_DIR,LABELS_DIR,exp):
    core.global_hydra.GlobalHydra.instance().clear()    
    initialize(config_path='../config')
    cfg = compose(
        config_name='config',
        overrides=[
            'experiment.save_dir=./logs/',                
            f'+experiment={exp}',
            f'data.dataset_dir={DATASET_DIR}',
            f'data.labels_dir={DATASET_DIR+LABELS_DIR}',
            'data.version=v1.0-trainval',
            # 'loader.batch_size=10',
        ]
    )
    return cfg

def _get_nusc_metric():
    nusc_path, pc_range = '/media/hcis-s20/SRL/nuscenes/trainval/', [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
    return CustomNuscMetric(nusc_path, pc_range)

def calculate_iou(model,network,loader, device, metric_mode):
    score_threshold = 0.6
    print("score_threshold:", score_threshold)
    if metric_mode == 'iou' or metric_mode == 'box_projection':
        iou_metrics = ['iou_ped','iou_vehicle']
        for k,m in model.metrics.items():
            if k not in iou_metrics:
                continue
            m.thresholds = m.thresholds.to(device)
            m.tp = m.tp.to(device)
            m.fp = m.fp.to(device)
            m.fn = m.fn.to(device)
    elif metric_mode == 'nusc':
        nusc_metric = _get_nusc_metric()
        nusc_metric._set_current_state('val')

    network.to(device)
    network.eval()
    with torch.no_grad():
        for i,batch in enumerate(loader):
    #         start = time.time()
            print(i,end='\r')
            for k, v in batch.items():
                if k!='features' or k!='center':
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                    elif isinstance(v, list):
                        if isinstance(v[0],torch.Tensor):
                            batch[k] = [_v.to(device) for _v in v]
                    else:
                        batch[k] = v

            pred = network(batch)

            if metric_mode == 'box_projection':
            # project box 
                b = batch['bev'].shape[0]
                render = np.zeros((b,2,200,200),np.float32)
                pred_box = box_cxcywh_to_xyxy(pred['pred_boxes'].detach() * 200, transform=False).cpu().numpy()
                # pred_box = pred_box * 100 - 50 
                scores, labels = pred['pred_logits'].softmax(-1)[..., :-1].max(-1)
                # pred_logits = pred['pred_logits'][0].detach().softmax(1).argmax(1).cpu().numpy() # N, num_classes
                # for box, logit in zip(pred_box,pred_logits):
                for j in range(b):
                    for (x1,y1,x2,y2), score, label in zip(pred_box[j], scores[j], labels[j]):
                        if score < score_threshold:
                            continue
                        label = 0 if label != 5 else 1
                        cv2.rectangle(render[j][label], (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
                pred['bev'] = torch.from_numpy(render[:,0:1]).to(device)
                pred['ped'] = torch.from_numpy(render[:,1:2]).to(device)

            if metric_mode == 'iou' or metric_mode == 'box_projection':
                model.metrics.update(pred,batch)
            elif metric_mode == 'nusc':
                nusc_metric.update(pred,batch)
    print()
    if metric_mode == 'iou' or metric_mode == 'box_projection':
        for k,m in model.metrics.items():
            print(k,':\n\t',m.compute(),'\n','='*50)
            if k in iou_metrics:
                print(m.compute_recall())
    elif metric_mode == 'nusc':
        nusc_out = nusc_metric.compute(verbose=True)
        del nusc_out['traffic_cone_mAP']
        del nusc_out['barrier_mAP']
        del nusc_out['mAP']
        count = 0.0
        for _, v in nusc_out.items():
            count += v
        nusc_out['mAP'] = count/8.0
        print(nusc_out)