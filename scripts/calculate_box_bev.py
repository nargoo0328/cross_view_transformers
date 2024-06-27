from sklearn.cluster import DBSCAN
import torch
import cv2
from common import get_cfg, prepare_val
import numpy as np
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy

DATASET_DIR = '/media/hcis-s20/SRL/nuscenes/trainval/'

def apply_dbscan(data, eps, min_samples):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = clustering.labels_
    return labels

def map2points(data):
    xs, ys = np.where(data == 1)
    return np.array((xs, ys)).transpose()
    h, w = data.shape
    out = [[i,j] for j in range(w) for i in range(h) if data[i,j] == 1]
    # for i in range(h):
    #     for j in range(w):
    #         if data[i,j] == 1:
    #             out.append([i,j])
    return np.array(out)

def get_min_max(pts_list):
    min_x, max_x, min_y, max_y = 200, -1, 200, -1
    for (x,y) in pts_list:
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    return (min_x, min_y), (max_x, max_y)

def get_bev_from_box(batch, class_index=[[4, 5, 6, 7, 8, 10, 11,12]]):
    label = [batch['bev'][:, idx].max(1, keepdim=True).values for idx in class_index]
    label = torch.cat(label, 1)
    render = np.zeros((200,200),np.uint8)
    bev_pts = map2points(label[0,0])
    if len(bev_pts) == 0:
        return render
    clusters = apply_dbscan(bev_pts,1.0,3)
    for j in range(clusters.max()+1):
        tmp_index = np.where(clusters==j)[0]
        (y1,x1),(y2,x2) = get_min_max(bev_pts[tmp_index])
        cv2.rectangle(render, (x1, y1), (x2, y2), 1, -1)
    return render

if __name__ == '__main__':
    version = 'cvt_labels_nuscenes_v4'
    cfg1 = get_cfg(DATASET_DIR, version, 'cvt_nuscenes_det') # cvt_nuscenes_multiclass
    device = torch.device('cpu') # cuda:5
    # best resnet: 0830_232653, origin cvt: 0824_024032
    CHECKPOINT_PATH = '../logs/cross_view_transformers_test/0331_025303/checkpoints/last.ckpt'
    model, network, loader, viz, _ = prepare_val(cfg1,device,CHECKPOINT_PATH,mode='val',batch_size=1)
    model.metrics.reset()
    for k,m in model.metrics.items():
        m.thresholds = m.thresholds.to(device)
        m.tp = m.tp.to(device)
        m.fp = m.fp.to(device)
        m.fn = m.fn.to(device)

    with torch.no_grad():
        for i,batch in enumerate(loader):
            print(i,end='\r')
            box_vehicle = get_bev_from_box(batch)
            box_ped = get_bev_from_box(batch,[[9]])
            pred = {'bev':torch.from_numpy(box_vehicle)[None,None].to(device),'ped':torch.from_numpy(box_ped)[None,None].to(device)}
            batch['bev'] = batch['bev'].to(device)
            batch['visibility'] = batch['visibility'].to(device)
            batch['visibility_ped'] = batch['visibility_ped'].to(device)
            model.metrics.update(pred,batch)
            
    print()
    for k,m in model.metrics.items():
        print(k,':\n\t',m.compute(),'\n\t','='*50)
    