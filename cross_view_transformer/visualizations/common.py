import torch
import numpy as np
import cv2

from torch.nn.functional import softmax
from matplotlib.pyplot import get_cmap

from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy, sincos2quaternion
from nuscenes.utils import data_classes
from cross_view_transformer.data.common import INTERPOLATION
# many colors from
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
COLORS = {
    # static
    'lane':                 (110, 110, 110),
    'road_segment':         (90, 90, 90),

    # dividers
    'road_divider':         (255, 200, 0),
    'lane_divider':         (255, 200, 0),# (130, 130, 130),

    # dynamic
    'car':                  (255, 158, 0),
    'truck':                (255, 99, 71),
    'bus':                  (255, 127, 80),
    'trailer':              (255, 140, 0),
    'construction':         (233, 150, 70),
    'pedestrian':           (0, 0, 230),
    'motorcycle':           (255, 61, 99),
    'bicycle':              (220, 20, 60),
    'drivable_area':               (255, 127, 80),
    'ped_crossing':                    (255, 61, 99),
    'walkway':                (0, 207, 191),
    'carpark_area':                (34, 139, 34),
    'emergency':                (34, 139, 34),
    'stop_line':               (138, 43, 226),
    'nothing':              (200, 200, 200)
}
MAP_PALETTE = {
    "DRIVABLE": (166, 206, 227),
    "lane": (51, 160, 44),
    "PED": (251, 154, 153),
    "WALKWAY": (227, 26, 28),
    "STOPLINE": (253, 191, 111),
    "CARPARK": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "DIVIDER": (106, 61, 154),
}
COLORS_2 = {
    # static
    'STATIC':                 (110, 110, 110),
    # 'road_segment':         (90, 90, 90),

    # dividers
    'DIVIDER':                (255, 200, 0),
    # 'lane_divider':         (130, 130, 130),

    # dynamic
    'bev':                    (255, 158, 0),
    'VEHICLE':                 (255, 158, 0),
    # 'truck':                (255, 99, 71),
    # 'bus':                  (255, 127, 80),
    # 'trailer':              (255, 140, 0),
    # 'construction':         (233, 150, 70),
    'PED':                    (0, 0, 230),
    # 'motorcycle':           (255, 61, 99),
    # 'bicycle':              (220, 20, 60),
    'RD':                     (255, 200, 0),
    'LD':                     (130, 130, 130),
    'nothing':                (200, 200, 200),
    'DRIVABLE':               (255, 127, 80),
    'CROSSING':                (255, 61, 99),
    'WALKWAY':                (0, 207, 191),
    'CARPARK':                (34, 139, 34),
    'STOPLINE':               (138, 43, 226)
}
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
    # 'emergency',
]
label_colors = np.array([(200, 200, 200),(0, 0, 230),(255, 158, 0),(255, 200, 0),(110, 110, 110)])

label_colors_2 = np.array([(110, 110, 110),(255, 200, 0),(255, 158, 0),(0, 0, 230)])
def decode_segmap(image, nc=5):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=0)
    return rgb.transpose(1,2,0)

def colorize(x, colormap='winter'):
    """
    x: (h w) np.uint8 0-255
    colormap
    """
    try:
        return (255 * get_cmap(colormap)(x)[..., :3]).astype(np.uint8)
    except:
        pass

    if x.dtype == np.float32:
        x = (255 * x).astype(np.uint8)

    if colormap is None:
        return x[..., None].repeat(3, 2)

    return cv2.applyColorMap(x, getattr(cv2, f'COLORMAP_{colormap.upper()}'))


def get_colors(semantics):
    # change to COLORS_2
    return np.array([COLORS[s] for s in semantics], dtype=np.uint8)


def to_image(x):
    return (255 * x).byte().cpu().numpy().transpose(1, 2, 0)


def greyscale(x):
    return (255 * x.repeat(3, 2)).astype(np.uint8)


def resize(src, dst=None, shape=None, idx=0):
    if dst is not None:
        ratio = dst.shape[idx] / src.shape[idx]
    elif shape is not None:
        ratio = shape[idx] / src.shape[idx]

    width = int(ratio * src.shape[1])
    height = int(ratio * src.shape[0])

    return cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)


class BaseViz:
    SEMANTICS = []

    def __init__(self, label_indices, key = ['STATIC','DIVIDER','bev','ped'], flip=False, box='', bev=True, orientation=False, mask=False, box_3d=True):
        
        self.label_indices = label_indices[0]
        SEMANTICS = [self.SEMANTICS[i] for i in self.label_indices]
        self.colors = get_colors(SEMANTICS)
        self.cmap = COLORS_2
        self.key = key
        self.flip = flip
        self.box = box
        self.bev = bev
        self.orientation = orientation
        self.mask = mask
        self.box_3d = box_3d

    def draw_ego(self, img, view):

        points = np.array([
                [-4.0 / 2 + 0.3, -1.73 / 2, 1],
                [-4.0 / 2 + 0.3,  1.73 / 2, 1],
                [ 4.0 / 2 + 0.3,  1.73 / 2, 1],
                [ 4.0 / 2 + 0.3, -1.73 / 2, 1],
            ])
        if self.flip:
            tmp = np.array(points[:,0])
            points[:,0] = points[:,1]
            points[:,1] = tmp
        
        points = view @ points.T
        cv2.fillPoly(img, [points.astype(np.int32)[:2].T], color=(164, 0, 0))
        return img

    def visuaulize_pred(self, pred, view, threshold, mask=None):
        # class_num = pred.shape[0]
        # pred[pred>threshold]=1
        # pred[pred<=threshold]=0
        # pre=torch.zeros(200,200,3).type(torch.uint8)
        # pre += 200
        # for i, k in enumerate(self.key):
        #     # if i==1 or i==3:
        #     #     continue
        #     pre[...,0][pred[i]==1]=self.cmap[k][0]
        #     pre[...,1][pred[i]==1]=self.cmap[k][1]
        #     pre[...,2][pred[i]==1]=self.cmap[k][2]

        # pre = pre.cpu().numpy()
        # pre = pre.astype(np.uint8)
        pred = pred * mask
        pred_vis = pred.squeeze(0).cpu().numpy() * 255
        pred_vis = pred_vis.astype(np.uint8)
        pred_vis = colorize(pred_vis)

        pred_vis = self.draw_ego(pred_vis, view)
        return pred_vis

    def visualize_bev(self, bev, view, scalar=1e-5):
        """
        (c, h, w) torch [0, 1] float

        returns (h, w, 3) np.uint8
        """
        if isinstance(bev, torch.Tensor):
            bev = bev.cpu().numpy().transpose(1, 2, 0)

        h, w, c = bev.shape

        # Prioritize higher class labels
        # c = 12
        # bev = bev[..., :c]
        c = len(self.label_indices)
        bev = bev[..., self.label_indices]
        eps = (scalar * np.arange(c))[None, None]
        idx = (bev + eps).argmax(axis=-1)
        
        # idx = (torch.tensor([c-1]).expand(h,w).type(torch.int64)).numpy()
        
        val = np.take_along_axis(bev, idx[..., None], -1)

        # Spots with no labels are light grey
        empty = np.uint8(COLORS['nothing'])[None, None]

        result = (val * self.colors[idx]) + ((1 - val) * empty)
        result = np.uint8(result)
        
        result = self.draw_ego(result, view)
        return result

    def visualize_custom(self, batch, pred_dict, b, threshold):
        if pred_dict is None:
            return []
        bev = batch['bev']
        view = batch['view'][0].cpu().numpy()
        mask = pred_dict['mask'][b].detach().cpu() if 'mask' in pred_dict else None
        # out = [self.visualize_pred(bev[b], pred[s][b].sigmoid()) for s in ['bev','ped']]

        tmp = list()
        for k in self.key:
            tmp.append(pred_dict[k][b].detach().cpu().sigmoid())
        pred = torch.cat(tmp, dim=1)

        right = self.visuaulize_pred(pred, view, threshold, mask)

        aux_list = []
        if 'aux' in pred_dict:
            for aux_pred in pred_dict['aux']:
                tmp = list()
                for k in self.key:
                    tmp.append(aux_pred[k][b].detach().cpu().sigmoid())
                pred = torch.cat(tmp, dim=1)
                aux_list.append(self.visuaulize_pred(pred, view, threshold))

        return [right] + aux_list
    
    # copied from data.transforms
    def _prepare_augmented_boxes(self, bev_aug, points, inverse=True):
        points_in = np.copy(points)
        Rquery = np.zeros((3, 3))
        if inverse:
            # Inverse query aug:
            # Ex: when tx=10, the query is 10/res meters front,
            # so points are fictivelly 10/res meters back.
            Rquery[:3, :3] = bev_aug[:3, :3].T
            tquery = np.array([-1, -1, 1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ (X + tquery)
            points_out = (Rquery @ (points_in[:3, :] + tquery))
        else:
            Rquery[:3, :3] = bev_aug[:3, :3]
            tquery = np.array([1, 1, -1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ X + tquery
            points_out = ((Rquery @ points_in[:3, :]) + tquery)

        return points_out
    
    def visualize_det(self, batch, b, pred=None, threshold=0.6):

        def check_index(x1, y1, x2, y2):
            return x1>=0 and x1<200 and x2>=0 and x2<200 and y1>=0 and y1<200 and y2>=0 and y2<200
        
        def parse_xx(x1, x2):
            if x1 >= x2:
                return x1, x2
            else:
                return x2, x1
                    
        gtBox_bev = np.zeros((200,200,3),np.uint8)
        gtBox_bev += 200

        view = batch['view'][b].cpu().numpy()
        labels = batch['labels'][b].clone().detach()
        gt_box = batch['boxes'][b].clone().detach()

        gt_box[:,2:4] = gt_box[:,2:4].exp()
        gt_box[:,5:6] = gt_box[:,5:6].exp()
        gt_box = gt_box.cpu().numpy()

        if self.orientation:
            for box, label in zip(gt_box, labels):
                translation = [box[0],box[1],box[4]]
                size = [box[2],box[3],box[5]]
                rotation = sincos2quaternion(box[6],box[7])
                box = data_classes.Box(translation, size, rotation)
                p = box.bottom_corners()[:2]                                              # 2 4
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                       # 3 4
                p = (view @ p )[:2]
                color = COLORS[DYNAMIC[label]]
                cv2.fillPoly(gtBox_bev, [p.round().astype(np.int32).T], color, INTERPOLATION)
        else:
            gt_box = gt_box[:,:4]
            gt_box = box_cxcywh_to_xyxy(gt_box, transform=True)

            for (x1,y1,x2,y2), label in zip(gt_box, labels):
                pts = np.array([[x1,y1,1],[x2,y2,1]]).transpose()
                # if 'bev_augm' in batch:
                #     pts = self._prepare_augmented_boxes(batch['bev_augm'][b].cpu().numpy(), pts)
                pts = view @ pts
                pts = pts.astype(np.uint8)
                x1, y1 = pts[:2,0] 
                x2, y2 = pts[:2,1]

                # x1, x2 = parse_xx(x1, x2)
                # y1, y2 = parse_xx(y1, y2)
                
                # if not check_index(x1, y1, x2, y2):
                #     continue
                color = COLORS[DYNAMIC[label]]
                cv2.rectangle(gtBox_bev, (x1, y1), (x2, y2), color, -1)
        
        gtBox_bev = self.draw_ego(gtBox_bev, view)
        if pred is None or 'pred_boxes' not in pred:
            return [gtBox_bev]
        
        predBox_bev = np.zeros((200,200,3),np.uint8)
        predBox_bev += 200

        pred_boxes = pred['pred_boxes'].clone().detach()
        pred_logits = pred['pred_logits'].clone().detach()

        if pred_logits.ndim == 4:
            pred_boxes = pred_boxes[-1]
            pred_logits = pred_logits[-1]
            
        pred_boxes = pred_boxes[b]
        pred_logits = pred_logits[b]
        pred_boxes[:,2:4] = pred_boxes[:,2:4].exp()
        pred_boxes[:,5:6] = pred_boxes[:,5:6].exp()
        scores, labels = pred_logits.softmax(-1)[:, :-1].max(-1) # N, num_classes
        if self.orientation:
            rots = torch.atan2(pred_boxes[...,6:7], pred_boxes[...,7:8]).cpu().numpy()
            pred_boxes = pred_boxes.cpu().numpy()
            for box, score, label, rot in zip(pred_boxes, scores, labels, rots):
                if score < threshold:
                    continue
                translation = [box[0],box[1],box[4]]
                size = [box[2],box[3],box[5]]
                rotation = sincos2quaternion(np.sin(rot[0]),np.cos(rot[0]))
                box = data_classes.Box(translation, size, rotation)
                p = box.bottom_corners()[:2]                                              # 2 4
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                      # 3 4
                p = (view @ p )[:2]
                color = COLORS[DYNAMIC[label]]
                cv2.fillPoly(predBox_bev, [p.round().astype(np.int32).T], color, INTERPOLATION)
    
        else:
            pred_box = pred_boxes[:,:4].cpu().numpy()
            pred_box = box_cxcywh_to_xyxy(pred_box, transform=True)
            # pred_box = pred_box * 100 - 50 
            # for box, logit in zip(pred_box,pred_logits):
            for (x1,y1,x2,y2), score, label in zip(pred_box, scores, labels):
                if score < threshold:
                    continue
                pts = np.array([[x1,y1,1],[x2,y2,1]]).transpose()
                pts = view @ pts
                pts = pts.astype(np.uint8)
                x1,y1 = pts[:2,0] 
                x2,y2 = pts[:2,1]
                if not check_index(x1, y1, x2, y2):
                    continue
                color = COLORS[DYNAMIC[label]]
                cv2.rectangle(predBox_bev, (x1, y1), (x2, y2), color, -1)
        
        predBox_bev = self.draw_ego(predBox_bev, view)
        return [gtBox_bev, predBox_bev]
    
    def visualize_det_2d(self, batch, b, pred=None, threshold=0.6):

        def check_index(x1, y1, x2, y2):
            return x1>=0 and x1<200 and x2>=0 and x2<200 and y1>=0 and y1<200 and y2>=0 and y2<200
        
        view = batch['view'][b].cpu().numpy()
        gtBox_bev = np.zeros((200,200,3),np.uint8)
        gtBox_bev += 200
        labels = batch['labels'][b].clone().detach()
        gt_box = batch['boxes'][b].clone().detach()
        gt_box = gt_box * 200
        gt_box = gt_box.cpu().numpy()
        gt_box = box_cxcywh_to_xyxy(gt_box, transform=False)

        for pts, label in zip(gt_box, labels):
            # if not check_index(x1, y1, x2, y2):
            #     continue
            color = COLORS[DYNAMIC[label]]
            x1,y1,x2,y2 = pts.round()
            cv2.rectangle(gtBox_bev, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        
        gtBox_bev = self.draw_ego(gtBox_bev, view)
        if pred is None or 'pred_boxes' not in pred:
            return [gtBox_bev]
        
        predBox_bev = np.zeros((200,200,3),np.uint8)
        predBox_bev += 200
        pred_boxes = pred['pred_boxes'].clone().detach()
        pred_logits = pred['pred_logits'].clone().detach()

        if pred_logits.ndim == 4:
            pred_boxes = pred_boxes[-1]
            pred_logits = pred_logits[-1]
            
        pred_boxes = pred_boxes[b]
        pred_logits = pred_logits[b]

        scores, labels = pred_logits.softmax(-1)[:, :-1].max(-1) # N, num_classes
        
        pred_boxes = (pred_boxes * 200).cpu().numpy()
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes, transform=False)

        for pts, score, label in zip(pred_boxes, scores, labels):
            if score < threshold:
                continue
            color = COLORS[DYNAMIC[label]]
            x1,y1,x2,y2 = pts.round()
            cv2.rectangle(predBox_bev, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        
        predBox_bev = self.draw_ego(predBox_bev, view)
        return [gtBox_bev, predBox_bev]
    
    def visualize_mask(self, batch, b, pred):
        if not 'mask' in pred:
            return []
        mask = pred['mask'][b,0].float().cpu().numpy()
        img = np.stack([mask, mask, mask], axis=-1)
        img = (img * 255).astype(np.uint8)

        return [img]

    @torch.no_grad()
    def visualize(self, batch, pred=None, b_max=8, threshold=0.5 ,**kwargs):
        bev = batch['bev']
        batch_size = bev.shape[0]
        for b in range(min(batch_size, b_max)):
            # if pred is not None:
            #     right = self.visualize_pred(bev[b], pred['bev'][b].sigmoid())
            # else:
                # right = self.visualize_bev(bev[b])
            right = []

            if self.bev:
                right = self.visualize_bev(bev[b],batch['view'][0].cpu().numpy())
                right = [right] + self.visualize_custom(batch, pred, b,threshold)

            if self.box:
                if self.box_3d:
                    right = right + self.visualize_det(batch, b, pred)
                else:
                    right = right + self.visualize_det_2d(batch, b, pred)

            if self.mask:
                right = right + self.visualize_mask(batch, b, pred)
            
            for x in right:
                x[:,-1] = [0,0,0]
                
            right = [x for x in right if x is not None]
            right = np.hstack(right)

            image = None if not hasattr(batch.get('image'), 'shape') else batch['image']

            if image is not None:
                imgs = [to_image(image[b][i]) for i in range(image.shape[1])]

                if len(imgs) == 6:
                    a = np.hstack(imgs[:3])
                    b = np.hstack(imgs[3:])
                    left = resize(np.vstack((a, b)), right)
                else:
                    left = np.hstack([resize(x, right) for x in imgs])

                yield np.hstack((left, right))
            else:
                yield right

    def __call__(self, batch=None, pred=None, **kwargs):
        return list(self.visualize(batch=batch, pred=pred, **kwargs))
