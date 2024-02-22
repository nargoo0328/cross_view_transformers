import torch
import numpy as np
import cv2

from torch.nn.functional import softmax
from matplotlib.pyplot import get_cmap


# many colors from
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
COLORS = {
    # static
    'lane':                 (110, 110, 110),
    'road_segment':         (90, 90, 90),

    # dividers
    'road_divider':         (255, 200, 0),
    'lane_divider':         (130, 130, 130),

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
    'vehicle':                 (255, 158, 0),
    # 'truck':                (255, 99, 71),
    # 'bus':                  (255, 127, 80),
    # 'trailer':              (255, 140, 0),
    # 'construction':         (233, 150, 70),
    'ped':                    (0, 0, 230),
    # 'motorcycle':           (255, 61, 99),
    # 'bicycle':              (220, 20, 60),
    'RD':                     (255, 200, 0),
    'LD':                     (130, 130, 130),
    'nothing':                (200, 200, 200),
    'DRIVABLE':               (255, 127, 80),
    'PED':                    (255, 61, 99),
    'WALKWAY':                (0, 207, 191),
    'CARPARK':                (34, 139, 34),
    'STOPLINE':               (138, 43, 226)
}

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

def colorize(x, colormap=None):
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

    def __init__(self, label_indices=None, colormap='inferno',key = ['STATIC','DIVIDER','bev','ped'],flip=False):
        self.label_indices = label_indices
        self.colors = get_colors(self.SEMANTICS)
        self.colormap = colormap
        self.cmap = COLORS_2#MAP_PALETTE if 'DRIVABLE' in key else COLORS_2
        self.key = key
        self.flip = flip

    def visualize_pred(self, bev, pred, threshold=None):
        """
        (c, h, w) torch float {0, 1}
        (c, h, w) torch float [0-1]
        """
        if isinstance(bev, torch.Tensor):
            bev = bev.cpu().numpy().transpose(1, 2, 0)

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy().transpose(1, 2, 0)

        if self.label_indices is not None:
            bev = [bev[..., idx].max(-1) for idx in self.label_indices]
            bev = np.stack(bev, -1)

        if threshold is not None:
            pred = (pred > threshold).astype(np.float32)

        result = colorize((255 * pred.squeeze(2)).astype(np.uint8), self.colormap)
        return result

    def visuaulize_pred_v2(self,pred,view,threshold):
        class_num = pred.shape[0]
        pred[pred>=threshold]=1
        pred[pred<threshold]=0
        pre=torch.zeros(200,200,3).type(torch.uint8)
        pre+=200
        for i,k in enumerate(self.key):
            # if i==1 or i==3:
            #     continue
            pre[...,0][pred[i]==1]=self.cmap[k][0]
            pre[...,1][pred[i]==1]=self.cmap[k][1]
            pre[...,2][pred[i]==1]=self.cmap[k][2]

        pre = pre.cpu().numpy()
        pre = pre.astype(np.uint8)

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


        cv2.fillPoly(pre, [points.astype(np.int32)[:2].T], color=(164, 0, 0))
        return pre

    # def visualize_pred_unused(self, bev, pred, threshold=None):
    #     h, w, c = pred.shape

    #     img = np.zeros((h, w, 3), dtype=np.float32)
    #     img[...] = 0.5
    #     colors = np.float32([
    #         [0, .6, 0],
    #         [1, .7, 0],
    #         [1,  0, 0]
    #     ])
    #     tp = (pred > threshold) & (bev > threshold)
    #     fp = (pred > threshold) & (bev < threshold)
    #     fn = (pred <= threshold) & (bev > threshold)

    #     for channel in range(c):
    #         for i, m in enumerate([tp, fp, fn]):
    #             img[m[..., channel]] = colors[i][None]

    #     return (255 * img).astype(np.uint8)

    def visualize_bev(self, bev, view,scalar=1e-5):
        """
        (c, h, w) torch [0, 1] float

        returns (h, w, 3) np.uint8
        """
        if isinstance(bev, torch.Tensor):
            bev = bev.cpu().numpy().transpose(1, 2, 0)

        h, w, c = bev.shape
        assert c == len(self.SEMANTICS), c

        # Prioritize higher class labels
        c = 12
        bev = bev[...,:c]
        eps = (scalar * np.arange(c))[None, None]
        idx = (bev + eps).argmax(axis=-1)
        
        # idx = (torch.tensor([c-1]).expand(h,w).type(torch.int64)).numpy()
        
        val = np.take_along_axis(bev, idx[..., None], -1)

        # Spots with no labels are light grey
        empty = np.uint8(COLORS['nothing'])[None, None]

        result = (val * self.colors[idx]) + ((1 - val) * empty)
        result = np.uint8(result)
        
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


        cv2.fillPoly(result, [points.astype(np.int32)[:2].T], color=(164, 0, 0))
        return result

    def visualize_custom(self, batch, pred, b, threshold):
        bev = batch['bev']
        # out = [self.visualize_pred(bev[b], pred[s][b].sigmoid()) for s in ['bev','ped']]

        tmp = list()
        for k in self.key:
            tmp.append(pred[k].sigmoid())
        pred = torch.cat(tmp,dim=1)
        right = self.visuaulize_pred_v2(pred[b],batch['view'][0].cpu().numpy(),threshold)

        # out = [self.visualize_bev(pred[b],c) for c in np.linspace(0.1,0.5,9)]
        return [right] #out+[right]

    @torch.no_grad()
    def visualize(self, batch, pred=None, b_max=8, threshold=0.5 ,**kwargs):
        bev = batch['bev']
        batch_size = bev.shape[0]
        for b in range(min(batch_size, b_max)):
            # if pred is not None:
            #     right = self.visualize_pred(bev[b], pred['bev'][b].sigmoid())
            # else:
                # right = self.visualize_bev(bev[b])
            right = self.visualize_bev(bev[b],batch['view'][0].cpu().numpy())
            right = [right] + self.visualize_custom(batch, pred, b,threshold)
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
