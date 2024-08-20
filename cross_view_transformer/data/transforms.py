import pathlib

import torch
import torchvision
from torch.nn import functional as F
import numpy as np
import cv2

from PIL import Image
from .common import encode, decode, apply_dbscan, map2points, get_min_max, INTERPOLATION
from .augmentations import StrongAug, GeometricAug, RandomTransformImage, RandomTransformationBev
import os
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy, sincos2quaternion
from nuscenes.utils.data_classes import Box

def get_5d_view(view):
    # device = view.device
    x,y,z = view[0]
    i,j,k = view[1]
    return torch.tensor(
        [
            [x,y,0,0,z],
            [i,j,0,0,k],
            [0,0,x,y,z],
            [0,0,i,j,k],
            [0,0,0,0,1],
        ]
    )

MAX_PTS = 10000
z_stats = np.array([
            [0.9496, 1.7372], # car
            [1.5563, 2.8328], # truck
            [1.8626, 3.5100], # bus
            [2.1127, 3.8164], # trailer
            [1.3749, 2.5279], # construction
            [1.0667, 1.7676], # pedestrian
            [0.8392, 1.4717], # motorcycle
            [0.7497, 1.3034], # bicycle
            [0, 0] # no_class
        ])

class Sample(dict):
    def __init__(
        self,
        token,
        scene,
        intrinsics,
        extrinsics,
        images,
        view,
        bev,
        radar= None,
        lidar= None,
        gt_box= None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Used to create path in save/load
        self.token = token
        self.scene = scene

        self.view = view
        self.bev = bev

        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.radar = radar
        self.lidar = lidar
        self.gt_box = gt_box

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val

        return super().__setattr__(key, val)


class SaveDataTransform:
    """
    All data to be saved to .json must be passed in as native Python lists
    """
    def __init__(self, labels_dir):
        self.labels_dir = pathlib.Path(labels_dir)

    def get_cameras(self, batch: Sample):
        return {
            'images': batch.images,
            'intrinsics': batch.intrinsics,
            'extrinsics': batch.extrinsics
        }

    def get_bev(self, batch: Sample):
        result = {
            'view': batch.view,
        }

        scene_dir = self.labels_dir / batch.scene

        bev_path = f'bev_{batch.token}.png'
        with open(scene_dir / bev_path, 'wb') as f:
            Image.fromarray(encode(batch.bev)).save(f)

        result['bev'] = bev_path

        # Auxilliary labels
        if batch.get('aux') is not None:
            aux_path = f'aux_{batch.token}.npz'
            np.savez_compressed(scene_dir / aux_path, aux=batch.aux)

            result['aux'] = aux_path

        if batch.get('aux_ped') is not None:
            aux_path = f'aux_ped_{batch.token}.npz'
            np.savez_compressed(scene_dir / aux_path, aux_ped=batch.aux_ped)

            result['aux_ped'] = aux_path

        # Visibility mask
        if batch.get('visibility') is not None:
            visibility_path = f'visibility_{batch.token}.png'
            with open(scene_dir / visibility_path, 'wb') as f:
                Image.fromarray(batch.visibility).save(f)

            result['visibility'] = visibility_path

        if batch.get('visibility_ped') is not None:
            visibility_path = f'visibility_ped_{batch.token}.png'
            with open(scene_dir / visibility_path, 'wb') as f:
                Image.fromarray(batch.visibility_ped).save(f)

            result['visibility_ped'] = visibility_path

        if batch.get('radar') is not None:
            radar_path = f'radar_{batch.token}.npz'
            radar_bev, radar_points = batch.radar
            np.savez_compressed(scene_dir / radar_path, radar=radar_bev, points=radar_points)
            result['radar'] = radar_path

        if batch.get('lidar') is not None:
            lidar_path = f'lidar_{batch.token}.npz'
            lidar_bev, lidar_pts_path, lidar_pose = batch.lidar
            np.savez_compressed(scene_dir / lidar_path, lidar=lidar_bev)
            result['lidar'] = lidar_path
            result['lidar_pose'] = lidar_pose
            result['lidar_pts'] = lidar_pts_path
        
        if batch.get('gt_box') is not None:
            gt_box_path = f'gt_box_{batch.token}.npz'
            gt_box = batch.gt_box
            np.savez_compressed(scene_dir / gt_box_path, gt_box=gt_box)
            result['gt_box'] = gt_box_path

        return result

    def __call__(self, batch):
        """
        Save sensor/label data and return any additional info to be saved to json
        """
        result = {}
        result.update(self.get_cameras(batch))
        result.update(self.get_bev(batch))
        result.update({k: v for k, v in batch.items() if k not in result})
        # result.update(self.parse_box(batch))

        return result


class LoadDataTransform(torchvision.transforms.ToTensor):
    def __init__(self, dataset_dir, labels_dir, image_config, num_classes, image_data=True, lidar=None, box='',split_intrin_extrin=False, orientation=False, augment_img=False, augment_bev=False, no_class=False, img_params=None, bev_aug_conf=None, training=True, box_3d=True, **kwargs):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes
        self.image_data = image_data
        self.lidar = lidar
        assert box in ['', 'gt', 'pseudo']
        self.box = box
        self.orientation = orientation
        self.no_class = no_class
        self.box_3d = box_3d
        self.split_intrin_extrin = split_intrin_extrin

        self.img_transform = torchvision.transforms.ToTensor()

        self.augment_img = RandomTransformImage(img_params, training) if augment_img else None
        self.augment_bev = RandomTransformationBev(bev_aug_conf, training) if augment_bev else None

        self.to_tensor = super().__call__

    def get_cameras(self, sample: Sample, h, w, top_crop):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        intrinsics = list()
        lidar2img = list()

        for image_path, I_original, extrinsic in zip(sample.images, sample.intrinsics, sample.extrinsics):
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir / image_path)

            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))
            images.append(self.img_transform(image_new))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop
            extrinsic = np.float32(extrinsic)
            
            if not self.split_intrin_extrin:
                viewpad = np.float32(np.eye(4))
                viewpad[:I.shape[0], :I.shape[1]] = I
                lidar2img.append(torch.tensor(viewpad @ extrinsic))
            else:
                intrinsics.append(torch.tensor(I))

        result = {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
        }

        sensor = {}
        if not self.split_intrin_extrin:
            sensor = {
                'lidar2img': torch.stack(lidar2img, 0),
            }
        else:
            sensor = {
                'intrinsics': torch.stack(intrinsics, 0),
                'extrinsics': torch.tensor(np.float32(sample.extrinsics)),
            }

        result.update(sensor)
        return result
    
    def get_cameras_augm(self, sample: Sample, **kwargs):
        images = list()
        intrinsics = list()
        extrinsics = list()

        for image_path, intrinsic, extrinsic in zip(sample.images, sample.intrinsics, sample.extrinsics):
            image = Image.open(self.dataset_dir / image_path)
            images.append(image)

            intrinsic = np.float32(intrinsic)
            extrinsic = np.float32(extrinsic)
            intrinsics.append(intrinsic)
            extrinsics.append(torch.tensor(extrinsic))

        result = {'image': images}
        result.update({'intrinsics':intrinsics, 'extrinsics':extrinsics})

        result = self.augment_img(result)
        result['image'] = torch.stack(result['image'], 0)
        result['intrinsics'] = torch.stack(result['intrinsics'], 0)
        result['extrinsics'] = torch.stack(result['extrinsics'], 0)

        lidar2img = list()
        for intrinsic,  extrinsic in zip(result['intrinsics'], result['extrinsics']):
            viewpad = torch.eye(4, dtype=torch.float32)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img.append(viewpad @ extrinsic)
        result['lidar2img'] = torch.stack(lidar2img, 0)

        result['cam_idx'] = torch.LongTensor(sample.cam_ids)

        return result

    def get_bev(self, sample: Sample):
        scene_dir = self.labels_dir / sample.scene
        bev = None

        if sample.bev is not None:
            bev = Image.open(scene_dir / sample.bev)
            bev = decode(bev, self.num_classes)
            bev = (255 * bev).astype(np.uint8)
            bev = self.to_tensor(bev)

        result = {
            'bev': bev,
        }

        if 'visibility' in sample:
            visibility = Image.open(scene_dir / sample.visibility)
            result['visibility'] = np.array(visibility, dtype=np.uint8)

        if 'visibility_ped' in sample:
            visibility = Image.open(scene_dir / sample.visibility_ped)
            result['visibility_ped'] = np.array(visibility, dtype=np.uint8)

        if 'aux' in sample:
            aux = np.load(scene_dir / sample.aux)['aux']
            result['segmentation'] = self.to_tensor(aux[..., 0])
            result['center'] = self.to_tensor(aux[..., 1])
            result['offset'] = self.to_tensor(aux[..., 2:4])
            result['hw'] = self.to_tensor(aux[..., -2:])

        if 'aux_ped' in sample:
            aux_ped = np.load(scene_dir / sample.aux_ped)['aux_ped']
            result['center_ped'] = self.to_tensor(aux_ped[..., 1])

        if 'pose' in sample:
            result['pose'] = np.float32(sample['pose'])  

        if 'pose_inverse' in sample:
            result['pose_inverse'] = np.float32(sample['pose_inverse'])        

        # if 'lidar' in sample:
        #     lidar = np.load(scene_dir / sample.lidar) # 16
        #     lidar = lidar['lidar'].astype(np.float32)
        #     lidar = torch.from_numpy(lidar).permute(2,0,1)
        #     # n_pts = radar.shape[-1]
        #     # if n_pts<MAX_PTS:
        #     #     radar = F.pad(radar, (0, MAX_PTS-n_pts), value=0)
        #     # else:
        #     #     radar = radar[...,:MAX_PTS]

        #     result['lidar'] = lidar

        result['token'] = sample['token']
        return result
    
    def get_lidar(self, sample: Sample):
        T = 10

        def preprocess(lidar):
        # shuffling the points
            np.random.shuffle(lidar)

            voxel_coords = ((lidar[:, :3] - np.array([-50, -50, -5])) / (
                            0.1, 0.1, 0.2)).astype(int)

            # convert to  (D, H, W)
            voxel_coords = voxel_coords[:,[2,1,0]]
            voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, \
                                                    return_inverse=True, return_counts=True)

            voxel_features = []

            for i in range(len(voxel_coords)):
                voxel = np.zeros((T, 7), dtype=np.float32)
                pts = lidar[inv_ind == i]
                if voxel_counts[i] > T:
                    pts = pts[:T, :]
                    voxel_counts[i] = T
                # augment the points
                voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
                voxel_features.append(voxel)
            return np.array(voxel_features), voxel_coords
        
        def shuffle(arr):
            return np.random.randint(len(arr),size=MAX_PTS)
        
        from nuscenes.utils.data_classes import LidarPointCloud
        from .common import get_pose

        current_pc = LidarPointCloud.from_file(sample['lidar_pose'])
        current_pc.remove_close(1.0)
        current_pc.transform(get_pose(sample['lidar_pts']['rotation'], sample['lidar_pts']['translation']))
        pts = np.transpose(current_pc.points)
        # lidar, voxel_coords = preprocess(pts)
        # idx = shuffle(lidar)
        # lidar, voxel_coords = lidar[idx], voxel_coords[idx]
        return {'lidar': pts}#, 'coords': voxel_coords}
    
    def get_box(self, sample: Sample):
        scene_dir = self.labels_dir / sample.scene

        if self.box == 'pseudo':
            det = np.load(scene_dir / sample.boxes)['boxes']
            tmp_boxes = np.zeros((len(det),6)) # cx, cy, w, h, z, l
            tmp_boxes[:,:4] = (det[:,:4] * 100) - 50
            labels = det[:,4].astype(np.int_)
            tmp_boxes[:,4:] = z_stats[labels]
            
            boxes = np.zeros((tmp_boxes.shape))
            boxes[:,0] = (tmp_boxes[:,0] + tmp_boxes[:,2]) / 2
            boxes[:,1] = (tmp_boxes[:,1] + tmp_boxes[:,3]) / 2
            boxes[:,2] = tmp_boxes[:,0] - tmp_boxes[:,2]
            boxes[:,3] = tmp_boxes[:,1] - tmp_boxes[:,3]
            boxes[:,2:4] = np.where(boxes[:,2:4] == 0, 0.5, boxes[:,2:4])
            boxes[:,2:4] = boxes[:,2:4] * 1.1
            boxes[:,4:] = tmp_boxes[:,4:]
        
        elif self.box == 'gt':
            det = np.load(scene_dir / sample.gt_box, allow_pickle=True)['gt_box'][:, :-1] # ignore visibility
            mask = (det[:, 0] >= -50) & (det[:, 1] >= -50) & (det[:, 0] <= 50) & (det[:, 1] <= 50)
            det = det[mask]
            if self.orientation:
                boxes = np.zeros((len(det),8)) # cx, cy, w, h, z, l, yaw.cos, yaw.sin
                yaw = det[...,6]
                yaw = -yaw - np.pi / 2
                boxes[...,6] = np.sin(yaw)
                boxes[...,7] = np.cos(yaw)
            else:
                boxes = np.zeros((len(det),6))

            boxes[...,:6] = det[...,:6]

            labels = det[:,-1].astype(np.int_)

        if (labels == 8).any() or len(boxes)==0:
            dimension = 8 if self.orientation else 6
            return {'labels': np.empty((0)).astype(np.int_),'boxes':np.empty((0,dimension)).astype(np.float32)}
        
        boxes[:,2:4] = np.log(boxes[:,2:4])
        boxes[:,5] = np.log(boxes[:,5])

        if self.no_class:
            labels = np.zeros_like(labels)

        return {'labels':labels, 'boxes':boxes.astype(np.float32)}
    
    def get_box_2d(self, sample: Sample, view):
        scene_dir = self.labels_dir / sample.scene

        assert self.box == 'pseudo'
        det = np.load(scene_dir / sample.boxes)['boxes']
        tmp_boxes = (det[:, :4] * 100) - 50
        labels = det[:,4].astype(np.int_)

        if (labels == 8).any() or len(det)==0:
            return {'labels': np.empty((0)).astype(np.int_),'boxes':np.empty((0,4)).astype(np.float32)}
        
        # lidar coordinate -> image coordinate
        pts = np.concatenate((tmp_boxes, np.ones((len(tmp_boxes),1))), axis=1).transpose()
        tmp_boxes = (view.numpy() @ pts)[:4].transpose()

        boxes = np.zeros((tmp_boxes.shape))
        boxes[:,0] = (tmp_boxes[:,0] + tmp_boxes[:,2]) / 2
        boxes[:,1] = (tmp_boxes[:,1] + tmp_boxes[:,3]) / 2
        boxes[:,2] = tmp_boxes[:,2] - tmp_boxes[:,0]
        boxes[:,3] = tmp_boxes[:,3] - tmp_boxes[:,1]
        boxes[:,2:4] = np.where(boxes[:,2:4] == 0, 1.0, boxes[:,2:4])

        # normalized
        boxes = boxes / 200.0

        if self.no_class:
            labels = np.zeros_like(labels)

        return {'labels':labels, 'boxes':boxes.astype(np.float32)}

    # copied from PointBEV
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
    
    def get_bev_from_gtbbox(self, sample: Sample, bev_augm):
        scene_dir = self.labels_dir / sample.scene
        gt_box = np.load(scene_dir / sample.gt_box, allow_pickle=True)['gt_box']
        V = sample.view

        # bev segmentation
        bev = np.zeros((8, 200, 200), dtype=np.uint8)

        # center & offset
        center_score = np.zeros((200, 200), dtype=np.float32)
        center_offset = np.zeros((200, 200, 2), dtype=np.float32)
        visibility = np.full((200, 200), 255, dtype=np.uint8)

        buf = np.zeros((200, 200), dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(200), np.arange(200)), -1).astype(np.float32)
        sigma = 1

        # height
        center_z = np.zeros((200, 200), dtype=np.float32)
        height = np.zeros((200, 200), dtype=np.float32)

        # box
        tmp = []
        # lidar2img @ bev_augm @ pts
        for box_data in gt_box:
            if len(box_data) == 0:
                continue
            class_idx = int(box_data[7])
            if class_idx == 5: 
                continue
            translation = [box_data[0],box_data[1],box_data[4]]
            size = [box_data[2],box_data[3],box_data[5]]
            yaw = box_data[6]
            yaw = -yaw - np.pi / 2
            # class_idx = int(box_data[7])
            visibility_token = box_data[8]
            box = Box(translation, size, sincos2quaternion(np.sin(yaw),np.cos(yaw)))
            
            points = box.bottom_corners()
            center = points.mean(-1)[:, None] # unsqueeze 1

            homog_points = np.ones((4, 4))
            homog_points[:3, :] = points
            homog_points[-1, :] = 1
            points = self._prepare_augmented_boxes(bev_augm, homog_points)
            points[2] = 1 # add 1 for next matrix matmul
            points = (V @ points)[:2]
            cv2.fillPoly(bev[class_idx], [points.round().astype(np.int32).T], 1, INTERPOLATION)

            # ignore pedestrians
            if class_idx != 5: 
                # center, offsets, height
                homog_points = np.ones((4, 1))
                homog_points[:3, :] = center
                homog_points[-1, :] = 1
                center = self._prepare_augmented_boxes(bev_augm, homog_points)
                center[2] = 1 # add 1 for next matrix matmul
                center = (V @ center)[:2, 0] # squeeze 1

                buf.fill(0)
                cv2.fillPoly(buf, [points.round().astype(np.int32).T], 1, INTERPOLATION)
                mask = buf > 0
                center_offset[mask] = center[None] - coords[mask]
                center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (2 * sigma ** 2))
                
                # height
                cv2.fillPoly(center_z, [points.round().astype(np.int32).T], box.center[-1], INTERPOLATION)
                cv2.fillPoly(height, [points.round().astype(np.int32).T], box.wlh[-1], INTERPOLATION)

                # visibility
                visibility[mask] = visibility_token

            x1 = np.min(points[0])
            x2 = np.max(points[0])
            y1 = np.min(points[1])
            y2 = np.max(points[1])
            tmp.append([x1, y1, x2, y2, int(box_data[-1]), box.wlh[-1]])
            # TODO: add offset & centerness
        
        bev = self.to_tensor(255 * bev.transpose(1,2,0))
        center_score = self.to_tensor(center_score)
        center_offset = self.to_tensor(center_offset)

        # height
        center_z = self.to_tensor(center_z)
        height = self.to_tensor(height)
        tmp = np.array(tmp)

        if len(tmp) == 0:
            return bev, center_score, center_offset, {'labels': np.empty((0)).astype(np.int_),'boxes':np.empty((0, 4)).astype(np.float32)}, height, center_z, visibility

        boxes = np.zeros((len(tmp), 5))
        labels = tmp[:,4].astype(np.int_)

        boxes[:,0] = (tmp[:, 0] + tmp[:,2]) / 2.0
        boxes[:,1] = (tmp[:,1] + tmp[:,3]) / 2.0
        boxes[:,2] = tmp[:,2] - tmp[:,0]
        boxes[:,3] = tmp[:,3] - tmp[:,1]

        # normalized
        boxes = boxes / 200.0
        boxes[:, 4] = tmp[:, -1]
        
        return bev, center_score, center_offset, {'labels':labels, 'boxes':boxes.astype(np.float32)}, height, center_z, visibility
    
    def _parse_bev_augm(self, result, bev_augm):
        box_cxcy = result['boxes'][:,:2] # N 2
        box_cz = result['boxes'][:,4:5]

        if len(box_cxcy) > 0:
            pts = np.concatenate((box_cxcy, box_cz), axis=1).transpose() # 3 N
            pts = self._prepare_augmented_boxes(bev_augm.numpy(), pts).transpose()
            result['boxes'][:,:2] = pts[:, :2]
            result['boxes'][:,4:5] = pts[:, 2:3]

        return result

    def get_bbox_from_bev(self, bev, view):
        # pts_bev = V @ augm @ pts_lidar
        # augm^-1 @ V^-1 @ pts_bev = pts_lidar
        
        tmp = []
        view_inv = np.linalg.inv(view.numpy())
        for i in range(8):
            label = bev[4+i].numpy()
            bev_pts = map2points(label)

            if len(bev_pts) == 0:
                continue

            clusters = apply_dbscan(bev_pts, 1.0, 3)
            for j in range(clusters.max()+1):
                tmp_index = np.where(clusters==j)[0]
                (x1,y1), (x2,y2) = get_min_max(bev_pts[tmp_index])
                
                if self.box_3d:
                    pts = np.array([[x1,y1,1],[x2,y2,1]]).transpose()
                    pts = view_inv @ pts
                    # pts = self._prepare_augmented_boxes(bev_augm.numpy(), pts, inverse=False)
                    x1, y1, x2, y2 = pts[:2].transpose().reshape(-1)

                tmp.append([x1, y1, x2, y2, i])

        dimension = 6 if self.box_3d else 4    
        tmp = np.array(tmp)
        # print(tmp)
        if len(tmp) == 0:
            return {'labels': np.empty((0)).astype(np.int_),'boxes':np.empty((0,dimension)).astype(np.float32)}

        boxes = np.zeros((len(tmp), dimension))
        labels = tmp[:,4].astype(np.int_)

        boxes[:,0] = (tmp[:, 0] + tmp[:,2]) / 2.0
        boxes[:,1] = (tmp[:,1] + tmp[:,3]) / 2.0
        boxes[:,2] = tmp[:,2] - tmp[:,0]
        boxes[:,3] = tmp[:,3] - tmp[:,1]

        if self.box_3d:
            boxes[:, 2:4] = -boxes[:, 2:4]

        boxes[:,2:4] = np.where(boxes[:,2:4] == 0, 1.0, boxes[:,2:4])

        # normalized
        if self.box_3d:
            boxes[:,4:] = z_stats[labels]
            boxes[:,2:4] = np.log(boxes[:,2:4])
            boxes[:,5] = np.log(boxes[:,5])
        else:
            boxes = boxes / 200.0
        
        return {'labels':labels, 'boxes':boxes.astype(np.float32)}

    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)
        
        result = dict()
        result['view'] = torch.tensor(batch.view)
        result['5d_view'] = get_5d_view(result['view'])

        if self.image_data:
            get_cameras = self.get_cameras_augm if self.augment_img is not None else self.get_cameras 
            result.update(get_cameras(batch, **self.image_config))
        
        result.update(self.get_bev(batch))

        if self.box and self.augment_bev is None:
            if self.box_3d:
                result.update(self.get_box(batch))
            else:
                result.update(self.get_box_2d(batch, result['5d_view']))

        if self.augment_bev is not None:
            bev_augm = self.augment_bev()
            augm_bev_gt, augm_center_score, augm_center_offset, gtbox_3d, height, center_z, visibility = self.get_bev_from_gtbbox(batch, bev_augm)

            result['bev'][4:12] = augm_bev_gt
            result['center'] = augm_center_score
            result['offset'] = augm_center_offset
            result['visibility'] = visibility
            result['height'] = height
            result['center_z'] = center_z

            if self.box == 'pseudo':
                result.update(self.get_bbox_from_bev(result['bev'], result['view']))
            elif self.box == 'gt':
                result.update(gtbox_3d)

            bev_augm = torch.from_numpy(bev_augm)
            result['extrinsics'] = result['extrinsics'] @ bev_augm
            result['lidar2img'] = result['lidar2img'] @ bev_augm
            # result = self._parse_bev_augm(result, bev_augm)
            # result['bev_augm'] = bev_augm

        if self.lidar:
            result.update(self.get_lidar(batch))

        return result