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

        if batch.get('bev') is not None:
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
            lidar_pts_path, lidar_pose = batch.lidar
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
    def __init__(self, 
                dataset_dir, 
                labels_dir, 
                image_config, 
                num_classes, 
                image_data=True, 
                lidar=None, 
                box='',
                split_intrin_extrin=False, 
                orientation=False, 
                augment_img=False, 
                augment_bev=False, 
                no_class=False, 
                img_params=None, 
                bev_aug_conf=None, 
                training=True, 
                box_3d=True, 
                bev=True,
                depth='',
                **kwargs
        ):
        super().__init__()
        assert box in ['', 'gt', 'pseudo']
        assert depth in ['', 'generate', 'generated']

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes
        self.image_data = image_data
        self.lidar = lidar
        self.bev = bev
        self.depth = depth
        self.box = box
        self.orientation = orientation
        self.no_class = no_class
        self.box_3d = box_3d
        self.split_intrin_extrin = split_intrin_extrin
        self.img_transform = torchvision.transforms.ToTensor()

        self.training = training
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
        depths = list()

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
            
            depth_path = image_path.replace('samples', 'depths')
            if self.depth == 'generate':
                depths.append(self.dataset_dir / depth_path)
            elif self.depth == 'generated':
                depth = Image.open(self.dataset_dir / depth_path)
                depth_new = depth.resize((w_resize, h_resize), resample=Image.BILINEAR)
                depth_new = depth_new.crop((0, top_crop, depth_new.width, depth_new.height))
                depths.append(self.img_transform(depth_new))

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

        if self.depth == 'generate':
            result['depth'] = depths
        elif self.depth == 'generated':
            result['depth'] = torch.stack(depths, 0) * 255
    
        return result
    
    def get_cameras_augm(self, sample: Sample, **kwargs):
        images = list()
        intrinsics = list()
        extrinsics = list()
        depths = list()

        for image_path, intrinsic, extrinsic in zip(sample.images, sample.intrinsics, sample.extrinsics):
            image = Image.open(self.dataset_dir / image_path)
            images.append(image)

            if self.depth:
                depth_path = image_path.replace('samples', 'depths')
                depth = Image.open(self.dataset_dir / depth_path)
                depths.append(depth)

            intrinsic = np.float32(intrinsic)
            extrinsic = np.float32(extrinsic)
            intrinsics.append(intrinsic)
            extrinsics.append(torch.tensor(extrinsic))

        result = {'image': images}
        result.update({'intrinsics':intrinsics, 'extrinsics':extrinsics})

        if self.depth:
            result['depth'] = depths

        result = self.augment_img(result)
        result['image'] = torch.stack(result['image'], 0)
        if self.depth:
            result['depth'] = torch.stack(result['depth'], 0) * 255 / 80 * 61.2
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
        
        from nuscenes.utils.data_classes import LidarPointCloud
        current_pc = LidarPointCloud.from_file(sample['lidar_pts'])
        current_pc.remove_close(1.0)
        pose = np.float32(sample['lidar_pose'])
        current_pc.transform(pose)
        pts = current_pc.points[:3]
        # lidar, voxel_coords = preprocess(pts)
        # idx = shuffle(lidar)
        # lidar, voxel_coords = lidar[idx], voxel_coords[idx]
        return pts #, 'coords': voxel_coords}
    
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
        bev = np.zeros((9, 200, 200), dtype=np.uint8)

        # center & offset
        center_score = np.zeros((200, 200), dtype=np.float32)
        center_offset = np.zeros((200, 200, 2), dtype=np.float32) + 255
        visibility = np.full((200, 200), 255, dtype=np.uint8)

        buf = np.zeros((200, 200), dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(200), np.arange(200)), -1).astype(np.float32)
        sigma = 1

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
            # if self.training: # as we apply BEV augmentation, we should filter boxes that are out of range
            #     if (points[0, :] > 50.0).all() or (points[0, :] < -50.0).all() or (points[1, :] > 50.0).all() or (points[1, :] < -50.0).all():
            #         continue
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
                center = self._prepare_augmented_boxes(bev_augm, homog_points).astype(np.float32)
                center[2] = 1 # add 1 for next matrix matmul
                center = (V @ center)[:2, 0].astype(np.float32) # squeeze 1

                buf.fill(0)
                cv2.fillPoly(buf, [points.round().astype(np.int32).T], 1, INTERPOLATION)
                mask = buf > 0
                # center_offset[mask] = center[None] - coords[mask]
                center_off = center[None] - coords
                center_offset[mask] = center_off[mask]
                g = np.exp(-(center_off ** 2).sum(-1) / (2 * sigma ** 2))
                center_score = np.maximum(center_score, g)
                # center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (2 * sigma ** 2))
                
                # visibility
                visibility[mask] = visibility_token

            # x1 = np.min(points[0])
            # x2 = np.max(points[0])
            # y1 = np.min(points[1])
            # y2 = np.max(points[1])
            # tmp.append([x1, y1, x2, y2, int(box_data[-1]), box.wlh[-1]])
        
        bev = self.to_tensor(255 * bev.transpose(1,2,0))
        center_score = self.to_tensor(center_score)
        center_offset = self.to_tensor(center_offset)

        # height

        # if len(tmp) == 0:
        #     return bev, center_score, center_offset, {'labels': np.empty((0)).astype(np.int_),'boxes':np.empty((0, 4)).astype(np.float32)}, height, center_z, visibility
        
        return bev, center_score, center_offset, visibility
    
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

    def get_depth(self, result, lidar_points, bev_augm):

        def fill_zeros_with_nearest(data):
            # Step 1: Identify zeros in the data
            zero_mask = data == 0
            # Step 2: Create an initial distance map where non-zero values have a distance of 0
            # Invert the mask: non-zero -> 0, zero -> 1
            # distance_map = zero_mask.float()
            # Step 3: Use convolution to propagate nearest non-zero values
            for _ in range(0):
                # Apply a convolution with a kernel of all ones
                # conv_filter = torch.ones(1, 1, 3, 3, device=data.device)
                # propagated_data = F.conv2d(data.unsqueeze(1), conv_filter, padding=1).squeeze(1)
                # propagated_count = F.conv2d(distance_map.unsqueeze(1), conv_filter, padding=1).squeeze(1)
                # # Avoid division by zero by masking out zero locations
                # propagated_count[propagated_count == 0] = 1  # Avoid division by zero
                # new_data = propagated_data / propagated_count

                # # Step 4: Update the data where it was zero
                # data[zero_mask] = new_data[zero_mask]

                dilated_data = F.max_pool2d(data.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        
                # Update the data: only replace zeros with the dilated values
                data[zero_mask] = dilated_data[zero_mask]
                zero_mask = data == 0

            return data
        
        scale = 1
        N, _, H, W = result['image'].shape
        lidar2img = result['lidar2img'] # N 4 4
        lidar2img = lidar2img @ bev_augm.inverse()

        lidar_points = torch.from_numpy(lidar_points) # 4 P
        lidar_points = torch.cat([lidar_points, torch.ones_like(lidar_points[0:1])], dim=0)
        lidar_points = torch.matmul(lidar2img, lidar_points)[:, :3] # N 3 P
        depth = lidar_points[:, 2:3]
        homo_nonzero = torch.maximum(depth, torch.zeros_like(depth) + 1e-6)
        lidar_points = lidar_points[:, 0:2] / homo_nonzero
        valid_mask = ((depth > 1e-6) \
            & (lidar_points[:, 1:2] > scale)
            & (lidar_points[:, 1:2] < H - scale)
            & (lidar_points[:, 0:1] > scale)
            & (lidar_points[:, 0:1] < W - scale)
        )

        lidar_points = lidar_points / scale
        depth_cam = torch.zeros((N, H // scale, W // scale))
        for i in range(N):
            lidar_points_camera = lidar_points[i][:, valid_mask[i,0]]
            lidar_points_camera = torch.round(lidar_points_camera).int()
            depth_cam[i, lidar_points_camera[1], lidar_points_camera[0]] = depth[i][0, valid_mask[i,0]]
        
        return {'lidar_depth': fill_zeros_with_nearest(depth_cam)}
    
    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)
        
        result = dict()
        result['view'] = torch.tensor(batch.view)

        if self.image_data:
            get_cameras = self.get_cameras_augm if self.augment_img is not None else self.get_cameras 
            result.update(get_cameras(batch, **self.image_config))
        
        if self.box and self.augment_bev is None:
            if self.box_3d:
                result.update(self.get_box(batch))
            else:
                result.update(self.get_box_2d(batch, result['5d_view']))

        bev_augm = torch.eye(4)
        if self.bev:
            if self.augment_bev is not None:
                result.update({'bev': torch.zeros((15,200,200))})
                result.update({'token': batch['token']})
                bev_augm = self.augment_bev()
                augm_bev_gt, augm_center_score, augm_center_offset, visibility = self.get_bev_from_gtbbox(batch, bev_augm)

                result['bev'][4:13] = augm_bev_gt
                result['center'] = augm_center_score
                result['offset'] = augm_center_offset
                # result['visibility'] = visibility
                # result['height'] = height
                # result['center_z'] = center_z

                # if self.box == 'pseudo':
                #     result.update(self.get_bbox_from_bev(result['bev'], result['view']))
                # elif self.box == 'gt':
                #     result.update(gtbox_3d)

                bev_augm = torch.from_numpy(bev_augm)
                result['extrinsics'] = result['extrinsics'] @ bev_augm
                result['lidar2img'] = result['lidar2img'] @ bev_augm
                result['bev_augm'] = bev_augm
                # result = self._parse_bev_augm(result, bev_augm)
                # result['bev_augm'] = bev_augm
            else:
                result.update(self.get_bev(batch))


        if self.lidar:
            pts = self.get_lidar(batch)
            result.update(self.get_depth(result, pts, bev_augm))

        return result
    
class LoadDataTransform_DepthAnything:
    def __init__(self, 
                dataset_dir, 
                labels_dir, 
                image,
                num_classes,
                transform=None, 
                training=True,
                **kwargs
        ):
        
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.transform = transform
    
    def get_cameras(self, sample: Sample):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        lidar2img = list()

        for image_path, I_original, extrinsic in zip(sample.images, sample.intrinsics, sample.extrinsics):

            image = cv2.imread(str(self.dataset_dir / image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            images.append(image)

            I = np.float32(I_original)
            extrinsic = np.float32(extrinsic)
            viewpad = np.float32(np.eye(4))
            viewpad[:I.shape[0], :I.shape[1]] = I
            lidar2img.append(torch.tensor(viewpad @ extrinsic))

        lidar2img = torch.stack(lidar2img, 0)

        return images, lidar2img
    
    def get_lidar(self, sample: Sample):
        
        from nuscenes.utils.data_classes import LidarPointCloud
        current_pc = LidarPointCloud.from_file(sample['lidar_pts'])
        current_pc.remove_close(1.0)
        pose = np.float32(sample['lidar_pose'])
        current_pc.transform(pose)
        pts = current_pc.points[:3]
        return pts 

    def get_depth(self, shape, lidar_points, lidar2img):

        def fill_zeros_with_nearest(data):
            # Step 1: Identify zeros in the data
            zero_mask = data == 0
            # Step 2: Create an initial distance map where non-zero values have a distance of 0
            # Invert the mask: non-zero -> 0, zero -> 1
            # distance_map = zero_mask.float()
            # Step 3: Use convolution to propagate nearest non-zero values
            for _ in range(0):
                # Apply a convolution with a kernel of all ones
                # conv_filter = torch.ones(1, 1, 3, 3, device=data.device)
                # propagated_data = F.conv2d(data.unsqueeze(1), conv_filter, padding=1).squeeze(1)
                # propagated_count = F.conv2d(distance_map.unsqueeze(1), conv_filter, padding=1).squeeze(1)
                # # Avoid division by zero by masking out zero locations
                # propagated_count[propagated_count == 0] = 1  # Avoid division by zero
                # new_data = propagated_data / propagated_count

                # # Step 4: Update the data where it was zero
                # data[zero_mask] = new_data[zero_mask]

                dilated_data = F.max_pool2d(data.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        
                # Update the data: only replace zeros with the dilated values
                data[zero_mask] = dilated_data[zero_mask]
                zero_mask = data == 0

            return data
        
        scale = 1
        H, W, _ = shape
        N = 6

        lidar_points = torch.from_numpy(lidar_points) # 4 P
        lidar_points = torch.cat([lidar_points, torch.ones_like(lidar_points[0:1])], dim=0)
        lidar_points = torch.matmul(lidar2img, lidar_points)[:, :3] # N 3 P
        depth = lidar_points[:, 2:3]
        homo_nonzero = torch.maximum(depth, torch.zeros_like(depth) + 1e-6)
        lidar_points = lidar_points[:, 0:2] / homo_nonzero
        valid_mask = ((depth > 1e-6) \
            & (lidar_points[:, 1:2] > scale)
            & (lidar_points[:, 1:2] < H - scale)
            & (lidar_points[:, 0:1] > scale)
            & (lidar_points[:, 0:1] < W - scale)
        )

        lidar_points = lidar_points / scale
        depth_cam = np.zeros((N, H // scale, W // scale))
        for i in range(N):
            lidar_points_camera = lidar_points[i][:, valid_mask[i,0]]
            lidar_points_camera = torch.round(lidar_points_camera).int()
            depth_cam[i, lidar_points_camera[1], lidar_points_camera[0]] = depth[i][0, valid_mask[i,0]]
        
        return depth_cam
    
    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)
        
        result = dict()
        result['view'] = torch.tensor(batch.view)

        images, lidar2img = self.get_cameras(batch)
        
        pts = self.get_lidar(batch)
        depths = self.get_depth(images[0].shape, pts, lidar2img)

        sample_images = []
        sample_depths = []
        valid_masks = []
        for image, depth in zip(images, depths):
            sample = self.transform({'image': image, 'depth': depth})
            sample_images.append(torch.from_numpy(sample['image']))
            sample_depths.append(torch.from_numpy(sample['depth']))
            valid_mask = (sample['depth'] <= 60) & (sample['depth'] >= 1)
            valid_masks.append(torch.from_numpy(valid_mask))
        
        sample_images, sample_depths, valid_masks = torch.stack(sample_images), torch.stack(sample_depths), torch.stack(valid_masks)
    
        return {'image': sample_images, 'depth':sample_depths, 'valid_mask': valid_masks}