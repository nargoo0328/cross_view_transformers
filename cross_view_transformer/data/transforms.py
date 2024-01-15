import pathlib

import torch
import torchvision
from torch.nn import functional as F
import numpy as np

from PIL import Image
from .common import encode, decode
from .augmentations import StrongAug, GeometricAug
import os

MAX_PTS = 512

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
        side= None,
        front= None,
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
        self.side = side
        self.front = front

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
        Image.fromarray(encode(batch.bev)).save(scene_dir / bev_path)

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
            Image.fromarray(batch.visibility).save(scene_dir / visibility_path)

            result['visibility'] = visibility_path

        if batch.get('visibility_ped') is not None:
            visibility_path = f'visibility_ped_{batch.token}.png'
            Image.fromarray(batch.visibility_ped).save(scene_dir / visibility_path)

            result['visibility_ped'] = visibility_path

        if batch.get('radar') is not None:
            radar_path = f'radar_{batch.token}.npz'
            radar_bev, radar_points = batch.radar
            np.savez_compressed(scene_dir / radar_path, radar=radar_bev, points=radar_points)
            result['radar'] = radar_path

        if batch.get('lidar') is not None:
            lidar_path = f'lidar_{batch.token}.npz'
            lidar_bev = batch.lidar
            np.savez_compressed(scene_dir / lidar_path, lidar=lidar_bev)
            result['lidar'] = lidar_path
        
        if batch.get('side') is not None:
            side_path = f'side_{batch.token}.png'
            Image.fromarray(encode(batch.side)).save(scene_dir / side_path)

            result['side'] = side_path

        if batch.get('front') is not None:
            front_path = f'front_{batch.token}.png'
            Image.fromarray(encode(batch.front)).save(scene_dir / front_path)

            result['front'] = front_path

        return result

    def __call__(self, batch):
        """
        Save sensor/label data and return any additional info to be saved to json
        """
        result = {}
        result.update(self.get_cameras(batch))
        result.update(self.get_bev(batch))
        result.update({k: v for k, v in batch.items() if k not in result})

        return result


class LoadDataTransform(torchvision.transforms.ToTensor):
    def __init__(self, dataset_dir, labels_dir, image_config, num_classes,radar,lidar,autoencoder,augment='none'):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes
        self.radar = radar
        self.lidar = lidar
        self.autoencoder = autoencoder

        xform = {
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)
        self.to_tensor = super().__call__

    def get_cameras(self, sample: Sample, h, w, top_crop):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        intrinsics = list()
        tensor = None
        # tensor = F.adaptive_avg_pool2d(tensor,(200,200))
        for image_path, I_original in zip(sample.images, sample.intrinsics):
            # tensor_path = 'features' + image_path[7:-3] + 'pt'
            h_resize = h + top_crop
            w_resize = w

            # tensor = torch.load(self.dataset_dir / tensor_path)

            image = Image.open(self.dataset_dir / image_path)

            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop
            
            images.append(self.img_transform(image_new))
            intrinsics.append(torch.tensor(I))
            # tensors.append(tensor)

        result = {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample.extrinsics)),
        }
        if tensor is not None:
            result.update({'features':tensor})
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
            'view': torch.tensor(sample.view),
        }
        
        if sample.side is not None:
            side = Image.open(scene_dir / sample.side)
            side = decode(side, 9)
            side = (255 * side).astype(np.uint8)
            side = self.to_tensor(side)
            result['side'] = side
 
        if sample.front is not None:
            front = Image.open(scene_dir / sample.front)
            front = decode(front, 9)
            front = (255 * front).astype(np.uint8)
            front = self.to_tensor(front)
            result['front'] = front

        if 'visibility' in sample:
            visibility = Image.open(scene_dir / sample.visibility)
            result['visibility'] = np.array(visibility, dtype=np.uint8)

        if 'visibility_ped' in sample:
            visibility = Image.open(scene_dir / sample.visibility_ped)
            result['visibility_ped'] = np.array(visibility, dtype=np.uint8)

        if 'aux' in sample:
            aux = np.load(scene_dir / sample.aux)['aux']
            result['center'] = self.to_tensor(aux[..., 1])
            result['offset'] = self.to_tensor(aux[..., 2:4])

        if 'aux_ped' in sample:
            aux_ped = np.load(scene_dir / sample.aux_ped)['aux_ped']
            result['center_ped'] = self.to_tensor(aux_ped[..., 1])

        if 'pose' in sample:
            result['pose'] = np.float32(sample['pose'])        

        if self.radar is not None:
            radar = np.load(scene_dir / sample.radar) # 16
            k = 'radar'
            radar = radar[k].astype(np.float32)
            radar = self.to_tensor(radar)
            result['radar'] = radar

        if self.lidar is not None:
            lidar = np.load(scene_dir / sample.lidar) # 16
            lidar = lidar['lidar'].astype(np.float32)
            lidar = self.to_tensor(lidar)
            result['lidar'] = lidar

        result['token'] = sample['token']
        return result

    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)
        
        result = dict()
        if self.autoencoder is None:
            result.update(self.get_cameras(batch, **self.image_config))
        result.update(self.get_bev(batch))

        return result
    
# python scripts/train.py +experiment=cvt_nuscenes_multiclass_fusion data.dataset_dir=/media/user/data4/nuscenes_data/ data.labels_dir=/media/user/data4/nuscenes_data/cvt_labels_nuscenes