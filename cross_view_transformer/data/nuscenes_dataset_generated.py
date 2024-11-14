import json
import torch

from pathlib import Path
from .common import get_split, get_view_matrix, get_pose, INTERPOLATION
from .transforms import Sample, LoadDataTransform, LoadDataTransform_DepthAnything
from .nuscenes_dataset import NuScenesSingleton

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    image=None,                         # image config
    depth_anything=False,
    map_seg=False,
    **dataset_kwargs
):
    out = []

    # for map seg, we need to load nusc at run time
    if map_seg:
        helper = NuScenesSingleton(dataset_dir, version)
        # training = True if split == 'train' else False

        # # Format the split name
        # split = f'mini_{split}' if version == 'v1.0-mini' else split
        # split_scenes = get_split(split, 'nuscenes')

        # for scene_name, scene_record in helper.get_scenes():
        #     if scene_name not in split_scenes:
        #         continue
        #     data = NuScenesMap(scene_name, scene_record, helper, **dataset_kwargs)
        #     out.append(data)

    else:
        dataset_dir = Path(dataset_dir)
        labels_dir = Path(labels_dir)

        # Override augment if not training
        training = True if split == 'train' else False
        transform = LoadDataTransform_DepthAnything if depth_anything else LoadDataTransform
        transform = transform(dataset_dir, labels_dir, image, num_classes, training=training, **dataset_kwargs)

        # Format the split name
        split = f'mini_{split}' if version == 'v1.0-mini' else split
        split_scenes = get_split(split, 'nuscenes')

        for s in split_scenes:
            tmp_dataset = NuScenesGeneratedDataset(s, labels_dir, transform=transform)
            out.append(tmp_dataset)
    return out
    # return [NuScenesGeneratedDataset(s, labels_dir, transform=transform) for s in split_scenes]

# 1045
class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = Sample(**self.samples[idx])

        if self.transform is not None:
            data = self.transform(data)

        return data

# class NuScenesMap(torch.utils.data.Dataset):
#     CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
#                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
#     def __init__(
#         self,
#         scene_name: str,
#         scene_record: dict,
#         helper: NuScenesSingleton,
#         cameras=[[0, 1, 2, 3, 4, 5]],
#         bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
#         **kwargs
#     ):
#         import numpy as np

#         self.scene_name = scene_name
#         self.nusc = helper.nusc
#         self.nusc_map = helper.get_map(scene_record['log_token'])
#         self.view = get_view_matrix(flip=False,**bev)
#         self.samples = self.parse_scene(scene_record, cameras)

#     def parse_scene(self, scene_record, camera_rigs):
#         data = []
#         sample_token = scene_record['first_sample_token']

#         while sample_token:
#             sample_record = self.nusc.get('sample', sample_token)

#             for camera_rig in camera_rigs:
#                 data.append(self.parse_sample_record(sample_record, camera_rig))

#             sample_token = sample_record['next']

#         return data
    
#     def parse_pose(self, record, *args, **kwargs):
#         return get_pose(record['rotation'], record['translation'], *args, **kwargs)

#     def parse_sample_record(self, sample_record, camera_rig):
#         """
#             box: world coordinate
#             parse_pose: pose @ point , LiDAR -> world 
#                    inv: pose_inv @ point , LiDAR <- world
#             sensor, ego_sensor world 
#         """
        
#         lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])

#         # calibrated_lidar = self.nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
#         egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])

#         # world_from_lidarflat = self.parse_pose(calibrated_lidar, flat=True)
#         # lidarflat_from_world = self.parse_pose(calibrated_lidar, flat=True,inv=True)

#         world_from_egolidarflat = self.parse_pose(egolidar, flat=True)
#         egolidarflat_from_world = self.parse_pose(egolidar, flat=True, inv=True)

#         cam_channels = []
#         images = []
#         intrinsics = []
#         extrinsics = []

#         for cam_idx in camera_rig:
#             cam_channel = self.CAMERAS[cam_idx]
#             cam_token = sample_record['data'][cam_channel]

#             cam_record = self.nusc.get('sample_data', cam_token)
#             egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
#             cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

#             cam_from_egocam = self.parse_pose(cam, inv=True)
#             egocam_from_world = self.parse_pose(egocam, inv=True)

#             E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat # @ world_from_lidarflat
#             I = cam['camera_intrinsic']

#             full_path = Path(self.nusc.get_sample_data_path(cam_token))
#             image_path = str(full_path.relative_to(self.nusc.dataroot))

#             cam_channels.append(cam_channel)
#             intrinsics.append(I)
#             extrinsics.append(E.tolist())
#             images.append(image_path)

#         return {
#             'scene': self.scene_name,
#             'token': sample_record['token'],

#             # 'pose': (world_from_egolidarflat @ world_from_lidarflat).tolist(),
#             # 'pose_inverse': (lidarflat_from_world @ egolidarflat_from_world).tolist(),

#             'pose': world_from_egolidarflat.tolist(),
#             'pose_inverse': egolidarflat_from_world.tolist(),
#             'lidar_record': egolidar,
#             # 'cam_ids': list(camera_rig),
#             # 'cam_channels': cam_channels,
#             'intrinsics': intrinsics,
#             'extrinsics': extrinsics,
#             'images': images,
#         }
    
#     def get_line_layers(self, sample, layers, patch_radius=150, thickness=2):
#         h, w = self.bev_shape[:2]
#         V = self.view
#         M_inv = np.array(sample['pose_inverse'])
#         S = np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 0, 1],
#         ])

#         lidar2global = (V @ S @ M_inv)
#         rotation = lidar2global[:3, :3]
#         v = np.dot(rotation, np.array([1, 0, 0]))
#         yaw = np.arctan2(v[1], v[0])
#         angle = (yaw / np.pi * 180)

#         map_mask = self.nusc_map.get_map_mask((sample['pose'][0][-1],sample['pose'][1][-1],100,100), angle, layers, (h,w))
#         result = [np.flipud(i) for i in map_mask]

#         return 255 * np.stack(result, -1)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         dividers = self.get_line_layers(sample, DIVIDER)