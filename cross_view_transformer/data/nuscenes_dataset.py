import torch
import numpy as np
import cv2

from pathlib import Path
from functools import lru_cache

from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon

from .common import INTERPOLATION, get_view_matrix, get_pose, get_split
from .transforms import Sample, SaveDataTransform
import os
from functools import reduce

STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
    # 'emergency',
]
STATIC2 = ['ped_crossing','walkway','carpark_area']
topology = True
if topology:
    CLASSES = STATIC + DIVIDER + DYNAMIC + STATIC2
else:
    CLASSES = STATIC + DIVIDER + DYNAMIC 
     
NUM_CLASSES = len(CLASSES)

def mask_out(out_l,y=[0,200],x=[0,200]):
    mask = np.ones(out_l.shape[1], dtype=bool)
    mask = np.logical_and(mask, out_l[0, :] < y[1])
    mask = np.logical_and(mask, out_l[0, :] >= y[0])
    mask = np.logical_and(mask, out_l[1, :] >= x[0])
    mask = np.logical_and(mask, out_l[1, :] < x[1])
    return mask

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    dataset='unused',                   # ignore
    augment='unused',                   # ignore
    image='unused',                     # ignore
    label_indices='unused',             # ignore
    num_classes=NUM_CLASSES,            # in here to make config consistent
    **dataset_kwargs
):
    assert num_classes == NUM_CLASSES

    helper = NuScenesSingleton(dataset_dir, version)
    transform = SaveDataTransform(labels_dir)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    result = list()

    for scene_name, scene_record in helper.get_scenes():
        if scene_name not in split_scenes:
            continue
        data = NuScenesDataset(scene_name, scene_record, helper,
                               transform=transform, **dataset_kwargs)
        result.append(data)

    return result


class NuScenesSingleton:
    """
    Wraps both nuScenes and nuScenes map API

    This was an attempt to sidestep the 30 second loading time in a "clean" manner
    """
    def __init__(self, dataset_dir, version):
        """
        dataset_dir: /path/to/nuscenes/
        version: v1.0-trainval
        """
        self.dataroot = str(dataset_dir)
        self.nusc = self.lazy_nusc(version, self.dataroot)

    @classmethod
    def lazy_nusc(cls, version, dataroot):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.nuscenes import NuScenes

        if not hasattr(cls, '_lazy_nusc'):
            cls._lazy_nusc = NuScenes(version=version, dataroot=dataroot)

        return cls._lazy_nusc

    def get_scenes(self):
        for scene_record in self.nusc.scene:
            yield scene_record['name'], scene_record

    @lru_cache(maxsize=16)
    def get_map(self, log_token):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.map_expansion.map_api import NuScenesMap

        map_name = self.nusc.get('log', log_token)['location']
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        return nusc_map

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            obj = super(NuScenesSingleton, cls).__new__(cls)
            obj.__init__(*args, **kwargs)

            cls._singleton = obj

        return cls._singleton


class NuScenesDataset(torch.utils.data.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    RADARS = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]

    def __init__(
        self,
        scene_name: str,
        scene_record: dict,
        helper: NuScenesSingleton,
        transform=None,
        cameras=[[0, 1, 2, 3, 4, 5]],
        bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
        lidar=False,
        radar=False,
        tri_view=False,
        gt_box=False,
        **kwargs
    ):
        self.scene_name = scene_name
        self.transform = transform

        self.nusc = helper.nusc
        self.nusc_map = helper.get_map(scene_record['log_token'])

        self.view = get_view_matrix(flip=False,**bev)
        self.bev_shape = (bev['h'], bev['w'])

        self.samples = self.parse_scene(scene_record, cameras)
        self.radar = radar
        self.lidar = lidar
        self.tri_view = tri_view
        self.gt_box = gt_box

    def parse_scene(self, scene_record, camera_rigs):
        data = []
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = self.nusc.get('sample', sample_token)

            for camera_rig in camera_rigs:
                data.append(self.parse_sample_record(sample_record, camera_rig))

            sample_token = sample_record['next']

        return data

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def parse_sample_record(self, sample_record, camera_rig):
        """
            box: world coordinate
            parse_pose: pose @ point , LiDAR -> world 
                   inv: pose_inv @ point , LiDAR <- world
            sensor, ego_sensor world 
        """
        
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])

        # calibrated_lidar = self.nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])

        # world_from_lidarflat = self.parse_pose(calibrated_lidar, flat=True)
        # lidarflat_from_world = self.parse_pose(calibrated_lidar, flat=True,inv=True)

        world_from_egolidarflat = self.parse_pose(egolidar, flat=True)
        egolidarflat_from_world = self.parse_pose(egolidar, flat=True, inv=True)

        cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []

        for cam_idx in camera_rig:
            cam_channel = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_channel]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            cam_from_egocam = self.parse_pose(cam, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)

            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat # @ world_from_lidarflat
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))

            cam_channels.append(cam_channel)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)

        return {
            'scene': self.scene_name,
            'token': sample_record['token'],

            # 'pose': (world_from_egolidarflat @ world_from_lidarflat).tolist(),
            # 'pose_inverse': (lidarflat_from_world @ egolidarflat_from_world).tolist(),

            'pose': world_from_egolidarflat.tolist(),
            'pose_inverse': egolidarflat_from_world.tolist(),
            'lidar_record': egolidar,
            'cam_ids': list(camera_rig),
            'cam_channels': cam_channels,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'images': images,
        }

    def get_dynamic_objects(self, sample, annotations):
        h, w = self.bev_shape[:2]

        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_score = np.zeros((h, w), dtype=np.float32)
        center_offset = np.zeros((h, w, 2), dtype=np.float32)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)
        buf = np.zeros((h, w), dtype=np.uint8)

        visibility = np.full((h, w), 255, dtype=np.uint8)

        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)
        count = 0
        for ann, p in zip(annotations, self.convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            sigma = 1
            segmentation[mask] = count + 1
            count += 1
            center_offset[mask] = center[None] - coords[mask]
            center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (2 * sigma ** 2))

            # orientation, h/2, w/2
            center_ohw[mask, 0:2] = ((front - center) / (np.linalg.norm(front - center) + 1e-6))[None]
            center_ohw[mask, 2:3] = np.linalg.norm(front - center)
            center_ohw[mask, 3:4] = np.linalg.norm(left - center)

            visibility[mask] = ann['visibility_token']

        segmentation = np.float32(segmentation[..., None])
        center_score = center_score[..., None]

        result = np.concatenate((segmentation, center_score, center_offset, center_ohw), 2)

        # (h, w, 1 + 1 + 2 + 2)
        return result, visibility

    def convert_to_box(self, sample, annotations):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.utils import data_classes

        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        for a in annotations:
            box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

            corners = box.bottom_corners()                                              # 3 4
            center = corners.mean(-1)                                                   # 3
            front = (corners[:, 0] + corners[:, 1]) / 2.0                               # 3
            left = (corners[:, 0] + corners[:, 3]) / 2.0                                # 3

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)      # 3 7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                        # 4 7
            p = V @ S @ M_inv @ p                                                       # 3 7

            yield p                                                                     # 3 7

    def convert_to_box_tri(self, sample, annotations,mode_in):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.utils import data_classes

        def get_view():
            mode, h, w = mode_in
            h_meters = 100 if h == 200 else 8
            w_meters = 100 if w == 200 else 8
            sh = h / h_meters
            sw = w / w_meters
            if mode == 0:
                return np.float32([
                [ 0., -sw,          w/2.],
                [-sh,  0.,          h/2.],
                [ 0.,  0.,            1.]
            ])
            elif mode == 1:
                return np.float32([
                [ sw,  0.,          w/2.],
                [ 0., -sh,          h/2.],
                [ 0.,  0.,            1.]
            ])
            else:
                return np.float32([
                [-sw,  0.,          w/2.],
                [ 0., -sh,          h/2.],
                [ 0.,  0.,            1.]
            ])

        M_inv = np.array(sample['pose_inverse'])
        if mode_in[0] == 0:
            S = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ])
        elif mode_in[0] == 1:
            S = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
        else:
            S = np.array([
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
        V = get_view()

        if mode_in[0] == 0:
            corners_index = [2, 3, 7, 6]
        elif mode_in[0] == 1:
            corners_index = [0, 4, 7, 3]
        else:
            corners_index = [2, 3, 0, 1]

        for a in annotations:
            box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

            corners = box.corners()[:, corners_index]                                    # 3 4
            center = corners.mean(-1)                                                   # 3
            front = (corners[:, 0] + corners[:, 1]) / 2.0                               # 3
            left = (corners[:, 0] + corners[:, 3]) / 2.0                                # 3

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)      # 3 7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                        # 4 7
            p = V @ S @ M_inv @ p                                                       # 3 7

            yield p

    def get_category_index(self, name, categories):
        """
        human.pedestrian.adult
        """
        tokens = name.split('.')

        for i, category in enumerate(categories):
            if category in tokens:
                return i

        return None

    def get_annotations_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            # if int(a['visibility_token']) == 1:
            #     continue
            if idx is not None:
                result[idx].append(a)

        return result

    def get_line_layers(self, sample, layers, patch_radius=150, thickness=2):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        # box_coords = (sample['pose'][0][-1] - patch_radius, sample['pose'][1][-1] - patch_radius,
        #               sample['pose'][0][-1] + patch_radius, sample['pose'][1][-1] + patch_radius)
        # records_in_patch = self.nusc_map.get_records_in_patch(box_coords, layers, 'intersect')

        # result = list()

        # for layer in layers:
        #     render = np.zeros((h, w), dtype=np.uint8)

        #     for r in records_in_patch[layer]:
        #         polygon_token = self.nusc_map.get(layer, r)
        #         line = self.nusc_map.extract_line(polygon_token['line_token'])

        #         p = np.float32(line.xy)                                     # 2 n
        #         p = np.pad(p, ((0, 1), (0, 0)), constant_values=0.0)        # 3 n
        #         p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)        # 4 n
        #         p = V @ S @ M_inv @ p                                       # 3 n
        #         p = p[:2].round().astype(np.int32).T                        # n 2

        #         cv2.polylines(render, [p], False, 1, thickness=thickness)

        #     result.append(render)

        lidar2global = (V @ S @ M_inv)
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        angle = (yaw / np.pi * 180)

        map_mask = self.nusc_map.get_map_mask((sample['pose'][0][-1],sample['pose'][1][-1],100,100), angle, layers, (h,w))
        result = [np.flipud(i) for i in map_mask]

        return 255 * np.stack(result, -1)

    def get_static_layers(self, sample, layers, patch_radius=150):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        box_coords = (sample['pose'][0][-1] - patch_radius, sample['pose'][1][-1] - patch_radius,
                      sample['pose'][0][-1] + patch_radius, sample['pose'][1][-1] + patch_radius)
        records_in_patch = self.nusc_map.get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)

                if layer == 'drivable_area': polygon_tokens = polygon_token['polygon_tokens']
                else: polygon_tokens = [polygon_token['polygon_token']]

                for p in polygon_tokens:
                    polygon = self.nusc_map.extract_polygon(p)
                    polygon = MultiPolygon([polygon])

                    exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in exteriors]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in exteriors]
                    exteriors = [V @ S @ M_inv @ p for p in exteriors]
                    exteriors = [p[:2].round().astype(np.int32).T for p in exteriors]

                    cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

                    interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in interiors]
                    interiors = [V @ S @ M_inv @ p for p in interiors]
                    interiors = [p[:2].round().astype(np.int32).T for p in interiors]

                    cv2.fillPoly(render, interiors, 0, INTERPOLATION)

            result.append(render)
        
        return 255 * np.stack(result, -1)

    def get_dynamic_layers(self, sample, anns_by_category):
        h, w = self.bev_shape[:2]
        result = list()

        for anns in anns_by_category:
            render = np.zeros((h, w), dtype=np.uint8)

            for p in self.convert_to_box(sample, anns):
                p = p[:2, :4]

                cv2.fillPoly(render, [p.round().astype(np.int32).T], 1, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)

    def get_dynamic_layers_triview(self, sample, anns_by_category):
        # 0: bev, 1: side, 2: front
        result = []
        for i in range(3):
            if i == 0:
                h, w = 200, 200
            elif i == 1:
                h, w = 32, 200
            else:
                h, w = 32, 200

            tmp_result = list()

            for anns in anns_by_category:
                render = np.zeros((h, w), dtype=np.uint8)

                for p in self.convert_to_box_tri(sample, anns,[i,h,w]):
                    p = p[:2, :4]

                    cv2.fillPoly(render, [p.round().astype(np.int32).T], 1, INTERPOLATION)

                tmp_result.append(render)

            result.append(255 * np.stack(tmp_result, -1))
        return result

    def get_radar(self,sample,use_radar_filters = False):
        from nuscenes.utils.data_classes import RadarPointCloud
        sample_rec = self.nusc.get('sample', sample['token'])
        min_distance = 1.0

        if use_radar_filters:
            RadarPointCloud.default_filters()
        else:
            RadarPointCloud.disable_filters()

        out_l = []
        for radar_name in self.RADARS:
            sample_data_token = sample_rec['data'][radar_name]
            current_sd_rec = self.nusc.get('sample_data', sample_data_token)
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = RadarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = self.parse_pose(current_pose_rec)
            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = self.parse_pose(current_cs_rec)

            # Fuse four transformation matrices into one and perform transform. lidar_sensor, lidar ego, radar ego, radar sensor
            trans_matrix = reduce(np.dot, [sample['pose_inverse'],global_from_car, car_from_current]) 
            current_pc.transform(trans_matrix)
            out_l.append(current_pc.points)

        out_lidar = np.concatenate(out_l,1)
        V = self.view
        S = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ])
        out_lidar[:3] = V @ S @ np.vstack((out_lidar[:3],np.ones(out_lidar.shape[1])))
        mask = mask_out(out_lidar)
        out_lidar = out_lidar[:,mask]
        radar_map = np.zeros((200,200,16),dtype=float)
        radar_map[(out_lidar[1,:].astype(int)),(out_lidar[0,:].astype(int))] = np.transpose(np.vstack((np.ones((1,out_lidar.shape[1]),dtype=float),out_lidar[3:,:])))
        return radar_map, out_lidar
    
    def get_lidar(self, sample):
        from nuscenes.utils.data_classes import LidarPointCloud
        sample_rec = self.nusc.get('sample', sample['token'])
        min_distance = 1.0

        sample_data_token = sample_rec['data']['LIDAR_TOP']
        current_sd_rec = self.nusc.get('sample_data', sample_data_token)
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = self.parse_pose(current_cs_rec)

        current_pc.transform(car_from_current)

        out_lidar = current_pc.points
        V = self.view
        S = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ])
        z = np.array(out_lidar[2,:])
        out_lidar[:3] = V @ S @ np.vstack((out_lidar[:3],np.ones(out_lidar.shape[1])))
        out_lidar[2,:] = z
        mask = mask_out(out_lidar)
        out_lidar = out_lidar[:,mask]
        lidar_map = np.zeros((200,200,3),dtype=float)
        lidar_map[(out_lidar[1,:].astype(int)),(out_lidar[0,:].astype(int))] = np.transpose(np.vstack((np.ones((1,out_lidar.shape[1]),dtype=float),out_lidar[2:,:])))
        return lidar_map, os.path.join(self.nusc.dataroot, current_sd_rec['filename']), current_cs_rec
    
    def get_gt_box(self, lidar_record, anns_by_category):
        from nuscenes.utils import data_classes
        """ 
            Return: 
                List[bounding boxes] 
                bounding boxes 8 dimensions: cx,cy,w,l,cz,h,yaw,class
        """
        gt_boxes = []
        for class_index, anns in enumerate(anns_by_category):
            for ann in anns:
                box = data_classes.Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

                # project box global -> lidar
                yaw = Quaternion(lidar_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(lidar_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
                cx, cy, cz = box.center
                w, l, h = box.wlh
                yaw = box.orientation.yaw_pitch_roll[0]
                gt_boxes.append(np.array([cx, cy, l, w, cz, h, yaw, class_index]))
        
        gt_boxes = np.stack(gt_boxes,0) if len(gt_boxes) != 0 else np.zeros((0,8))
        return gt_boxes
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Raw annotations
        anns_dynamic = self.get_annotations_by_category(sample, DYNAMIC)
        anns_vehicle = self.get_annotations_by_category(sample, ['vehicle'])[0]
        anns_ped = self.get_annotations_by_category(sample, ['pedestrian'])[0]

        static = self.get_static_layers(sample, STATIC)                             # 200 200 2
        dividers = self.get_line_layers(sample, DIVIDER)                            # 200 200 2
        # dynamic = self.get_dynamic_layers(sample, anns_dynamic)                     # 200 200 8
        if self.tri_view:
            dynamic = self.get_dynamic_layers_triview(sample, anns_dynamic) 
        else:
            dynamic = self.get_dynamic_layers(sample, anns_dynamic)
        if topology:
            static_v2 = self.get_static_layers(sample, STATIC2)                             # 200 200 2
            # dividers_v2 = self.get_line_layers(sample, DIVIDER_v2)     
            bev = np.concatenate((static, dividers, dynamic,static_v2), -1)                     
        else:
            bev = np.concatenate((static, dividers, dynamic[0]), -1)                       # 200 200 12
        assert bev.shape[2] == NUM_CLASSES

        # Additional labels for vehicles only.
        aux, visibility = self.get_dynamic_objects(sample, anns_vehicle)
        aux_ped, visibility_ped = self.get_dynamic_objects(sample, anns_ped)
        radar, lidar, gt_box = None, None, None
        if self.radar:
            radar = self.get_radar(sample)
        if self.lidar:
            lidar = self.get_lidar(sample)
        if self.tri_view:
            side, front = dynamic[1:]
        else:
            side = front = None

        if self.gt_box:
            gt_box = self.get_gt_box(sample['lidar_record'], anns_dynamic)

        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            bev=bev,
            aux=aux,
            aux_ped=aux_ped,
            visibility=visibility,
            visibility_ped=visibility_ped,
            radar=radar,
            lidar=lidar,
            side=side,
            front=front,
            gt_box=gt_box,
            **sample
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
