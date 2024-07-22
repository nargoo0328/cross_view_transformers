"""
https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/transforms.py
"""
import imgaug.augmenters as iaa
from scipy.spatial.transform import Rotation as R

import torchvision
import torch
import numpy as np

from PIL import Image

class AugBase(torchvision.transforms.ToTensor):
    def __init__(self):
        super().__init__()

        self.augment = self.get_augment().augment_image

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        return self.augment(x)


class StrongAug(AugBase):
    def get_augment(self):
        return iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
        ])


class GeometricAug(AugBase):
    def get_augment(self):
        return iaa.Affine(rotate=(-2.5, 2.5),
                          translate_percent=(-0.05, 0.05),
                          scale=(0.95, 1.05),
                          mode='symmetric')

class RandomTransformImage(object):
    def __init__(self, ida_aug_conf=None, training=True):
        self.ida_aug_conf = ida_aug_conf
        self.training = training
        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, results):
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        shape = 4 if 'lidar2img' in results else 3
        for i in range(len(results['image'])):
            # img = Image.fromarray(np.uint8(results['img'][i]))
            img = results['image'][i]
            
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
            img, ida_mat = self.img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
                shape=shape,
            )
            results['image'][i] = self.transform(img) # np.array(img).astype(np.uint8)
            if 'lidar2img' in results:
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]
            elif 'intrinsics' in results:
                results['intrinsics'][i] = ida_mat @ results['intrinsics'][i]

        results['image'] = torch.stack(results['image'], 0)
        if 'lidar2img' in results:
            results['lidar2img'] = torch.stack(results['lidar2img'], 0)
        elif 'intrinsics' in results:
            results['intrinsics'] = torch.stack(results['intrinsics'], 0)
            results['extrinsics'] = torch.stack(results['extrinsics'], 0)
        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate, shape):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L48
        """
        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(shape)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat

    def sample_augmentation(self):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L247
        """
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']

        if self.training:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            
            crop_h = int((newH - fH)/2)
            crop_w = int((newW - fW)/2)

            crop_offset = self.ida_aug_conf['crop_offset']
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate
    
class RandomTransformationBev(object):
    def __init__(self, bev_aug_conf=None, training=True):
        self.training = training
        self.bev_aug_conf = bev_aug_conf

    def get_random_ref_matrix(self):
        """
        Use scipy to create a random reference transformation matrix.
        """
        coeffs = self.bev_aug_conf
        trans_coeff, rot_coeff = coeffs[:3], coeffs[3:]

        # Initialize in homogeneous coordinates.
        mat = np.eye(4, dtype=np.float32)

        # Translate
        mat[:3, 3] = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
            trans_coeff
        )

        # Rotate
        random_zyx = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
            rot_coeff
        )
        mat[:3, :3] = R.from_euler("zyx", random_zyx, degrees=True).as_matrix()

        return mat
    
    def __call__(self):
        if self.training:
            return self.get_random_ref_matrix()
        else:
            return np.eye(4, dtype=np.float32)
    