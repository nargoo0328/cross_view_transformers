import torch
import torch.nn.functional as F
import json
import hydra
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from cross_view_transformer.common import setup_config, setup_data_module
import threading

# import depth anything here
import sys
metric_depth = True
if metric_depth:
    sys.path.append("../../Depth-Anything-V2/metric_depth")
else:
    sys.path.append("../../Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

input_size = 518
transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
])

def prepare_model(input_size=518, encoder='vitl'):
    model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    if metric_depth:
        dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 80 # 20 for indoor model, 80 for outdoor model

        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        model.load_state_dict(torch.load(f'/media/hcis-s20/SRL/Depth-Anything-V2/ckpt/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    else:
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'/media/hcis-s20/SRL/Depth-Anything-V2/ckpt/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    
    return model

def image2tensor(raw_image, DEVICE=torch.device('cpu')):        
                
    images = []
    for image in raw_image:
        # image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image)
        
        image = image.to(DEVICE)
        images.append(image)
    
    return torch.stack(images)

def save_depth(file_name, depth):
    cv2.imwrite(file_name, depth)
    return

def setup(cfg):
    # Don't change these
    # cfg.data.dataset = cfg.data.dataset.replace('_generated', '')
    cfg.data.augment = 'none'
    cfg.data.bev = False
    cfg.data.augment_img = False
    cfg.data.image.h = 900
    cfg.data.image.w = 1600
    cfg.data.image.top_crop = 0
    cfg.data.depth = 'generate'
    cfg.loader.batch_size = 1
    cfg.loader.persistent_workers = False
    cfg.loader.drop_last = False
    cfg.loader.shuffle = False
    cfg.loader.__delattr__("train_batch_size")
    cfg.loader.__delattr__("val_batch_size")

@hydra.main(config_path=str(Path.cwd() / 'config'), config_name='config.yaml')
def main(cfg):
    """
    Creates the following dataset structure

    cfg.data.labels_dir/
        01234.json
        01234/
            bev_0001.png
            bev_0002.png
            ...

    If the 'visualization' flag is passed in,
    the generated data will be loaded from disk and shown on screen
    """
    setup_config(cfg, setup)

    data = setup_data_module(cfg)

    depths_dir = Path(cfg.data.dataset_dir)
    depths_dir = depths_dir / 'depths'
    depths_dir.mkdir(parents=False, exist_ok=True)
    for sensor in ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        (depths_dir / sensor).mkdir(parents=False, exist_ok=True)
    
    device = torch.device('cuda:0')
    model = prepare_model()
    model = model.to(device).eval()
    for split in ['val', 'train']:
        print(f'Generating split: {split}')

        for episode in tqdm(data.get_split(split, loader=False), position=0, leave=False):
            # scene_dir = labels_dir / episode.scene_name
            # scene_dir.mkdir(exist_ok=True, parents=False)

            loader = torch.utils.data.DataLoader(episode, collate_fn=list, **cfg.loader)

            for i, batch in enumerate(tqdm(loader, position=1, leave=False)):
                # list[dict]
                batch = batch[0]
                images = [img.permute(1,2,0).numpy() for img in batch['image']]
                h, w = images[0].shape[:2]
                images = image2tensor(images, device)
                with torch.no_grad():
                    depths = model(images)
                    depths = F.interpolate(depths[:, None], (h, w), mode="bilinear", align_corners=True)[:, 0].cpu().numpy()

                threads = []
                for j in range(6):
                    threads.append(threading.Thread(target=save_depth, args=(str(batch['depth'][j]), depths[j])))

                for thread in threads:
                    thread.start()

                for thread in threads:
                    thread.join()

                # for j in range(6):
                #     cv2.imwrite(str(batch['depth'][j]), depths[j])
            
if __name__ == '__main__':
    main()