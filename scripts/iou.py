import argparse

from common import get_cfg, prepare_val, calculate_iou
import torch 

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model no.",
                        type=str)
    parser.add_argument("--mode", help="run which metrics",
                        type=str)
    return parser.parse_args()

def main(args):
    DATASET_DIR = '/media/hcis-s20/SRL/nuscenes/trainval/'
    version = 'cvt_labels_nuscenes_v1'
    cfg1 = get_cfg(DATASET_DIR,version,'SparseBEVSeg_Det') # cvt_nuscenes_multiclass
    device = torch.device('cuda:0')
    CHECKPOINT_PATH = './logs/cross_view_transformers_test/' + args.model + '/checkpoints/last.ckpt'
    model1, network1, loader1, _, _ = prepare_val(cfg1,device,CHECKPOINT_PATH, mode='val',batch_size=10)
    calculate_iou(model1, network1, loader1, device, args.mode)
    
if __name__ == '__main__':
    args = get_parser()
    main(args)