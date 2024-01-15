import timm
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from mmseg.apis import init_model

class swinT_backbone(nn.Module):
    def __init__(self,model_name='swinv2_cr_tiny_ns_224.sw_in1k',image_height=224,image_width=480,out_indices=[0,2]):
        super().__init__()
        self.model = timm.create_model(model_name,pretrained=True,img_size=(image_height,image_width),features_only=True,out_indices=out_indices)

        dummy = torch.rand(1, 3, image_height, image_width)
        self.output_shapes = [y.shape for y in self.model(dummy)]

    def forward(self,x):
        x = self.model(x)
        return x
    
class ResNet101(nn.Module):
    def __init__(self, image_height, image_width,out_indices=[2,4], fpn=False):
        super().__init__()
        self.model = timm.create_model('resnet101', features_only=True, pretrained=True,out_indices=out_indices)
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self.model(dummy)]
        self.output_shapes = output_shapes
        self.fpn = None
        if fpn:
            in_channels = [i[1] for i in output_shapes]
            self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels, 256)

    def forward(self,x):
        x = self.model(x)
        if self.fpn is not None:
            _in = OrderedDict()
            for i,tmp_x in enumerate(x):
                _in[i] = tmp_x
            x = self.fpn(_in)
            x = [v for _, v in x.items()]
        return x
    
class ResNet101_torchvision(nn.Module):
    def __init__(self, image_height, image_width):
        super().__init__()
        resnet = torchvision.models.resnet101(pretrained=True)
        layer1 = nn.Sequential(*list(resnet.children())[:-5])
        layer2 = resnet.layer2
        layer3 = resnet.layer3
        layer4 = resnet.layer4
        self.backbone = nn.ModuleList([layer1,layer2,layer3,layer4])
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self.forward(dummy)]
        self.output_shapes = output_shapes

    def forward(self,x):
        out = []
        for l in self.backbone:
            x = l(x)
            out.append(x)
        return out
    
class ResNet101_pretrained_cityscapes(nn.Module):
    def __init__(self, image_height, image_width,out_indices=[2,4]):
        super().__init__()
        config_file = '/media/user/data/mmsegmentation/mask2former_r101_8xb2-90k_cityscapes-512x1024.py'
        checkpoint_file = '/media/user/data/mmsegmentation/mask2former_r101_8xb2-90k_cityscapes-512x1024_20221130_031628-43e68666.pth'
        model = init_model(config_file, checkpoint_file, device='cpu')
        self.backbone = model.backbone
        self.out_indices = out_indices
        dummy = torch.rand(1, 3, image_height, image_width)#.to('cuda:0')
        output_shapes = [x.shape for x in self.forward(dummy)]
        self.output_shapes = output_shapes
        self.backbone = self.backbone.to('cpu')

    def forward(self,x):
        out = []
        x = self.backbone(x)
        for i in self.out_indices:
            out.append(x[i-1])
        return out

if __name__ == '__main__':
    swinT = swinT_backbone(model_name='vit_small_patch16_384')
    print(swinT.output_shapes)