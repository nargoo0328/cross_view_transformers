import timm
import torch
import torch.nn as nn

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
    def __init__(self, image_height, image_width,out_indices=[2,4]):
        super().__init__()
        self.model = timm.create_model('resnet101', features_only=True, pretrained=True,out_indices=out_indices)
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self.model(dummy)]

        self.output_shapes = output_shapes

    def forward(self,x):
        x = self.model(x)
        return x
