import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
from torchvision.models.resnet import resnet18

from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy
from .layers import LayerNorm2d

norm = nn.InstanceNorm2d # LayerNorm2d nn.InstanceNorm2d

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, factor, skip_dim=0, residual=False):
        super().__init__()

        dim = out_channels # // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)
    
class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 1, stride=2),
            nn.Conv2d(in_channels*2, in_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*2, in_channels*2, 1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)

class BEVDecoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, factor, dim, residual)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x, view=None):
        y = x

        for i, layer in enumerate(self.layers):
            y = layer(y, x)

        return y
    
class BEVUNet(nn.Module):
    def __init__(self, dim, num_blocks, factor=2, sparse_input=False):
        super().__init__()

        self.num_blocks = num_blocks

        # self.threshold = nn.Linear(dim, 1)
        # nn.init.zeros_(self.threshold.weight)

        self.in_c = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        layers = list()
        channels = dim

        for _ in range(num_blocks):
            layer = EncoderBlock(channels)
            layers.append(layer)

            channels = channels * 2
        self.encoder = nn.Sequential(*layers)

        layers = list()

        for _ in range(num_blocks):
            layer = DecoderBlock(channels, channels//2, factor)
            layers.append(layer)

            channels = channels // 2
        self.decoder = nn.Sequential(*layers)

        self.sparse_input = sparse_input

    def forward(self, x, view=None, grid_idx=None):

        if self.sparse_input:
            b, N = x.shape[:2]
            device = x.device
            batch_idx = torch.arange(0, b).view(-1,1,1).expand(-1, N, 1).int().to(device)
            grid_idx = torch.cat([batch_idx, grid_idx], dim=-1).flatten(0,1)
            x = spconv.SparseConvTensor(x.flatten(0,1), grid_idx.int(), [200,200], b).dense()

        x = self.in_c(x)
        tmp = []
        for layer in self.encoder:
            tmp.append(x)
            x = layer(x)
            # import matplotlib.pyplot as plt
            # plt.imshow(x[0,0].cpu().numpy())


        for i, layer in enumerate(self.decoder):
            x = layer(x)
            x = x + tmp[self.num_blocks-i-1]
        # import matplotlib.pyplot as plt
        # plt.imshow(x[0,0].cpu().numpy())
        return x
    
class SparseConvDecoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for i, layer in enumerate(self.layers):
            y = layer(y, x)

        return y
    
class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip
    
class SimpleBEVDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        backbone = resnet18(zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

    def forward(self, x, **kwargs):
        b, c, h, w = x.shape

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class BevEncode(nn.Module):
    def __init__(self, in_channels):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, in_channels, kernel_size=3, padding=1, bias=False),
            norm(in_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x
    
class SegHead(nn.Module):
    def __init__(self, 
            dim_last, 
            multi_head, 
            outputs,
            sparse=False,
        ):
        super().__init__()

        self.multi_head = multi_head
        self.outputs = outputs

        dim_total = 0
        dim_max = 0
        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total
        if sparse:
            algo = spconv.ConvAlgo.Native
            if multi_head:
                layer_dict = {}
                for k, (start, stop) in outputs.items():
                    layer_dict[k] = spconv.SparseSequential(
                        spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                        nn.InstanceNorm1d(dim_last, momentum=0.1),
                        nn.ReLU(inplace=False),
                        spconv.SubMConv2d(
                            dim_last, out_channels=stop-start, kernel_size=1, padding=0, algo=algo
                        )
                    )
                self.to_logits = nn.ModuleDict(layer_dict)
            else:
                self.to_logits = spconv.SparseSequential(
                    spconv.SubMConv2d(dim_last, dim_last, 3, padding=1, bias=False, algo=algo),
                    nn.BatchNorm1d(dim_last, momentum=0.1),
                    nn.ReLU(inplace=False),
                    spconv.SubMConv2d(dim_last, dim_max, 1, algo=algo),
                )
        else:
            if multi_head:
                layer_dict = {}
                for k, (start, stop) in outputs.items():
                    layer_dict[k] = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    norm(dim_last),
                    nn.GELU(),
                    nn.Conv2d(dim_last, stop-start, 1)
                )
                self.to_logits = nn.ModuleDict(layer_dict)
            else:
                self.to_logits = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    norm(dim_last),
                    nn.GELU(),
                    nn.Conv2d(dim_last, dim_max, 1)
                )

    def forward_head(self, x, aux=False):
        if self.multi_head:
            if aux:
                return {'VEHICLE': self.to_logits['VEHICLE'](x)}
            else:
                return {k: v(x) for k, v in self.to_logits.items()}
        else:
            x = self.to_logits(x)
            return {k: x[:, start:stop] for k, (start, stop) in self.outputs.items()}
        
    def forward(self, x, aux=False):
        if self.multi_head:
            if aux:
                return {'VEHICLE': self.to_logits['VEHICLE'](x)}
            else:
                return {k: v(x) for k, v in self.to_logits.items()}
        else:
            x = self.to_logits(x)
            return {k: x[:, start:stop] for k, (start, stop) in self.outputs.items()}

def box_to_bev(x, H, W, view, threshold):
    pred_boxes = x['pred_boxes'][..., :4].clone()
    box_feats = x['box_feats'].clone()
    pred_logits = x['pred_logits'].clone()

    B, _, C = box_feats.shape
    device = box_feats.device

    bev_feats = box_feats.new_zeros([B, H, W, C])
    
    pred_boxes[:,2:4] = pred_boxes[:,2:4].exp()
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes, transform=True)

    scores, _ = pred_logits.softmax(-1)[..., :-1].max(-1)
    # idx = scores > threshold

    for i in range(B):

        # Filter score < threshold
        idx = scores[i] > threshold # threshold[i].mean()
        BOX_exp = pred_boxes[i,idx]
        F_exp = box_feats[i,idx]
        N = BOX_exp.shape[0]

        if N == 0:
            continue

        # Project from Lidar -> BEV
        coords = F.pad(BOX_exp,(0, 1), value=1).transpose(0,1)
        coords = (view[i] @ coords)[:4].transpose(0,1)

        x1, y1, x2, y2 = coords.unbind(1)
        x1 = (x1 / x1.max()).clamp(0, 1) * (W - 1)
        y1 = (y1 / y1.max()).clamp(0, 1) * (H - 1)
        x2 = (x2 / x2.max()).clamp(0, 1) * (W - 1)
        y2 = (y2 / y2.max()).clamp(0, 1) * (H - 1)
        x1, y1, x2, y2 = x1.int(), y1.int(), x2.int(), y2.int()
        for j in range(N):
            x1, y1, x2, y2 = coords[j].unbind(-1)
            x1 = (x1 / x1.max()).clamp(0, 1) * (W - 1)
            y1 = (y1 / y1.max()).clamp(0, 1) * (H - 1)
            x2 = (x2 / x2.max()).clamp(0, 1) * (W - 1)
            y2 = (y2 / y2.max()).clamp(0, 1) * (H - 1)
            x1, y1, x2, y2 = x1.int(), y1.int(), x2.int(), y2.int()
            bev_feats[i, x1:x2, y1:y2] += F_exp[j]
            
    return bev_feats.permute(0,3,1,2) # B C H W

