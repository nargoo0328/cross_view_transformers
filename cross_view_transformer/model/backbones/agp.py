from typing import Iterable, Optional
from collections import OrderedDict

import torch
import torch.nn as nn

import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck

class AlignRes(nn.Module):
    """Align resolutions of the outputs of the backbone."""

    def __init__(
        self,
        mode="upsample",
        scale_factors: Iterable[int] = [1, 2],
        in_channels: Iterable[int] = [256, 512, 1024, 2048],
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        if mode == "upsample":
            for s in scale_factors:
                if s != 1:
                    self.layers.append(
                        nn.Upsample(
                            scale_factor=s, mode="bilinear", align_corners=False
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        elif mode == "conv2dtranspose":
            for i, in_c in enumerate(in_channels):
                if scale_factors[i] != 1:
                    self.layers.append(
                        nn.ConvTranspose2d(
                            in_c, in_c, kernel_size=2, stride=2, padding=0
                        )
                    )
                else:
                    self.layers.append(nn.Identity())

        else:
            raise NotImplementedError
        return

    def forward(self, x):
        return [self.layers[i](xi) for i, xi in enumerate(x)]


class PrepareChannel(nn.Module):
    """Transform the feature map to align with Network."""

    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],
        interm_c=128,
        out_c: Optional[int] = 128,
        mode="doubleconv",
        tail_mode="identity",
        depth_num=0,
    ):
        super().__init__()
        assert mode in ["simpleconv", "doubleconv", "doubleconv_w_depth_layer"]
        assert tail_mode in ["identity", "conv2d"]

        in_c = sum(in_channels)
        if "simpleconv" in mode:
            self.layers = nn.Sequential(
                nn.Conv2d(in_c, interm_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(interm_c),
            )

        elif "doubleconv" in mode:
            # Used in SimpleBEV
            self.layers = nn.Sequential(
                nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(interm_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(interm_c, interm_c, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(interm_c),
                nn.ReLU(inplace=True),
            )
            if depth_num != 0:
                self.depth_layers = nn.Sequential(
                    nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm2d(interm_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(interm_c, interm_c, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm2d(interm_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(interm_c, depth_num, kernel_size=1, padding=0)
                )
                # nn.init.zeros_(self.depth_layers[-1].weight)
                # nn.init.zeros_(self.depth_layers[-1].bias)
            else:
                self.depth_layers = None

        if tail_mode == "identity":
            self.tail = nn.Identity()
            self.out_c = interm_c
        elif tail_mode == "conv2d":
            # Used in SimpleBEV
            self.tail = nn.Conv2d(interm_c, out_c, kernel_size=1, padding=0)
            self.out_c = out_c

        return

    def forward(self, x, pseudo_depth):
        if pseudo_depth is not None:
            H, W = x.shape[-2:]
            pseudo_depth = pseudo_depth.flatten(0,1) / 61.2
            pseudo_depth = nn.functional.interpolate(pseudo_depth, size=[H,W], mode='bilinear')
            x = torch.cat((x, pseudo_depth), dim=1)
        
        if self.depth_layers is not None:
            depth = self.depth_layers(x)
        else:
            depth = None
            
        return self.tail(self.layers(x)), depth


class AGPNeck(nn.Module):
    """
    Upsample outputs of the backbones, group them and align them to be compatible with Network.

    Note: mimics UpsamplingConcat in SimpleBEV.
    """

    def __init__(
        self,
        align_res_layer,
        prepare_c_layer,
        group_method=lambda x: torch.cat(x, dim=1),
        list_output=False,
    ):
        """
        Args:
            - align_res_layer: upsample layers at different resolution to the same.
            - group_method: how to gather the upsampled layers.
            - prepare_c_layer: change the channels of the upsampled layers in order to align with the network.
        """
        super().__init__()

        self.align_res_layer = align_res_layer
        self.group_method = group_method
        self.prepare_c_layer = prepare_c_layer
        self.out_c = prepare_c_layer.out_c
        self.list_output = list_output

    def forward(self, x: Iterable[torch.Tensor], pseudo_depth=None):
        if x[0].ndim == 5:
            x = [y.flatten(0,1) for y in x]
        # Align resolution of inputs.
        x = self.align_res_layer(x)

        # Group inputs.
        x = self.group_method(x)

        # Change channels of final input.
        x, depth = self.prepare_c_layer(x, pseudo_depth)
        assert x.shape[1] == self.out_c
        if self.list_output:
            return [x], depth
        return x, depth

class Depth_Neck(nn.Module):
    def __init__(
        self,
        in_channels,
        scales,
        embed_dims,
        group_method=lambda x: torch.cat(x, dim=1),
        return_index=0,
        depth_num=0,
    ):
        super().__init__()
        self.up_layers = nn.ModuleList()
        scales_ = [scale // scales[return_index] for scale in scales[return_index:]]
        for s in scales_:
            if s != 1:
                self.up_layers.append(
                    nn.Upsample(
                        scale_factor=s, mode="bilinear", align_corners=False
                    )
                )
            else:
                self.up_layers.append(nn.Identity())
        self.group_method = group_method

        # aggregate features for return index
        self.return_index = return_index
        in_c = sum(in_channels[return_index:])
        self.layers = nn.Sequential(
                nn.Conv2d(in_c, embed_dims, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(embed_dims), # InstanceNorm2d
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(embed_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, padding=0)
        )

        # aggregate features for depth estimation on highest resolution
        if depth_num > 0:
            self.up_layers_depth = nn.ModuleList()
            for s in scales:
                if s != 1:
                    self.up_layers_depth.append(
                        nn.Upsample(
                            scale_factor=s, mode="bilinear", align_corners=False
                        )
                    )
                else:
                    self.up_layers_depth.append(nn.Identity())
    
            in_c = sum(in_channels)
            self.depth_layer = nn.Sequential(
                nn.Conv2d(in_c, embed_dims, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(embed_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(embed_dims),
                nn.ReLU(inplace=True),
                # BasicBlock(embed_dims, embed_dims),
                # BasicBlock(embed_dims, embed_dims),
                nn.Conv2d(embed_dims, depth_num, kernel_size=1, padding=0)
        )
        else:
            self.depth_layer = None

    def forward(self, x):
        feats_x = x[self.return_index:]
        feats_x = [self.up_layers[i](xi) for i, xi in enumerate(feats_x)]
        feats_x = self.group_method(feats_x)
        feats_x = self.layers(feats_x)

        if self.depth_layer is not None:
            depth = [self.up_layers_depth[i](xi) for i, xi in enumerate(x)]
            depth = self.group_method(depth)
            depth = self.depth_layer(depth)
        else:
            depth = None

        return [feats_x], depth
