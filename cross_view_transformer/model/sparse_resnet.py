import spconv.pytorch as spconv
from einops import rearrange

import torch
import torch.nn as nn

from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy

algo = spconv.ConvAlgo.Native

class SubMConvNormAct(spconv.SparseSequential):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size=3,
        padding=1,
        bias=False,
        norm=nn.BatchNorm1d,
        activ=nn.ReLU,
    ):
        super().__init__(
            spconv.SubMConv2d(
                in_c,
                out_c,
                kernel_size,
                padding=padding,
                groups=1,
                bias=bias,
                algo=algo,
            ),
            norm(out_c, momentum=0.1),
            activ(inplace=False),
        )


class SubMConvNormReLU(spconv.SparseSequential):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size=3,
        stride=1,
        groups=1,
        activ=nn.BatchNorm1d,
        is_3d=False,
    ):
        padding = (kernel_size - 1) // 2

        if is_3d:
            layer = spconv.SubMConv3d
        else:
            layer = spconv.SubMConv2d
        super(SubMConvNormReLU, self).__init__(
            layer(
                in_c,
                out_c,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
                algo=algo,
            ),
            activ(out_c, momentum=0.1),
            nn.ReLU(inplace=False),
        )


class SparseUpsamplingAdd(spconv.SparseModule):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size,
        indice_key,
        activ=nn.BatchNorm1d,
        bias=False,
        is_3d=False,
    ):
        super().__init__()
        if is_3d:
            layer = spconv.SparseInverseConv3d
        else:
            layer = spconv.SparseInverseConv2d
        self.upsample_layer = spconv.SparseSequential(
            layer(
                in_c, out_c, kernel_size, indice_key=indice_key, bias=bias, algo=algo
            ),
            activ(out_c, momentum=0.1),
        )
        return

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip
# Sparse UNET
class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self, in_c, out_c, kernels=[3, 3], strides=[1, 1], downsample=None, is_3d=False
    ):
        """Basic Block for Sparse ResNet. It consists on two convolutions and a skip connection.

        O = Activ( I + Down( ConvNorm(ConvActivNorm(I))) )
        """
        super().__init__()
        if is_3d:
            layer = spconv.SubMConv3d
        else:
            layer = spconv.SubMConv2d
        conv1 = layer(in_c, out_c, kernels[0], strides[0], 1, bias=False, algo=algo)
        conv2 = layer(out_c, out_c, kernels[1], strides[1], 1, bias=False, algo=algo)
        norm1 = nn.BatchNorm1d(out_c, momentum=0.1)
        norm2 = nn.BatchNorm1d(out_c, momentum=0.1)

        self.conv1_bn_relu = spconv.SparseSequential(
            conv=conv1, bn=norm1, relu=nn.ReLU(inplace=True)
        )
        self.conv2_bn = spconv.SparseSequential(conv=conv2, bn=norm2)

        self.activ = spconv.SparseReLU(inplace=True)
        self.downsample = downsample
        self.iden_for_fx_match = spconv.SparseIdentity()
        self.skip = in_c == out_c

    def forward(self, x: spconv.SparseConvTensor):
        identity = x
        out = self.conv1_bn_relu(x)
        out = self.conv2_bn(out)

        if self.downsample:
            identity = self.downsample(x)

        if self.skip:
            out = out.replace_feature(out.features + identity.features)

        # cf. INT8_GUIDE
        out = out.replace_feature(self.activ(out.features))
        return out


# 2D UNet
class SparseEncoder(nn.Module):
    def __init__(
        self, in_c, mid_c=64, down_mode="maxpool", with_large_kernels: bool = False
    ):
        super().__init__()
        # Activation
        activ = nn.BatchNorm1d

        self.first_conv = SubMConvNormReLU(in_c, mid_c, kernel_size=7, activ=activ)

        # (H,W)
        k = 5 if with_large_kernels else 3
        self.layer_1 = spconv.SparseSequential(
            SparseBasicBlock(mid_c, mid_c, kernels=[k, k]),
            SparseBasicBlock(mid_c, mid_c, kernels=[k, k]),
        )

        # (H/4,W/4)
        mid_c_lev2 = mid_c * 2
        self.layer_2 = spconv.SparseSequential(
            SparseBasicBlock(
                mid_c,
                mid_c_lev2,
                kernels=[k, k],
                # downsample=spconv.SparseSequential(
                #     spconv.SubMConv2d(mid_c, mid_c_lev2, 1, bias=False, algo=algo),
                #     activ(mid_c_lev2, momentum=0.1),
                # ),
            ),
            SparseBasicBlock(mid_c_lev2, mid_c_lev2, kernels=[k, k]),
        )
        # (H/8,W/8)
        mid_c_lev3 = mid_c * 4
        self.out_c = mid_c * 4
        self.layer_3 = spconv.SparseSequential(
            SparseBasicBlock(
                mid_c_lev2,
                mid_c_lev3,
                kernels=[3, 3],
                # downsample=spconv.SparseSequential(
                #     spconv.SubMConv2d(mid_c_lev2, mid_c_lev3, 1, bias=False, algo=algo),
                #     activ(mid_c_lev3, momentum=0.1),
                # ),
            ),
            SparseBasicBlock(mid_c_lev3, mid_c_lev3, kernels=[3, 3]),
        )

        # Downsample layers
        if down_mode == "maxpool":
            self.down_1 = spconv.SparseMaxPool2d(
                k, 2, (k - 1) // 2, indice_key="cp1", algo=algo
            )
            self.down_2 = spconv.SparseMaxPool2d(
                k, 2, (k - 1) // 2, indice_key="cp2", algo=algo
            )
            self.down_3 = spconv.SparseMaxPool2d(3, 2, 1, indice_key="cp3", algo=algo)
        else:
            self.down_1 = spconv.SparseConv2d(
                mid_c,
                mid_c,
                3,
                stride=2,
                padding=1,
                bias=False,
                indice_key="cp1",
                algo=algo,
            )
            self.down_2 = spconv.SparseConv2d(
                mid_c * 2,
                mid_c * 2,
                3,
                stride=2,
                padding=1,
                bias=False,
                indice_key="cp2",
                algo=algo,
            )
            self.down_3 = spconv.SparseConv2d(
                mid_c * 4,
                mid_c * 4,
                3,
                stride=2,
                padding=1,
                bias=False,
                indice_key="cp3",
                algo=algo,
            )

    def forward(self, x_sp):
        skip_x = {"1": x_sp}
        x_sp = self.first_conv(x_sp)
        x_sp = self.down_1(self.layer_1(x_sp))
        skip_x["2"] = x_sp
        x_sp = self.down_2(self.layer_2(x_sp))
        skip_x["3"] = x_sp
        x_sp = self.down_3(self.layer_3(x_sp))
        return x_sp, skip_x


class SparseDecoder(nn.Module):
    def __init__(
        self, mid_c, in_c, with_large_kernels: bool = False, bias: bool = False
    ):
        super().__init__()
        # Activation
        activ = nn.BatchNorm1d

        # Upsampling
        self.up_3 = SparseUpsamplingAdd(
            mid_c * 4,
            mid_c * 2,
            kernel_size=3,
            indice_key="cp3",
            activ=activ,
            bias=bias,
        )

        k = 5 if with_large_kernels else 3
        self.up_2 = SparseUpsamplingAdd(
            mid_c * 2, mid_c, kernel_size=k, indice_key="cp2", activ=activ, bias=bias
        )
        self.up_1 = SparseUpsamplingAdd(
            mid_c, in_c, kernel_size=k, indice_key="cp1", activ=activ, bias=bias
        )
        return

    def forward(self, x_sp, skip_x):
        x_sp = self.up_3(x_sp, skip_x["3"])
        x_sp = self.up_2(x_sp, skip_x["2"])
        x_sp = self.up_1(x_sp, skip_x["1"])
        return x_sp


class SparseUNet(nn.Module):
    def __init__(
        self,
        in_c,
        mid_c=64,
        out_c=128,
        with_tail_conv: bool = False,
        with_large_kernels: bool = False,
        with_decoder_bias: bool = False,
        top_k: int = 100,
    ):
        super().__init__()
        self.encoder = SparseEncoder(in_c, mid_c, "maxpool", with_large_kernels)
        self.decoder = SparseDecoder(mid_c, in_c, with_large_kernels, with_decoder_bias)
        self.in_c = in_c

        if with_tail_conv:
            self.tail_conv = spconv.SparseSequential(
                SubMConvNormReLU(in_c, out_c, kernel_size=7),
                spconv.SubMConv2d(out_c, out_c, kernel_size=1, bias=False, algo=algo),
            )
        else:
            self.tail_conv = nn.Identity()
        
        self.top_k = top_k

    def forward(
        self,
        x, 
        view=None,
        from_dense=False,
        grid_idx=None,
        spatial_shape=None, 
        batch_size=None,
        **kwargs,
    ):
        if not from_dense:
            # feats, indices, batch_size = box_to_bev(x, view, self.top_k)
            if batch_size is not None:
                x_sp = spconv.SparseConvTensor(x, grid_idx, spatial_shape, batch_size)
            else:
                b, N = x.shape[:2]
                device = x.device

                # pad batch index
                batch_idx = torch.arange(0, b).view(-1,1,1).expand(-1, N, 1).int().to(device)
                grid_idx = torch.cat([batch_idx, grid_idx], dim=-1).flatten(0,1)
                x_sp = spconv.SparseConvTensor(x.flatten(0,1), grid_idx.int(), spatial_shape, b)

        else:
            # feats = rearrange(feats, "b c h w -> b h w c")
            x_sp = spconv.SparseConvTensor.from_dense(x)
        
        x_sp, skip_x = self.encoder(x_sp)
        x_sp = self.decoder(x_sp, skip_x)
        x_sp = self.tail_conv(x_sp)
        
        return x_sp
    
def box_to_bev(pred, view, topk):

    pred_boxes_coords = pred['pred_boxes'][..., :4].clone()
    box_feats = pred['box_feats'].clone()
    pred_logits = pred['pred_logits'].clone()
    
    pred_boxes[:,2:4] = pred_boxes[:,2:4].exp()
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes, transform=True)

    B, N, C = box_feats.shape
    device = box_feats.device

    if topk == 0:

        pred_boxes_coords = nn.functional.pad(pred_boxes_coords,(0, 1), value=1) # b filter_N 3
        pred_boxes_coords = (torch.einsum('b i j, b N j -> b N i', view, pred_boxes_coords)[..., :2]).int()
        batch_idx = torch.arange(0, B).view(-1,1,1).expand(-1,N,1).int().to(device)
        pred_boxes_coords = torch.cat([batch_idx, pred_boxes_coords], dim=-1).flatten(0,1)
        box_feats = box_feats.flatten(0,1)
    
    else:
        scores, _ = pred_logits.softmax(-1)[..., :-1].max(-1)
        filter_idx = torch.topk(scores, k=topk, dim=-1).indices

        # Expand dimensions for filter_idx for matching with pred_boxes_coords
        filter_idx_expand = filter_idx.unsqueeze(-1).expand(*filter_idx.shape, pred_boxes_coords.shape[-1])
        pred_boxes_coords = torch.gather(pred_boxes_coords, 1, filter_idx_expand)
        
        filter_idx_expand = filter_idx.unsqueeze(-1).expand(*filter_idx.shape, C)
        box_feats = torch.gather(box_feats, 1, filter_idx_expand).flatten(0,1)

        pred_boxes_coords = nn.functional.pad(pred_boxes_coords,(0, 1), value=1) # b filter_N 3
        pred_boxes_coords = (torch.einsum('b i j, b N j -> b N i', view, pred_boxes_coords)[..., :4]).int()

        batch_idx = torch.arange(0, B).view(-1,1,1).expand(-1,topk,1).int().to(device)
        pred_boxes_coords = torch.cat([batch_idx, pred_boxes_coords], dim=-1).flatten(0,1)

    return box_feats, pred_boxes_coords, B


