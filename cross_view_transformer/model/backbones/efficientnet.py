import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


# Precomputed aliases
MODELS = {
    'efficientnet-b0': [
        ('reduction_1', (0, 2)),
        ('reduction_2', (2, 4)),
        ('reduction_3', (4, 6)),
        ('reduction_4', (6, 12))
    ],
    'efficientnet-b4': [
        ('reduction_1', (0, 3)),
        ('reduction_2', (3, 7)),
        ('reduction_3', (7, 10)),
        ('reduction_4', (10, 22)),
    ]
}


class EfficientNetExtractor(torch.nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = EfficientNetExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, layer_names, image_height, image_width, model_name='efficientnet-b4', checkpointing=False):
        super().__init__()

        assert model_name in MODELS
        assert all(k in [k for k, v in MODELS[model_name]] for k in layer_names)

        idx_max = -1
        layer_to_idx = {}

        # Find which blocks to return
        for i, (layer_name, _) in enumerate(MODELS[model_name]):
            if layer_name in layer_names:
                idx_max = max(idx_max, i)
                layer_to_idx[layer_name] = i+1

        # We can set memory efficient swish to false since we're using checkpointing
        net = EfficientNet.from_pretrained(model_name)
        self.checkpointing = checkpointing
        if checkpointing:
            net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        blocks = [nn.Sequential(net._conv_stem, net._bn0, net._swish)]

        # Only run needed blocks
        for idx in range(idx_max+1):
            l, r = MODELS[model_name][idx][1]
            block = SequentialWithArgs(*[(net._blocks[i], [i * drop]) for i in range(l, r)])
            blocks.append(block)

        self.layers = nn.Sequential(*blocks)
        self.layer_names = layer_names
        self.idx_pick = [layer_to_idx[l] for l in layer_names]
        # Pass a dummy tensor to precompute intermediate shapes
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True)

        result = []

        for layer in self.layers:
            if self.training and self.checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)

            result.append(x)

        return [result[i] for i in self.idx_pick]


class SequentialWithArgs(nn.Sequential):
    def __init__(self, *layers_args):
        layers = [layer for layer, args in layers_args]
        args = [args for layer, args in layers_args]

        super().__init__(*layers)

        self.args = args

    def forward(self, x):
        for l, a in zip(self, self.args):
            x = l(x, *a)

        return x

class Encoder_eff(nn.Module):
    def __init__(self, C=None, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        # if self.downsample == 16:
        #     if self.version == 'b0':
        #         upsampling_in_channels = 320 + 112
        #     elif self.version == 'b4':
        #         upsampling_in_channels = 448 + 160
        #     upsampling_out_channels = 512
        # elif self.downsample == 8:
        #     if self.version == 'b0':
        #         upsampling_in_channels = 112 + 40
        #     elif self.version == 'b4':
        #         upsampling_in_channels = 160 + 56
        #     upsampling_out_channels = 128
        # else:
        #     raise ValueError(f'Downsample factor {self.downsample} not handled.')

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        # x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return [input_2, input_1]

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        # x = self.depth_layer(x)  # feature and depth head
        return x
    
class EfficientNet_PointBEV(nn.Module):
    def __init__(self, image_height, image_width, version="b4", checkpoint=False):
        super().__init__()

        self.version = version
        self.checkpoint = checkpoint
        self._init_efficientnet(version)

        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes
        
    def _init_efficientnet(self, version):
        trunk = EfficientNet.from_pretrained(f"efficientnet-{version}")

        self._conv_stem, self._bn0, self._swish = (
            trunk._conv_stem,
            trunk._bn0,
            trunk._swish,
        )
        self.drop_connect_rate = trunk._global_params.drop_connect_rate

        self._blocks = nn.ModuleList()
        for idx, block in enumerate(trunk._blocks):
            if version == "b0" and idx > 10 or version == "b4" and idx > 21:
                break
            self._blocks.append(block)

        del trunk

    def forward(self, x, return_all=False):
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            if self.training and self.checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, drop_connect_rate)
            else:
                x = block(x, drop_connect_rate=drop_connect_rate)
            # x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f"reduction_{len(endpoints)+1}"] = prev_x
            prev_x = x

            if self.version == "b0" and idx == 10:
                break
            if self.version == "b4" and idx == 21:
                break

        # Head
        endpoints[f"reduction_{len(endpoints)+1}"] = x

        if not return_all:
            list_keys = ["reduction_3", "reduction_4"]
        else:
            list_keys = list(endpoints.keys())

        return [endpoints[k] for k in list_keys]

if __name__ == '__main__':
    """
    Helper to generate aliases for efficientnet backbones
    """
    device = torch.device('cuda')
    dummy = torch.rand(6, 3, 224, 480).to(device)

    for model_name in ['efficientnet-b0', 'efficientnet-b4']:
        net = EfficientNet.from_pretrained(model_name)
        net = net.to(device)
        net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        conv = nn.Sequential(net._conv_stem, net._bn0, net._swish)

        record = list()

        x = conv(dummy)
        print(x.shape)
        px = x
        pi = 0

        # Terminal early to save computation
        for i, block in enumerate(net._blocks):
            x = block(x, i * drop)
            if px.shape[-2:] != x.shape[-2:]:
                record.append((f'reduction_{len(record)+1}', (pi, i+1)))
                print(len(record)+1,x.shape)
                pi = i + 1
                px = x

        print(model_name, ':', {k: v for k, v in record})
