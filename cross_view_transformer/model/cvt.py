import torch.nn as nn
import torch
import torch.nn.functional as F
from cross_view_transformer.common import load_backbone
import os
from cross_view_transformer.util.box_ops import box_cxcywh_to_xyxy

class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]},
        multi_head: bool =False,
        autoencoder = None,
        ae_config = None,
        transformer = None,
        num_classes = 0,
        num_queries = 0, # maxv maxp: 67/50, total 80
        class_specific = False,
    ):
        super().__init__()
        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs
        self.multi_head = multi_head
        
        self.class_specific = class_specific
        self.transformer = transformer
        if transformer is not None:
            dim = 128
            if class_specific:
                self.class_embed_vehicle = nn.Linear(dim, 1)
                self.bbox_embed_vehicle = MLP(dim, dim, 4, 3)
                self.class_embed_ped = nn.Linear(dim, 1)
                self.bbox_embed_ped = MLP(dim, dim, 4, 3)
            else:
                self.class_embed = nn.Linear(dim, num_classes + 1)
                self.bbox_embed = MLP(dim, dim, 4, 3)

            self.query_embed = nn.Embedding(num_queries, dim)
            self.input_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))
        
        if multi_head:
            self.to_logits = nn.ModuleDict(
                {k: 
                nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim_last),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_last, o_ - i_, 1),
                ) for k,(i_,o_) in outputs.items()}
                )
        self.autoencoder = autoencoder
        if autoencoder is not None:
            self._load_autoencoder(ae_config)


    def forward(self, batch):
        x = self.encoder(batch)
        y, inter_bev = self.decoder(x)
        result = {}
        if self.transformer is not None:
            hs, _ = self.transformer(self.input_proj(inter_bev), self.query_embed.weight)
            if self.class_specific:
                n_q = hs[0].shape[1]
                v_q, p_q = hs[0][:,:n_q//2], hs[0][:,n_q//2:]
                outputs_class_vehicle = self.class_embed_vehicle(v_q)
                outputs_coord_vehicle = self.bbox_embed_vehicle(v_q).sigmoid()
                outputs_class_ped = self.class_embed_ped(p_q)
                outputs_coord_ped = self.bbox_embed_ped(p_q).sigmoid()
                out = {'pred_logits_vehicle': outputs_class_vehicle, 'pred_boxes_vehicle': outputs_coord_vehicle,
                    'pred_logits_ped': outputs_class_ped, 'pred_boxes_ped': outputs_coord_ped
                }
            else:
                outputs_class = self.class_embed(hs[0])
                outputs_coord = self.bbox_embed(hs[0]).sigmoid()
                out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

            result.update(out)
        if self.multi_head:
            out = {k:self.to_logits[k](y) for k in self.outputs}
        else:
            z = self.to_logits(y)
            out = {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}

        result.update(out)
        if False:
            view = batch['view']
            pred_box = box_cxcywh_to_xyxy(outputs_coord, transform=True)
            pred_box = pred_box * 100 - 50 
            x0, y0, x1, y1 = pred_box.unbind(-1)
            p1, p2 = torch.stack([x0,y0], dim=-1), torch.stack([x1,y1], dim=-1)
            p1, p2 = F.pad(p1,(0,1), value=1).permute(0,2,1), F.pad(p2,(0,1), value=1).permute(0,2,1)
            p1 = torch.einsum('b i j, b j n -> b i n', view, p1).permute(0,2,1)[...,:2]
            p2 = torch.einsum('b i j, b j n -> b i n', view, p2).permute(0,2,1)[...,:2]
            new_points = torch.cat([p1,p2],dim=-1)
            pred_logits = outputs_class.softmax(-1) # b, N, num_classes
            # for box, logit in zip(pred_box,pred_logits):
            # out['bev'] = 
            # out['ped'] = 
            b, n = new_points.shape[:2]
            bev, ped = torch.zeros_like(out['bev']), torch.zeros_like(out['ped'])
            for i in range(b):
                for j in range(n):
                    x1,y1,x2,y2 = new_points[i,j]
                    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                    bev[i,0,x1:x2,y1:y2] = pred_logits[i,j][0]
                    ped[i,0,x1:x2,y1:y2] = pred_logits[i,j][1]
            out['bev'] = + bev
            out['ped'] = + ped
        if self.autoencoder is not None:
            latent_features = self.autoencoder.encode(batch['bev'])
            out['features'] = latent_features
            out['pred_features'] = x
        
        return result
    
    def _load_autoencoder(self, ae_config):
        ckpt = '../../../logs/cross_view_transformers_test/' + ae_config['ckpt'] + '/checkpoints/last.ckpt'
        print(f"Loading pre-trained autoencoder. Checkpoint path: {ckpt}")
        self.autoencoder = load_backbone(ckpt,backbone=self.autoencoder)
        # TODO decoder & to_logits not used
        if ae_config['use_decoder']:
            self.decoder = self.autoencoder.decoder
            self.to_logits = self.autoencoder.to_logits
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.to_logits.parameters():
                param.requires_grad = False

        for param in self.autoencoder.parameters():
            param.requires_grad = False

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x