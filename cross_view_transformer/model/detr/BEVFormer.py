import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional
import copy

from einops import rearrange, repeat
from ..ops.defattn.modules import MSDeformAttn

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class BEVFomerHead(nn.Module):
    """
    Predict bbox -> project to BEV -> sparse_conv2d/conv2d
    """
    def __init__(self,
                with_box_refine=True,
                transformer=None,
                embed_dims=128,
                pc_range=None,
                **kwargs):
        
        super().__init__()

        self.with_box_refine = with_box_refine
        self.transformer = transformer
        self.embed_dims = embed_dims
        self.pc_range = pc_range

        self._init_box_layers(**kwargs)
        self.init_weights()
            
    def _init_box_layers(self, num_classes=0, num_query=100, num_reg_fcs=2, **kwargs):
        
        self.num_classes = num_classes
        self.num_query = num_query

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.reference_points = nn.Linear(self.embed_dims, 3)
        # torch.nn.init.xavier_uniform_(self.reference_points, distribution='uniform', bias=0.)
        
        cls_branch = []
        for _ in range(num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes + 1))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 6))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = self.transformer.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.reference_points.weight.data)
        torch.nn.init.constant_(self.reference_points.bias.data, 0.0)
    
    def forward(self, bev_feats):

        query_embeds = self.query_embedding.weight
        bs = bev_feats.shape[0]
        query_pos, query = torch.split(
            query_embeds, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()

        hs, init_reference, inter_references = self.transformer( 
            query,
            reference_points,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            value=bev_feats,
            query_pos=query_pos,
        )

        # hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        cls_scores = torch.stack(outputs_classes)
        bbox_preds = torch.stack(outputs_coords)

        box_output = {
            'pred_logits': cls_scores[-1],
            'pred_boxes': bbox_preds[-1],
            'aux_outputs': [{'pred_logits': outputs_class, 'pred_boxes': outputs_coord} 
                            for outputs_class, outputs_coord in zip(cls_scores[:-1],bbox_preds[:-1])],
        }
        return box_output

class DetectionTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, n_layer=1, return_intermediate=True, **kwargs):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_layers = n_layer

        self._init_layers(n_layer, **kwargs)

    def _init_layers(self, n_layer, **kwargs):
        layer = DetectionTransformerDecoderLayer(**kwargs)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(n_layer)])

    def forward(self,
                query,
                reference_points,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        init_reference_out = reference_points

        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                reference_points=reference_points_input,
                **kwargs)
            
            # output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            # output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), init_reference_out, torch.stack(
                intermediate_reference_points)

        return output, init_reference_out, reference_points
    
class DetectionTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dims, nheads, dropout=0.1, **kwargs):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dims, nheads, dropout=dropout)
        self.cross_attn = SADefnAttn(embed_dims, nheads, dropout)
        self.ffn = MLP(embed_dims, embed_dims*2, embed_dims, 2)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self,
        query=None, 
        query_pos=None, 
        value=None,
        reference_points=None,
        **kwargs
    ):
        q = k = self.with_pos_embed(query, query_pos)
        tgt = self.self_attn(q, k, value=query)[0]
        query = query + tgt
        query = self.norm1(query)
        query = self.cross_attn(
            query,
            query_pos,
            value,
            reference_points
        )
        query = self.norm2(query)
        query = self.norm3(self.ffn(query))

        return query


class SADefnAttn(nn.Module):
    def __init__(
        self,
        in_c=128,
        n_heads=4,
        dropout=0.1,
        msdef_kwargs={"n_levels": 1, "n_points": 8},
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=in_c, n_heads=n_heads, **msdef_kwargs)
        self.mlp_out = nn.Linear(in_c, in_c)

    def forward(self, 
            query=None, 
            query_pos=None, 
            value=None,
            reference_points=None,
            **kwargs
        ):
        # Alias
        B, N, C = query.shape
        device = query.device

        # Get residual
        query_residual = query.clone()

        # Add positional encoding
        if query_pos is not None:
            query = query + query_pos

        # Define shapes
        input_spatial_shapes = query.new_full([1, 2], fill_value=200).long()
        input_level_start_index = query.new_zeros([1]).long()

        value = rearrange(value, 'b c h w -> b (h w) c')
        # Process
        queries = self.deformable_attention(
            query,
            reference_points,
            value,
            input_spatial_shapes,
            input_level_start_index,
        )

        return self.dropout(self.mlp_out(queries)) + query_residual

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, add_identity=True, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.add_identity = add_identity
        self.dropout = nn.Dropout(dropout) if dropout != 0.0 else nn.Identity()

    def forward(self, x):
        if self.add_identity:
            res = x.clone()

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        
        if self.add_identity:
            return res + self.dropout(x)
        return self.dropout(x)