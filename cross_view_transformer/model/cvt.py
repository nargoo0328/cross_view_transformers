import torch.nn as nn
import torch
from .encoder import Normalize
from einops import rearrange

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]},
        multi_head: bool =False,
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

    def forward(self, batch):
        x = self.encoder(batch)
        y = self.decoder(x)
        if self.multi_head:
            return {k:self.to_logits[k](y) for k in self.outputs}
        else:
            z = self.to_logits(y)

        return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
