import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

class AutoEncoder(nn.Module):
    def __init__(
        self,
        dim,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]},
        multi_head: bool =False,
        label_indices: list = [],
        prob: float = 0.0,
    ):
        super().__init__()

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.outputs = outputs
        self.encoder = Encoder(len(label_indices), dim)
        self.decoder = decoder
        self.multi_head = multi_head
        self.label_indices = label_indices
        self.mask = torch.distributions.bernoulli.Bernoulli(1-prob) if prob > 0.0 else None

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
        else:
            self.to_logits = nn.Sequential(
                nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim_last),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_last, dim_max, 1)
            )

    def forward(self,batch):
        x = self.encode(batch['lidar'])
        return self.prediction(x)
    
    def _parse_input(self, x):
        x = [x[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
        x = torch.cat(x, 1) 
        if self.mask is not None:
            b, _, h, w = x.shape
            mask = self.mask.sample(sample_shape=(b,1,h,w)).to(x.device)
            x = x * mask
        return x 

    def encode(self, x):
        # x = self._parse_input(x)
        return self.encoder(x)
    
    def prediction(self, x):
        y = self.decoder(x)
        if self.multi_head:
            return {k:self.to_logits[k](y) for k in self.outputs}
        else:
            z = self.to_logits(y)
            return{k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
        
class Encoder(nn.Module):
    def __init__(self,in_dim, out_dim):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim,64,kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.out_conv = nn.Sequential(
            nn.Conv2d(128, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.SELU(inplace=True),
        )

    def forward(self,x):
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.out_conv(x)
        return x