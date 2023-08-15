import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
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
        is_ce: bool = False,
        is_fuse: bool = False,
        ablation: int = 1,
        fuse_in: int = 128,
        radar: bool = False,
        multi_head: bool =False,
        n_bev_embedding: int = 1,
    ):
        super().__init__()
        if  n_bev_embedding == 1:
            dim_total = 0
            dim_max = 0

            for _, (start, stop) in outputs.items():
                assert start < stop

                dim_total += stop - start
                dim_max = max(dim_max, stop)

            assert dim_max == dim_total
        else:
            dim_max = 1
            assert len(outputs) == n_bev_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs
        self.is_ce = is_ce
        self.is_fuse = is_fuse
        self.ablation = ablation
        self.fusn_in = fuse_in
        self.radar = radar
        self.multi_head = multi_head
        self.n_bev_embedding = n_bev_embedding

        cvt_in = self.decoder.out_channels
        if is_fuse:
            self.rescale = nn.AdaptiveAvgPool2d(200)
            if self.ablation == 1:
                self.reduce_layer = ConvModule(
                    fuse_in + 64,
                    128,
                    3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
                self.fuse_layer = SE_Block(128)
                cvt_in = 128
            elif self.ablation == 2:
                self.reduce_layer = ConvModule(
                    fuse_in,
                    64,
                    3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
        if radar:
            # self.norm = Normalize(mean=[1.939,31.86,4.592,1.305,0.023,-0.117,-0.009,2.587,18.005,17.923,4.155,2.07,15.269,2.776],
            # std=[1.59,26.229,7.48,6.035,2.682,1.843,1.054,0.953,5.155,5.123,5.221,1.549,4.395,0.788]
            # )
            dim = 128
            self.radar_conv = nn.Sequential(
                nn.Conv2d(14,dim,3,padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim,dim,1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )
            self.reduce_layer = ConvModule(
                64+dim,
                64,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

        self.to_logits = nn.Sequential(
            nn.Conv2d(cvt_in, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))
        if multi_head:
            self.center_head = nn.Sequential(
            nn.Conv2d(cvt_in, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, 1, 1),
            nn.Sigmoid())

            self.offset_head = nn.Sequential(
            nn.Conv2d(cvt_in, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, 2, 1))
        if is_ce:
            self.out_relu = nn.ReLU(inplace=True)

    def forward(self, batch,inspect=False):
        if self.radar:
            radar = self.radar_conv(batch['radar'])
            x = self.encoder(batch,radar)
        else:    
            x, mid_features = self.encoder(batch,inspect=inspect)
        
        # x = torch.Size([1, 128, 25, 25])
        if self.n_bev_embedding !=1:
            x = torch.stack(x) # n_bev,b,...
            x = rearrange(x,'b n ... -> (b n) ...')

        y = self.decoder(x)
        if self.is_fuse:
            b = y.shape[0]
            fuse = batch['features']
            fuse = self.rescale(fuse)
            # origin 
            if self.ablation == 1:
                y = torch.cat((y,fuse),dim=1)
                y = self.fuse_layer(self.reduce_layer(y))
            elif self.ablation == 2:
                fuse = self.reduce_layer(fuse)
                y = y + fuse
            # add
        elif self.radar:
            y = torch.cat((y,radar),dim=1)
            y = self.reduce_layer(y)
        # y = torch.Size([1, 64, 200, 200])
        z = self.to_logits(y)
        if self.multi_head:
            center = self.center_head(y)
            offset = self.offset_head(y)
        if self.is_ce:
            z = self.out_relu(z)
        if self.multi_head:
            z = torch.cat((z,center,offset),dim=1)
        if self.n_bev_embedding !=1:
            b = z.shape[0]
            z = rearrange(z,'(b n) ... -> b n ...',b=b,n=self.n_bev_embedding)
            return {k: z[:,i] for i,(k,_) in self.outputs.items()}
        if inspect:
            return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}, mid_features
        return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
    
    
class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]},
        is_ce: bool = False,
        is_fuse: bool = False,
        ablation: int = 1,
        fuse_in: int = 128,
        radar: bool = False,
        multi_head: bool =False,
        n_bev_embedding: int = 1,
    ):
        super().__init__()
        if  n_bev_embedding == 1:
            dim_total = 0
            dim_max = 0

            for _, (start, stop) in outputs.items():
                assert start < stop

                dim_total += stop - start
                dim_max = max(dim_max, stop)

            assert dim_max == dim_total
        else:
            dim_max = 1
            assert len(outputs) == n_bev_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs
        self.is_ce = is_ce
        self.is_fuse = is_fuse
        self.ablation = ablation
        self.fusn_in = fuse_in
        self.radar = radar
        self.multi_head = multi_head
        self.n_bev_embedding = n_bev_embedding

        cvt_in = self.decoder.out_channels
        if is_fuse:
            self.rescale = nn.AdaptiveAvgPool2d(200)
            if self.ablation == 1:
                self.reduce_layer = ConvModule(
                    fuse_in + 64,
                    128,
                    3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
                self.fuse_layer = SE_Block(128)
                cvt_in = 128
            elif self.ablation == 2:
                self.reduce_layer = ConvModule(
                    fuse_in,
                    64,
                    3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
        if radar:
            # self.norm = Normalize(mean=[1.939,31.86,4.592,1.305,0.023,-0.117,-0.009,2.587,18.005,17.923,4.155,2.07,15.269,2.776],
            # std=[1.59,26.229,7.48,6.035,2.682,1.843,1.054,0.953,5.155,5.123,5.221,1.549,4.395,0.788]
            # )
            dim = 128
            self.radar_conv = nn.Sequential(
                nn.Conv2d(14,dim,3,padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim,dim,1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )
            self.reduce_layer = ConvModule(
                64+dim,
                64,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

        self.to_logits = nn.Sequential(
            nn.Conv2d(cvt_in, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))
        if multi_head:
            self.center_head = nn.Sequential(
            nn.Conv2d(cvt_in, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, 1, 1),
            nn.Sigmoid())

            self.offset_head = nn.Sequential(
            nn.Conv2d(cvt_in, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, 2, 1))
        if is_ce:
            self.out_relu = nn.ReLU(inplace=True)

    def forward(self, batch,inspect=False):
        if self.radar:
            radar = self.radar_conv(batch['radar'])
            x = self.encoder(batch,radar)
        else:    
            x, mid_features = self.encoder(batch,inspect=inspect)
        
        # x = torch.Size([1, 128, 25, 25])
        if self.n_bev_embedding !=1:
            x = torch.stack(x) # n_bev,b,...
            x = rearrange(x,'b n ... -> (b n) ...')

        y = self.decoder(x)
        if self.is_fuse:
            b = y.shape[0]
            fuse = batch['features']
            fuse = self.rescale(fuse)
            # origin 
            if self.ablation == 1:
                y = torch.cat((y,fuse),dim=1)
                y = self.fuse_layer(self.reduce_layer(y))
            elif self.ablation == 2:
                fuse = self.reduce_layer(fuse)
                y = y + fuse
            # add
        elif self.radar:
            y = torch.cat((y,radar),dim=1)
            y = self.reduce_layer(y)
        # y = torch.Size([1, 64, 200, 200])
        z = self.to_logits(y)
        if self.multi_head:
            center = self.center_head(y)
            offset = self.offset_head(y)
        if self.is_ce:
            z = self.out_relu(z)
        if self.multi_head:
            z = torch.cat((z,center,offset),dim=1)
        if self.n_bev_embedding !=1:
            b = z.shape[0]
            z = rearrange(z,'(b n) ... -> b n ...',b=b,n=self.n_bev_embedding)
            return {k: z[:,i] for i,(k,_) in self.outputs.items()}
        if inspect:
            return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}, mid_features
        return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
