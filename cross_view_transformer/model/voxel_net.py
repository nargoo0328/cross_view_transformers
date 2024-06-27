import torch.nn as nn
import torch.nn.functional as F
import torch
import math

xrange = [-50.0, 50.0]
yrange = [-50.0, 50.0]
zrange = [-5.0, 3.0]
voxel_size = [0.5, 0.5, 1.0]

W = math.ceil((xrange[1] - xrange[0]) / voxel_size[0])
H = math.ceil((yrange[1] - yrange[0]) / voxel_size[1])
D = math.ceil((zrange[1] - zrange[0]) / voxel_size[2])

MAX_PTS = 10000
# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self,in_channels,out_channels,k,s,p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=k,stride=s,padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation
    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x=self.bn(x)
        if self.activation:
            return F.relu(x,inplace=True)
        else:
            return x

# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)

# Fully Connected Network
class FCN(nn.Module):

    def __init__(self,cin,cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self,x):
        # KK is the stacked k across batch
        # *shape, dim = x.shape
        # x = self.linear(x.view(-1,dim))
        # x = F.relu(self.bn(x))
        # return x.view(*shape,-1)
        kk, t, _ = x.shape
        x = self.linear(x.view(kk*t,-1))
        x = F.relu(self.bn(x))
        return x.view(kk,t,-1)

# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self,cin,cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin,self.units)

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)
        #locally aggregated feature
        laf = torch.max(pwf,1)[0].unsqueeze(1).repeat(1,10,1)
        # point-wise concat feature
        pwcf = torch.cat((pwf,laf),dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf

# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7,32)
        self.vfe_2 = VFE(32,128)
        self.fcn = FCN(128,128)
    def forward(self, x):
        mask = torch.ne(torch.max(x,2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x,1)[0]
        return x

# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x

class VoxelNet(nn.Module):

    def __init__(self):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE()
        self.cml = CML()

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]
        b = sparse_features.shape[0] // MAX_PTS
        dense_feature = torch.zeros(dim, b, D, H, W).to(sparse_features.device)
        print(sparse_features.shape,coords.shape)
        dense_feature[:, coords, coords[:,0], coords[:,1], coords[:,2]]= sparse_features

        return dense_feature.transpose(0,1)

    def forward(self, batch):
        voxel_features, voxel_coords = batch['lidar'], batch['coords']
        voxel_features = voxel_features.reshape(-1,10,7)
        voxel_coords = voxel_coords.reshape(-1,3)
        # feature learning network
        vwfs = self.svfe(voxel_features)
        vwfs = self.voxel_indexing(vwfs,voxel_coords)

        # convolutional middle network
        cml_out = self.cml(vwfs)

        return cml_out
