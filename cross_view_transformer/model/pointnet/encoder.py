import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class PointNetEncoder(nn.Module):
    def __init__(self,radar_dim):
        super(PointNetEncoder, self).__init__()

        self.sa1 = PointNetSetAbstraction(256, 0.1, 32, radar_dim + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(64, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(16, 0.4, 32, 128 + 3, [128, 128, 256], False)
        
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])


    def forward(self, xyz):
        xyz = xyz[:,0]
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return self.project_bev(l0_points,l0_xyz)
    
    def project_bev(self,points,coords):
        b = points.shape[0]
        device = points.device
        dtype = torch.LongTensor if device.type == 'cpu' else torch.cuda.LongTensor
        coords = torch.floor(coords)
        bev_map = torch.zeros((b,128,200,200)).to(device)
        for i in range(b):
            bev_map[i,:,coords[i,1,:].type(dtype),coords[i,0,:].type(dtype)] = points[i]
        return bev_map