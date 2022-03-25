"""
Author: Zhiyuan Zhang
Date: Dec 2021
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from riconv2_utils import RIConv2SetAbstraction

class get_model(nn.Module):
    def __init__(self,num_class, n, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 64
        self.normal_channel = normal_channel
        
        self.sa0 = RIConv2SetAbstraction(npoint=512*n, radius=0.12,  nsample=8, in_channel= 0+in_channel, mlp=[32],  group_all=False)
        self.sa1 = RIConv2SetAbstraction(npoint=256*n,  radius=0.16,  nsample=16, in_channel=32 + in_channel, mlp=[64],  group_all=False)
        self.sa2 = RIConv2SetAbstraction(npoint=128*n,  radius=0.24,  nsample=32, in_channel=64 + in_channel, mlp=[128],  group_all=False)
        self.sa3 = RIConv2SetAbstraction(npoint=64*n,  radius=0.48,  nsample=64, in_channel=128 + in_channel, mlp=[256],  group_all=False)
        self.sa4 = RIConv2SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + in_channel, mlp=[512],  group_all=True)

        channels = [512, 256]
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            # compute the LRA and use as normal
            norm = None
        
        l0_xyz, l0_norm, l0_points = self.sa0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sa1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sa2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sa3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sa4(l3_xyz, l3_norm, l3_points)
        x = l4_points.view(B, 512)
        # x = torch.max(l4_points, 2)[0]
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss
