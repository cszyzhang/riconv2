"""
Author: Zhiyuan Zhang
Date: Dec 2021
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from riconv2_utils import RIConv2SetAbstraction, RIConv2FeaturePropagation_v2
from time import time


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 64
        self.normal_channel = normal_channel

        self.sa0 = RIConv2SetAbstraction(npoint=512, radius=0.2,  nsample=8, in_channel= 0+in_channel, mlp=[64],  group_all=False)
        self.sa1 = RIConv2SetAbstraction(npoint=256,  radius=0.4, nsample=16, in_channel=64 + in_channel, mlp=[128],  group_all=False)
        self.sa2 = RIConv2SetAbstraction(npoint=128,  radius=0.6, nsample=32, in_channel=128 + in_channel, mlp=[256],  group_all=False)
        self.sa3 = RIConv2SetAbstraction(npoint=64,  radius=0.8,  nsample=64, in_channel=256 + in_channel, mlp=[512],  group_all=False)

        self.fp3 = RIConv2FeaturePropagation_v2(radius=1.5, nsample=8, in_channel=512+64, in_channel_2=512+256, mlp=[512], mlp2=[512])
        self.fp2 = RIConv2FeaturePropagation_v2(radius=0.8, nsample=16, in_channel=512+64, in_channel_2=512+128, mlp=[512], mlp2=[256])
        self.fp1 = RIConv2FeaturePropagation_v2(radius=0.48, nsample=32, in_channel=256+64, in_channel_2=256+64, mlp=[256], mlp2=[128])
        self.fp0 = RIConv2FeaturePropagation_v2(radius=0.48, nsample=32,  in_channel=128+64, in_channel_2=128+16, mlp=[128], mlp2=[])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1)

    def forward(self, xyz):
        B, N, C = xyz.shape
        if self.normal_channel:
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]

        l0_xyz, l0_norm, l0_points = self.sa0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sa1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sa2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sa3(l2_xyz, l2_norm, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_norm, l3_norm, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_norm, l2_norm, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_norm, l1_norm, l0_points, l1_points)
        l0_points = self.fp0(xyz, l0_xyz, norm, l0_norm, None, l0_points)


        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))