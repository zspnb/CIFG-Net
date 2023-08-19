import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.segformer.segformer import mit_b2, mit_b3, mit_b4
import math
import torch
import torch.nn as nn
import torch.nn.functional as F




# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         return x

#
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         mip = max(16, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)# 变化到相同维度可以cat
#         #print(x_h.shape, x_w.shape) # n, c, h, 1
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)# n, 8, 2 * h, 1
#         #print(y.shape)
#         x_h, x_w = torch.split(y, [h, w], dim=2)# 分开
#         x_w = x_w.permute(0, 1, 3, 2)
#         #print(x_w.shape, x_h.shape)# n, 8, 1, h    n, 8, h, 1
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#         #print(a_w.shape, a_h.shape)# n, 3, 1, h   n, 3, h, 1
#         #print(identity.shape)# n, c, h, w
#         out = identity * a_w * a_h
#
#         return out
#
# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x
#
# class CFM(nn.Module):
#     def __init__(self, channel):
#         super(CFM, self).__init__()
#         self.relu = nn.ReLU(True)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#
#         self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#         self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
#         self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
#
#
#     def forward(self, x1, x2, x3, g):
#
#         x1_1 = x1
#         x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 + x2 * g
#         x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3 + x3 * g
#
#         x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
#         x2_2 = self.conv_concat2(x2_2)
#
#         x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
#         x3_2 = self.conv_concat3(x3_2)
#
#         x1 = self.conv4(x3_2)
#
#         return x1
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class sMLPBlock(nn.Module):
#     def __init__(self, h=44, w=44, c=64):
#         super().__init__()
#         self.proj_h = nn.Linear(h, h)
#         self.proj_w = nn.Linear(w, w)
#         self.fuse = nn.Linear(3 * c, c)
#
#     def forward(self, x):
#         x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
#         x_w = self.proj_w(x)
#         x_id = x
#         x_fuse = torch.cat([x_h, x_w, x_id], dim=1)
#         out = self.fuse(x_fuse.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         return out
#
# class RFB_modified(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(RFB_modified, self).__init__()
#         self.relu = nn.ReLU(True)
#         self.branch0 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
#         )
#         self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
#         self.conv_res = BasicConv2d(in_channel, out_channel, 1)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
#
#         x = self.relu(x_cat + self.conv_res(x))
#         return x
#
#
#
# class gap(nn.Module):
#     def __init__(self):
#         super(gap, self).__init__()
#         self.glob = nn.AdaptiveAvgPool2d(1)
#         self.glob_c1 = nn.Conv2d(64, 16, kernel_size=1)
#         self.glob_c2 = nn.Conv2d(16, 64, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.glob(x)
#         x = self.glob_c2(self.relu(self.glob_c1(x)))
#         return self.sigmoid(x)



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
#
# class CFM(nn.Module):
#     def __init__(self, channel):
#         super(CFM, self).__init__()
#         self.relu = nn.ReLU(True)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#
#         self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#         self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
#         self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
#
#
#     def forward(self, x1, x2, x3, g):
#
#         x1_1 = x1
#         x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 * g
#         x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
#
#         x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
#         x2_2 = self.conv_concat2(x2_2)
#
#         x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
#         x3_2 = self.conv_concat3(x3_2)
#
#         x1 = self.conv4(x3_2)
#
#         return x1

class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)


    def forward(self, x1, x2, x3, g):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 + x2 * g
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3 + x3 * g

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

# class CFM1(nn.Module):
#     def __init__(self, channel):
#         super(CFM1, self).__init__()
#         self.relu = nn.ReLU(True)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#
#         self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
#         self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
#         self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
#
#
#     def forward(self, x1, x2, x3, g):
#
#         x1_1 = x1
#         x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 * g # 加个残差
#         x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3 * g
#
#         x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
#         x2_2 = self.conv_concat2(x2_2)
#
#         x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
#         x3_2 = self.conv_concat3(x3_2)
#
#         x1 = self.conv4(x3_2)
#
#         return x1


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)



class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



# class RFB_modified(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(RFB_modified, self).__init__()
#         self.relu = nn.ReLU(True)
#         self.branch0 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
#         )
#         self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
#         self.conv_res = BasicConv2d(in_channel, out_channel, 1)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
#
#         x = self.relu(x_cat + self.conv_res(x))
#         return x



class gap(nn.Module):
    def __init__(self):
        super(gap, self).__init__()
        self.glob = nn.AdaptiveAvgPool2d(1)
        self.glob_c1 = nn.Conv2d(64, 16, kernel_size=1)
        self.glob_c2 = nn.Conv2d(16, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.glob(x)
        x = self.glob_c2(self.relu(self.glob_c1(x)))
        return self.sigmoid(x)


# class gap(nn.Module):
#     def __init__(self):
#         super(gap, self).__init__()
#         self.glob = nn.AdaptiveAvgPool2d(1)
#         self.glob_c1 = nn.Conv2d(64, 16, kernel_size=1)
#         self.glob_c2 = nn.Conv2d(16, 64, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.glob(x)
#         x = self.glob_c2(self.relu(self.glob_c1(x)))
#         return self.sigmoid(x)


class backbone(nn.Module):
    def __init__(self, mode='segformer'):
        super(backbone, self).__init__()
        print("segformer...!")
        self.backbone = mit_b3().cuda()
        path = r'E:\zyh\Code\Polyp-PVT-main\backbone\segformer\mit_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        #channel = [64, 128, 320, 512]

        # 版本1:上下两条支路，下面保持原来不变，为级联上采样+空间注意力,上面融合1为相加，融合2也为相加，融合2的尺寸按照小的，上采样八倍
        print(state_dict)
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        channel = 64
        # print(state_dict)
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        self.CFM = CFM(channel)
        self.conv_out1 = nn.Conv2d(channel, 1, 1)
        self.conv_out2 = nn.Conv2d(channel, 1, 1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        # self.glob = nn.AdaptiveAvgPool2d(1)
        self.glob = gap()
        self.confuse1 = BasicConv2d(64 * 2, channel, 1)
        self.rbf = RFB_modified(64, 64)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)


    def forward(self, x):
        # backbone:
        # seg_former:
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # cnn:
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # 分支融合
        # x_f = x1 + p2
        x_f = torch.cat([x1, p2], dim=1)
        x_f = self.confuse1(x_f)
        x_f = self.rbf(x_f)

        # 下面的 高级的信息
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        g = self.glob(x4_t)
        cfm_feature = self.CFM(x4_t, x3_t, x2_t, g)
        # 直接×空间注意力！
        cfm_feature = cfm_feature * self.sa(cfm_feature)
        cfm_feature_out = self.conv_out1(cfm_feature)
        # 1*1卷积之后 直接4倍上采样！！！
        prediction1_8 = F.interpolate(cfm_feature_out, scale_factor=8, mode='bilinear')

        # 上面的 丰富的信息
        # 上下间融合 相×试试？
        x_f = self.Translayer2_0(x_f)
        x_f = self.down(x_f)
        x_f = x_f + cfm_feature
        x_f = x_f * self.ca(x_f)
        x_f = self.conv_out2(x_f)
        prediction2_8 = F.interpolate(x_f, scale_factor=8, mode='bilinear')
        return prediction1_8, prediction2_8
    #     channel = 64
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    #     self.pool1 = nn.MaxPool2d(2)
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    #     self.pool2 = nn.MaxPool2d(2)
    #
    #     self.Translayer2_0 = BasicConv2d(64, channel, 1)
    #     self.Translayer2_1 = BasicConv2d(128, channel, 1)
    #     self.Translayer3_1 = BasicConv2d(320, channel, 1)
    #     self.Translayer4_1 = BasicConv2d(512, channel, 1)
    #     self.CFM = CFM(channel)
    #     self.conv_out1 = nn.Conv2d(channel, 1, 1)
    #     self.conv_out2 = nn.Conv2d(channel, 1, 1)
    #     self.sa = SpatialAttention()
    #     self.ca = ChannelAttention(64)
    #     # self.glob = nn.AdaptiveAvgPool2d(1)
    #     self.glob = gap()
    #     self.confuse1 = BasicConv2d(64 * 2, channel, 1)
    #     self.rbf = RFB_modified(64, 64)
    #     self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
    # def forward(self, x):
    #
    #     # backbone:
    #     # seg_former:
    #     pvt = self.backbone(x)
    #     x1 = pvt[0]
    #     x2 = pvt[1]
    #     x3 = pvt[2]
    #     x4 = pvt[3]
    #     # cnn:
    #     c1 = self.conv1(x)
    #     p1 = self.pool1(c1)
    #     c2 = self.conv2(p1)
    #     p2 = self.pool2(c2)
    #     # 分支融合
    #     # x_f = x1 + p2
    #     x_f = torch.cat([x1, p2], dim=1)
    #     x_f = self.confuse1(x_f)
    #     x_f = self.rbf(x_f)
    #
    #     # 下面的 高级的信息
    #     x2_t = self.Translayer2_1(x2)
    #     x3_t = self.Translayer3_1(x3)
    #     x4_t = self.Translayer4_1(x4)
    #     g = self.glob(x4_t)
    #     cfm_feature = self.CFM(x4_t, x3_t, x2_t, g)
    #     # 直接×空间注意力！
    #     cfm_feature = cfm_feature * self.sa(cfm_feature)
    #     cfm_feature_out = self.conv_out1(cfm_feature)
    #     # 1*1卷积之后 直接4倍上采样！！！
    #     prediction1_8 = F.interpolate(cfm_feature_out, scale_factor=8, mode='bilinear')
    #
    #     # 上面的 丰富的信息
    #     # 上下间融合 相×试试？
    #     x_f = self.Translayer2_0(x_f)
    #     x_f = self.down(x_f)
    #     x_f = x_f + cfm_feature
    #     x_f = x_f * self.ca(x_f)
    #     x_f = self.conv_out2(x_f)
    #     prediction2_8 = F.interpolate(x_f, scale_factor=8, mode='bilinear')
    #     return prediction1_8, prediction2_8


if __name__ == '__main__':
    model = backbone().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.shape, prediction2.shape)
    #print(model)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)




#
# if __name__ == '__main__':
#     input=torch.randn(16,64,88,88)
#     smlp=sMLPBlock(h=88,w=88)
#     out=smlp(input)
#     print(out.shape)
#



#
#
# if __name__ == '__main__':
#     model = backbone().cuda()
#     input_tensor = torch.randn(1, 3, 352, 352).cuda()
#
#     prediction1 = model(input_tensor)
#     print(prediction1.shape)
#     #print(model)
#     num_params = 0
#     for param in model.parameters():
#         num_params += param.numel()
#     print(num_params)

    # a = SAM(num_in=64)
    # in1 = torch.Tensor(4,64,208,208)
    # in2 = torch.Tensor(4,64,208,208)
    # out = a(in1, in2)
    # print(out.shape)

    # model = upsample_aspp().cuda()
    # input_tensor = torch.randn(8, 32, 44, 44).cuda()
    #
    # p = model(input_tensor)
    # print(p.size())
    # print(model)
    # num_params = 0
    # for param in model.parameters():
    #     num_params += param.numel()
    # print(num_params)




'''
备份：加了空间注意力，之前跑的高的， 怎么改变了优化器变回来就不行了
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.cswin.cswin import CSWin
from backbone.p2t.p2t import p2t_tiny
from backbone.pvt.pvt_v2 import pvt_v2_b2, pvt_v2_b3
from backbone.rest.rest import rest_base, rest_large
from backbone.segformer.segformer import mit_b2, mit_b3, mit_b4
from backbone.swin.swin import SwinTransformer
from backbone.twins.twins import pcpvt_base_v0, pcpvt_large
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class backbone(nn.Module):
    def __init__(self, mode='segformer'):
        super(backbone, self).__init__()
        if mode == 'pvtv2':
            self.backbone = pvt_v2_b3()  # [64, 128, 320, 512]
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\pvt\pvt_v2_b3.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'cswin':
            self.backbone = CSWin(embed_dim=96,
                  depth=[2, 4, 32, 2],
                  num_heads=[4, 8, 16, 32],
                  split_size=[1, 2, 7, 7])
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\cswin\cswin_base_224.pth'
            save_model = torch.load(path)['state_dict_ema']
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [96, 192, 384, 768]
        elif mode == 'p2t':
            self.backbone = p2t_tiny().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\p2t\p2t_tiny.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'rest':
            self.backbone = rest_large().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\rest\rest_large.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [96, 192, 384, 768]
        elif mode == 'segformer':
            print("segformer...!")
            self.backbone = mit_b3().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\segformer\mit_b3.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'swin':
            self.backbone = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32], window_size=7)
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\Swin\swin_base_patch4_window7_224.pth'
            save_model = torch.load(path)['model']
            for key in save_model:
                print(key)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [128, 256, 512, 1024]
        elif mode == 'twins':
            self.backbone = pcpvt_large()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\twins\pcpvt_large.pth'
            save_model = torch.load(path)
            # for key in save_model:
            #     print(key)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]


        channel = 64
        print(state_dict)
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        self.CFM = CFM(channel)
        self.conv = nn.Conv2d(channel, 1, 1)

        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)


    def forward(self, x):
        # backbone:
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        cfm_feature = self.CFM(x4_t, x3_t, x2_t)
        # 再加两个注意力试试呢： 直接×注意力！
        # cfm_feature = cfm_feature * self.ca(cfm_feature)
        cfm_feature = cfm_feature * self.sa(cfm_feature)


        cfm_feature = self.conv(cfm_feature)
        # 1*1卷积之后 直接4倍上采样！！！
        prediction1_8 = F.interpolate(cfm_feature, scale_factor=8, mode='bilinear')

        return prediction1_8


if __name__ == '__main__':
    model = backbone().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1 = model(input_tensor)
    print(prediction1.shape)
    #print(model)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)

    # a = SAM(num_in=64)
    # in1 = torch.Tensor(4,64,208,208)
    # in2 = torch.Tensor(4,64,208,208)
    # out = a(in1, in2)
    # print(out.shape)

    # model = upsample_aspp().cuda()
    # input_tensor = torch.randn(8, 32, 44, 44).cuda()
    #
    # p = model(input_tensor)
    # print(p.size())
    # print(model)
    # num_params = 0
    # for param in model.parameters():
    #     num_params += param.numel()
    # print(num_params)

'''

# 备份2：加了两条路，gloab加了，cnn也加了，
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.cswin.cswin import CSWin
from backbone.p2t.p2t import p2t_tiny
from backbone.pvt.pvt_v2 import pvt_v2_b2, pvt_v2_b3
from backbone.rest.rest import rest_base, rest_large
from backbone.segformer.segformer import mit_b2, mit_b3, mit_b4
from backbone.swin.swin import SwinTransformer
from backbone.twins.twins import pcpvt_base_v0, pcpvt_large
import math
import torch
import torch.nn as nn
import torch.nn.functional as F




# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(16, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)# 变化到相同维度可以cat
        #print(x_h.shape, x_w.shape) # n, c, h, 1
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)# n, 8, 2 * h, 1
        #print(y.shape)
        x_h, x_w = torch.split(y, [h, w], dim=2)# 分开
        x_w = x_w.permute(0, 1, 3, 2)
        #print(x_w.shape, x_h.shape)# n, 8, 1, h    n, 8, h, 1
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        #print(a_w.shape, a_h.shape)# n, 3, 1, h   n, 3, h, 1
        #print(identity.shape)# n, c, h, w
        out = identity * a_w * a_h

        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)


    def forward(self, x1, x2, x3, g):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 * g
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class sMLPBlock(nn.Module):
    def __init__(self, h=44, w=44, c=64):
        super().__init__()
        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)
        self.fuse = nn.Linear(3 * c, c)

    def forward(self, x):
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id], dim=1)
        out = self.fuse(x_fuse.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class backbone(nn.Module):
    def __init__(self, mode='segformer'):
        super(backbone, self).__init__()
        if mode == 'pvtv2':
            self.backbone = pvt_v2_b3()  # [64, 128, 320, 512]
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\pvt\pvt_v2_b3.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'cswin':
            self.backbone = CSWin(embed_dim=96,
                  depth=[2, 4, 32, 2],
                  num_heads=[4, 8, 16, 32],
                  split_size=[1, 2, 7, 7])
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\cswin\cswin_base_224.pth'
            save_model = torch.load(path)['state_dict_ema']
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [96, 192, 384, 768]
        elif mode == 'p2t':
            self.backbone = p2t_tiny().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\p2t\p2t_tiny.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'rest':
            self.backbone = rest_large().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\rest\rest_large.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [96, 192, 384, 768]
        elif mode == 'segformer':
            print("segformer...!")
            self.backbone = mit_b3().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\segformer\mit_b3.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'swin':
            self.backbone = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32], window_size=7)
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\Swin\swin_base_patch4_window7_224.pth'
            save_model = torch.load(path)['model']
            for key in save_model:
                print(key)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [128, 256, 512, 1024]
        elif mode == 'twins':
            self.backbone = pcpvt_large()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\twins\pcpvt_large.pth'
            save_model = torch.load(path)
            # for key in save_model:
            #     print(key)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]

        # 版本1:上下两条支路，下面保持原来不变，为级联上采样+空间注意力,上面融合1为相加，融合2也为相加，融合2的尺寸按照小的，上采样八倍
        channel = 64
        print(state_dict)
        model_dict.update(state_dict)

        self.backbone.load_state_dict(model_dict)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        self.CFM = CFM(channel)
        self.conv_out1 = nn.Conv2d(channel, 1, 1)
        self.conv_out2 = nn.Conv2d(channel, 1, 1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.glob = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
    def forward(self, x):

        # backbone:
        # seg_former:
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # cnn:
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # 分支融合
        x_f = x1 + p2

        # 下面的 高级的信息
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        g = self.glob(x4_t)
        cfm_feature = self.CFM(x4_t, x3_t, x2_t, g)
        # 直接×空间注意力！
        cfm_feature = cfm_feature * self.sa(cfm_feature)
        cfm_feature_out = self.conv_out1(cfm_feature)
        # 1*1卷积之后 直接4倍上采样！！！
        prediction1_8 = F.interpolate(cfm_feature_out, scale_factor=8, mode='bilinear')

        # 上面的 丰富的信息
        # 上下间融合 相×试试？
        x_f = self.Translayer2_0(x_f)
        x_f = self.down(x_f)
        x_f = x_f + cfm_feature
        x_f = self.conv_out1(x_f)
        prediction2_8 = F.interpolate(x_f, scale_factor=8, mode='bilinear')
        return prediction1_8, prediction2_8


if __name__ == '__main__':
    model = backbone().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.shape, prediction2.shape)
    #print(model)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)

'''


'''
版本2
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.cswin.cswin import CSWin
from backbone.p2t.p2t import p2t_tiny
from backbone.pvt.pvt_v2 import pvt_v2_b2, pvt_v2_b3
from backbone.rest.rest import rest_base, rest_large
from backbone.segformer.segformer import mit_b2, mit_b3, mit_b4
from backbone.swin.swin import SwinTransformer
from backbone.twins.twins import pcpvt_base_v0, pcpvt_large
import math
import torch
import torch.nn as nn
import torch.nn.functional as F




# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.relu2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(16, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)# 变化到相同维度可以cat
        #print(x_h.shape, x_w.shape) # n, c, h, 1
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)# n, 8, 2 * h, 1
        #print(y.shape)
        x_h, x_w = torch.split(y, [h, w], dim=2)# 分开
        x_w = x_w.permute(0, 1, 3, 2)
        #print(x_w.shape, x_h.shape)# n, 8, 1, h    n, 8, h, 1
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        #print(a_w.shape, a_h.shape)# n, 3, 1, h   n, 3, h, 1
        #print(identity.shape)# n, c, h, w
        out = identity * a_w * a_h

        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)


    def forward(self, x1, x2, x3, g):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 * g
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class sMLPBlock(nn.Module):
    def __init__(self, h=44, w=44, c=64):
        super().__init__()
        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)
        self.fuse = nn.Linear(3 * c, c)

    def forward(self, x):
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id], dim=1)
        out = self.fuse(x_fuse.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class backbone(nn.Module):
    def __init__(self, mode='segformer'):
        super(backbone, self).__init__()
        if mode == 'pvtv2':
            self.backbone = pvt_v2_b3()  # [64, 128, 320, 512]
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\pvt\pvt_v2_b3.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'cswin':
            self.backbone = CSWin(embed_dim=96,
                  depth=[2, 4, 32, 2],
                  num_heads=[4, 8, 16, 32],
                  split_size=[1, 2, 7, 7])
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\cswin\cswin_base_224.pth'
            save_model = torch.load(path)['state_dict_ema']
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [96, 192, 384, 768]
        elif mode == 'p2t':
            self.backbone = p2t_tiny().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\p2t\p2t_tiny.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'rest':
            self.backbone = rest_large().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\rest\rest_large.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [96, 192, 384, 768]
        elif mode == 'segformer':
            print("segformer...!")
            self.backbone = mit_b3().cuda()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\segformer\mit_b3.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]
        elif mode == 'swin':
            self.backbone = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32], window_size=7)
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\Swin\swin_base_patch4_window7_224.pth'
            save_model = torch.load(path)['model']
            for key in save_model:
                print(key)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [128, 256, 512, 1024]
        elif mode == 'twins':
            self.backbone = pcpvt_large()
            path = r'E:\zyh\Code\Polyp-PVT-main\backbone\twins\pcpvt_large.pth'
            save_model = torch.load(path)
            # for key in save_model:
            #     print(key)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            channel = [64, 128, 320, 512]

        # 版本1:上下两条支路，下面保持原来不变，为级联上采样+空间注意力,上面融合1为相加，融合2也为相加，融合2的尺寸按照小的，上采样八倍
        channel = 64
        print(state_dict)
        model_dict.update(state_dict)

        self.backbone.load_state_dict(model_dict)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        self.CFM = CFM(channel)
        self.conv_out1 = nn.Conv2d(channel, 1, 1)
        self.conv_out2 = nn.Conv2d(channel, 1, 1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.glob = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
    def forward(self, x):

        # backbone:
        # seg_former:
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # cnn:
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # 分支融合
        x_f = x1 #+ p2

        # 下面的 高级的信息
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        g = self.glob(x4_t)
        cfm_feature = self.CFM(x4_t, x3_t, x2_t, g)
        # 直接×空间注意力！
        cfm_feature = cfm_feature * self.sa(cfm_feature)
        cfm_feature_out = self.conv_out1(cfm_feature)
        # 1*1卷积之后 直接4倍上采样！！！
        prediction1_8 = F.interpolate(cfm_feature_out, scale_factor=8, mode='bilinear')

        # 上面的 丰富的信息
        # 上下间融合 相×试试？
        x_f = self.Translayer2_0(x_f)
        x_f = self.down(x_f)
        x_f = x_f + cfm_feature
        x_f = x_f * self.ca(x_f)
        x_f = self.conv_out2(x_f)
        prediction2_8 = F.interpolate(x_f, scale_factor=8, mode='bilinear')
        return prediction1_8, prediction2_8


if __name__ == '__main__':
    model = backbone().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.shape, prediction2.shape)
    #print(model)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)
'''