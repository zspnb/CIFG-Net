from turtle import right
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib1.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat



'''CFP MODEL'''


class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_2_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)

        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)

        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)

        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)
        output_4 = torch.cat([o4_1, o4_2, o4_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4], 1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input



""" Adaptive Selection Module    ASM  ***ACSNET"""


class ASM(nn.Module):
    def __init__(self, in_channels, all_channels):
        super(ASM, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.selayer = SELayer(all_channels)

    def forward(self, lc, fuse):
        fuse = self.non_local(fuse)
        fuse = torch.cat([lc, fuse], dim=1)
        fuse = self.selayer(fuse)

        return fuse


"""
Squeeze and Excitation Layer

https://arxiv.org/abs/1709.01507

"""


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


"""
Non Local Block

https://arxiv.org/abs/1711.07971
"""


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

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


# Efficient self attention
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PPM(nn.Module):
    def __init__(self, pooling_sizes=(1, 3, 5)):
        super().__init__()
        self.layer = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(size, size)) for size in pooling_sizes])

    def forward(self, feat):
        b, c, h, w = feat.shape
        output = [layer(feat).view(b, c, -1) for layer in self.layer]
        output = torch.cat(output, dim=-1)
        return output


class ESA_layer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.ppm = PPM(pooling_sizes=(1, 3, 5))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # input x (b, c, h, w)
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # q/k/v shape: (b, inner_dim, h, w)
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.heads)  # q shape: (b, head, n_q, d)

        k, v = self.ppm(k), self.ppm(v)  # k/v shape: (b, inner_dim, n_kv)
        k = rearrange(k, 'b (head d) n -> b head n d', head=self.heads)  # k shape: (b, head, n_kv, d)
        v = rearrange(v, 'b (head d) n -> b head n d', head=self.heads)  # v shape: (b, head, n_kv, d)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # shape: (b, head, n_q, n_kv)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)  # shape: (b, head, n_q, d)
        out = rearrange(out, 'b head n d -> b n (head d)')
        return self.to_out(out)


class ESA_blcok(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()
        self.ESAlayer = ESA_layer(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        b, c, h, w = x.shape
        out = rearrange(x, 'b c h w -> b (h w) c')
        out = self.ESAlayer(x) + out
        out = self.ff(out) + out
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)

        return out


# Cascading Context Module
class CCM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=[1, 3, 5], in_channel_list=[],
                 out_channel_list=[256, 128 / 2], cascade=False):
        super(CCM, self).__init__()
        self.cascade = cascade
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []

        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(len(self.out_channel_list)):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, self.out_channel_list[i], 3, 1, 1),
                                           nn.BatchNorm2d(self.out_channel_list[i]),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout2d(),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x, y=None):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))
        global_context = torch.cat(global_context, dim=1)
        output = []
        for i in range(len(self.GCoutmodel)):
            out = self.GCoutmodel[i](global_context)
            if self.cascade is True and y is not None:
                out = out + y[i]
            output.append(out)
        return output

        # BDGM

    class RFB_modified(nn.Module):
        def __init__(self, in_channel, out_channel):
            super(RFB_modified, self).__init__()
            self.relu = nn.ReLU(True)
            self.branch0 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
            )
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
                nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(out_channel),
                nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
                nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
                nn.BatchNorm2d(out_channel),
                nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
            )
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
                nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
                nn.BatchNorm2d(out_channel),
                nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
            )
            self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
            self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        def forward(self, x):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

            x = self.relu(x_cat + self.conv_res(x))
            return x

    class Agg(nn.Module):
        def __init__(self, channel=64):
            super(Agg, self).__init__()
            self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
            self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

            # stage 1
            self.h2h_1 = nn.Sequential(
                Conv2dReLU(channel, channel, 3, 1, 1)
            )
            self.h2l_1 = nn.Sequential(
                Conv2dReLU(channel, channel, 3, 1, 1)
            )
            self.l2h_1 = nn.Sequential(
                Conv2dReLU(channel, channel, 3, 1, 1)
            )
            self.l2l_1 = nn.Sequential(
                Conv2dReLU(channel, channel, 3, 1, 1)
            )

            # stage 2
            self.h2h_2 = nn.Sequential(
                Conv2dReLU(channel, channel, 3, 1, 1)
            )
            self.l2h_2 = nn.Sequential(
                Conv2dReLU(channel, channel, 3, 1, 1)
            )

        def forward(self, h, l):
            # stage 1
            h2h = self.h2h_1(h)
            h2l = self.h2l_1(self.h2l_pool(h))
            l2l = self.l2l_1(l)
            l2h = self.l2h_1(self.l2h_up(l))
            h = h2h + l2h
            l = l2l + h2l

            # stage 2
            h2h = self.h2h_2(h)
            l2h = self.l2h_2(self.l2h_up(l))
            out = h2h + l2h
            return out

    class BDMM(nn.Module):
        def __init__(self, inplanes: list, midplanes=32, upsample=8):
            super(BDMM, self).__init__()
            assert len(inplanes) == 3

            self.rfb1 = RFB_modified(inplanes[0], midplanes)
            self.rfb2 = RFB_modified(inplanes[1], midplanes)
            self.rfb3 = RFB_modified(inplanes[2], midplanes)

            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

            self.agg1 = Agg(midplanes)
            self.agg2 = Agg(midplanes)

            self.conv_out = nn.Sequential(
                Conv2dReLU(midplanes, 1, 3, padding=1),
                nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True),
            )

        def forward(self, x1, x2, x3):
            x1 = self.rfb1(x1)
            x2 = self.rfb2(x2)
            x3 = self.rfb3(x3)

            x2 = self.agg1(x2, x3)
            x1 = self.agg2(x1, x2)

            out = self.conv_out(x1)
            # print("out", out.shape)
            return out

            # BDGM    *********************************************************


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Agg(nn.Module):
    def __init__(self, channel=64):
        super(Agg, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 1
        self.h2h_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.h2l_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2h_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2l_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )

        # stage 2
        self.h2h_2 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2h_2 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )

    def forward(self, h, l):
        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        h = h2h + l2h
        l = l2l + h2l

        # stage 2
        h2h = self.h2h_2(h)
        l2h = self.l2h_2(self.l2h_up(l))
        out = h2h + l2h
        return out


class BDMM(nn.Module):
    def __init__(self, channel1, channel2, channel3, midplanes=32, upsample=8):
        super(BDMM, self).__init__()

        self.rfb1 = RFB_modified(channel1, midplanes)
        self.rfb2 = RFB_modified(channel2, midplanes)
        self.rfb3 = RFB_modified(channel3, midplanes)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.agg1 = Agg(midplanes)
        self.agg2 = Agg(midplanes)

        self.conv_out = nn.Sequential(
            Conv2dReLU(midplanes, 1, 3, padding=1),
            nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True),
        )

    def forward(self, x1, x2, x3):
        # bx1: torch.Size([1, 128, 44, 44])
        # bx2: torch.Size([1, 320, 22, 22])
        # bx3: torch.Size([1, 512, 11, 11])
        # x1: torch.Size([1, 32, 44, 44])
        # x2: torch.Size([1, 32, 22, 22])
        # x3: torch.Size([1, 32, 11, 11])

        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)

        x2 = self.agg1(x2, x3)
        x1 = self.agg2(x1, x2)

        out = self.conv_out(x1)
        # print("out",out.shape)
        return out

        #   AA_kernel


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1), stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1, stride=1, padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3), stride=1, padding=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx


class PolypPVT(nn.Module):
    def __init__(self, channel=576):
        super(PolypPVT, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r'/media/1/hzp/Polyp-PVT/pretrained_pth/pvt_v2_b2.pth'
        #path = r'D:\Code\Polyp-PVT\pretrained_pth\pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # self.Translayer2_0 = BasicConv2d(64, channel, 1)
        # self.Translayer2_1 = BasicConv2d(128, channel, 1)
        # self.Translayer3_1 = BasicConv2d(320, channel, 1)
        # self.Translayer4_1 = BasicConv2d(512, channel, 1)

        # CFP
        self.CFP = CFPModule(64, d=8)

        # BDM
        self.agg = BDMM(channel1=128, channel2=320, channel3=512, upsample=2)

        # Efficient self-attention block
        self.esa1 = ESA_blcok(dim=64)
        self.esa2 = ESA_blcok(dim=128)
        self.esa3 = ESA_blcok(dim=320)
        self.esa4 = ESA_blcok(dim=512)

        self.ca = ChannelAttention(64)
        #self.ca2 = ChannelAttention(512)
        self.sa = SpatialAttention()

        # cascade context module  #64 32 16 16
        self.ccm4 = CCM(512, 64, pool_size=[1, 3, 5], in_channel_list=[], out_channel_list=[320, 128, 64, 64])
        self.ccm3 = CCM(320, 32, pool_size=[2, 6, 10], in_channel_list=[128, 64, 64], out_channel_list=[128, 64, 64],
                        cascade=True)
        self.ccm2 = CCM(128, 16, pool_size=[3, 9, 15], in_channel_list=[64, 64], out_channel_list=[64, 64],
                        cascade=True)
        self.ccm1 = CCM(64, 16, pool_size=[4, 12, 20], in_channel_list=[64], out_channel_list=[64], cascade=True)

        # adaptive selection module
        self.asm1 = ASM(512, 832)
        self.asm2 = ASM(128, 960)

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        # self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)
        self.out_CFM2 = nn.Conv2d(512, 1, 1)
        self.out_CFM3 = nn.Conv2d(2, 1, 1)
        self.out_CFM4 = nn.Conv2d(64, 1, 1)

        # AA_kernel

        self.aa_kernel_1 = AA_kernel(64, 64)

        self.ra1_conv1 = Conv(64, 64, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(64, 64, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(512, 64, 3, 1, padding=1, bn_acti=True)

    def forward(self, x):
        # backbone
        # print("backbone-START")
        pvt = self.backbone(x)
        x1 = pvt[0]  # x1 : torch.Size([1, 64, 88, 88])
        x2 = pvt[1]  # x2 : torch.Size([1, 128, 44, 44])
        x3 = pvt[2]  # x3 : torch.Size([1, 320, 22, 22])
        x4 = pvt[3]  # x4 : torch.Size([1, 512, 11, 11])

        # BDM
        #bdm = self.agg(x2, x3, x4)
        # bdm: torch.Size([1, 1, 352, 352])
        bdm = self.agg(x2, x3, x4)



        # ESA
        e4 = self.esa4(x4)
        e3 = self.esa3(x3)
        e2 = self.esa2(x2)

        # CCM
        cascade_context4 = self.ccm4(e4)
        cascade_context3 = self.ccm3(e3, cascade_context4[1:])
        cascade_context2 = self.ccm2(e2, cascade_context3[1:])

        ccm4 = cascade_context4[0]  # ccm4: torch.Size([1, 256, 22, 22])
        ccm3 = cascade_context3[0]  # ccm3: torch.Size([1, 128, 44, 44])
        ccm2 = cascade_context2[0]  # ccm2: torch.Size([1, 64, 88, 88])

        # upsample
        u4 = nn.functional.interpolate(ccm4, scale_factor=4, mode='bilinear')
        u3 = nn.functional.interpolate(ccm3, scale_factor=2, mode='bilinear')
        u2 = ccm2
        # u4: torch.Size([1, 320, 88, 88])
        # u3: torch.Size([1, 128, 88, 88])
        # u2: torch.Size([1, 64, 88, 88])

        # # ASM
        # #print("ASM-START")
        # comb1 = self.asm1(u3, u4)
        # # torch.Size([1, 832, 88, 88])
        # print("comb1",comb1.shape)
        # comb2 = self.asm2(comb1, u2)

        # concat
        # print("concat-START")
        ca1 = torch.cat([u4, u3, u2], dim=1)
        # print("cat1",ca1.shape)
        # ca1: torch.Size([1, 512, 88, 88])

        # # CIM
        # # print("CIM-START")
        # x1 = self.ca(x1) * x1  # channel attention
        # cim_feature = self.sa(x1) * x1  # spatial attention
        # # x1: torch.Size([1, 64, 88, 88])



        # CFP
        cim_feature = self.CFP(x1)  # 64 - 64


        # catcim_feature = torch.cat([ca1, cim_feature], dim=1)
        # # catcim_feature torch.Size([1, 2368, 88, 88])

        # prediction = self.out_CFM(catcim_feature)
        # # torch.Size([1, 1, 88, 88])
        # prediction = F.interpolate(prediction, scale_factor=4, mode='bilinear')
        # # torch.Size([1, 1, 352, 352])
        # prediction = torch.cat([prediction, bdm], dim=1)
        # # torch.Size([1, 2, 352, 352])
        # prediction = self.out_CFM3(prediction)
        # # torch.Size([1, 1, 352, 352])

        cfeature = self.out_CFM2(ca1)  # torch.Size([1, 1, 88, 88])

        rafeature = -1 * (torch.sigmoid(bdm)) + 1  # torch.Size([1, 1, 88, 88])
        #  cim_feature : torch.Size([1, 64, 88, 88])
        aa_atten_1 = self.aa_kernel_1(cim_feature)  # torch.Size([1, 64, 88, 88])
        aa_atten_1_o = rafeature.expand(-1, 64, -1, -1).mul(aa_atten_1)  # torch.Size([1, 64, 88, 88])

        # aa mode
        ra_1 = self.ra1_conv1(aa_atten_1_o)  # 32 - 32
        ra_1 = self.ra1_conv2(ra_1)  # 32 - 32
        # ra_1 = self.ra1_conv3(ra_1)
        # ra_1: torch.Size([1, 64, 88, 88])
        # ra_1 = self.ra1_conv3(ra_1)  # 32 - 1

        ca1 = self.ra1_conv3(ca1)  # 512 - 64

        ra_b = ra_1 + ca1  # ra_b: torch.Size([1, 64, 88, 88])
        fin = self.ca(ra_b) * ra_b
        #fin = torch.matmul(fin, ca1)
        fin = self.out_CFM4(fin)

        bdm = F.interpolate(bdm, scale_factor=4, mode='bilinear')

        fin =  F.interpolate(fin, scale_factor=4, mode='bilinear')

        right = F.interpolate(cfeature, scale_factor=4, mode='bilinear')



        prediction1 = bdm
        prediction2 = fin
        prediction3 = right

        # prediction1 = self.out_CFM(cfm_feature)
        # prediction2 = self.out_SAM(sam_feature)
        #
        # prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear')
        # prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        return prediction1, prediction2, prediction3


if __name__ == '__main__':
    model = PolypPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    prediction1, prediction2, prediction3 = model(input_tensor)
    print(prediction1.size(), prediction2.size(), prediction3.size())
