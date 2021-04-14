import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNorm(nn.Module):
    def __init__(self, in_channels):
        super(ConvNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)

class ConvBatchRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(ConvBatchRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(ConvNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

'''
# CWise: Channel에 Weight를 부과하는 Channelwise Attention --> CBAM Base
# ##########
# ##########
# parameter
# input_c --> input channel of Discriminator
# ##########
# output --> feature 0 - 1
# ##########
'''
class CWise(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=1):
        super(CWise, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // ratio), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)

'''
# CWise: Channel에 Weight를 부과하는 Channelwise Attention --> CBAM Base
# ##########
# ##########
# parameter
# input_c --> input channel of Discriminator
# ##########
# output --> feature 0 - 1
# ##########
'''
class SegmentB(nn.Module):
    def __init__(self, backbone):
        super(SegmentB, self).__init__()
        channels = backbone.channels
        self.backbone = backbone
        self.c_wise = CWise(channels[4], channels[4], ratio=4)
        self.conv_s16 = ConvBatchRelu(channels[4], channels[3], 5, stride=1, padding=2)
        self.conv_s8 = ConvBatchRelu(channels[3], channels[2], 5, stride=1, padding=2)
        self.conv_lr = ConvBatchRelu(channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False,
                                     with_relu=False)

    def forward(self, img):
        e_feature = self.backbone.forward(img)
        low_level2, low_level4, low_level32 = e_feature[0], e_feature[1], e_feature[4]

        low_level32 = self.c_wise(low_level32)
        segment_lower = F.interpolate(low_level32, scale_factor=2, mode='bilinear', align_corners=False)
        segment_lower = self.conv_s16(segment_lower)
        segment_upper = F.interpolate(segment_lower, scale_factor=2, mode='bilinear', align_corners=False)
        segment_upper = self.conv_s8(segment_upper)

        lr = self.conv_lr(segment_upper)
        segment = torch.sigmoid(lr)

        return segment, segment_upper, [low_level2, low_level4]


'''
# CWise: Channel에 Weight를 부과하는 Channelwise Attention --> CBAM Base
# ##########
# ##########
# parameter
# input_c --> input channel of Discriminator
# ##########
# output --> feature 0 - 1
# ##########
'''
class SubB(nn.Module):

    def __init__(self, hr_channels, enc_channels):
        super(SubB, self).__init__()

        self.to_d_e2 = ConvBatchRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_e2 = ConvBatchRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.to_d_e4 = ConvBatchRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_e4 = ConvBatchRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_d4 = nn.Sequential(
            ConvBatchRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            ConvBatchRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            ConvBatchRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_d2 = nn.Sequential(
            ConvBatchRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            ConvBatchRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            ConvBatchRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            ConvBatchRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_d = nn.Sequential(
            ConvBatchRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            ConvBatchRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, e2, e4, segment8):
        img2 = F.interpolate(img, scale_factor=1 / 2, mode='bilinear', align_corners=False)
        img4 = F.interpolate(img, scale_factor=1 / 4, mode='bilinear', align_corners=False)

        e2 = self.to_d_e2(e2)
        hr4x = self.conv_e2(torch.cat((img2, e2), dim=1))

        e4 = self.to_d_e4(e4)
        hr4x = self.conv_e4(torch.cat((hr4x, e4), dim=1))

        lr4x = F.interpolate(segment8, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv_d4(torch.cat((hr4x, lr4x, img4), dim=1))

        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_d2(torch.cat((hr2x, e2), dim=1))

        hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)
        hr = self.conv_d(torch.cat((hr, img), dim=1))
        pred_detail = torch.sigmoid(hr)

        return pred_detail, hr2x


'''
# Fusion Branch
# ##########
# ##########
# parameter
# input_c --> input channel of Discriminator
# ##########
# output --> feature 0 - 1
# ##########
'''
class FusionBranch(nn.Module):
    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_s4 = ConvBatchRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)

        self.conv_f2 = ConvBatchRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            ConvBatchRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            ConvBatchRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, s8, d2):
        s4 = F.interpolate(s8, scale_factor=2, mode='bilinear', align_corners=False)
        s4 = self.conv_s4(s4)
        s2 = F.interpolate(s4, scale_factor=2, mode='bilinear', align_corners=False)

        f2 = self.conv_f2(torch.cat((s2, d2), dim=1))
        f = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        alpha = torch.sigmoid(f)

        return alpha