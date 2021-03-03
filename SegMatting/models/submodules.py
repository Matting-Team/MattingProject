import torch
import torch.nn as nn
from random import randint
import torch.nn.functional as F

# conv_batch, residual etc...
class MixtureNorm(nn.Module):
    def __init__(self, channel):
        super(MixtureNorm, self).__init__()
        self.batch_channel = channel//2
        self.instance_channel = channel - self.batch_channel

        self.batch_norm = nn.BatchNorm2d(self.batch_channel, affine=True)
        self.instance_norm = nn.InstanceNorm2d(self.instance_channel, affine=False)

    def forward(self, x):
        b_c = self.batch_norm(x[:,:self.batch_channel, ...].contiguous())
        i_c = self.instance_norm(x[:, self.batch_channel:, ...].contiguous())
        return torch.cat([b_c, i_c], dim=1)

#ConvBatch
class ConvBatch(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False, norm="mixture", relu=False):
        super(ConvBatch, self).__init__()
        sequence = nn.ModuleList([
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        ])
        if norm == "batch":
            sequence.append(nn.BatchNorm2d(out_channel))
        elif norm == "mixture":
            sequence.append(MixtureNorm(out_channel))
        elif norm =="instance":
            sequence.append(nn.InstanceNorm2d(out_channel))
        if relu:
            sequence.append(nn.ReLU(inplace=True))

        self.sequence = nn.Sequential(*sequence)
    def forward(self, x):
        return self.sequence(x)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBatch(in_channels, out_channels, norm="batch", relu=True)
        self.conv2 = ConvBatch(out_channels, out_channels, norm="batch")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

#motivated by cbam
class ChannelWise(nn.Module):
    def __init__(self, in_channel, out_channel, ratio):
        super(ChannelWise, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(in_channel, in_channel//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//ratio, out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        B, C, H, W = input.shape
        feature = self.pool(input).view(B, C)
        feature = self.linear(feature).view(B, C, 1, 1)
        return input * feature.expand_as(input)


# Predict Semantic Segmentation with Backbone
# Pretrained MobileNetv2 Used...
class SegmentNet(nn.Module):
    def __init__(self, backbone):
        super(SegmentNet, self).__init__()
        self.backbone = backbone

        channels = backbone.channels

        self.c_wise = ChannelWise(channels[4], channels[4], ratio=4)
        self.conv_16 = ConvBatch(channels[4], channels[3], 5, stride=1, padding=2, norm="mixture")
        self.conv_8 = ConvBatch(channels[3], channels[2], 5, stride=1, padding=2, norm="mixture")
        self.conv_last = nn.Conv2d(channels[2], 1, 3, stride=2, padding=1)

    def forward(self, input):
        feature = self.backbone.forward(input)
        feature_2, feature_4, feature_32 = feature[0], feature[1], feature[4]

        feature_32 = self.c_wise(feature_32)
        feature_16 = F.interpolate(feature_32, scale_factor=2, mode="bilinear", align_corners=False)
        feature_16 = self.conv_16(feature_16)
        feature_8 = F.interpolate(feature_16, scale_factor=2, mode="bilinear", align_corners=False)
        feature_8 = self.conv_8(feature_8)

        feature_1 = self.conv_last(feature_8)
        segment = torch.sigmoid(feature_1)

        return segment, feature_8, [feature_2, feature_4]

# Predict Detail Edge
# Spatial Attention Used
class DetailNet(nn.Module):
    def __init__(self, basic_channel, features):
        super(DetailNet, self).__init__()
        self.first_conv = ConvBatch(features[0], basic_channel, 1, stride=1, padding=0, norm="mixture")
        self.conv_2 = ConvBatch(basic_channel+3, basic_channel, 3, stride=2, padding=1, norm="mixture")

        self.first_conv_2 = ConvBatch(features[1], basic_channel, 1, stride=1, padding=0, norm="mixture")
        self.conv_4 = ConvBatch(2 * basic_channel, 2 * basic_channel, 3, stride=1, padding=1, norm="mixture")

        self.conv_block1 = nn.Sequential(
            ConvBatch(2*basic_channel, 2*basic_channel, 3, stride=1, padding=1, norm="mixture"),
            ConvBatch(2*basic_channel, basic_channel, 3, stride=1, padding=1, norm="mixture"),
            ConvBatch(basic_channel, basic_channel, 3, stride=1, padding=1, norm="mixture"),
            ConvBatch(basic_channel, basic_channel, 3, stride=1, padding=1, norm="mixture")
        )

        self.last_conv = nn.Sequential(
            ConvBatch(basic_channel + 3, basic_channel, 3, stride=1, padding=1),
            nn.Conv2d(basic_channel, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, img, feature_2, feature_4, lr_8):
        img_half = F.interpolate(img, scale_factor=1/2, mode="bilinear", align_corners=False)
        img_quad = F.interpolate(img, scale_factor=1/4, mode="bilinear", align_corners=False)

        feature_2 = self.first_conv(feature_2)
        return input


class ChannelPool(nn.Module):
    def forward(self, input, mode="max"):
        return torch.max(input, 1)[0].unsqueeze(1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel = 7, stride=1 ,padding=3):
        super(SpatialAttention, self).__init__()
        self.pooling = ChannelPool()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel, stride= stride, padding=padding)
    def forward(self, input):
        x = self.pooling(input)
        x = torch.sigmoid(self.conv(x))
        return input*x.expand_as(input)

class DetailNet(nn.Module):
    def __init__(self, basic_channel, channels):
        super(DetailNet, self).__init__()
        self.feature_extract_2 = ConvBatch(channels[0], basic_channel, 1, stride=1, padding=0)
        self.conv_2 = ConvBatch(basic_channel+3, basic_channel, 3, stride=2, padding=1)#1/4 size로

        self.feature_extract_4 = ConvBatch(channels[1], basic_channel, 1, stride=1, padding=0)
        self.conv_4 = ConvBatch(basic_channel*2, basic_channel*2, 3, stride=1, padding=1)

        self.conv_detail_4 = nn.Sequential(
        ConvBatch(basic_channel*3+3, 2*basic_channel,3, stride=1, padding=1),
        SpatialAttention(),
        ConvBatch(2*basic_channel, 2*basic_channel, 3,stride=1, padding=1),
        ConvBatch(2*basic_channel, basic_channel, 3, stride=1, padding=1)
        )

        self.conv_detail_2= nn.Sequential(
        ConvBatch(basic_channel*2, basic_channel*2, 3, stride=1, padding=1),
        SpatialAttention(),
        ConvBatch(basic_channel*2, basic_channel, 3, stride=1, padding=1),
        ConvBatch(basic_channel, basic_channel, 3, stride=1, padding=1),
        ConvBatch(basic_channel, basic_channel, 3, stride=1, padding=1)
        )

        self.last_conv = nn.Sequential(
        ConvBatch(basic_channel+3, basic_channel, 3, stride=1, padding=1),
        nn.Conv2d(basic_channel, 1, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, img, feature_2, feature_4, segment_8):
        img_2 = F.interpolate(img, scale_factor=1/2, mode="bilinear", align_corners=False)
        img_4 = F.interpolate(img, scale_factor=1/4, mode="bilinear", align_corners=False)
        segment_4 = F.interpolate(segment_8, scale_factor=2, mode="bilinear", align_corners=False)
        e_feature2 = self.feature_extract_2(feature_2)
        hr_4 = self.conv_2(torch.cat([img_2, e_feature2], dim=1))

        e_feature4 = self.feature_extract_4(feature_4)
        hr_4 = self.conv_4(torch.cat([hr_4, e_feature4], dim=1))
        hr_4 = self.conv_detail_4(torch.cat([hr_4, segment_4, img_4], dim=1))
        hr_2 = F.interpolate(hr_4, scale_factor=2, mode="bilinear", align_corners=False)
        hr_2 = self.conv_detail_2(torch.cat([hr_2, e_feature2], dim=1))

        hr = F.interpolate(hr_2, scale_factor=2, mode="bilinear", align_corners=False)
        hr = self.last_conv(torch.cat([hr, img], dim=1))
        result = torch.sigmoid(hr)

        return result, hr_2


class FusionNet(nn.Module):
    def __init__(self, basic_channel, channels):
        super(FusionNet, self).__init__()
        self.conv_segment_4 = ConvBatch(channels[2], basic_channel, 5, stride=1, padding=2)

        self.conv_feature_2 = ConvBatch(2*basic_channel, basic_channel, 3, stride=1,padding=1)
        self.conv_feature = nn.Sequential(
            ConvBatch(basic_channel+3, basic_channel//2, 3, stride=1, padding=1),
            nn.Conv2d(basic_channel//2, 1, 1, stride=1, padding=0)
        )
    def forward(self, img, segment_8, detail_2):
        segment_4 = F.interpolate(segment_8, scale_factor=2, mode="bilinear", align_corners=False)
        segment_4 = self.conv_segment_4(segment_4)
        segment_2 = F.interpolate(segment_4, scale_factor=2, mode="bilinear", align_corners=False)

        feature_2 = self.conv_feature_2(torch.cat([segment_2, detail_2], dim=1))
        feature = F.interpolate(feature_2, scale_factor=2, mode="bilinear", align_corners=False)
        feature = self.conv_feature(torch.cat([feature, img], dim=1))
        alpha_map = torch.sigmoid(feature)
        return alpha_map

'''
cwise = ChannelPool()
swise = SpatialAttention()
tensor= torch.randn(1, 3, 256, 256)
enc_2x = torch.randn(1, 16, 128, 128)
enc_4x = torch.randn(1, 24, 64, 64)
hr_8 = torch.randn(1, 32, 32, 32)
print(swise(tensor).shape)
channels = [16, 24, 32, 96, 1280]
detail = DetailNet(32, channels)
result = detail(tensor, enc_2x, enc_4x, hr_8)
print(result[0].shape)
print(result[1].shape)
'''