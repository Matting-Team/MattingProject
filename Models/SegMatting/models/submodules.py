import torch
import torch.nn as nn
from random import randint
import torch.nn.functional as F


"""
ConvLayer:
#################################
Parameter
 norm - ['BN', 'IN']batch or instance normalization
#################################
Input
 x - feature --> in_channels feature
"""
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


"""
TransposedConvLayer:
#################################
Parameter
 norm - ['BN', 'IN']batch or instance normalization
#################################
Input
 x - feature --> in_channels feature, 2x scale
"""
class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

"""
UpsampleConvLayer:
#################################
Parameter
 norm - ['BN', 'IN']batch or instance normalization
#################################
Input
 x - feature --> in_channels feature, 2 x scale
"""
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

"""
UpConvLayerShuffle:
#################################
Parameter
 norm - ['BN', 'IN']batch or instance normalization
#################################
Input
 x - feature --> in_channels feature, 2x scale
"""
class UpConvLayerShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=5, stride=1, padding=0, norm=None, activation=None):
        super(UpConvLayerShuffle, self).__init__()

        self.in_plane = in_channels
        self.out_plane = out_channels
        self.scale = scale
        self.planes = self.out_plane * scale ** 2

        self.conv0 = nn.Conv2d(self.in_plane, self.planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
        self.icnr(scale=scale)
        self.shuf = nn.PixelShuffle(self.scale)

        self.activation = nn.ReLU()

    def icnr(self, scale=2, init=nn.init.kaiming_normal_):
        ni, nf, h, w = self.conv0.weight.shape
        ni2 = int(ni / (scale ** 2))
        k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale ** 2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        self.conv0.weight.data.copy_(k)

    def forward(self, x):

        x = self.shuf(self.conv0(x))
        x = self.activation(x)
        return x


"""
MixtureNorm:
#################################
Input
 x - feature --> in_channels feature
"""
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

"""
ConvBatch: Conv & BatchNormalization with Mixture Normalization
#################################
Parameter
 norm - ['batch', 'instance', 'mixture'] choose one way
#################################
Input
 x - feature --> in_channels feature
"""
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


"""
ResidualBlock: Generate Residual Block
#################################
Input
 x - feature --> in_channels feature
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, norm="batch"):
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

"""
ChannelWise: Channelwise Attention with 
#################################
Parameter
 ratio - reduce channel ratio (2 --> 1/2, 4 --> 1/4 ...)
#################################
Input
 x - feature --> in_channels feature
Output
 x - feature --> c wise attention feature
"""
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


"""
SegmentNet: extract high level feature with segment net
segment network --> mobilenet v2
#################################
Parameter
 bbackbone --> backbone module(mobilenet v2)
#################################
Input
 x - feature --> in_channels feature
Output
 get 1x, 1/8x, [1/2x, 1/4x] features.
 - feature, feature, list( length=2 ) -
"""
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

        self.masking = CBAM(basic_channel*3+3, 16)
        self.conv_detail_4 = nn.Sequential(
        ConvBatch(basic_channel*3+3, 2*basic_channel,3, stride=1, padding=1),
        ConvBatch(2*basic_channel, 2*basic_channel, 3,stride=1, padding=1),
        ConvBatch(2*basic_channel, basic_channel, 3, stride=1, padding=1)
        )

        self.masking2 = CBAM(basic_channel*2, 16)
        self.conv_detail_2= nn.Sequential(
        ConvBatch(basic_channel*2, basic_channel*2, 3, stride=1, padding=1),
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
        input_4 = torch.cat([hr_4, segment_4, img_4], dim=1)
        mask = self.masking(input_4)
        hr_4 = self.conv_detail_4(input_4)*mask
        hr_2 = F.interpolate(hr_4, scale_factor=2, mode="bilinear", align_corners=False)
        input_2 = torch.cat([hr_2, e_feature2], dim=1)
        mask2 = self.masking2(input_2)
        hr_2 = self.conv_detail_2(input_2)*mask2

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


class ConvBatchBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=1, dilation=1, relu=True, bn=True, bias=False):
        super(ConvBatchBlock, self).__init__()
        self.out_c = out_c
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU(0.02) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, channel, ratio):
        super(ChannelGate, self).__init__()
        self.channel = channel
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel // ratio),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel)
        )

    def forward(self, x):
        avg_c = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_c = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # avg_c = self.mlp(avg_c)
        max_c = self.mlp(max_c)
        channel_x = max_c  # + avg_c
        scale = torch.sigmoid(channel_x).unsqueeze(2).unsqueeze(3).expand_as(x)
        # print(scale.shape)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel = 7
        self.compress = ChannelPool()
        self.spatial = ConvBatchBlock(2, 1, kernel_size=kernel, stride=1, padding=(kernel - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class SpatialGate2(nn.Module):
    def __init__(self, in_c):
        super(SpatialGate2, self).__init__()
        kernel = 7
        self.spatial_e = nn.Sequential(
            ConvBatchBlock(in_c, in_c, kernel_size=(kernel, 1), stride=1, padding=((kernel - 1) // 2, 0), relu=True),
            ConvBatchBlock(in_c, 1, kernel_size=(1, kernel), stride=1, padding=(0, (kernel - 1) // 2), relu=True)
        )
        self.spatial_e2 = nn.Sequential(
            ConvBatchBlock(in_c, in_c, kernel_size=(1, kernel), stride=1, padding=(0, (kernel - 1) // 2), relu=True),
            ConvBatchBlock(in_c, 1, kernel_size=(kernel, 1), stride=1, padding=((kernel - 1) // 2, 0), relu=True)
        )
        self.spatial = ConvBatchBlock(2, 1, kernel_size=1, stride=1, padding=0, relu=False)

    def forward(self, x):
        spatial_1 = self.spatial_e(x)
        spatial_2 = self.spatial_e2(x)
        feature = torch.cat([spatial_1, spatial_2], dim=1)
        x_out = self.spatial(feature)
        scale = torch.sigmoid(x_out)
        return scale


class CBAM(nn.Module):
    def __init__(self, gate_channel, ratio):
        super(CBAM, self).__init__()
        self.cgate = ChannelGate(gate_channel, ratio)
        self.conv1 = ConvBatchBlock(gate_channel, gate_channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.sgate = SpatialGate2(gate_channel)

    def forward(self, x):
        x_out = self.cgate(x)
        x_out = self.conv1(x_out)
        x_out2 = self.sgate(x_out)
        return x_out2
