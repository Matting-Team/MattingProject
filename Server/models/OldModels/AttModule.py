import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBatchBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=1, dilation=1,  relu=True, bn=True, bias=False):
        super(ConvBatchBlock, self).__init__()
        self.out_c = out_c
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride,padding=padding, dilation=dilation, bias=bias)
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
            nn.Linear(channel, channel//ratio),
            nn.ReLU(),
            nn.Linear(channel//ratio, channel)
        )
    def forward(self, x):
        avg_c = F.avg_pool2d(x, (x.size(2), x.size(3)), stride= (x.size(2), x.size(3)))
        max_c = F.max_pool2d(x, (x.size(2), x.size(3)), stride= (x.size(2), x.size(3)))
        #avg_c = self.mlp(avg_c)
        max_c = self.mlp(max_c)
        channel_x = max_c# + avg_c
        scale = torch.sigmoid(channel_x).unsqueeze(2).unsqueeze(3).expand_as(x)
        #print(scale.shape)
        return x*scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel = 7
        self.compress = ChannelPool()
        self.spatial = ConvBatchBlock(2, 1, kernel_size=kernel, stride=1, padding = (kernel-1)//2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x*scale

class SpatialGate2(nn.Module):
    def __init__(self, in_c):
        super(SpatialGate2, self).__init__()
        kernel = 7
        self.spatial_e = nn.Sequential(
            ConvBatchBlock(in_c, in_c, kernel_size=(kernel,1), stride=1, padding = ((kernel-1)//2, 0), relu=True),
            ConvBatchBlock(in_c, 1, kernel_size=(1, kernel), stride=1, padding=(0, (kernel - 1) // 2), relu=True)
        )
        self.spatial_e2 = nn.Sequential(
            ConvBatchBlock(in_c, in_c, kernel_size=(1, kernel), stride=1, padding=(0, (kernel-1)//2), relu=True),
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

class CBAM2(nn.Module):
    def __init__(self, gate_channel, ratio):
        super(CBAM2, self).__init__()
        self.cgate = ChannelGate(gate_channel, ratio)
        self.conv1 = ConvBatchBlock(gate_channel, gate_channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.sgate = SpatialGate2(gate_channel)
    def forward(self, x):
        x_out = self.cgate(x)
        x_out = self.conv1(x_out)
        x_out2 = self.sgate(x_out)
        return x_out, x_out2

class CBAttention(nn.Module):
    def __init__(self, in_c):
        super(CBAttention, self).__init__()
        self.w_wise_conv = ConvBatchBlock(in_c,in_c, (1, 7), stride=1, padding=0, relu=True)
        self.h_wise_conv = ConvBatchBlock(in_c, in_c, (7, 1), stride=1, padding=0, relu=True)
    def forward(self, x):
        w_feature = self.w_wise_conv(x)
        w_feature = torch.mean(w_feature, dim=3).unsqueeze(3)
        h_feature = self.h_wise_conv(x)
        h_feature = torch.mean(h_feature, dim=2).unsqueeze(2)

        return torch.sigmoid(w_feature*h_feature)


class DPA(nn.Module):
    def __init__(self,in_c, out_c, upsample=True):
        super(DPA, self).__init__()
        self.upsample=upsample
        self.c_wise = ConvBatchBlock(in_c, out_c, 1, stride=1, padding=0, relu=True)
        self.pooling = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            ConvBatchBlock(in_c, out_c, 1, relu=True)
        )
        self.aspp_6 = ConvBatchBlock(in_c, out_c, 3, stride=1, padding=6, dilation=6, relu=True)
        self.aspp_12 = ConvBatchBlock(in_c, out_c, 3, stride=1, padding=12, dilation=12, relu=True)
        self.aspp_18 = ConvBatchBlock(in_c, out_c, 3, stride=1, padding=18, dilation=18, relu=True)
        self.totalize = nn.Sequential(
            ConvBatchBlock(out_c * 5, out_c, 3, stride=1, padding=1),
            nn.LeakyReLU(0.02)
        )
        self.cbam = CBAM2(out_c, 16)
        self.simpleUpSample = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        x0 = self.c_wise(x)
        x1 = self.pooling(x)
        x2 = self.aspp_6(x)
        x3 = self.aspp_12(x)
        x4 = self.aspp_18(x)
        x1 = F.interpolate(x1, size=x0.shape[2:], mode='nearest')
        feature = torch.cat([x0, x1, x2, x3, x4], dim=1)

        result = self.totalize(feature)
        if  self.upsample:
            result = self.simpleUpSample(result)

        channel, spatial = self.cbam(result)
        return spatial,channel
