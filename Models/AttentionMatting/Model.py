import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import Models.AttentionMatting.Network as net

class Encoder(nn.Module):
    def __init__(self, basic_channel=32, depth=3):
        super(Encoder, self).__init__()
        self.m_list = nn.ModuleList([])
        c = basic_channel
        for idx in range(depth):
            m = nn.Sequential(
                net.ConvBatchBlock(c, c * 2, 4, stride=2, padding=1),
                net.ResBlock(c*2),
                nn.ReLU(inplace=True)
            )
            self.m_list.append(m)
            c = c*2
    def forward(self, input):
        x = input
        result = []
        result.append(x)
        for m in self.m_list:
            x = m(x)
            result.append(x)
        return x, result

class Decoder(nn.Module):
    def __init__(self, basic_channel=32, depth=3):
        super(Decoder, self).__init__()
        self.m_list = nn.ModuleList([])
        c= basic_channel*(2**depth)

        for idx in range(depth):
            m = nn.Sequential(
                net.ConvTBatchBlock(c, c//2, 4, stride=2, padding=1),
                net.ResBlock(c//2),
                nn.ReLU(inplace=True)
            )
            self.m_list.append(m)
            c=c//2

    def forward(self, input, skip_con=[]):
        x = input
        idx = 0
        sk_len = len(skip_con)
        for m in self.m_list:
            x = m(x)
            idx+=1
            if sk_len !=0:
                x = x + skip_con[sk_len-(idx+1)]
        return x

class FusionNet(nn.Module):
    def __init__(self, channel):
        super(FusionNet, self).__init__()
        self.net = nn.Sequential(
            net.ConvBatchBlock(channel, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            net.ConvBatchBlock(256, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            net.ConvBatchBlock(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            net.ConvBatchBlock(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            net.ConvBatchBlock(64, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self, input):
        return self.net(input)

class MattingNet(nn.Module):
    def __init__(self, basic_channel):
        super(MattingNet, self).__init__()
        self.FeatureExtractor = nn.Sequential(
            net.ConvBatchBlock(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            net.ConvBatchBlock(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.Encoder = nn.ModuleList([])
        extractor = nn.Sequential(
            net.ConvBatchBlock(3, basic_channel, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.Encoder.append(extractor)
        encoder = Encoder(basic_channel=basic_channel)
        self.Encoder.append(encoder)
        self.Decoder1 = Decoder(basic_channel=basic_channel)
        self.lastConv1 = nn.Sequential(
            nn.Conv2d(basic_channel, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
        self.Decoder2 = Decoder(basic_channel=basic_channel)
        self.lastConv2 = nn.Sequential(
            nn.Conv2d(basic_channel, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
        self.FusionNet = FusionNet(128+(basic_channel*2))
    def forward(self, input):
        feature = self.FeatureExtractor(input)#피쳐 뽑아냄
        first_layer = self.Encoder[0](input)
        e_feature, skip_connection = self.Encoder[1](first_layer)
        fg_feature = self.Decoder1(e_feature, skip_connection)
        fg = self.lastConv1(fg_feature)
        bg_feature = self.Decoder2(e_feature, skip_connection)
        bg = self.lastConv2(bg_feature)

        cat_feature = torch.cat([feature, fg_feature, bg_feature],dim=1)
        result = self.FusionNet(cat_feature)
        return result, fg, bg