import torch
import torch.nn as nn
import torch.nn.functional as F

from Network.SegMatting.models.submodules import *
from Network.SegMatting.backbones import BACKBONE_LIST

"""
TotalNet: Low Level & High Level data.

"""
class TotalNet(nn.Module):
    def __init__(self, input_channel=3, basic_channel=32, backbone='mobilenetv2', pretrained=True):
        super(TotalNet, self).__init__()

        self.input_channel = input_channel
        self.basic_channel = basic_channel
        self.backbone_name = backbone
        self.pretrained = pretrained

        self.backbone = BACKBONE_LIST[self.backbone_name](input_channel)
        self.segmentNet = SegmentNet(self.backbone)
        self.detailNet = DetailNet(self.basic_channel, self.backbone.channels)
        self.fusion = FusionNet(self.basic_channel, self.backbone.channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.pretrained:
            self.backbone.load_pretrained_ckpt()

    def forward(self, img):
        segment, segment_8, [feature_2, feature_4] = self.segmentNet(img)
        detail, detail_feature_2 = self.detailNet(img, feature_2, feature_4, segment_8)
        alpha = self.fusion(img, segment_8, detail_feature_2)
        return segment, detail, alpha

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)

    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

class Discriminator(nn.Module):
    def __init__(self, input_c):
        super(Discriminator, self).__init__()
        #input shape is 512 512
        channels = [64, 128, 256, 256, 512, 1]
        layer = nn.ModuleList([])
        current_c = input_c
        for c in channels:
            layer.append(
                nn.Sequential(
                    nn.Conv2d(current_c, c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(c),
                    nn.LeakyReLU(0.01)
                )
            )
            current_c = c
        self.model = nn.Sequential(*layer)
    def forward(self, x):
        return torch.sigmoid(self.model(x))

