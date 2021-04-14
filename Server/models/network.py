import torch
import torch.nn as nn
from backbones.wrapper import MobileNetV2Backbone
from models.submodules import SegmentB, SubB, FusionBranch

'''
# CamNet : MobileNetv2를 이용하여 
# ##########
# parameter
# input_c --> input channel of CANet
# base_c --> Basic Channel of Network(32)
# pretrained --> is pretrained or not
# ##########
# output --> feature 0 - 1
# ##########
'''
class CamNet(nn.Module):
    def __init__(self, input_c=3, base_c=32, pretrained=True):
        super(CamNet, self).__init__()

        self.in_c = input_c
        self.base_c = base_c
        self.pretrained = pretrained

        self.backbone = MobileNetV2Backbone(input_c)
        self.segment_branch = SegmentB(self.backbone)
        self.detail_branch = SubB(self.base_c, self.backbone.channels)
        self.fusionNet = FusionBranch(self.base_c, self.backbone.channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.initialize(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self.initialize_norm(m)

        if self.pretrained:
            self.backbone.load_backbone()

    def forward(self, img):
        segment, low_level8, [low_level2, low_level4] = self.segment_branch(img)

        boundary, high_level2 = self.detail_branch(img, low_level2, low_level4, low_level8)
        alpha = self.fusionNet(img, low_level8, high_level2)
        return alpha

    def initialize(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def initialize_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)


'''
# Discriminator: Discriminator는 RGB 입력이미지 + Alpha map을 입력으로 받아 진위여부를 판별한다.
# ##########
# parameter
# input_c --> input channel of Discriminator
# ##########
# output --> feature 0 - 1
# ##########
'''
class Discriminator(nn.Module):
    def __init__(self, input_c):
        super(Discriminator, self).__init__()
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
