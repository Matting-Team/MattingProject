import torch
import torch.nn as nn
from Server.backbones.wrapper import MobileNetV2Backbone
from Server.models.submodules import SegmentB, SubB, FusionBranch

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
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.pretrained:
            self.backbone.load_pretrained_ckpt()

    def forward(self, img):
        segment, s8, [e2, e4] = self.segment_branch(img)
        boundary, d2 = self.detail_branch(img, e2, e4, s8)
        alpha = self.fusionNet(img, s8, d2)
        return alpha

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)

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
