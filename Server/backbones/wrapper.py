import os
from functools import reduce

import torch
import torch.nn as nn
from backbones.mobilenetv2 import MobileNetV2

class BaseBackbone(nn.Module):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone 
    """

    def __init__(self, in_channels):
        super(MobileNetV2Backbone, self).__init__(in_channels)

        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.channels = [16, 24, 32, 96, 1280]
    def forward(self, x):
        x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        enc2x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        enc4x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        enc8x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        enc16x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained model https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = 'backbones/pretrained/mobilenetv2_human_seg.ckpt'
        if not os.path.exists(ckpt_path):
            print('there is no Pretrained File!')
            exit()
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
