import os
from functools import reduce

import torch
import torch.nn as nn
from backbones.mobilenetv2 import MobileNetV2

#from .mobilenetv2 import MobileNetV2
#from mobilenetv2 import MobileNetV2
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
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch 
        #ckpt_path = './pretrained/mobilenetv2.ckpt'
        ckpt_path = 'pretrained/mobilenetv2.ckpt'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()
        
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)


'''
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

Backbone = MobileNetV2Backbone(3)
Backbone.load_pretrained_ckpt()
input = Image.open("india.jpg")
input = input.resize((512, 512))
input = transform(input).unsqueeze(0)
print(input.shape)


result = Backbone(input)

def tensor_plot(tensor):
    tensor = tensor.detach().cpu()
    tensor = torch.mean(tensor, dim=1)
    img = torchvision.utils.make_grid(tensor)
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()
tensor_plot(result[-1])
print(result[3].shape) # 뭔가를 얻는게 아니라, feature를 뽑는거라서 이런듯?
#생각만큼 segmentation의 결과가 좋지는 않다?
'''