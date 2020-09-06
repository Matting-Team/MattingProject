import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# custom Class
import Network as net
# net has ConvBatch//ConvTransBatch//UNet etc

class Encoder(nn.Module):
    def __init__(self, basic_channel=32, depth=3):
        super(Encoder, self).__init__()
        self.m_list = nn.ModuleList([])
        c = basic_channel
        for idx in range(depth):
            m = nn.Sequential(
                net.ConvBatchBlock(c, c * 2, 4, stride=2, padding=1),
                net.ResBlock(c*2),
                nn.ReLU(inplace=True)#Activation of Res Block
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

dummy = torch.randn((1, 32, 128, 128))
encoder = Encoder()
decoder = Decoder()
result, list = encoder(dummy)
print(result.shape, 'result shape')
print(list[0].shape, 'first shape')
result = decoder(result, list)
print(result.shape, 'final result')