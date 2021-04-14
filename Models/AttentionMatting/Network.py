import torch
import torchvision
import torch.nn as nn
import torch.functional as F

class ConvBatchBlock(nn.Module):
    def __init__(self, in_c, out_c, k_s=3, stride = 1, padding =1, dilation=1):
        super(ConvBatchBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_s,stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_c)
        )
    def forward(self, input):
        return self.block(input)

class ConvTBatchBlock(nn.Module):
    def __init__(self, in_c, out_c, k_s=4, stride=2, padding=1):
        super(ConvTBatchBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k_s, stride=stride, padding = padding),
            nn.BatchNorm2d(out_c)
        )
    def forward(self,input):
        return self.block(input)

class ResBlock(nn.Module):
    def __init__(self, channel, kernel_size=3 ,res_num=3, dilation=1, activation = nn.ReLU(inplace=True)):
        super(ResBlock, self).__init__()
        self.list = nn.ModuleList([])
        stride = 1
        padding = (kernel_size-stride)//2+(dilation-1)
        for i in range(res_num):
            seq = nn.Sequential(
                ConvBatchBlock(channel, channel, kernel_size, stride, padding=padding, dilation=dilation),
                activation
            )
            self.list.append(seq)

    def forward(self, input):
        x = input
        for seq in self.list:
            x = seq(x)+x
        return x

class DenseBlock(nn.Module):
    def __init__(self, c, k_s=3, length=3):
        super(DenseBlock, self).__init__()
        self.m_list = nn.ModuleList([])
        padding = k_s//2
        for i in range(length):
            m = nn.Sequential(
                nn.Conv2d(c, c, k_s, stride=1, padding=padding),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            )
            self.m_list.append(m)
    def forward(self, input):
        x = input
        result_list = []
        for m in self.m_list:
            
            x = m(x)


class Unet(nn.Module):
    def __init__(self, in_c=64, depth=3, res_num=5, activation = nn.ReLU(inplace=True)):
        super(Unet, self).__init__()
        self.down_list = nn.ModuleList([])
        self.up_list = nn.ModuleList([])
        self.activation = activation
        current_c = in_c
        for d in range(depth):
            seq = nn.Sequential(
                ConvBatchBlock(current_c, current_c*2, 4, stride=2, padding=1),
                activation,
                ConvBatchBlock(current_c*2, current_c*2,3, stride=1, padding=1),
                activation
            )
            self.down_list.append(seq)
            current_c=current_c*2
        self.resBlock = ResBlock(current_c, res_num=res_num)
        for d in range(depth):
            seq = nn.Sequential(
                ConvTBatchBlock(current_c, current_c//2),
                activation,
                ConvBatchBlock(current_c//2, current_c//2, 3, stride=1, padding=1),
            )
            current_c=current_c//2
            self.up_list.append(seq)
        self.lastConv = nn.Sequential(
            ConvBatchBlock(current_c, current_c, 3, stride=1, padding=1),
            activation,
            ConvBatchBlock(current_c, current_c, 1, stride=1, padding=1),
            activation
        )

    def forward(self, input):
        result_list = []
        x = input
        for seq in self.down_list:
            result_list.append(x)
            x = seq(x)
        x = self.resBlock(x)
        idx = 0
        for seq in self.up_list:
            x=seq(x)
            x = x+result_list[len(result_list)-(1+idx)]
            x = self.activation(x)
            idx+=1
        self.lastConv(x)
        return x+input

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.firstConv = nn.Sequential(
            ConvBatchBlock(3, 64),
            nn.ReLU()
        )
        self.Unet = Unet()
        self.lastConv = nn.Sequential(
            ConvBatchBlock(64, 3),
            nn.Tanh()
        )
    def forward(self, input):
        x = input
        x = self.firstConv(x)
        x = self.Unet(x)
        x = self.lastConv(x)
        return x