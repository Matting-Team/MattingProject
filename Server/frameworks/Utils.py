import torch.optim as optim

import torch.nn as nn
import numpy as np
import math
import torch
import scipy
import torchvision
import os
import matplotlib.pyplot as plt


def print_progress(epoch, idx, cur_epoch, cur_idx):
    print("___current epoch is %d__________ %d/%d"%(cur_epoch, cur_epoch, epoch))
    print("___current index is %d__________ %d/%d"%(cur_idx, cur_idx, idx))
    print("■■■■■■■■■■■■■■■■■■■■■■■■")


def optimizer_setting(optimizer_config, param, lr):
    optimizer = None
    if optimizer_config == "Adam":
        optimizer = optim.Adam(param, lr)# 0.2, 0.999?
    return optimizer

def write_text(path, text):
    f = open(path+'/log.txt', 'a')
    print(text, file= f)
    f.close()

def print_tensor(tensor):
    tensor = tensor.detach().cpu().squeeze(0)
    np_img = torchvision.utils.make_grid(tensor).numpy()
    img = np.transpose(np_img, (1, 2, 0))
    plt.imshow(img)
    plt.show()

def save_tensor(tensor, path,name="notitle.png"):
    tensor = tensor.detach().cpu().squeeze(0)
    np_img = torchvision.utils.make_grid(tensor).numpy()
    img = np.transpose(np_img, (1, 2, 0))
    plt.imsave(path+"/"+name, img)
    plt.show()

def valid_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)

class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),
            nn.Conv2d(channels, channels, self.kernel_size,
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))