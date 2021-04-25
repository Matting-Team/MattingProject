import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import Models.AttentionMatting.Model as model
import Models.AttentionMatting.ImageLoader as loader
import Models.AttentionMatting.Network as net
import Models.AttentionMatting.Loss as loss

from torch.utils.data import DataLoader
from Utils.BasicUtil import print_img

base_dir = 'C:/Users/kjm04/PycharmProjects/AdainNet/AdainMattingNet/Dataset'
trimap_dir = '/trimap'
input_dir = '/input'
gt_dir = '/gt'

loader = loader.ImageLoader(base_dir+input_dir, base_dir+gt_dir)

if __name__ == '__main__':
    print('Process Start')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(loader, batch_size=1, shuffle=True, num_workers=4)

    hp_gradient=0.5
    hp_bce=0.5
    basic_channel = 64

    network = model.MattingNet(basic_channel=64)
    network = network.to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(network.parameters(), lr=0.0002)

    epoch=100
    print(device)

    for i in range(epoch):
        total_loss = 0.
        for idx, data in enumerate(data_loader):
            img, gt = data
            img = img.to(device)
            gt = gt.to(device)
            blend, fg, bg = network(img)
            blend = blend*0.5+0.5
            fg = fg*0.5+0.5
            bg = bg*0.5+0.5
            alpha=blend*fg + (1-blend)*(1-bg)

            p_loss = loss.probability(fg, gt)
            g_loss = loss.gradientLoss(fg, gt)
            bce_loss = criterion(fg, gt)
            fusion_loss = loss.fusionLoss(alpha, gt)
            current_loss = p_loss+g_loss+bce_loss+fusion_loss
            total_loss+=current_loss

            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            if idx%300==299:
                print("|__idx_is__%d__|__loss is %f__|"%(idx, total_loss/300))
                total_loss=0.
                print_img(alpha)
                print_img(gt)
                print_img(fg)
                print_img(bg)
                print_img(blend)


