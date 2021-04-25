import ImageLoader as loader
import SegmentNet as segnet
import Model as model
import Loss as loss
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Utils.BasicUtil import print_img, print_cv2, print_tensor, save_tensor_cv, save_tensor_plt




lr = 0.0002
epoch = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset Directory:
base_dir = 'C:/Users/kjm04/PycharmProjects/AdainNet/AdainMattingNet/Dataset'
trimap_dir = '/trimap'
input_dir = '/input'
gt_dir = '/gt'

model_path = 'model/'
loader = loader.ImageLoader(base_dir+input_dir, base_dir+gt_dir)


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

    model = model.features.eval()
    data_loader = DataLoader(loader, batch_size=1, shuffle=True, num_workers=4)
    #net define
    encoder = segnet.Encoder(model)
    decoder = segnet.DoubleDecoder()
    encoder.to(device)
    decoder.to(device)

    for param in encoder.parameters():
        param.requires_grad = False
    criterion = nn.BCELoss()

    optimizer = optim.Adam(decoder.parameters(), lr=0.0002)

    for i in range(epoch):
        total_loss = 0.
        for idx, data in enumerate(data_loader):
            img, gt = data
            img = img.to(device)
            gt = gt.to(device)
            feature, skip_connection = encoder(img)
            fg, bg = decoder(feature, skip_connection)

            fg = fg*0.5+0.5
            bg = bg*0.5+0.5
            predict_fg = 1-bg

            p_loss = loss.probability(fg, gt)+loss.probability(predict_fg, gt)
            g_loss = loss.gradientLoss(fg, gt) + loss.gradientLoss(predict_fg,gt)
            bce_loss = criterion(fg, gt) + criterion(predict_fg, gt)
            current_loss = p_loss+g_loss+bce_loss
            total_loss = total_loss + current_loss

            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            if idx%300==299:
                print("|__idx_is__%d__|__loss is %f __|"%(idx+1, current_loss))
                total_loss=0.
                cat_img = torch.cat([gt, fg, bg], dim=2)
                save_tensor_cv(cat_img, '%depoch_idx%d.jpg' % (i, idx))
        dict = {}
        dict['model'] = decoder.state_dict()
        dict['optim'] = optimizer.state_dict()
        torch.save(dict,'%smodel%d.pt'%(model_path,i))

