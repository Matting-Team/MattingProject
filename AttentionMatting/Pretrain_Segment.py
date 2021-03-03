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

def print_image(img, title='title'):
    img = img.detach().cpu()
    img = torchvision.utils.make_grid(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.transpose(img, (1, 2, 0))
    img = img*std + mean
    plt.imshow(img)
    plt.savefig(title)
    plt.clf()

def print_cv2(img):
    cv2.imshow("read", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_img(img, title='title'):
    path ='result/'+title
    img = img.detach().cpu()
    img = torchvision.utils.make_grid(img)
    img = np.transpose(img, (1,2,0))
    #img = img*0.5 + 0.5
    print(img.shape)
    cv_img = np.array(img)
    #아래 부분은 필요 없다. 어차피 CV2로 읽었기 때문.
    #cv_img = cv_img[:, :, ::-1].copy()
    cv_img = cv2.convertScaleAbs(cv_img, alpha=(255.0))
    cv2.imwrite(path, cv_img)


lr = 0.0002
epoch = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
base_dir = 'C:/Users/kjm04/PycharmProjects/AdainNet/AdainMattingNet/Dataset'
trimap_dir = '/trimap'
input_dir = '/input'
gt_dir = '/gt'

model_path = 'model/'
loader = loader.ImageLoader(base_dir+input_dir, base_dir+gt_dir)


if __name__ == '__main__':
    #~~
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
                print_img(cat_img,'%depoch_idx%d.jpg'%(i,idx))
        dict = {}
        dict['model'] = decoder.state_dict()
        dict['optim'] = optimizer.state_dict()
        torch.save(dict,'%smodel%d.pt'%(model_path,i))

