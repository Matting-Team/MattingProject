import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2 as cv
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from scipy.ndimage import grey_dilation, grey_erosion
from frameworks.Reader import Opener
from frameworks.Loader import TorchLodader
from models.network import CamNet, Discriminator
from frameworks.Utils import GaussianBlurLayer, print_tensor, valid_path, write_text, save_tensor

blurer = GaussianBlurLayer(1, 3)

yaml_path = "config.yaml"
opener = Opener(yaml_path)

config = opener.conf

def get_trimap(gt, device):
    b, c, h, w = gt.shape
    gt = gt.data.cpu().numpy()
    boundaries = []
    for sdx in range(0, b):
        alpha = np.transpose(gt[sdx], (1, 2, 0))
        k_size = 10
        iterations = 10#np.random.randint(1, 20)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv.dilate(alpha, kernel, iterations)
        eroded = cv.erode(alpha, kernel, iterations)

        trimap = np.zeros(alpha.shape)
        trimap.fill(0.5)
        trimap[eroded >= 1.0] = 1.0
        trimap[dilated <= 0] = 0
        trimap = torch.from_numpy(trimap).permute(2, 0, 1).unsqueeze(0)
        boundaries.append(trimap)
    boundaries = torch.cat(boundaries, dim=0)
    boundaries = torch.tensor(boundaries).float().to(device)
    return boundaries



def Training(config):
    train_config = config['Training']
    load_config = config['Loader']
    network_config = config['Network']

    pretrained = train_config['Pretrained']
    crop_shape = load_config['CropShape']
    epoch = train_config['EPOCH']
    pt_path = train_config['PtPath']
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    loader = TorchLodader(load_config, aug=True)
    batch_loader = DataLoader(loader, batch_size=train_config['BATCH_SIZE'], shuffle=True, drop_last=True)

    network = CamNet(input_c=network_config['INPUT_CHANNEL'], base_c = network_config['BASIC_CHANNEL']).to(device)
    discriminator = Discriminator(input_c=4).to(device)
    valid_path(pt_path)

    state_dict = {}

    blurer.to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.0002, betas=(0.5, 0.999))#optim.SGD(network.parameters(), lr=train_config['LR'], momentum=0.9)#optim.Adam(network.parameters(), lr=train_config['LR'])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.MSELoss()
    l1_loss = nn.L1Loss()
    gan_criterion = nn.BCELoss()
    base_loss = 0.0
    start_point = 0

    start_point = 0
    if pretrained:
        state_dict = torch.load(pt_path)
        network.load_state_dict(state_dict['model'])
        discriminator.load_state_dict(state_dict['discriminator'])
        start_point = state_dict['e']
        optimizer.load_state_dict(state_dict['optimizer'])


    for e in range(start_point, epoch):
        output_path = train_config['OUTPUT_PATH'] + "/epoch%d" % (e)
        valid_path(output_path)
        for idx, data in enumerate(batch_loader):

            # G Train
            real_label = torch.ones((data[1].size(0), 1, 8, 8)).to(device)
            fake_label = torch.zeros((data[1].size(0), 1, 8, 8)).to(device)

            img, gt= data
            img = img.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            sudo_trimap = get_trimap(gt, device)
            boundaries = (sudo_trimap < 0.5) + (sudo_trimap > 0.5)
            segment_gt = F.interpolate(gt, scale_factor=1/16, mode="bilinear")
            segment_gt = blurer(segment_gt)

            segmentation, detail, alpha = network(img)

            fake_img = torch.cat([alpha, img], dim=1)
            real_img = torch.cat([gt, img], dim=1)

            segment_loss = criterion(segmentation, segment_gt)*10

            boundary = torch.where(boundaries, sudo_trimap, detail)
            boundary_gt = torch.where(boundaries, sudo_trimap, alpha)
            detail_loss = l1_loss(boundary, boundary_gt) * 10

            boundary_alpha = torch.where(boundaries, sudo_trimap, alpha)
            alpha_l1_loss = F.l1_loss(alpha, gt) + (4.0 * F.l1_loss(boundary, alpha))
            alpha_comp_loss = F.l1_loss(img * alpha, img * gt) + 4.0 * F.l1_loss(img * boundary_alpha, img * gt)
            alpha_loss = torch.mean(alpha_l1_loss + alpha_comp_loss)

            gan_loss = gan_criterion(discriminator(fake_img), real_label)*0.001

            loss = segment_loss + detail_loss + alpha_loss + gan_loss
            base_loss+=loss.item()
            loss.backward()
            optimizer.step()
            # D Train
            optimizer_d.zero_grad()
            real_loss = gan_criterion(discriminator(real_img), real_label)
            fake_loss = gan_criterion(discriminator(fake_img.detach()), fake_label)
            d_loss = (real_loss + fake_loss)/2
            d_loss.backward()
            optimizer_d.step()

            if idx%100 == 99:
                print("Current Process...")
                print("Current Epoch%d --- %d/%d"%(e, idx+1, len(batch_loader)))
                print("Base loss is...%f"%(base_loss/100))

        state_dict['model'] = network.state_dict()
        state_dict['e'] = e
        state_dict['optimizer'] = optimizer.state_dict()
        torch.save(state_dict, "%s/%depoch.pt" % (pt_path, e))
        Validation(e, network, config)




def Validation(e, network, config):
    print("Validation")
    train_config = config['Training']
    load_config = config['Loader']
    network_config = config['Network']
    output_path = train_config['OUTPUT_PATH']+"/epoch%d"%(e)
    log_path = train_config['LogPath']
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    loader = TorchLodader(load_config, validation=True, aug=False)
    batch_loader = DataLoader(loader, batch_size=1, shuffle=False)

    valid_path(log_path)
    valid_path(output_path)

    blurer.to(device)
    criterion = nn.MSELoss()
    l1_loss = nn.L1Loss()

    total_segment = 0.0
    total_alpha = 0.0
    total_detail = 0.0
    img_length = len(batch_loader)



    for idx, data in enumerate(batch_loader):

        img, gt = data
        img = img.to(device)
        gt = gt.to(device)
        shape = img.shape[2:]
        reshaper = nn.Upsample([(shape[0]//128) * 128, (shape[1]//128) * 128])
        img = reshaper(img)
        gt = reshaper(gt)

        sudo_trimap = get_trimap(gt, device)
        boundaries = (sudo_trimap < 0.5) + (sudo_trimap > 0.5)

        segment_gt = F.interpolate(gt, scale_factor=1 / 16, mode="bilinear")
        segment_gt = blurer(segment_gt)

        segmentation, detail, alpha = network(img)

        segment_loss = criterion(segmentation, segment_gt) * 10

        boundary = torch.where(boundaries, sudo_trimap, detail)
        boundary_gt = torch.where(boundaries, sudo_trimap, alpha)
        detail_loss = l1_loss(boundary, boundary_gt) * 10

        boundary_alpha = torch.where(boundaries, sudo_trimap, alpha)
        alpha_l1_loss = F.l1_loss(alpha, gt) + (4.0 * F.l1_loss(boundary, alpha))
        alpha_comp_loss = F.l1_loss(img * alpha, img * gt) + 4.0 * F.l1_loss(img * boundary_alpha, img * gt)
        alpha_loss = torch.mean(alpha_l1_loss + alpha_comp_loss)

        # 이미지 출력해주기 --->
        result_tensor = torch.cat([gt, alpha], dim=2)
        save_tensor(result_tensor, output_path, "epoch_%d_indes%d.png"%(e, idx))
        total_alpha = alpha_loss.item()
        total_detail = detail_loss.item()
        total_segment = segment_loss.item()

    print("--- Segment: %f --- Detail %f --- Total %f ---"%(total_segment/img_length, total_detail/img_length, total_alpha/img_length))
    write_text(log_path, "--- Segment: %f --- Detail %f --- Total %f ---"%(total_segment/img_length, total_detail/img_length, total_alpha/img_length))


reader = Opener("../configs/config.yaml")
conf = reader.conf
Training(conf)
