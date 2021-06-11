import torch
from Utils.Loss import SSIM
import torch.nn as nn
import torch.nn.functional as f
import tqdm
from Utils.ImageUtil import psudo_detail

# This function trains a specific epoch.
# Learning is performed using GAN, and generators, discriminators, and optimizers are received as inputs.
#############################################################################################
def trainer(epoch, generator, dataset, discriminator, optimizer_g, optimizer_d, device):
    generator.train()
    discriminator.train()
    loss_module = loss_estimator()
    current_epoch_loss = 0.0
    print("{} epoch train start...".format(epoch))
    for data in tqdm.tqdm(dataset):
        optimizer_g.zero_grad()

        real_label = torch.ones((data[1].size(0), 1, 8, 8)).to(device)
        fake_label = torch.zeros((data[1].size(0), 1, 8, 8)).to(device)

        input_data, gt = data
        input_data = input_data.to(device)
        gt = gt.to(device)

        alpha = generator(input_data)
        loss = loss_module(gt, alpha)
        g_loss = gan_loss_estimator(discriminator, gt, alpha, input_data, real_label, fake_label, mode="g")
        total_g_loss = loss + g_loss*0.0001
        total_g_loss.backward()
        optimizer_g.step()
        with torch.no_grad():
            current_epoch_loss += g_loss
        #######################################################
        optimizer_d.zero_grad()
        d_loss = gan_loss_estimator(discriminator, gt, alpha.detach(), input_data, real_label, fake_label, mode="d")
        d_loss.backward()
        optimizer_d.step()
    print("Current epoch is {} epoch loss is {}".format(epoch, current_epoch_loss/len(dataset)))

# This function performs validation on the validation set.
# However, discriminator loss that may be flexible is not included in this validation.
###################################################################
def validation(epoch, generator, dataset, device):
    generator.eval()
    loss_module = loss_estimator()
    current_epoch_loss = 0.0
    print("{} epoch validation start...".format(epoch))
    for data in tqdm.tqdm(dataset):

        input_data, gt = data
        input_data = input_data.to(device)
        gt = gt.to(device)

        alpha = generator(input_data)
        loss = loss_module(gt, alpha)
        current_epoch_loss += loss

    print("Current epoch is {} epoch loss is {}".format(epoch, current_epoch_loss / len(dataset)))


def evaluation_testset(epoch, generator, dataset, device):
    generator.eval()
    mse = nn.MSELoss()

    current_epoch_loss = 0.0
    print("{} epoch validation start...".format(epoch))
    for data in tqdm.tqdm(dataset):

        input_data, gt = data
        input_data = input_data.to(device)
        gt = gt.to(device)

        alpha = generator(input_data)

        current_epoch_loss += loss

    print("Current epoch is {} epoch loss is {}".format(epoch, current_epoch_loss / len(dataset)))


# This function perform training without GAN
###################################################################
def trainer_without_gan(epoch, generator, dataset, optimizer_g, device):
    generator.train()
    loss_module = loss_estimator()
    current_epoch_loss = 0.0
    print("{} epoch train start...".format(epoch))
    for data in tqdm.tqdm(dataset):

        optimizer_g.zero_grad()
        input_data, gt = data
        input_data = input_data.to(device)
        gt = gt.to(device)

        alpha = generator(input_data)
        loss = loss_module(gt, alpha)
        loss.backward()
        optimizer_g.step()

        with torch.no_grad():
            current_epoch_loss += loss
    print("Current epoch is {} epoch loss is {}".format(epoch, current_epoch_loss / len(dataset)))



class loss_estimator(nn.Module):
    def __init__(self):
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM(window_size=11)

    def forward(self, gt, predict, hint=False):
        l1 = self.l1_loss(predict, gt)
        ssim = 1-self.ssim_loss(predict, gt)
        loss = l1+ssim
        if hint:
            psudo_p = psudo_detail(predict, predict.device)
            psudo_gt = psudo_detail(gt, gt.device)
            loss += self.l1_loss(psudo_p, psudo_gt)
        return loss


def gan_loss_estimator(discriminator, gt, predict, image, real_label, fake_label, mode="g"):
    real_image = gt * image
    fake_image = predict * image
    if mode=="g":
        fake_d_label = discriminator(fake_image,)
        loss = f.mse_loss(fake_d_label, real_label)
    elif mode=="d":
        real_d_label = discriminator(real_image)
        fake_d_label = discriminator(fake_image)
        loss_r = f.mse_loss(real_d_label, real_label)
        loss_f = f.mse_loss(fake_d_label, fake_label)
        loss = (loss_r + loss_f)/2

    return loss