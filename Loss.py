import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def L1Loss(x1, x2):
    return torch.mean(torch.abs(x1 - x2))

def L2loss(x1, x2):
    return torch.mean((x1 -x2)**2)

#######################################
###########probability_Loss############
#######################################
def probability(gt, predict):
    l1_result = torch.abs(predict-gt)
    l2_result = (gt - predict)**2
    mask_1 = (gt == -1) | (gt == 1)
    mask_2 = (gt < 1) & (gt > -1)
    result = torch.masked_select(l1_result, mask_2).mean() + torch.masked_select(l2_result, mask_1).mean()
    return result.mean()

#######################################
###########gradient Loss###############
#######################################
#calc gradient
def gradient(x):
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
    dx, dy = right - left, bottom - top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    return dx, dy
#gradient loss
def gradientLoss(predict, gt):
    gradient_p = gradient(predict)
    gradient_gt = gradient(gt)
    return L1Loss(gradient_p[0], gradient_gt[0]) + L1Loss(gradient_p[1], gradient_gt[1])

#######################################
###########cross entropy Loss##########
#######################################
