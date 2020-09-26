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
def probability(predict, gt):
    l1_result = torch.abs(predict-gt)
    l2_result = (gt - predict)**2
    mask_1 = (predict == 0) | (predict == 1)
    mask_2 = (predict < 1) & (predict > -0)
    result = torch.masked_select(l1_result, mask_2).mean()
    l2 = torch.masked_select(l2_result, mask_1).mean()
    if not torch.isnan(l2):
        result = result+l2
    return result

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


#######################################
###########p_loss_fusion net###########
#######################################
def fusionLoss(predict, gt):
    l1_result = torch.abs(predict-gt)
    l1_weighted = l1_result*0.1
    loss = torch.where((gt>0 )& (gt<1), l1_result, l1_weighted)
    return loss.mean()