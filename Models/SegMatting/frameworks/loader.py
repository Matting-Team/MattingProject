from torch.utils.data import DataLoader
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import torch.nn as nn

from Utils.ImageUtil import ToRandomRotationAndCrop

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])
to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def horizontal_flip(tensor):
    return torch.flip(tensor, [0, 2])

def vertical_flip(tensor):
    return torch.flip(tensor, [0, 1])

"""
TorchLoader: Augmentation is performed in the same way as reading the image and gt.
Parameter: yaml config, validation set, whether augmentation is in progress
 - config: yaml file saved as opener class --> dict type
 - validation: Whether to validate. The file read path is changed.
 - aug: Whether augmentation or not
############################################################################
Input:
 - index: file index within length
"""
class TorchLodader(DataLoader):
    def __init__(self, data_config, validation = False, aug=False):
        self.input_type = data_config['INPUT_TYPE']
        self.sub_input = data_config['SUB_INPUT']
        if not validation:
            self.input_path = data_config['INPUT_PATH']
            self.gt_path = data_config['GT_PATH']
        else:
            self.input_path = data_config['Valid_Path']
            self.gt_path = data_config['GT_valid']

        self.hint_path = data_config['HINT_PATH']

        self.input_w = data_config['INPUT_W']
        self.input_h = data_config['INPUT_H']

        self.crop_shape = data_config['CropShape']

        self.augmentation = aug
        self.reshaper = nn.Upsample(self.crop_shape, align_corners=False)

        self.input_list = os.listdir(self.input_path)
        self.gt_list = os.listdir(self.gt_path)
        self.hint_list = os.listdir(self.hint_path) if self.sub_input != 0 else None


    def __len__(self):
        return len(self.input_list)
    """
    safe_crop: Safe cropping that doesn't go out of range
    #####################################################
    Input
     - img: 3 channel RGB Image
     - gt: 1 channel Grayscale Alphamap
     - size: crop size
    """
    def safe_crop(self, img, gt, size):
        h, w, c = img.shape
        if h<=size[0] or w<=size[1]:#resize
            img = img.resize(size)
            gt = gt.resize(size)
            img = np.array(img)[..., :3]
            gt = np.array(gt)
        else:
            img = np.array(img)[..., :3]
            gt = np.array(gt)
            startPoint_x = random.randrange(0,w-size[1])
            startPoint_y = random.randrange(0, h-size[0])
            img = img[startPoint_y:startPoint_y+size[0], startPoint_x:startPoint_x+size[1]]
            gt = gt[startPoint_y:startPoint_y + size[0], startPoint_x:startPoint_x + size[1]]
        self.noise_intensity=0.005
        return img, gt

    def __getitem__(self, idx):
        input = Image.open(os.path.join(self.input_path, self.input_list[idx]))
        gt = Image.open(os.path.join(self.gt_path, self.gt_list[idx])).convert('L')

        if self.augmentation:
            if input.size[0] > self.crop_shape[0] and input.size[1] > self.crop_shape[1]:
                cropper = ToRandomRotationAndCrop(45, self.crop_shape)
                cropper.set()
                input, gt = augmentation_img(input, gt, cropper)

            else:
                input = input.resize(self.crop_shape)
                gt = gt.resize(self.crop_shape)

        input = transform(input)
        gt = to_tensor(gt)
        if self.augmentation:
            flip_h = random.randint(0, 1)
            flip_w = random.randint(0, 1)
            input, gt = augmentation_tensor(input, gt, flip_h, flip_w)
        else:
            reshaper = nn.Upsample([512, 512])
            input = reshaper(input.unsqueeze(0)).squeeze(0)
            gt = reshaper(gt.unsqueeze(0)).squeeze(0)
        return input, gt

# augmentation (ToRandomRotate & Cropper) Apply
# input and gt must be given the same value
def augmentation_img(input, gt, cropper):
    input = cropper.apply(input)
    gt = cropper.apply(gt)

    return input, gt

# There are augmentations in the tensor state, horizon and vertical flip.
def augmentation_tensor(input, gt, flip_h, flip_w):
    if flip_h == 1:
        input = horizontal_flip(input)
        gt = horizontal_flip(gt)
    if flip_w == 1:
        input = vertical_flip(input)
        gt = vertical_flip(gt)
    return input, gt